"""
step6_monte_carlo.py — Step 6: Monte Carlo career simulations

Fits a variance + correlation model from 21 years of historical panel data
(residuals from pooled-league-mean demeaning, calibrated to step-4 CV RMSE).
Simulates 40k careers per (FA-eligible player, candidate length 1–5) for all
non-refused Marcel projections.  Each simulated career draws:
  - Persistent talent shock (one draw/career, correlated across components)
  - Transient luck shock (new draw each year, uncorrelated across components)
  - Playing-time factor (log-normal per age, includes catastrophic-injury tail)
  - Incremental aging from step-2 curves for years 2+

Track A (value): park-neutral Marcel rates → Nationals Stadium re-adjustment →
    personal preference weights (infield defense ×1.20 for 2B/3B/SS; pitcher
    HR suppression ×1.15) → quadratic $/RAR curve from step 3.

Track B (market price): step-5 ridge model applied to Marcel-derived features.

Surplus = Track A cumulative value − Track B AAV × candidate length.

Inputs:
  intermediate/batting_neutral.csv
  intermediate/pitching_neutral.csv
  intermediate/fielding_raw.csv
  intermediate/aging_curves_smooth.csv
  intermediate/marcel_projections.csv
  intermediate/marcel_hyperparams.csv
  intermediate/linear_weights.csv
  intermediate/curve_coefficients.csv
  intermediate/market_model_coefficients.csv
  frostfire_data/players.csv
  frostfire_data/team_batting_YYYY.csv
  frostfire_data/team_pitching_YYYY.csv
  frostfire_data/ballparks.json

Outputs:
  intermediate/mc_variance_model.csv          — σ per component (total/persistent/transient)
  intermediate/mc_correlation_batting.csv     — batting component correlation matrix
  intermediate/mc_correlation_fielding.csv    — fielding component correlation matrix
  intermediate/mc_correlation_pitching.csv    — pitching component correlation matrix
  intermediate/mc_playtime_bat.csv            — batter PT log-normal params by age group
  intermediate/mc_playtime_pit.csv            — pitcher PT log-normal params by age group
  intermediate/mc_surplus_distributions.csv   — per (player_id, candidate_length) surplus stats
"""

import glob, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = Path("frostfire_data")
INT  = Path("intermediate")

RNG  = np.random.default_rng(seed=42)

N_SIMS        = 40_000
L_CANDIDATES  = [1, 2, 3, 4, 5]
PERSIST_FRAC  = 0.50        # 50% of Marcel RMSE variance is persistent (true talent)

# Survivorship bias correction: clip positive aging deltas at age >= this threshold.
# Players who survive to 34+ are positively selected; apparent improvement is bias, not signal.
SURVIVORSHIP_CLIP_AGE = 34

# Forecast horizon uncertainty: sigma grows by this fraction per additional year into projection.
# Accounts for the fact that Marcel accuracy degrades significantly beyond year 1.
HORIZON_SIGMA_EXPANSION = 0.20   # 20% more uncertain per year out (year 5 = 1.8× year-1 sigma)

# Maximum age at end of contract — prevents recommending a 5-year deal for a 42-year-old.
MAX_CONTRACT_END_AGE = 44

# Performance cliff: age-specific probability per year that a player suddenly can't
# contribute (catastrophic injury, sudden decline, retirement). Once triggered, all
# subsequent years in that simulated career produce zero value.
# These probabilities are not in the historical panel (survivorship filter eliminates
# cliff years), so they are set as informed priors.
CLIFF_PROBS = [(35, 38, 0.04), (38, 41, 0.08), (41, 99, 0.14)]

# Luxury tax: progressive AAV penalty reflecting salary-cap pressure.
# Tiers: $35-50M at 30% surcharge, $50M+ at 60% surcharge.
# Effective cost = AAV + tier1_excess * 0.30 + tier2_excess * 0.60
LUXURY_TAX_TIER1_M     = 35.0   # tax kicks in here at 30%
LUXURY_TAX_TIER2_M     = 50.0   # rate steepens here at 60%
LUXURY_TAX_RATE1       = 0.30
LUXURY_TAX_RATE2       = 0.60

VALID_TIDS    = {1,3,4,6,7,8,9,13,16,17,18,20,21,23,24,25,26,27,29,30,301,302}
POS_ADJ       = {2:12.5, 3:-12.5, 4:2.5, 5:2.5, 6:7.5, 7:-7.5, 8:2.5, 9:-7.5, 10:-17.5}
REPL_PER_600PA = 20.0
REPL_PER_162IP = 27.0

# Market-track retro feature params (must match step 5 training methodology exactly)
GAMMA_MKT   = 0.9   # same GAMMA as step 5 retro_bat / retro_pit
L_RETRO_MKT = 3     # same L_RETRO as step 5
MIN_PA        = 100
MIN_BF        = 100
MIN_IP_FLD    = 50
CURRENT_YEAR  = 2035
PROJ_YEAR     = 2036
SALARY_FLOOR_M = 0.75       # replacement salary in $M

# Personal preference weights (Track A only)
PREF_INDEF_POS   = {4, 5, 6}   # 2B, 3B, SS
PREF_INDEF_MULT  = 1.20
PREF_HRASUP_MULT = 1.15

# Components
BAT_COMPS = ["hr_pa", "xbh_pa", "single_pa", "bb_pa", "k_pa", "ubr_g", "hp_pa"]
FLD_COMPS = ["zr_rate", "arm_rate", "framing_rate"]
PIT_COMPS = ["k_bf", "bb_hbp_bf", "hra_bf", "ha_n_bf"]

# Physical upper limits for clipping simulated rates
BAT_CLIP_MAX = np.array([0.15, 0.22, 0.38, 0.22, 0.45, 0.35, 0.04])  # per PA
PIT_CLIP_MAX = np.array([0.50, 0.28, 0.12, 0.45])                      # per BF

# Extension player IDs (excluded from FA market)
EXT_PLAYER_IDS = {535,23192,25566,29390,29759,31187,35118,37118,39658,40316,
                  40437,40770,41021,41778,43209,43318}

print("=" * 70)
print("STEP 6: Monte Carlo career simulations")
print("=" * 70)

# ─── 1. LOAD PLAYERS ─────────────────────────────────────────────────────────
print("\n[1] Loading players...")
players = pd.read_csv(DATA / "players.csv",
                      usecols=["ID","First Name","Last Name","date_of_birth","bats","Pos","Role",
                                "mlb_service_years","is_active","Level","Retired"])
players.rename(columns={"ID":"player_id","Pos":"primary_pos"}, inplace=True)
players["birth_year"] = pd.to_datetime(players["date_of_birth"], errors="coerce").dt.year
players["age_2035"]   = CURRENT_YEAR - players["birth_year"]
bio = players.set_index("player_id")
print(f"  {len(players):,} total players loaded")

# ─── 2. LOAD PANELS ──────────────────────────────────────────────────────────
print("\n[2] Loading historical panels...")

# Batting (split_id=1, aggregate stints)
_bc = ["player_id","year","split_id","pa","g","hr_n","d_n","t_n",
       "singles_n","bb","hp","k","ubr"]
bat_raw = pd.read_csv(INT / "batting_neutral.csv", usecols=_bc)
bat_raw = bat_raw[bat_raw["split_id"] == 1]
bat_py = bat_raw.groupby(["player_id","year"]).agg(
    pa=("pa","sum"), g=("g","sum"),
    hr_n=("hr_n","sum"), d_n=("d_n","sum"), t_n=("t_n","sum"),
    singles_n=("singles_n","sum"), bb=("bb","sum"),
    hp=("hp","sum"), k=("k","sum"), ubr=("ubr","sum"),
).reset_index()
bat_py = bat_py[bat_py["pa"] >= MIN_PA].copy()
bat_py["hr_pa"]     = bat_py["hr_n"]                        / bat_py["pa"]
bat_py["xbh_pa"]    = (bat_py["d_n"] + bat_py["t_n"])       / bat_py["pa"]
bat_py["single_pa"] = bat_py["singles_n"]                    / bat_py["pa"]
bat_py["bb_pa"]     = bat_py["bb"]                           / bat_py["pa"]
bat_py["k_pa"]      = bat_py["k"]                            / bat_py["pa"]
bat_py["hp_pa"]     = bat_py["hp"]                           / bat_py["pa"]
bat_py["ubr_g"]     = bat_py["ubr"]                          / bat_py["g"].clip(lower=1)
bat_py["d_frac"]    = bat_py["d_n"] / (bat_py["d_n"] + bat_py["t_n"]).clip(lower=1e-9)
print(f"  Batting player-years (pa>={MIN_PA}): {len(bat_py):,}")

# Pitching
_pc = ["player_id","year","team_id","split_id","bf","outs","g","gs",
       "k","bb","hp","ha_n","hra_n"]
pit_raw = pd.read_csv(INT / "pitching_neutral.csv", usecols=_pc)
pit_raw = pit_raw[(pit_raw["split_id"] == 1) & (pit_raw["team_id"].isin(VALID_TIDS))]
pit_py = pit_raw.groupby(["player_id","year"]).agg(
    bf=("bf","sum"), outs=("outs","sum"),
    g=("g","sum"), gs=("gs","sum"),
    k=("k","sum"), bb=("bb","sum"), hp=("hp","sum"),
    ha_n=("ha_n","sum"), hra_n=("hra_n","sum"),
).reset_index()
pit_py["ip"]        = pit_py["outs"] / 3
pit_py = pit_py[pit_py["bf"] >= MIN_BF].copy()
pit_py["k_bf"]      = pit_py["k"]                    / pit_py["bf"]
pit_py["bb_hbp_bf"] = (pit_py["bb"] + pit_py["hp"]) / pit_py["bf"]
pit_py["hra_bf"]    = pit_py["hra_n"]                 / pit_py["bf"]
pit_py["ha_n_bf"]   = pit_py["ha_n"]                  / pit_py["bf"]
pit_py["bb_frac"]   = pit_py["bb"] / (pit_py["bb"] + pit_py["hp"]).clip(lower=1e-9)
pit_py["sp_flag"]   = (pit_py["gs"] / pit_py["g"].clip(lower=1)) >= 0.5
pit_py["role"]      = pit_py["sp_flag"].map({True:"SP", False:"RP"})
print(f"  Pitching player-years (bf>={MIN_BF}): {len(pit_py):,}")

# Lookup dict for retro market-track feature computation (matches step 5 methodology)
pit_py_by_pid = {int(pid): grp for pid, grp in pit_py.groupby("player_id")}

# Fielding
fld_raw = pd.read_csv(INT / "fielding_raw.csv",
                      usecols=["player_id","year","position","ip","zr","arm","framing"])
fld_raw = fld_raw[fld_raw["ip"] >= 1]
fld_py = fld_raw.groupby(["player_id","year"]).agg(
    ip=("ip","sum"), zr=("zr","sum"),
    arm=("arm","sum"), framing=("framing","sum"),
).reset_index()
fld_py = fld_py[fld_py["ip"] >= MIN_IP_FLD].copy()
fld_py["zr_rate"]      = fld_py["zr"]      / fld_py["ip"] * 1000
fld_py["arm_rate"]     = fld_py["arm"]     / fld_py["ip"] * 1000
fld_py["framing_rate"] = fld_py["framing"] / fld_py["ip"] * 1000
print(f"  Fielding player-years (ip>={MIN_IP_FLD}): {len(fld_py):,}")

# ─── 3. LEAGUE AVERAGES ───────────────────────────────────────────────────────
print("\n[3] Computing pooled league averages...")

tb_files = sorted(f for f in glob.glob(str(DATA / "team_batting_*.csv"))
                  if "vsL" not in f and "vsR" not in f)
_rows = []
for f in tb_files:
    df = pd.read_csv(f)
    _rows.append(df[(df["tid"].isin(VALID_TIDS)) & (df["split_id"] == 1)])
team_bat = pd.concat(_rows, ignore_index=True)
team_bat = team_bat[team_bat["pa"] > 0]
_lg = team_bat[["s","d","t","hr","bb","hp","k","pa"]].sum()
LG = {col: float(_lg[col] / _lg["pa"]) for col in ["s","d","t","hr","bb","hp","k"]}
print(f"  LG bat: s={LG['s']:.4f} d={LG['d']:.4f} t={LG['t']:.4f} "
      f"hr={LG['hr']:.4f} bb={LG['bb']:.4f} k={LG['k']:.4f}")

tp_files = sorted(f for f in glob.glob(str(DATA / "team_pitching_*.csv"))
                  if "vsL" not in f and "vsR" not in f)
_rows = []
for f in tp_files:
    df = pd.read_csv(f)
    _rows.append(df[(df["tid"].isin(VALID_TIDS)) & (df["split_id"] == 1)])
team_pit = pd.concat(_rows, ignore_index=True)
team_pit = team_pit[team_pit["bf"] > 0]
_lp = team_pit[["ha","hra","bb","hp","k","bf"]].sum()
LG_PIT = {col: float(_lp[col] / _lp["bf"]) for col in ["ha","hra","bb","hp","k"]}
print(f"  LG pit: ha={LG_PIT['ha']:.4f} hra={LG_PIT['hra']:.4f} "
      f"bb={LG_PIT['bb']:.4f} k={LG_PIT['k']:.4f}")

# ─── 4. LINEAR WEIGHTS & CURVE ────────────────────────────────────────────────
print("\n[4] Loading linear weights and dollar curve...")

lw_df  = pd.read_csv(INT / "linear_weights.csv")
lw_bat = dict(zip(lw_df[lw_df["side"]=="batting"]["component"],
                   lw_df[lw_df["side"]=="batting"]["weight"]))
lw_pit = dict(zip(lw_df[lw_df["side"]=="pitching"]["component"],
                   lw_df[lw_df["side"]=="pitching"]["weight"]))
lg_ra_per_bf = (lw_pit["ha"] * LG_PIT["ha"] +
                lw_pit["hra_extra"] * LG_PIT["hra"] +
                lw_pit["bb"] * LG_PIT["bb"] +
                lw_pit["hbp"] * LG_PIT["hp"] +
                lw_pit["k"] * LG_PIT["k"] +
                lw_pit["intercept"])
print(f"  lg_ra_per_bf = {lg_ra_per_bf:.5f}")

crv   = pd.read_csv(INT / "curve_coefficients.csv").iloc[0]
C_INT = float(crv["intercept_M"])
C_B   = float(crv["b_M_per_run"])
C_C   = float(crv["c_M_per_run2"])
# Pitcher Track A discount: partial correction for pooled-curve over-estimation.
# Step-3 residual shows SPs overpredicted by ~$4M, but step-5 market regression already
# captures ~half that gap (it's a pitcher-aware ridge model). Net residual ~$2M.
# Full -$4M eliminates all pitcher recommendations; -$2M gives ~12 positive pitchers.
PITCHER_TRACK_A_DISCOUNT_M = -2.0
print(f"  Curve: {C_INT:.3f} + {C_B:.4f}·RAR + {C_C:.6f}·RAR²  (pitcher Track A offset {PITCHER_TRACK_A_DISCOUNT_M:+.1f}M)")


def rar_to_value_M(rar, is_pitcher=False):
    """Convert RAR to dollar value in $M. is_pitcher applies the pooling-bias correction."""
    base = C_INT + C_B * rar + C_C * rar**2
    if is_pitcher:
        base = base + PITCHER_TRACK_A_DISCOUNT_M
    return np.clip(base, SALARY_FLOOR_M, None)

# ─── 5. MARKET MODEL (TRACK B) ───────────────────────────────────────────────
print("\n[5] Loading step-5 market model...")

mc_coef = pd.read_csv(INT / "market_model_coefficients.csv")


def _load_ridge(player_type, target):
    """Extract a fitted ridge model's features, coefficients, scaler params, and intercept from the market_model_coefficients table for the given player_type/target."""
    sub = mc_coef[(mc_coef["player_type"] == player_type) &
                  (mc_coef["target"] == target)]
    feat_rows = sub[(sub["feature"] != "intercept") &
                    sub["scaler_mean"].notna()].copy()
    return {
        "features":   feat_rows["feature"].tolist(),
        "coef":       feat_rows["coef_scaled"].values,
        "means":      feat_rows["scaler_mean"].values,
        "stds":       feat_rows["scaler_std"].values,
        "intercept":  float(sub[sub["feature"] == "intercept"]["coef_scaled"].iloc[0]),
    }


bat_aav_mdl = _load_ridge("batter",  "log_aav")
bat_yrs_mdl = _load_ridge("batter",  "years")
pit_aav_mdl = _load_ridge("pitcher", "log_aav")
pit_yrs_mdl = _load_ridge("pitcher", "years")


def ridge_pred(mdl, x_raw):
    """Apply a loaded ridge model's scaler and coefficients to a raw feature vector, returning the predicted value."""
    x_sc = (np.asarray(x_raw, float) - mdl["means"]) / mdl["stds"]
    return float(x_sc @ mdl["coef"] + mdl["intercept"])

# ─── 6. AGING CURVES ─────────────────────────────────────────────────────────
print("\n[6] Loading aging curves...")
aging_df = pd.read_csv(INT / "aging_curves_smooth.csv",
                        usecols=["group_label","component","age",
                                 "loess50_delta","poly3_delta"])
aging_df["delta"] = aging_df["loess50_delta"].fillna(aging_df["poly3_delta"]).fillna(0.0)
AGING_LKP = {
    (str(r.group_label), str(r.component), int(r.age)): float(r.delta)
    for r in aging_df.itertuples()
}
print(f"  {len(AGING_LKP):,} aging curve entries")


def _aging_delta(group_label, comp, age):
    """Look up the one-year aging delta for (group_label, comp, age), clamping positive deltas to 0 at/above the survivorship-bias clip age."""
    a = int(np.clip(age, 20, 40))
    delta = AGING_LKP.get((str(group_label), str(comp), a), 0.0)
    # Survivorship bias correction: players who survive to SURVIVORSHIP_CLIP_AGE+ are
    # positively selected, making apparent improvement an artifact of selection, not skill.
    # Cap positive deltas to 0 (project flat rather than continued improvement).
    if a >= SURVIVORSHIP_CLIP_AGE and delta > 0:
        return 0.0
    return delta


def _effective_aav(nominal_aav_M):
    """Apply progressive luxury tax to AAV; returns effective cost for surplus calculation."""
    excess1 = max(0.0, nominal_aav_M - LUXURY_TAX_TIER1_M)
    excess2 = max(0.0, nominal_aav_M - LUXURY_TAX_TIER2_M)
    return nominal_aav_M + excess1 * LUXURY_TAX_RATE1 + excess2 * LUXURY_TAX_RATE2


def _cliff_prob(age):
    """Per-year probability that a player's career ends due to catastrophic injury/decline."""
    for lo, hi, p in CLIFF_PROBS:
        if lo <= age < hi:
            return p
    return 0.0


def aging_cumulative(group_label, comps, age_proj, n_years):
    """
    Returns array (n_comp, n_years).
    Index [c, y] = cumulative additional aging from projection year to year y.
    Year 0 = 0 (Marcel already applied 1 year of aging into the projection).
    Year 1 = delta(age_proj), year 2 = delta(age_proj)+delta(age_proj+1), etc.
    """
    n_comp = len(comps)
    result = np.zeros((n_comp, n_years))
    for ci, c in enumerate(comps):
        cum = 0.0
        for y in range(1, n_years):
            cum += _aging_delta(group_label, c, age_proj + y - 1)
            result[ci, y] = cum
    return result

# ─── 7. VARIANCE MODEL ───────────────────────────────────────────────────────
print("\n[7] Fitting variance model...")

hp_df = pd.read_csv(INT / "marcel_hyperparams.csv")
RMSE  = dict(zip(hp_df["component"], hp_df["rmse"]))
print("  CV RMSE per component (sigma_total):")
for c, r in RMSE.items():
    print(f"    {c:20s}: {r:.6f}")


def _make_psd(M):
    """Force matrix M to be positive semi-definite by clipping negative eigenvalues, then renormalize to a correlation matrix if applicable."""
    eig, vec = np.linalg.eigh(M)
    eig = np.clip(eig, 1e-12, None)
    M2  = vec @ np.diag(eig) @ vec.T
    # Re-normalize to correlation matrix if it was a corr matrix
    d = np.sqrt(np.diag(M2))
    if np.all(d > 0):
        M2 = M2 / np.outer(d, d)
        np.fill_diagonal(M2, 1.0)
    return M2


def build_corr_and_sigmas(panel, comp_cols):
    """
    Compute correlation matrix from pooled-league-mean-subtracted rates.
    Returns (corr_psd, {comp: sigma_total}).
    """
    lg_means = {c: float(panel[c].mean()) for c in comp_cols}
    res = panel[comp_cols].copy().sub(pd.Series(lg_means))
    corr = res.corr().fillna(0.0).values.copy()
    np.fill_diagonal(corr, 1.0)
    return _make_psd(corr)


def build_covariance_matrices(comp_cols, corr, rmse_dict, persist_frac):
    """
    Sigma_total[i,j] = corr[i,j] * rmse_i * rmse_j
    Sigma_p  = persist_frac * Sigma_total   (correlated persistent shocks)
    Sigma_t  = (1-persist_frac) * diag(rmse^2)   (uncorrelated transient shocks)
    """
    sigmas = np.array([rmse_dict.get(c, 0.02) for c in comp_cols])
    Sigma_total = corr * np.outer(sigmas, sigmas)
    Sigma_p = persist_frac * Sigma_total
    Sigma_t = (1 - persist_frac) * np.diag(sigmas**2)

    def psd(M):
        eig, vec = np.linalg.eigh(M)
        eig = np.clip(eig, 1e-12, None)
        return vec @ np.diag(eig) @ vec.T

    return psd(Sigma_p), psd(Sigma_t)


# Batting
corr_bat = build_corr_and_sigmas(bat_py, BAT_COMPS)
Sigma_p_bat, Sigma_t_bat = build_covariance_matrices(BAT_COMPS, corr_bat, RMSE, PERSIST_FRAC)
print(f"\n  Batting corr matrix (top-left 3×3):\n"
      f"  {corr_bat[:3,:3].round(3)}")

# Fielding
corr_fld = build_corr_and_sigmas(fld_py, FLD_COMPS)
Sigma_p_fld, Sigma_t_fld = build_covariance_matrices(FLD_COMPS, corr_fld, RMSE, PERSIST_FRAC)

# Pitching
corr_pit = build_corr_and_sigmas(pit_py, PIT_COMPS)
Sigma_p_pit, Sigma_t_pit = build_covariance_matrices(PIT_COMPS, corr_pit, RMSE, PERSIST_FRAC)
print(f"  Pitching corr matrix:\n  {corr_pit.round(3)}")

# Save correlation matrices
for label, corr_m, comps in [
    ("batting",  corr_bat, BAT_COMPS),
    ("fielding", corr_fld, FLD_COMPS),
    ("pitching", corr_pit, PIT_COMPS),
]:
    pd.DataFrame(corr_m, index=comps, columns=comps).to_csv(
        INT / f"mc_correlation_{label}.csv")

# Variance summary
var_rows = []
for c in BAT_COMPS + FLD_COMPS + PIT_COMPS:
    sig = RMSE.get(c, 0.02)
    var_rows.append({"component": c, "sigma_total": sig,
                     "sigma_p": np.sqrt(PERSIST_FRAC) * sig,
                     "sigma_t": np.sqrt(1 - PERSIST_FRAC) * sig,
                     "persist_frac": PERSIST_FRAC})
pd.DataFrame(var_rows).to_csv(INT / "mc_variance_model.csv", index=False)
print("  Saved mc_variance_model.csv, mc_correlation_*.csv")

# ─── 8. PLAYING-TIME MODEL ────────────────────────────────────────────────────
print("\n[8] Fitting playing-time model...")


def fit_pt_model(panel, vol_col):
    """
    Year-over-year volume ratio model: ratio = actual_vol[T] / actual_vol[T-1].
    For each age group: fit log-normal + catastrophic injury probability.
    Catastrophic = ratio < 0.15 (severe injury / role loss).
    """
    birth_yr = bio["birth_year"].to_dict()
    panel = panel.copy()
    panel["birth_year"] = panel["player_id"].map(birth_yr)
    panel = panel.dropna(subset=["birth_year"])
    panel["age"] = (panel["year"] - panel["birth_year"]).astype(int)

    rows = []
    for pid, grp in panel.groupby("player_id"):
        grp = grp.sort_values("year")
        vols = grp[vol_col].values
        ages = grp["age"].values
        for i in range(1, len(grp)):
            if vols[i-1] >= MIN_PA * 0.5:   # qualifying prior year
                rows.append({"age": int(ages[i]),
                              "ratio": float(vols[i]) / max(float(vols[i-1]), 1.0)})

    ratio_df = pd.DataFrame(rows)

    age_bands = [(20, 25, "20-24"), (25, 28, "25-27"), (28, 31, "28-30"),
                 (31, 34, "31-33"), (34, 37, "34-36"), (37, 42, "37+")]
    model_rows = []
    for lo, hi, label in age_bands:
        sub = ratio_df[(ratio_df["age"] >= lo) & (ratio_df["age"] < hi)]
        n = len(sub)
        if n < 10:
            mu, sig, p_cat = 0.0, 0.25, 0.10
        else:
            p_cat = float((sub["ratio"] < 0.15).mean())
            good  = sub.loc[sub["ratio"] >= 0.15, "ratio"].clip(upper=3.0)
            if len(good) >= 5:
                log_r = np.log(good)
                mu, sig = float(log_r.mean()), float(log_r.std())
            else:
                mu, sig = 0.0, 0.25
        model_rows.append({
            "age_group": label, "age_lo": lo, "age_hi": hi,
            "mu_log": mu, "sigma_log": max(sig, 0.05),
            "p_catastrophic": p_cat, "n_pairs": n,
        })
    return pd.DataFrame(model_rows)


bat_pt = fit_pt_model(bat_py, "pa")
pit_pt = fit_pt_model(pit_py, "ip")
bat_pt.to_csv(INT / "mc_playtime_bat.csv", index=False)
pit_pt.to_csv(INT / "mc_playtime_pit.csv", index=False)
print("  Batter PT model:")
print(bat_pt[["age_group","mu_log","sigma_log","p_catastrophic","n_pairs"]].to_string(index=False))
print("  Pitcher PT model:")
print(pit_pt[["age_group","mu_log","sigma_log","p_catastrophic","n_pairs"]].to_string(index=False))


def _pt_params(model_df, age):
    """Look up (mu_log, sigma_log, p_catastrophic) for the age band containing age, falling back to the last row if no band matches."""
    age_c = int(np.clip(age, 20, 41))
    for _, r in model_df.iterrows():
        if r["age_lo"] <= age_c < r["age_hi"]:
            return float(r["mu_log"]), float(r["sigma_log"]), float(r["p_catastrophic"])
    r = model_df.iloc[-1]
    return float(r["mu_log"]), float(r["sigma_log"]), float(r["p_catastrophic"])

# ─── 9. NATIONALS PARK FACTORS ────────────────────────────────────────────────
print("\n[9] Loading Nationals Stadium park factors...")
with open(DATA / "ballparks.json") as f:
    bp_raw = json.load(f)
nat = next(p for p in bp_raw["ballparks"] if p["park_id"] == 32)

# Effective factor = (home_pf + 1) / 2  (half games at home, half at neutral road)
NAT = {
    "eff_avg_r": (nat["avg_r"] + 1) / 2,   # 0.985 vs RHP → RHB
    "eff_avg_l": (nat["avg_l"] + 1) / 2,   # 1.010 vs LHP → LHB
    "eff_avg":   (nat["avg"]   + 1) / 2,   # 0.9938 blended
    "eff_d":     (nat["d"]     + 1) / 2,   # 1.000
    "eff_t":     (nat["t"]     + 1) / 2,   # 0.960
    "eff_hr_r":  (nat["hr_r"]  + 1) / 2,   # 1.010 vs RHP
    "eff_hr_l":  (nat["hr_l"]  + 1) / 2,   # 1.000 vs LHP
    "eff_hr":    (nat["hr"]    + 1) / 2,   # 1.0065 blended
}
print(f"  eff_avg R/L: {NAT['eff_avg_r']:.4f}/{NAT['eff_avg_l']:.4f}  "
      f"eff_hr R/L: {NAT['eff_hr_r']:.4f}/{NAT['eff_hr_l']:.4f}  "
      f"eff_t: {NAT['eff_t']:.4f}")


def _bat_park_adj(bats):
    """Return (eff_avg, eff_hr) for Nationals Stadium given batter handedness."""
    if bats == 1:   # Right
        return NAT["eff_avg_r"], NAT["eff_hr_r"]
    if bats == 2:   # Left
        return NAT["eff_avg_l"], NAT["eff_hr_l"]
    # Switch: blend
    return (NAT["eff_avg_r"] + NAT["eff_avg_l"]) / 2, (NAT["eff_hr_r"] + NAT["eff_hr_l"]) / 2

# ─── 10. LOAD MARCEL PROJECTIONS ─────────────────────────────────────────────
print("\n[10] Loading Marcel projections...")
mp = pd.read_csv(INT / "marcel_projections.csv")
mp = mp.merge(bio[["bats"]].reset_index(), on="player_id", how="left")

fa = mp[(mp["mlb_service_years"] >= 5) &   # Frostfire: FA after 5 yrs
        (mp["is_active"] == 1) & (mp["Level"] == 1)].copy()

fa_bat = fa[(fa["Role"] == 0) & (~fa["refused_bat"])].copy()
fa_pit = fa[(fa["Role"].isin([11, 12, 13])) & (~fa["refused_pit"])].copy()

print(f"  FA-eligible batters (non-refused): {len(fa_bat)}")
print(f"  FA-eligible pitchers (non-refused): {len(fa_pit)}")

# Resolve primary position for batters (fld_pos > primary_pos as preference)
_POS_MAP = {2:"C",3:"1B",4:"2B",5:"3B",6:"SS",7:"LF",8:"CF",9:"RF",10:"DH"}


def _resolve_pos(row):
    """Resolve a batter's primary position, preferring fielding-derived position over the bio primary_pos, defaulting to 0 if neither is available."""
    fp = row.get("fld_pos")
    pp = row.get("primary_pos")
    try:
        if pd.notna(fp) and float(fp) > 0:
            return int(fp)
    except (ValueError, TypeError):
        pass
    try:
        if pd.notna(pp) and float(pp) > 0:
            return int(pp)
    except (ValueError, TypeError):
        pass
    return 0


fa_bat["resolved_pos"] = fa_bat.apply(_resolve_pos, axis=1)
fa_bat["resolved_pos"] = fa_bat["resolved_pos"].replace({10: 3})

# ─── 11. COMPUTE MARKET PRICES (TRACK B) ─────────────────────────────────────
print("\n[11] Computing Track B market prices...")


def _bat_raa(row):
    """Batting runs above average from Marcel projection."""
    pa = float(row.get("pa_proj") or 0)
    if pa < 10:
        return 0.0
    df   = float(row.get("d_frac") or 0.7)
    xbh  = float(row.get("xbh_pa_proj") or 0)
    d_r  = xbh * df
    t_r  = xbh * (1 - df)
    return pa * (
        lw_bat["single"] * (float(row.get("single_pa_proj") or 0) - LG["s"]) +
        lw_bat["double"] * (d_r - LG["d"]) +
        lw_bat["triple"] * (t_r - LG["t"]) +
        lw_bat["hr"]     * (float(row.get("hr_pa_proj") or 0) - LG["hr"]) +
        lw_bat["bb"]     * (float(row.get("bb_pa_proj") or 0) - LG["bb"]) +
        lw_bat["hbp"]    * (float(row.get("hp_pa_proj") or 0) - LG["hp"]) +
        lw_bat["k"]      * (float(row.get("k_pa_proj")  or 0) - LG["k"])
    )


def _pit_raa_and_rates(row):
    """Pitching RAA plus key rates from Marcel projection."""
    bf = float(row.get("bf_proj") or 0)
    if bf < 10:
        return 0.0, 0.0, 0.0
    k_r  = float(row.get("k_bf_proj") or 0)
    bh_r = float(row.get("bb_hbp_bf_proj") or 0)
    bf_f = float(row.get("bb_frac") or 0.85)
    bb_r = bh_r * bf_f
    hp_r = bh_r * (1 - bf_f)
    hr_r = float(row.get("hra_bf_proj") or 0)
    ha_r = float(row.get("ha_n_bf_proj") or 0)
    pit_ra = (lw_pit["ha"] * ha_r + lw_pit["hra_extra"] * hr_r +
              lw_pit["bb"] * bb_r + lw_pit["hbp"] * hp_r +
              lw_pit["k"]  * k_r  + lw_pit["intercept"])
    raa = (lg_ra_per_bf - pit_ra) * bf
    return raa, k_r, hr_r


def _bat_market_features(row, b_raa):
    """Build the batter feature dict (bat_raa, proj_ubr, proj_def, age, proj_pa, is_premium_def, proj_rar_sq, proj_rar) used by the step 5 market ridge model, matching its training-time def_runs definition (pos_adj folded into proj_def)."""
    pa   = float(row.get("pa_proj") or 0)
    ubr  = float(row.get("ubr_g_proj") or 0) * (pa / 4.0)
    pos  = int(row.get("resolved_pos") or 0)
    ip   = pa / 4.0 * 8.8
    zr   = float(row.get("zr_rate_proj") or 0) * ip / 1000
    arm  = float(row.get("arm_rate_proj") or 0) * ip / 1000
    frm  = float(row.get("framing_rate_proj") or 0) * ip / 1000
    # pos_adj folded into proj_def to match step 5 training, where def_runs = zr+arm+frm+pos_adj
    pos_adj_val = POS_ADJ.get(pos, 0) * (pa / 600)
    proj_def  = zr + arm + frm + pos_adj_val
    rep       = REPL_PER_600PA * pa / 600
    proj_rar  = b_raa + ubr + proj_def + rep
    is_prem   = int(pos in {2, 6})
    age       = float(row.get("age_proj") or 30)
    return {
        "bat_raa":        b_raa,
        "proj_ubr":       ubr,
        "proj_def":       proj_def,
        "age":            age,
        "proj_pa":        pa,
        "is_premium_def": is_prem,
        "proj_rar_sq":    proj_rar ** 2,
        "proj_rar":       proj_rar,
    }


def _retro_pit_mkt_features(pid):
    """
    Compute pitcher market-track features using the same 3-year gamma-weighted
    approach as step 5 retro_pit().  This keeps training/inference consistent:
    step 5 used actual-IP weighted averages (~65-70 IP for RPs), NOT Marcel's
    regression-to-mean IP projections (~55-58 IP), so inference must match.
    Returns a dict or None if insufficient panel data.
    """
    cutoff = CURRENT_YEAR
    lo     = cutoff - (L_RETRO_MKT - 1)

    grp = pit_py_by_pid.get(int(pid))
    if grp is None:
        return None
    s = grp[(grp["year"] >= lo) & (grp["year"] <= cutoff)].copy()
    if len(s) == 0:
        return None

    s["gap"] = cutoff - s["year"]
    s["w"]   = GAMMA_MKT ** s["gap"]
    W           = s["w"].sum()
    total_wbf   = (s["bf"] * s["w"]).sum()
    if total_wbf == 0:
        return None

    proj_ip = (s["ip"] * s["w"]).sum() / W
    proj_bf = total_wbf / W

    # Volume-and-recency weighted rates (counting columns already aggregated in pit_py)
    k_r   = (s["k"]     * s["w"]).sum() / total_wbf
    bb_r  = (s["bb"]    * s["w"]).sum() / total_wbf
    hp_r  = (s["hp"]    * s["w"]).sum() / total_wbf
    hr_r  = (s["hra_n"] * s["w"]).sum() / total_wbf
    ha_r  = (s["ha_n"]  * s["w"]).sum() / total_wbf

    pit_ra_per_bf = (lw_pit["ha"]        * ha_r +
                     lw_pit["hra_extra"]  * hr_r +
                     lw_pit["bb"]         * bb_r +
                     lw_pit["hbp"]        * hp_r +
                     lw_pit["k"]          * k_r  +
                     lw_pit["intercept"])
    pit_raa  = (lg_ra_per_bf - pit_ra_per_bf) * proj_bf
    rep_runs = REPL_PER_162IP * proj_ip / 162
    proj_rar = pit_raa + rep_runs

    total_wg  = (s["g"]  * s["w"]).sum()
    total_wgs = (s["gs"] * s["w"]).sum()
    sp_flag_v = int((total_wgs / max(total_wg, 1e-9)) >= 0.5)

    return {
        "proj_rar": proj_rar,
        "proj_ip":  proj_ip,
        "sp_flag":  sp_flag_v,
        "k_rate":   k_r,
        "hra_rate": hr_r,
    }


def _pit_market_features(proj_rar, ip, sp_flag_v, k_r, hr_r, age):
    """Build the pitcher feature dict used by the step 5 market ridge model (proj_rar, age, age_sq, log_proj_ip, sp_flag, proj_rar_sq, k_rate, age_x_krate, rar_x_hra)."""
    return {
        "proj_rar":    proj_rar,
        "age":         age,
        "age_sq":      age ** 2,
        "log_proj_ip": float(np.log(max(ip, 10))),
        "sp_flag":     sp_flag_v,
        "proj_rar_sq": proj_rar ** 2,
        "k_rate":      k_r,
        "age_x_krate": age * k_r,
        "rar_x_hra":   proj_rar * hr_r,
    }


bat_market_list = []
for _, row in fa_bat.iterrows():
    b_raa = _bat_raa(row)
    feats = _bat_market_features(row, b_raa)
    x_aav = np.array([feats[f] for f in bat_aav_mdl["features"]])
    x_yrs = np.array([feats[f] for f in bat_yrs_mdl["features"]])
    pred_aav = max(np.exp(ridge_pred(bat_aav_mdl, x_aav)) / 1e6, SALARY_FLOOR_M)
    pred_yrs = float(np.clip(ridge_pred(bat_yrs_mdl, x_yrs), 1.0, 5.0))
    bat_market_list.append({"player_id": int(row["player_id"]),
                             "pred_aav_M": pred_aav, "pred_years": pred_yrs,
                             **feats})

pit_market_list = []
n_retro_used = 0
for _, row in fa_pit.iterrows():
    pid      = int(row["player_id"])
    age_proj = float(row.get("age_proj") or 30)

    retro = _retro_pit_mkt_features(pid)
    if retro is not None:
        # Use retro weighted-average features — consistent with step 5 training
        proj_rar_mkt = retro["proj_rar"]
        ip_mkt       = retro["proj_ip"]
        sp_flag_mkt  = retro["sp_flag"]
        k_r_mkt      = retro["k_rate"]
        hr_r_mkt     = retro["hra_rate"]
        n_retro_used += 1
    else:
        # Fallback: Marcel features (no panel data — shouldn't happen for 6yr service)
        p_raa, k_r_mkt, hr_r_mkt = _pit_raa_and_rates(row)
        ip_mkt      = float(row.get("ip_proj") or 100)
        sp_flag_mkt = int(str(row.get("role") or "RP") == "SP")
        proj_rar_mkt = p_raa + REPL_PER_162IP * ip_mkt / 162

    feats = _pit_market_features(proj_rar_mkt, ip_mkt, sp_flag_mkt,
                                  k_r_mkt, hr_r_mkt, age_proj)
    x_aav = np.array([feats[f] for f in pit_aav_mdl["features"]])
    x_yrs = np.array([feats[f] for f in pit_yrs_mdl["features"]])
    pred_aav = max(np.exp(ridge_pred(pit_aav_mdl, x_aav)) / 1e6, SALARY_FLOOR_M)
    pred_yrs = float(np.clip(ridge_pred(pit_yrs_mdl, x_yrs), 1.0, 5.0))
    pit_market_list.append({"player_id": pid, "pred_aav_M": pred_aav,
                             "pred_years": pred_yrs, **feats})
print(f"  Retro features used for {n_retro_used}/{len(fa_pit)} pitchers "
      f"({len(fa_pit)-n_retro_used} Marcel fallback)")

bat_mkt = pd.DataFrame(bat_market_list).set_index("player_id")
pit_mkt = pd.DataFrame(pit_market_list).set_index("player_id")
print(f"  Batter AAV range: "
      f"[{bat_mkt['pred_aav_M'].min():.1f}–{bat_mkt['pred_aav_M'].max():.1f}]M  "
      f"n={len(bat_mkt)}")
print(f"  Pitcher AAV range: "
      f"[{pit_mkt['pred_aav_M'].min():.1f}–{pit_mkt['pred_aav_M'].max():.1f}]M  "
      f"n={len(pit_mkt)}")

# ─── 12. MONTE CARLO SIMULATION ───────────────────────────────────────────────
print(f"\n[12] Running Monte Carlo ({N_SIMS:,} sims × "
      f"{len(fa_bat)+len(fa_pit)} players × {len(L_CANDIDATES)} lengths)...")

L_MAX = max(L_CANDIDATES)


def _surplus_stats(surplus_vec, candidate_length, market_aav_M):
    """Compute summary stats from surplus distribution array."""
    m = float(surplus_vec.mean())
    s = float(surplus_vec.std())
    return {
        "candidate_length": candidate_length,
        "pred_aav_M":       market_aav_M,
        "mean_value_M":     m + market_aav_M * candidate_length,
        "mean_surplus_M":   m,
        "std_surplus_M":    s,
        "objective":        m - 1.15 * s,
        "p_underperform":   float((surplus_vec < 0).mean()),
        "pct5_surplus":     float(np.percentile(surplus_vec, 5)),
        "pct25_surplus":    float(np.percentile(surplus_vec, 25)),
        "pct75_surplus":    float(np.percentile(surplus_vec, 75)),
        "pct95_surplus":    float(np.percentile(surplus_vec, 95)),
    }


def simulate_batter_mc(row, market_aav_M):
    """Run MC for all candidate lengths; return list of result dicts."""
    pid    = int(row["player_id"])
    age_p  = int(float(row.get("age_proj") or 30))
    pos    = int(row.get("resolved_pos") or 0)
    bats_v = int(bio.loc[pid, "bats"]) if (pid in bio.index and
              pd.notna(bio.loc[pid, "bats"])) else 1
    pos_lbl = {2:"C",3:"1B",4:"2B",5:"3B",6:"SS",7:"LF",8:"CF",9:"RF"}.get(pos,"LF")

    # Marcel projected rates + volume
    bat_proj = np.array([float(row.get(f"{c}_proj") or 0) for c in BAT_COMPS])
    fld_proj = np.array([float(row.get(f"{c}_proj") or 0) for c in FLD_COMPS])
    d_frac   = float(row.get("d_frac") or 0.7)
    pa_proj  = max(float(row.get("pa_proj") or 300), 10.0)

    # Nationals park adjustment
    eff_avg, eff_hr = _bat_park_adj(bats_v)

    # Cumulative aging deltas (n_comp, L_MAX)
    bat_age = aging_cumulative(pos_lbl, BAT_COMPS, age_p, L_MAX)
    fld_age = aging_cumulative(pos_lbl, FLD_COMPS, age_p, L_MAX)

    # Preference multipliers
    zm = PREF_INDEF_MULT if pos in PREF_INDEF_POS else 1.0
    am = PREF_INDEF_MULT if pos in PREF_INDEF_POS else 1.0

    pos_adj_per_pa = POS_ADJ.get(pos, 0.0) / 600.0
    eff_aav_M = _effective_aav(market_aav_M)   # luxury-tax-adjusted cost

    # Filter candidate lengths by age cap
    valid_lengths = {L for L in L_CANDIDATES if age_p + L - 1 <= MAX_CONTRACT_END_AGE}

    # Draw persistent shocks (one per career)
    pers_bat = RNG.multivariate_normal(np.zeros(len(BAT_COMPS)), Sigma_p_bat, size=N_SIMS)
    pers_fld = RNG.multivariate_normal(np.zeros(len(FLD_COMPS)), Sigma_p_fld, size=N_SIMS)

    cumval    = np.zeros(N_SIMS)
    hit_cliff = np.zeros(N_SIMS, dtype=bool)   # once True, sim contributes 0 going forward
    results   = []

    for y in range(L_MAX):
        age_y = age_p + y

        # Performance cliff: catastrophic injury / sudden retirement
        p_cliff = _cliff_prob(age_y)
        if p_cliff > 0:
            hit_cliff |= RNG.uniform(size=N_SIMS) < p_cliff

        # Horizon uncertainty: transient shocks grow with forecast horizon to reflect
        # declining Marcel accuracy in years 2-5.
        horizon_scale = 1.0 + HORIZON_SIGMA_EXPANSION * y
        tran_bat = RNG.multivariate_normal(np.zeros(len(BAT_COMPS)), Sigma_t_bat,
                                           size=N_SIMS) * horizon_scale
        tran_fld = RNG.multivariate_normal(np.zeros(len(FLD_COMPS)), Sigma_t_fld,
                                           size=N_SIMS) * horizon_scale

        bat_r = np.clip(
            bat_proj[np.newaxis, :] + bat_age[:, y][np.newaxis, :] + pers_bat + tran_bat,
            0.0, BAT_CLIP_MAX
        )
        fld_r = fld_proj[np.newaxis, :] + fld_age[:, y][np.newaxis, :] + pers_fld + tran_fld

        # Playing-time factor
        mu_pt, sig_pt, p_cat = _pt_params(bat_pt, age_y)
        cat = RNG.uniform(size=N_SIMS) < p_cat
        log_ratio = RNG.normal(mu_pt, sig_pt, size=N_SIMS)
        pt = np.where(cat, RNG.uniform(0.0, 0.08, size=N_SIMS), np.exp(log_ratio))
        pt = np.clip(pt, 0.0, 2.5)
        pa_y   = pa_proj * pt
        g_y    = pa_y / 4.0
        ip_y   = g_y * 8.8

        # Decompose XBH into D and T; apply park factors
        hr_adj = bat_r[:, 0] * eff_hr
        d_r    = bat_r[:, 1] * d_frac          # already neutral, d_pf=1.0 at Nationals
        t_r    = bat_r[:, 1] * (1 - d_frac) * NAT["eff_t"]
        s_adj  = bat_r[:, 2] * eff_avg
        bb_r   = bat_r[:, 3]
        k_r    = bat_r[:, 4]
        ubr_r  = bat_r[:, 5]
        hp_r   = bat_r[:, 6]

        bat_raa_y = pa_y * (
            lw_bat["single"] * (s_adj  - LG["s"]) +
            lw_bat["double"] * (d_r    - LG["d"]) +
            lw_bat["triple"] * (t_r    - LG["t"]) +
            lw_bat["hr"]     * (hr_adj - LG["hr"]) +
            lw_bat["bb"]     * (bb_r   - LG["bb"]) +
            lw_bat["hbp"]    * (hp_r   - LG["hp"]) +
            lw_bat["k"]      * (k_r    - LG["k"])
        )
        ubr_y     = ubr_r  * g_y
        zr_y      = fld_r[:, 0] * ip_y / 1000 * zm
        arm_y     = fld_r[:, 1] * ip_y / 1000 * am
        frm_y     = fld_r[:, 2] * ip_y / 1000   # framing (only matters for C)
        pos_adj_y = pos_adj_per_pa * pa_y
        rep_y     = REPL_PER_600PA * pa_y / 600

        rar_y  = bat_raa_y + ubr_y + zr_y + arm_y + frm_y + pos_adj_y + rep_y
        val_y  = rar_to_value_M(rar_y, is_pitcher=False)
        val_y  = np.where(hit_cliff, 0.0, val_y)   # cliff zeroes out this year + all future
        cumval += val_y

        L = y + 1
        if L in L_CANDIDATES and L in valid_lengths:
            surplus = cumval - eff_aav_M * L
            d = _surplus_stats(surplus, L, market_aav_M)
            d["player_id"]    = pid
            d["player_type"]  = "batter"
            d["age"]          = float(age_p)
            d["proj_rar"]     = float(bat_mkt.loc[pid, "proj_rar"]) if pid in bat_mkt.index else 0.0
            results.append(d)

    return results


def simulate_pitcher_mc(row, market_aav_M):
    """Run MC for all candidate lengths; return list of result dicts."""
    pid    = int(row["player_id"])
    age_p  = int(float(row.get("age_proj") or 30))
    role   = str(row.get("role") or "RP")
    role_g = "SP" if role == "SP" else "RP"

    pit_proj = np.array([float(row.get(f"{c}_proj") or 0) for c in PIT_COMPS])
    bb_frac  = float(row.get("bb_frac") or 0.85)
    ip_proj  = max(float(row.get("ip_proj") or 80), 5.0)
    bf_proj  = float(row.get("bf_proj") or ip_proj * 4.3)
    bf_per_ip = bf_proj / ip_proj

    pit_age = aging_cumulative(role_g, PIT_COMPS, age_p, L_MAX)
    eff_aav_M = _effective_aav(market_aav_M)

    valid_lengths = {L for L in L_CANDIDATES if age_p + L - 1 <= MAX_CONTRACT_END_AGE}

    pers_pit = RNG.multivariate_normal(np.zeros(len(PIT_COMPS)), Sigma_p_pit, size=N_SIMS)

    cumval    = np.zeros(N_SIMS)
    hit_cliff = np.zeros(N_SIMS, dtype=bool)
    results   = []

    for y in range(L_MAX):
        age_y = age_p + y

        p_cliff = _cliff_prob(age_y)
        if p_cliff > 0:
            hit_cliff |= RNG.uniform(size=N_SIMS) < p_cliff

        horizon_scale = 1.0 + HORIZON_SIGMA_EXPANSION * y
        tran_pit = RNG.multivariate_normal(np.zeros(len(PIT_COMPS)), Sigma_t_pit,
                                           size=N_SIMS) * horizon_scale
        pit_r = np.clip(
            pit_proj[np.newaxis, :] + pit_age[:, y][np.newaxis, :] + pers_pit + tran_pit,
            0.0, PIT_CLIP_MAX
        )

        mu_pt, sig_pt, p_cat = _pt_params(pit_pt, age_y)
        cat = RNG.uniform(size=N_SIMS) < p_cat
        log_ratio = RNG.normal(mu_pt, sig_pt, size=N_SIMS)
        pt = np.where(cat, RNG.uniform(0.0, 0.08, size=N_SIMS), np.exp(log_ratio))
        pt = np.clip(pt, 0.0, 2.5)
        ip_y = ip_proj * pt
        bf_y = ip_y * bf_per_ip

        k_r   = pit_r[:, 0]
        bh_r  = pit_r[:, 1]
        hr_r  = pit_r[:, 2]
        ha_r  = pit_r[:, 3]
        bb_r  = bh_r * bb_frac
        hp_r  = bh_r * (1 - bb_frac)

        # Apply Nationals park factors to pitcher's allowed rates
        ha_adj = ha_r * NAT["eff_avg"]
        hr_adj = hr_r * NAT["eff_hr"]

        pit_ra_per_bf = (lw_pit["ha"] * ha_adj + lw_pit["hra_extra"] * hr_adj +
                         lw_pit["bb"] * bb_r + lw_pit["hbp"] * hp_r +
                         lw_pit["k"]  * k_r  + lw_pit["intercept"])

        # Decompose HR suppression component for preference weighting
        hra_contr = lw_pit["hra_extra"] * (LG_PIT["hra"] - hr_adj) * bf_y
        other_raa = (lg_ra_per_bf - pit_ra_per_bf) * bf_y - hra_contr
        pit_raa_y = other_raa + hra_contr * PREF_HRASUP_MULT

        rep_y  = REPL_PER_162IP * ip_y / 162
        rar_y  = pit_raa_y + rep_y
        val_y  = rar_to_value_M(rar_y, is_pitcher=True)
        val_y  = np.where(hit_cliff, 0.0, val_y)
        cumval += val_y

        L = y + 1
        if L in L_CANDIDATES and L in valid_lengths:
            surplus = cumval - eff_aav_M * L
            d = _surplus_stats(surplus, L, market_aav_M)
            d["player_id"]   = pid
            d["player_type"] = "pitcher"
            d["age"]         = float(age_p)
            d["proj_rar"]    = float(pit_mkt.loc[pid, "proj_rar"]) if pid in pit_mkt.index else 0.0
            results.append(d)

    return results


# ── Main simulation loop ──────────────────────────────────────────────────────
all_rows = []

for i, (_, row) in enumerate(fa_bat.iterrows()):
    pid = int(row["player_id"])
    if pid not in bat_mkt.index:
        continue
    mkt_aav = float(bat_mkt.loc[pid, "pred_aav_M"])
    if (i + 1) % 30 == 0 or i == 0:
        print(f"  Batter {i+1}/{len(fa_bat)}: pid={pid}  AAV=${mkt_aav:.2f}M")
    all_rows.extend(simulate_batter_mc(row, mkt_aav))

for i, (_, row) in enumerate(fa_pit.iterrows()):
    pid = int(row["player_id"])
    if pid not in pit_mkt.index:
        continue
    mkt_aav = float(pit_mkt.loc[pid, "pred_aav_M"])
    if (i + 1) % 30 == 0 or i == 0:
        print(f"  Pitcher {i+1}/{len(fa_pit)}: pid={pid}  AAV=${mkt_aav:.2f}M")
    all_rows.extend(simulate_pitcher_mc(row, mkt_aav))

# ─── 13. SAVE OUTPUTS ────────────────────────────────────────────────────────
print("\n[13] Saving outputs...")

out_df = pd.DataFrame(all_rows)
# Add predicted years from market model
bat_yrs = bat_mkt["pred_years"].reset_index().rename(columns={"pred_years":"market_pred_years"})
pit_yrs = pit_mkt["pred_years"].reset_index().rename(columns={"pred_years":"market_pred_years"})
yrs_df  = pd.concat([bat_yrs, pit_yrs], ignore_index=True)
out_df  = out_df.merge(yrs_df, on="player_id", how="left")

# Add player names for readability
_pnames = players[["player_id","First Name","Last Name"]].rename(
    columns={"First Name":"first_name","Last Name":"last_name"})
out_df = out_df.merge(_pnames, on="player_id", how="left")
out_df["player_name"] = out_df["first_name"].fillna("") + " " + out_df["last_name"].fillna("")

# Reorder columns
col_order = [
    "player_id","player_name","player_type","candidate_length","age","proj_rar",
    "pred_aav_M","market_pred_years",
    "mean_value_M","mean_surplus_M","std_surplus_M","objective","p_underperform",
    "pct5_surplus","pct25_surplus","pct75_surplus","pct95_surplus",
]
out_df = out_df[[c for c in col_order if c in out_df.columns]]
out_df.to_csv(INT / "mc_surplus_distributions.csv", index=False)
# Row count may vary if some players had lengths pruned by age cap
print(f"  Saved mc_surplus_distributions.csv ({len(out_df):,} rows)")

# Summary: best candidate length per player
best = out_df.loc[out_df.groupby(["player_id","player_type"])["objective"].idxmax()]
best = best.sort_values("objective", ascending=False)
print(f"\n  Top 15 by objective (best length):")
print(best[["player_name","player_type","age","candidate_length","pred_aav_M",
            "mean_surplus_M","std_surplus_M","objective","p_underperform"
            ]].head(15).to_string(index=False))

print(f"\n  Recommend 'do not sign' (best objective < 0): "
      f"{(best['objective'] < 0).sum()} / {len(best)} players")
print(f"\n  Candidate length distribution (best per player):")
print(best["candidate_length"].value_counts().sort_index().to_string())

print("\n" + "=" * 70)
print("STEP 6 COMPLETE")
print(f"  Players evaluated: {len(best)} "
      f"({(best['player_type']=='batter').sum()} bat, "
      f"{(best['player_type']=='pitcher').sum()} pit)")
print(f"  Outputs: mc_variance_model.csv, mc_correlation_*.csv, "
      f"mc_playtime_*.csv, mc_surplus_distributions.csv")
print("=" * 70)
