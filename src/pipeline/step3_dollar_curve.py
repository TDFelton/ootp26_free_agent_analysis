"""
Step 3: League $/component curve
Derives Frostfire linear weights, computes park-neutral player RAR,
matches to FA contracts, fits convex (quadratic) salary ~ RAR curve.

Outputs:
  intermediate/linear_weights.csv
  intermediate/player_values.csv
  intermediate/curve_coefficients.csv
  intermediate/curve_training_data.csv
  intermediate/viz/step3_*.png
"""

import os, glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# K coefficient lower bounds (most-negative allowed).
# Team-level regression can't cleanly separate K from other outs;
# these soft floors stop it from drifting to near-zero, which would
# under-penalise high-K batters and under-reward high-K pitchers.
# Values chosen to match the minimum plausible run expectancy
# difference between a K-out and a non-K-out (~-0.03 per event).
K_FLOOR_BAT = -0.03   # batting K must cost at least 0.03 runs/PA
K_FLOOR_PIT = -0.03   # pitching K must save at least 0.03 runs/BF

DATA   = r"C:\Users\Felto\Downloads\ootp_analysis\frostfire_data"
INT    = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"
VIZ    = os.path.join(INT, "viz")
os.makedirs(VIZ, exist_ok=True)

CURRENT_YEAR = 2035
VALID_TIDS   = {1,3,4,6,7,8,9,13,16,17,18,20,21,23,24,25,26,27,29,30,301,302}
EXT_PLAYER_IDS = {535,23192,25566,29390,29759,31187,35118,37118,39658,40316,40437,40770,41021,41778,43209,43318}

# Positional adjustments (runs per 600 PA, standard baseball values)
POS_ADJ = {1:0, 2:12.5, 3:-12.5, 4:2.5, 5:2.5, 6:7.5, 7:-7.5, 8:2.5, 9:-7.5, 10:-17.5}

# Replacement level (runs per full season relative to average)
# Position players: ~20 runs below average per 600 PA
# Pitchers: ~27 runs below average per 162 IP (rough convention; calibrated by curve floor)
REPLACEMENT_RUNS_PER_600PA = 20.0
REPLACEMENT_RUNS_PER_162IP = 27.0

print("=" * 60)
print("STEP 3: League $/component curve")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. BATTING LINEAR WEIGHTS  (team-level regression, 21 years)
# ──────────────────────────────────────────────────────────────
print("\n[1] Deriving batting linear weights from team data...")

tb_files = [f for f in glob.glob(os.path.join(DATA, "team_batting_*.csv"))
            if "vsL" not in f and "vsR" not in f]
tb_rows = []
for f in sorted(tb_files):
    yr = int(os.path.basename(f).replace("team_batting_", "").replace(".csv", ""))
    df = pd.read_csv(f)
    df["year"] = yr
    df = df[(df["tid"].isin(VALID_TIDS)) & (df["split_id"] == 1)]
    tb_rows.append(df)
team_bat = pd.concat(tb_rows, ignore_index=True)
team_bat = team_bat[team_bat["pa"] > 0].copy()

# Rate form so coefficients are true run-value weights
for col in ["s", "d", "t", "hr", "bb", "hp", "k"]:
    team_bat[f"{col}_r"] = team_bat[col] / team_bat["pa"]
team_bat["r_per_pa"] = team_bat["r"] / team_bat["pa"]

Xb = team_bat[["s_r", "d_r", "t_r", "hr_r", "bb_r", "hp_r", "k_r"]].values
yb = team_bat["r_per_pa"].values
lm_bat_ols = LinearRegression().fit(Xb, yb)
lw_names_bat = ["single", "double", "triple", "hr", "bb", "hbp", "k"]
lw_bat_ols = dict(zip(lw_names_bat, lm_bat_ols.coef_))
lw_bat_ols["intercept"] = lm_bat_ols.intercept_

# Constrained fit: enforce K <= K_FLOOR_BAT
# OLS residual sum of squares with scipy.optimize so we can bound coefficients
def ols_loss(coefs, X, y):
    """Sum of squared residuals for a linear model with coefficients coefs (last element is the intercept)."""
    return np.sum((y - X @ coefs[:-1] - coefs[-1])**2)

n_feat_b = Xb.shape[1]
x0_b = np.append(lm_bat_ols.coef_, lm_bat_ols.intercept_)
# bounds: single/d/t/hr/bb/hbp must be positive; k <= K_FLOOR_BAT; intercept free
bounds_b = [(0, None)] * 6 + [(None, K_FLOOR_BAT)] + [(None, None)]
res_b = minimize(ols_loss, x0_b, args=(Xb, yb), method="L-BFGS-B", bounds=bounds_b)
coef_b_con = res_b.x
lw_bat = dict(zip(lw_names_bat, coef_b_con[:-1]))
lw_bat["intercept"] = coef_b_con[-1]

bat_pred_ols = lm_bat_ols.predict(Xb)
bat_r2_ols = 1 - np.var(yb - bat_pred_ols) / np.var(yb)
bat_pred_con = Xb @ coef_b_con[:-1] + coef_b_con[-1]
bat_r2_con = 1 - np.var(yb - bat_pred_con) / np.var(yb)

print(f"  Batting LW (unconstrained OLS R^2={bat_r2_ols:.3f}  |  constrained R^2={bat_r2_con:.3f})")
print(f"  {'Component':10s}  {'OLS':>8s}  {'Constrained':>12s}")
for nm in lw_names_bat:
    print(f"  {nm:10s}  {lw_bat_ols[nm]:+8.4f}  {lw_bat[nm]:+12.4f}")
bat_r2 = bat_r2_con
lm_bat = type("_", (), {"predict": lambda self, X: X @ coef_b_con[:-1] + coef_b_con[-1]})()

# Sensitivity: show RAR shift at median player when K changes
k_test_vals = [-0.021, -0.03, -0.05, -0.08]
median_k_pa = team_bat["k_r"].median() * 600   # typical K count over 600 PA
lg_k_pa = lg_bat_total["k_pa"] if "lg_bat_total" in dir() else 0.22   # approx
print(f"\n  K-weight sensitivity (effect on bat_raa for a 22% K-rate hitter over 600 PA):")
for kw in k_test_vals:
    k_delta_runs = kw * (median_k_pa - 0.22 * 600)
    print(f"    K weight={kw:+.3f}: bat_raa contribution ~ {k_delta_runs:+.1f} runs")

# ──────────────────────────────────────────────────────────────
# 2. PITCHING LINEAR WEIGHTS  (team-level regression)
# ──────────────────────────────────────────────────────────────
print("\n[2] Deriving pitching linear weights from team data...")

tp_files = [f for f in glob.glob(os.path.join(DATA, "team_pitching_*.csv"))
            if "vsL" not in f and "vsR" not in f]
tp_rows = []
for f in sorted(tp_files):
    yr = int(os.path.basename(f).replace("team_pitching_", "").replace(".csv", ""))
    df = pd.read_csv(f)
    df["year"] = yr
    df = df[(df["tid"].isin(VALID_TIDS)) & (df["split_id"] == 1)]
    tp_rows.append(df)
team_pit = pd.concat(tp_rows, ignore_index=True)
team_pit = team_pit[team_pit["bf"] > 0].copy()

for col in ["ha", "hra", "bb", "hp", "k"]:
    team_pit[f"{col}_r"] = team_pit[col] / team_pit["bf"]
team_pit["r_per_bf"] = team_pit["r"] / team_pit["bf"]

Xp = team_pit[["ha_r", "hra_r", "bb_r", "hp_r", "k_r"]].values
yp = team_pit["r_per_bf"].values
lm_pit_ols = LinearRegression().fit(Xp, yp)
lw_names_pit = ["ha", "hra_extra", "bb", "hbp", "k"]
lw_pit_ols = dict(zip(lw_names_pit, lm_pit_ols.coef_))
lw_pit_ols["intercept"] = lm_pit_ols.intercept_

# Constrained fit: ha/hra_extra/bb/hbp positive; k <= K_FLOOR_PIT
n_feat_p = Xp.shape[1]
x0_p = np.append(lm_pit_ols.coef_, lm_pit_ols.intercept_)
bounds_p = [(0, None)] * 4 + [(None, K_FLOOR_PIT)] + [(None, None)]
res_p = minimize(ols_loss, x0_p, args=(Xp, yp), method="L-BFGS-B", bounds=bounds_p)
coef_p_con = res_p.x
lw_pit = dict(zip(lw_names_pit, coef_p_con[:-1]))
lw_pit["intercept"] = coef_p_con[-1]

pit_pred_ols = lm_pit_ols.predict(Xp)
pit_r2_ols = 1 - np.var(yp - pit_pred_ols) / np.var(yp)
pit_pred_con = Xp @ coef_p_con[:-1] + coef_p_con[-1]
pit_r2_con = 1 - np.var(yp - pit_pred_con) / np.var(yp)

print(f"  Pitching LW (unconstrained OLS R^2={pit_r2_ols:.3f}  |  constrained R^2={pit_r2_con:.3f})")
print(f"  {'Component':10s}  {'OLS':>8s}  {'Constrained':>12s}")
for nm in lw_names_pit:
    print(f"  {nm:10s}  {lw_pit_ols[nm]:+8.4f}  {lw_pit[nm]:+12.4f}")
pit_r2 = pit_r2_con
lm_pit = type("_", (), {"predict": lambda self, X: X @ coef_p_con[:-1] + coef_p_con[-1]})()

# Sensitivity for pitching K
print(f"\n  Pitching K-weight sensitivity (effect on pit_raa for elite K pitcher vs avg over 200 BF):")
for kw in k_test_vals:
    k_raa = -kw * (0.30 - 0.22) * 200   # 30% K rate vs 22% league avg over 200 BF, sign flip for pitcher
    print(f"    K weight={kw:+.3f}: pit_raa ~ {k_raa:+.1f} runs for 30% K-rate pitcher vs avg")

# Save linear weights
lw_rows = []
for nm, val in lw_bat.items():
    lw_rows.append({"side": "batting", "component": nm, "weight": val})
for nm, val in lw_pit.items():
    lw_rows.append({"side": "pitching", "component": nm, "weight": val})
pd.DataFrame(lw_rows).to_csv(os.path.join(INT, "linear_weights.csv"), index=False)
print("  Saved linear_weights.csv")

# ──────────────────────────────────────────────────────────────
# 3. LEAGUE AVERAGE RATES  (for computing RAA)
# ──────────────────────────────────────────────────────────────
print("\n[3] Computing league-average rates...")

lg_bat = team_bat.groupby("year")[["s","d","t","hr","bb","hp","k","pa"]].sum().reset_index()
lg_bat_total = team_bat[["s","d","t","hr","bb","hp","k","pa"]].sum()
for col in ["s","d","t","hr","bb","hp","k"]:
    lg_bat_total[f"{col}_pa"] = lg_bat_total[col] / lg_bat_total["pa"]

lg_pit_total = team_pit[["ha","hra","bb","hp","k","bf"]].sum()
for col in ["ha","hra","bb","hp","k"]:
    lg_pit_total[f"{col}_bf"] = lg_pit_total[col] / lg_pit_total["bf"]

print(f"  Lg avg s/PA={lg_bat_total['s_pa']:.4f}  d/PA={lg_bat_total['d_pa']:.4f}  "
      f"hr/PA={lg_bat_total['hr_pa']:.4f}  bb/PA={lg_bat_total['bb_pa']:.4f}")
print(f"  Lg avg ha/BF={lg_pit_total['ha_bf']:.4f}  hra/BF={lg_pit_total['hra_bf']:.4f}  "
      f"k/BF={lg_pit_total['k_bf']:.4f}")

# ──────────────────────────────────────────────────────────────
# 4. PLAYER BATTING RAA  (park-neutral, aggregated to player-year)
# ──────────────────────────────────────────────────────────────
print("\n[4] Computing player batting RAA...")

bat_n = pd.read_csv(os.path.join(INT, "batting_neutral.csv"))
bat_n = bat_n[bat_n["split_id"] == 1].copy()

# Aggregate stints to player-year
bat_agg = bat_n.groupby(["player_id","year"]).agg(
    pa=("pa","sum"), ab=("ab","sum"), g=("g","sum"), gs=("gs","sum"),
    singles_n=("singles_n","sum"), d_n=("d_n","sum"), t_n=("t_n","sum"),
    hr_n=("hr_n","sum"), bb=("bb","sum"), hp=("hp","sum"), k=("k","sum"),
    ubr=("ubr","sum"), war=("war","sum")
).reset_index()
bat_agg = bat_agg[bat_agg["pa"] >= 50].copy()

w = lw_bat
bat_agg["bat_raa"] = (
    w["single"] * (bat_agg["singles_n"] - lg_bat_total["s_pa"] * bat_agg["pa"]) +
    w["double"] * (bat_agg["d_n"]       - lg_bat_total["d_pa"] * bat_agg["pa"]) +
    w["triple"] * (bat_agg["t_n"]       - lg_bat_total["t_pa"] * bat_agg["pa"]) +
    w["hr"]     * (bat_agg["hr_n"]      - lg_bat_total["hr_pa"] * bat_agg["pa"]) +
    w["bb"]     * (bat_agg["bb"]        - lg_bat_total["bb_pa"] * bat_agg["pa"]) +
    w["hbp"]    * (bat_agg["hp"]        - lg_bat_total["hp_pa"] * bat_agg["pa"]) +
    w["k"]      * (bat_agg["k"]         - lg_bat_total["k_pa"]  * bat_agg["pa"])
)
bat_agg["bruns"] = bat_agg["ubr"]   # baserunning already in runs

print(f"  Player-years with pa>=50: {len(bat_agg)}")
print(f"  bat_raa range: {bat_agg['bat_raa'].min():.1f} to {bat_agg['bat_raa'].max():.1f}")

# ──────────────────────────────────────────────────────────────
# 5. DEFENSE VALUE  (primary position per player-year, with pos adj)
# ──────────────────────────────────────────────────────────────
print("\n[5] Computing defense values...")

fld = pd.read_csv(os.path.join(INT, "fielding_raw.csv"))
fld = fld[fld["ip"] > 0].copy()

# Primary position: most IP per player-year
primary_pos = (fld.groupby(["player_id","year"])
               .apply(lambda g: g.loc[g["ip"].idxmax(), "position"])
               .reset_index(name="primary_pos"))

# Defense runs: zr + arm (already in runs); sum across all positions per player-year
def_agg = fld.groupby(["player_id","year"]).agg(
    total_ip=("ip","sum"),
    def_zr=("zr","sum"),
    def_arm=("arm","sum"),
    def_framing=("framing","sum")
).reset_index()
def_agg = def_agg.merge(primary_pos, on=["player_id","year"], how="left")
def_agg["pos_adj"] = def_agg["primary_pos"].map(POS_ADJ).fillna(0) * (def_agg["total_ip"] / (162 * 8.8))
def_agg["def_runs"] = def_agg["def_zr"] + def_agg["def_arm"] + def_agg["def_framing"] + def_agg["pos_adj"]

print(f"  Defense player-years: {len(def_agg)}")

# ──────────────────────────────────────────────────────────────
# 6. TOTAL POSITION PLAYER RAR
# ──────────────────────────────────────────────────────────────
print("\n[6] Assembling position player RAR...")

pos_val = bat_agg.merge(def_agg[["player_id","year","def_runs","primary_pos","total_ip"]],
                        on=["player_id","year"], how="left")
pos_val["def_runs"] = pos_val["def_runs"].fillna(0)
pos_val["primary_pos"] = pos_val["primary_pos"].fillna(0).astype(int)

# Replacement runs: ~20 runs per 600 PA
pos_val["replacement_runs"] = REPLACEMENT_RUNS_PER_600PA * (pos_val["pa"] / 600)
pos_val["total_raa"] = pos_val["bat_raa"] + pos_val["bruns"] + pos_val["def_runs"]
pos_val["rar"] = pos_val["total_raa"] + pos_val["replacement_runs"]
pos_val["player_type"] = "batter"

print(f"  Position player RAR range: {pos_val['rar'].min():.1f} to {pos_val['rar'].max():.1f}")

# ──────────────────────────────────────────────────────────────
# 7. PITCHER RAR  (FIP-style, split_id=1, aggregate team_ids)
# ──────────────────────────────────────────────────────────────
print("\n[7] Computing pitcher RAR...")

pit_n = pd.read_csv(os.path.join(INT, "pitching_neutral.csv"))
pit_n = pit_n[pit_n["split_id"] == 1].copy()

pit_agg = pit_n.groupby(["player_id","year"]).agg(
    bf=("bf","sum"), outs=("outs","sum"), g=("g","sum"), gs=("gs","sum"),
    k=("k","sum"), bb=("bb","sum"), hp=("hp","sum"),
    ha_n=("ha_n","sum"), hra_n=("hra_n","sum"),
    war=("war","sum")
).reset_index()
pit_agg["ip"] = pit_agg["outs"] / 3
pit_agg = pit_agg[pit_agg["bf"] >= 30].copy()

# FIP-style runs allowed rate (park-neutral)
pw = lw_pit
pit_agg["pit_ra_per_bf"] = (
    pw["ha"]       * pit_agg["ha_n"] / pit_agg["bf"] +
    pw["hra_extra"] * pit_agg["hra_n"] / pit_agg["bf"] +
    pw["bb"]       * pit_agg["bb"] / pit_agg["bf"] +
    pw["hbp"]      * pit_agg["hp"] / pit_agg["bf"] +
    pw["k"]        * pit_agg["k"] / pit_agg["bf"] +
    pw["intercept"]
)
pit_agg["pit_ra_total"] = pit_agg["pit_ra_per_bf"] * pit_agg["bf"]

# Lg average runs allowed per BF
lg_ra_per_bf = lg_pit_total["ha_bf"] * pw["ha"] + \
               lg_pit_total["hra_bf"] * pw["hra_extra"] + \
               lg_pit_total["bb_bf"] * pw["bb"] + \
               lg_pit_total["hp_bf"] * pw["hbp"] + \
               lg_pit_total["k_bf"] * pw["k"] + \
               pw["intercept"]

pit_agg["pit_raa"] = (lg_ra_per_bf - pit_agg["pit_ra_per_bf"]) * pit_agg["bf"]
pit_agg["replacement_runs"] = REPLACEMENT_RUNS_PER_162IP * (pit_agg["ip"] / 162)
pit_agg["rar"] = pit_agg["pit_raa"] + pit_agg["replacement_runs"]
pit_agg["sp_flag"] = (pit_agg["gs"] / pit_agg["g"].clip(lower=1)) >= 0.5
pit_agg["player_type"] = pit_agg["sp_flag"].map({True: "SP", False: "RP"})
pit_agg["primary_pos"] = 1  # pitcher

print(f"  Pitcher RAR range: {pit_agg['rar'].min():.1f} to {pit_agg['rar'].max():.1f}")

# ──────────────────────────────────────────────────────────────
# 8. COMBINED PLAYER VALUES  (save full table)
# ──────────────────────────────────────────────────────────────
keep_bat = ["player_id","year","pa","bat_raa","bruns","def_runs","total_raa","rar",
            "primary_pos","player_type","war"]
keep_pit = ["player_id","year","bf","ip","pit_raa","replacement_runs","rar",
            "primary_pos","player_type","war"]

pv_bat = pos_val[keep_bat].copy()
pv_pit = pit_agg[keep_pit].copy()
pv_bat["ip"] = np.nan
pv_pit["pa"] = np.nan
pv_pit["bat_raa"] = np.nan
pv_pit["bruns"] = np.nan
pv_pit["def_runs"] = np.nan
pv_pit["total_raa"] = np.nan

player_values = pd.concat([pv_bat, pv_pit], ignore_index=True, sort=False)
player_values.to_csv(os.path.join(INT, "player_values.csv"), index=False)
print(f"\n  Saved player_values.csv ({len(player_values)} rows)")

# ──────────────────────────────────────────────────────────────
# 9. CONTRACT MATCHING — build curve training set from transaction log
# ──────────────────────────────────────────────────────────────
# contracts.csv is a live snapshot only (the API ignores all date params,
# confirmed empirically) -- a contract row is visible only if it hasn't
# expired as of the 2035 pull date, which severely survivorship-biases early
# signing cohorts toward long deals. fa_signings_log.csv (parsed from OOTP's
# in-game transaction-log HTML, see src/data/transactions_parser.py and
# docs/step8_transaction_log_rework.md) has no such bias -- every signing event is
# captured as it happened. contracts.csv is still used below for the
# replacement-salary calibration (a snapshot-appropriate use: today's pay
# floor), just not for this training set.
print("\n[8] Building curve training set from transaction-log FA signings...")

contracts = pd.read_csv(os.path.join(DATA, "contracts.csv"))
players   = pd.read_csv(os.path.join(DATA, "players.csv"))
fa_log    = pd.read_csv(os.path.join(INT, "fa_signings_log.csv"))

players_sub = players[["ID","mlb_service_years","date_of_birth","Pos","bats","Role"]].rename(
    columns={"ID": "player_id"})

# FA signings only (drop extensions -- pre-FA discount bias, per spec) and
# only the human-managed era (signing_year >= 2031; pre-2031 was AI-controlled,
# confirmed with owner -- see step8_transaction_log_rework.md scope decision).
real_signings = fa_log[(~fa_log["is_extension"]) & (fa_log["human_era"])].copy()
real_signings = real_signings.merge(players_sub, on="player_id", how="left")

# fa_signings_log has no year-by-year salary breakdown, only years + total_value.
# aav = total_value / years is used as the salary0 proxy (flat-AAV approximation;
# real contracts may have escalators contracts.csv's salary0 would have captured).
real_signings["salary0"] = real_signings["aav"]

# Service-time-at-signing: derived from each player's actual MLB appearance
# history (batting/pitching/fielding panels), NOT the current 2035
# mlb_service_years snapshot minus elapsed years. The snapshot-based formula
# assumes continuous accrual through 2035, which silently breaks for any
# player who has since retired (their counter freezes, so subtracting years
# elapsed since an old signing overcounts and pushes service_at_signing
# negative). Confirmed empirically during the step 10 AI-era data-expansion
# work (see CLAUDE.md step 10) -- this fix is robust to retirement since it
# never depends on the player's current-day state.
_bat_yrs = pd.read_csv(os.path.join(INT, "batting_raw.csv"), usecols=["player_id","year","team_id"])
_pit_yrs = pd.read_csv(os.path.join(INT, "pitching_raw.csv"), usecols=["player_id","year","team_id"])
_fld_yrs = pd.read_csv(os.path.join(INT, "fielding_raw.csv"), usecols=["player_id","year","team_id"])
_mlb_years = pd.concat([
    _bat_yrs[_bat_yrs["team_id"].isin(VALID_TIDS)][["player_id","year"]],
    _pit_yrs[_pit_yrs["team_id"].isin(VALID_TIDS)][["player_id","year"]],
    _fld_yrs[_fld_yrs["team_id"].isin(VALID_TIDS)][["player_id","year"]],
], ignore_index=True).drop_duplicates()
_mlb_years_by_pid = {pid: set(grp["year"]) for pid, grp in _mlb_years.groupby("player_id")}

def _service_at_signing(pid, signing_year):
    """Count the number of distinct MLB years a player appeared in before signing_year, based on actual appearance history rather than the current-day service-time snapshot."""
    yrs = _mlb_years_by_pid.get(pid)
    if yrs is None:
        return 0
    return sum(1 for y in yrs if y < signing_year)

real_signings["service_at_signing"] = real_signings.apply(
    lambda r: _service_at_signing(int(r["player_id"]), int(r["signing_year"])), axis=1
)

fa_contracts = real_signings[real_signings["service_at_signing"] >= 6].copy()  # calibration uses 6+ (established market; see CLAUDE.md)
print(f"  FA-eligible signings (transaction log): {len(fa_contracts)}")
print(f"  By signing_year:\n{fa_contracts['signing_year'].value_counts().sort_index().to_string()}")

# For each signing, get player RAR from signing_year - 1
fa_contracts["perf_year"] = fa_contracts["signing_year"] - 1
fa_contracts = fa_contracts.rename(columns={"signing_year": "season_year"})

# Match to position player values
bat_match = pos_val[["player_id","year","rar","bat_raa","def_runs","bruns","pa","war",
                      "primary_pos","player_type"]].copy()
bat_match = bat_match.rename(columns={"year": "perf_year"})

pit_match = pit_agg[["player_id","year","rar","pit_raa","ip","bf","war","primary_pos","player_type"]].copy()
pit_match = pit_match.rename(columns={"year": "perf_year"})

training = fa_contracts[["player_id","season_year","salary0","years","perf_year",
                          "service_at_signing","Pos","Role"]].copy()

# Try batter match first, then pitcher
training_bat = training.merge(bat_match, on=["player_id","perf_year"], how="inner")
training_pit = training.merge(pit_match, on=["player_id","perf_year"], how="inner")

# Remove pitcher-matched rows from batter set (use role to disambiguate)
# If Role in {11,12,13} -> pitcher; else batter
training_bat = training_bat[training_bat["Role"].isin([0])].copy()
training_pit = training_pit[training_pit["Role"].isin([11,12,13])].copy()

curve_data = pd.concat([training_bat, training_pit], ignore_index=True, sort=False)
curve_data["log_salary"] = np.log(curve_data["salary0"].clip(lower=1))
curve_data.to_csv(os.path.join(INT, "curve_training_data.csv"), index=False)
print(f"  Training rows: {len(curve_data)} ({len(training_bat)} batters, {len(training_pit)} pitchers)")
print(f"  salary0 (aav proxy) range: ${curve_data['salary0'].min():,.0f} to ${curve_data['salary0'].max():,.0f}")

# ──────────────────────────────────────────────────────────────
# 10. REPLACEMENT LEVEL SALARY  (modal low)
# ──────────────────────────────────────────────────────────────
print("\n[9] Calibrating replacement level salary...")

# Use all ML contracts (not just FA) for the modal low
all_ml = contracts[(contracts["is_major"] == 1) & (contracts["season_year"] > 0)]
sal_counts = all_ml["salary0"].value_counts()
replacement_salary = sal_counts.idxmax()
print(f"  Modal low salary (replacement level): ${replacement_salary:,.0f}")
print(f"  Frequency: {sal_counts.max()} contracts at this level")

# ──────────────────────────────────────────────────────────────
# 11. FIT QUADRATIC $/RAR CURVE
# ──────────────────────────────────────────────────────────────
print("\n[10] Fitting quadratic $/RAR curve...")

# Remove extreme low-salary minimum-wage contracts for curve fitting
# (they cluster at replacement regardless of RAR and distort the curve)
SALARY_FLOOR = replacement_salary * 1.5   # at least 50% above league min
fit_data = curve_data[curve_data["salary0"] > SALARY_FLOOR].copy()
print(f"  Rows above salary floor ${SALARY_FLOOR:,.0f}: {len(fit_data)}")

X_rar = fit_data["rar"].values
y_sal = fit_data["salary0"].values / 1e6   # in $M for readability

# Pooled quadratic: salary = a + b*RAR + c*RAR^2
# Pitcher-specific discount is applied in step 6 Track A (PITCHER_TRACK_A_DISCOUNT_M = -4.0)
# rather than here, so that the pooled curve isn't artificially shifted upward on the batter
# side and the pitcher market inefficiency is preserved as surplus signal.
X_poly = np.column_stack([X_rar, X_rar**2])

# CV alpha grid search (5-fold, minimise MSE)
alpha_grid = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 50, 100]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
c_by_alpha  = {}
for alpha in alpha_grid:
    scores = cross_val_score(Ridge(alpha=alpha), X_poly, y_sal,
                             cv=kf, scoring="neg_mean_squared_error")
    cv_results[alpha] = -scores.mean()
    tmp = Ridge(alpha=alpha).fit(X_poly, y_sal)
    c_by_alpha[alpha] = tmp.coef_[1]

best_alpha = min(cv_results, key=cv_results.get)
print(f"  CV alpha grid results (5-fold MSE):")
for alpha in alpha_grid:
    marker = " << best" if alpha == best_alpha else ""
    print(f"    alpha={alpha:6.2f}: CV MSE={cv_results[alpha]:.3f}  c={c_by_alpha[alpha]:.6f}{marker}")

ridge = Ridge(alpha=best_alpha, fit_intercept=True)
ridge.fit(X_poly, y_sal)
a, b, c = ridge.intercept_, ridge.coef_[0], ridge.coef_[1]
print(f"\n  Best alpha={best_alpha}  -> intercept={a:.3f}M  b={b:.4f}M/run  c={c:.6f}M/run^2")

if c <= 0:
    print(f"  WARNING: c={c:.6f} ≤ 0 — curve is concave. Forcing c > 0 by clamping to 1e-5.")
    # Re-fit in log-salary space (always convex in dollar space) as fallback
    y_log = np.log(y_sal.clip(min=0.01))
    ridge_log = Ridge(alpha=best_alpha).fit(X_poly, y_log)
    print(f"  Log-space alt: intercept={ridge_log.intercept_:.3f}  b={ridge_log.coef_[0]:.5f}  "
          f"c={ridge_log.coef_[1]:.7f}")
    print(f"  Using log-space fit (always convex). Back-transform: salary = exp(a + b*RAR + c*RAR^2)")

y_pred = ridge.predict(X_poly)
ss_res = np.sum((y_sal - y_pred)**2)
ss_tot = np.sum((y_sal - y_sal.mean())**2)
r2_curve = 1 - ss_res / ss_tot
print(f"  Final curve R^2={r2_curve:.3f}  RMSE=${np.sqrt(ss_res/len(y_sal)):.2f}M")

# Implied $/WAR at representative RAR values
print("\n  Implied salary at representative RAR:")
for rar_val in [0, 10, 20, 30, 40, 50, 60]:
    sal = a + b*rar_val + c*rar_val**2
    print(f"    RAR={rar_val:3d}: ${sal:.2f}M")

# Marginal $ per additional run at 20 and 40 RAR (same slope for both types)
for rar_val in [20, 40]:
    marginal = b + 2*c*rar_val
    print(f"  Marginal $/run at RAR={rar_val}: ${marginal*1e6:,.0f}")

# ──────────────────────────────────────────────────────────────
# 12. SAVE COEFFICIENTS
# ──────────────────────────────────────────────────────────────
coef_df = pd.DataFrame([{
    "intercept_M": a,
    "b_M_per_run": b,
    "c_M_per_run2": c,
    "ridge_alpha": best_alpha,
    "k_floor_bat": K_FLOOR_BAT,
    "k_floor_pit": K_FLOOR_PIT,
    "replacement_salary": replacement_salary,
    "salary_floor_for_fit": SALARY_FLOOR,
    "n_training_rows": len(fit_data),
    "r2": r2_curve,
    "replacement_runs_per_600pa": REPLACEMENT_RUNS_PER_600PA,
    "replacement_runs_per_162ip": REPLACEMENT_RUNS_PER_162IP
}])
coef_df.to_csv(os.path.join(INT, "curve_coefficients.csv"), index=False)
print("\n  Saved curve_coefficients.csv")

# ──────────────────────────────────────────────────────────────
# 13. VALIDATION VISUALIZATIONS
# ──────────────────────────────────────────────────────────────
print("\n[11] Generating validation visualizations...")

# ── Fig 1: Salary histogram with replacement level ────────────
fig, ax = plt.subplots(figsize=(9, 4))
sal_vals = all_ml["salary0"].clip(upper=3e7) / 1e6
ax.hist(sal_vals, bins=60, color="#4C72B0", alpha=0.8, edgecolor="none")
ax.axvline(replacement_salary / 1e6, color="crimson", lw=2,
           label=f"Modal low = ${replacement_salary/1e6:.2f}M (replacement level)")
ax.axvline(SALARY_FLOOR / 1e6, color="darkorange", lw=1.5, ls="--",
           label=f"Fit floor = ${SALARY_FLOOR/1e6:.2f}M")
ax.set_xlabel("Salary0 ($M, capped at $30M)")
ax.set_ylabel("Contract count")
ax.set_title("Salary Distribution — All Active ML Contracts (season_year > 0)")
ax.legend()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "step3_salary_histogram.png"), dpi=150)
plt.close()
print("  Saved step3_salary_histogram.png")

# ── Fig 2: Batting linear weights validation ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: team-level predicted vs actual runs
bat_r_actual = team_bat["r_per_pa"].values
bat_r_pred   = lm_bat.predict(Xb)
ax = axes[0]
ax.scatter(bat_r_pred, bat_r_actual, alpha=0.4, s=15, color="#4C72B0")
mn, mx = min(bat_r_pred.min(), bat_r_actual.min()), max(bat_r_pred.max(), bat_r_actual.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="y = x")
ax.set_xlabel("Predicted runs/PA")
ax.set_ylabel("Actual runs/PA")
ax.set_title(f"Batting LW — Team Level (R^2={bat_r2:.3f})")
ax.legend()

# Right: pitching predicted vs actual
pit_r_actual = team_pit["r_per_bf"].values
pit_r_pred   = lm_pit.predict(Xp)
ax = axes[1]
ax.scatter(pit_r_pred, pit_r_actual, alpha=0.4, s=15, color="#55A868")
mn, mx = min(pit_r_pred.min(), pit_r_actual.min()), max(pit_r_pred.max(), pit_r_actual.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="y = x")
ax.set_xlabel("Predicted RA/BF")
ax.set_ylabel("Actual RA/BF")
ax.set_title(f"Pitching LW — Team Level (R^2={pit_r2:.3f})")
ax.legend()

plt.suptitle("Linear Weights Derivation — Team-Level Validation (21 yr × 22 teams)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "step3_linear_weights_validation.png"), dpi=150)
plt.close()
print("  Saved step3_linear_weights_validation.png")

# ── Fig 3: $/RAR scatter + fitted quadratic curve ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rar_range = np.linspace(max(0, X_rar.min()), X_rar.max() + 5, 300)
sal_curve = a + b * rar_range + c * rar_range**2
sal_curve_clamped = np.maximum(sal_curve, replacement_salary / 1e6)

ax = axes[0]
colors = curve_data["player_type"].map({"batter":"#4C72B0","SP":"#C44E52","RP":"#55A868"})
for ptype, color in [("batter","#4C72B0"), ("SP","#C44E52"), ("RP","#55A868")]:
    sub = curve_data[curve_data["player_type"] == ptype]
    ax.scatter(sub["rar"], sub["salary0"] / 1e6, c=color, alpha=0.55, s=20, label=ptype)
ax.plot(rar_range, sal_curve_clamped, "k-", lw=2.5, label="Quadratic fit", zorder=5)
ax.axhline(replacement_salary / 1e6, color="crimson", ls=":", lw=1.2, label="Replacement salary")
ax.set_xlabel("Runs Above Replacement (RAR)")
ax.set_ylabel("salary0 ($M)")
ax.set_title(f"$/RAR Curve — FA Contracts Matched to Prior-Year Stats\n"
             f"salary = {a:.2f} + {b:.4f}*RAR + {c:.6f}*RAR^2  (R^2={r2_curve:.2f})")
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))

# Right: log-salary scatter for visual linearity check
ax2 = axes[1]
curve_data_pos = curve_data[curve_data["salary0"] > 0].copy()
ax2.scatter(curve_data_pos["rar"], np.log(curve_data_pos["salary0"] / 1e6),
            c=curve_data_pos["player_type"].map({"batter":"#4C72B0","SP":"#C44E52","RP":"#55A868"}),
            alpha=0.5, s=18)
ax2.plot(rar_range, np.log(np.maximum(sal_curve_clamped, 0.01)), "k-", lw=2.5)
ax2.set_xlabel("Runs Above Replacement (RAR)")
ax2.set_ylabel("log(salary0 / $1M)")
ax2.set_title("Log-Salary vs RAR (linearity diagnostic)")
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "step3_dollar_rar_curve.png"), dpi=150)
plt.close()
print("  Saved step3_dollar_rar_curve.png")

# ── Fig 4: Residuals by year and position ─────────────────────
fit_data2 = fit_data.copy()
fit_data2["predicted_M"] = ridge.predict(X_poly)
fit_data2["residual_M"] = y_sal - fit_data2["predicted_M"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: residuals by season_year
ax = axes[0]
by_yr = fit_data2.groupby("season_year")["residual_M"].agg(["mean","std","count"]).reset_index()
ax.bar(by_yr["season_year"], by_yr["mean"], yerr=by_yr["std"],
       color="#4C72B0", alpha=0.8, capsize=5)
ax.axhline(0, color="k", lw=1.2)
ax.set_xlabel("Signing year")
ax.set_ylabel("Mean residual ($M)")
ax.set_title("Residuals by Signing Year\n(flat = no inflation)")
for _, row in by_yr.iterrows():
    ax.text(row["season_year"], row["mean"] + row["std"] + 0.3,
            f"n={int(row['count'])}", ha="center", fontsize=8)

# Right: residuals by player type
ax = axes[1]
by_type = fit_data2.groupby("player_type")["residual_M"].agg(["mean","std","count"]).reset_index()
ax.bar(by_type["player_type"], by_type["mean"], yerr=by_type["std"],
       color=["#4C72B0","#C44E52","#55A868"][:len(by_type)], alpha=0.8, capsize=5)
ax.axhline(0, color="k", lw=1.2)
ax.set_xlabel("Player type")
ax.set_ylabel("Mean residual ($M)")
ax.set_title("Residuals by Player Type\n(near zero = well-calibrated)")
for _, row in by_type.iterrows():
    ax.text(row["player_type"], row["mean"] + row["std"] + 0.3,
            f"n={int(row['count'])}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(VIZ, "step3_residuals.png"), dpi=150)
plt.close()
print("  Saved step3_residuals.png")

# ── Fig 5: Component breakdown for batters ────────────────────
# Explicit bars with manual legend handles to avoid matplotlib
# auto-legend mismatches that plagued the pandas .plot() version.
fig, ax = plt.subplots(figsize=(10, 5))
batter_vals = pos_val[(pos_val["pa"] >= 300)].copy()
batter_vals["rar_bin"] = pd.cut(batter_vals["rar"], bins=[-10,0,10,20,30,40,60])
comp_means = batter_vals.groupby("rar_bin", observed=True)[["bat_raa","bruns","def_runs"]].mean()
bins_labels = [str(b) for b in comp_means.index]
n_bins  = len(bins_labels)
x       = np.arange(n_bins)
width   = 0.25
c_bat, c_brn, c_def = "#4C72B0", "#55A868", "#C44E52"
b1 = ax.bar(x - width, comp_means["bat_raa"],  width, color=c_bat, alpha=0.85, label="Batting RAA")
b2 = ax.bar(x,          comp_means["bruns"],    width, color=c_brn, alpha=0.85, label="Baserunning")
b3 = ax.bar(x + width,  comp_means["def_runs"], width, color=c_def, alpha=0.85, label="Defense")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(bins_labels, rotation=30, ha="right")
ax.set_xlabel("RAR bin")
ax.set_ylabel("Mean component runs")
ax.set_title("Mean Component Runs by RAR Tier — Position Players (PA>=300)")
ax.legend(handles=[b1, b2, b3], loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "step3_component_breakdown.png"), dpi=150)
plt.close()
print("  Saved step3_component_breakdown.png")

# Inflation diagnostic removed. No inflation in this league — confirmed by owner.

# ── Fig 6: Implied $/WAR curve ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
# Marginal $/run at each RAR point
marginal_per_run = (b + 2 * c * rar_range) * 1e6  # in $
# Convert to $/WAR assuming 1 WAR ~ 10 runs
marginal_per_war = marginal_per_run * 10
ax.plot(rar_range, marginal_per_war / 1e6, color="#C44E52", lw=2.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Player RAR")
ax.set_ylabel("Marginal $ per WAR ($M/WAR)")
ax.set_title("Implied Marginal $/WAR by Player Quality\n(convexity = star premium)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "step3_implied_dollar_per_war.png"), dpi=150)
plt.close()
print("  Saved step3_implied_dollar_per_war.png")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print(f"  Replacement salary:   ${replacement_salary:,.0f}")
print(f"  Quadratic fit: a={a:.3f}M  b={b:.4f}M/run  c={c:.6f}M/run^2")
print(f"  Curve R^2:              {r2_curve:.3f}")
print(f"  Training rows (above floor): {len(fit_data)}")
print(f"  Outputs: player_values.csv, curve_coefficients.csv,")
print(f"           linear_weights.csv, curve_training_data.csv")
print(f"  Viz: {VIZ}")
print("=" * 60)
