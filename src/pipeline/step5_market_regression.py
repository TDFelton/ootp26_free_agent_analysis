"""
Step 5: Market price regression

For each FA signing in the training set (2032-2035), compute a retrospective
Marcel-style projection (3-year gamma=0.9 weighted average) and convert it to
RAR using step 3's linear weights. Fit ridge models predicting log(AAV) and
years, with alpha selected jointly via LOO-CV. Separate models for batters
and pitchers.

Inputs:
  intermediate/batting_neutral.csv
  intermediate/pitching_neutral.csv
  intermediate/fielding_raw.csv
  intermediate/linear_weights.csv
  frostfire_data/contracts.csv
  frostfire_data/players.csv
  frostfire_data/contract_extensions.csv
  frostfire_data/team_batting_YYYY.csv
  frostfire_data/team_pitching_YYYY.csv

Outputs:
  intermediate/market_model_bat.csv     — batter ridge coefficients + scaler params
  intermediate/market_model_pit.csv     — pitcher ridge coefficients + scaler params
  intermediate/market_training_data.csv — features + targets + predictions
"""

import os, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
DATA  = r"C:\Users\Felto\Downloads\ootp_analysis\frostfire_data"
INT   = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"
VIZ   = os.path.join(INT, "viz")
os.makedirs(VIZ, exist_ok=True)

CURRENT_YEAR  = 2035
VALID_TIDS    = {1,3,4,6,7,8,9,13,16,17,18,20,21,23,24,25,26,27,29,30,301,302}
GAMMA         = 0.9   # gap-year decay (from step 4 CV)
L_RETRO       = 3     # lookback seasons for retrospective projection
MIN_PA        = 100   # min PA per season to qualify for batting lookback
MIN_BF        = 100   # min BF per season to qualify for pitching lookback
REPLACEMENT_PER_600PA = 20.0
REPLACEMENT_PER_162IP = 27.0
POS_ADJ = {2:12.5, 3:-12.5, 4:2.5, 5:2.5, 6:7.5, 7:-7.5, 8:2.5, 9:-7.5, 10:-17.5}
EXT_PLAYER_IDS = {535,23192,25566,29390,29759,31187,35118,37118,39658,40316,
                  40437,40770,41021,41778,43209,43318}
ALPHA_GRID = list(np.logspace(-2, 3, 50))   # 50 pts 0.01->1000, finer than discrete grid

# Premium defense positions (scarcity drives market premium beyond RAR)
PREMIUM_DEF_POS = {2, 6}   # C, SS

print("=" * 60)
print("STEP 5: Market price regression")
print("=" * 60)

# ── 1. LOAD BASE PANELS ───────────────────────────────────────────────────────
print("\n[1] Loading base panels...")

# 1a. Batting (park-neutral, split_id=1, aggregate stints to player-year)
bat_raw = pd.read_csv(
    os.path.join(INT, "batting_neutral.csv"),
    usecols=["player_id","year","split_id","pa","g","gs",
             "singles_n","d_n","t_n","hr_n","bb","hp","k","ubr"]
)
bat_raw = bat_raw[bat_raw["split_id"] == 1].copy()
bat_py = bat_raw.groupby(["player_id","year"], sort=False).agg(
    pa=("pa","sum"), g=("g","sum"), gs=("gs","sum"),
    singles_n=("singles_n","sum"), d_n=("d_n","sum"),
    t_n=("t_n","sum"), hr_n=("hr_n","sum"),
    bb=("bb","sum"), hp=("hp","sum"), k=("k","sum"), ubr=("ubr","sum"),
).reset_index()
bat_py = bat_py[bat_py["pa"] >= MIN_PA].copy()
print(f"  Batting player-years (pa>={MIN_PA}): {len(bat_py)}")

# 1b. Pitching (park-neutral, split_id=1, VALID_TIDS, aggregate to player-year)
pit_raw = pd.read_csv(
    os.path.join(INT, "pitching_neutral.csv"),
    usecols=["player_id","year","team_id","split_id",
             "bf","outs","g","gs","k","bb","hp","ha_n","hra_n","gb","fb"]
)
pit_raw = pit_raw[
    (pit_raw["split_id"] == 1) & (pit_raw["team_id"].isin(VALID_TIDS))
].copy()
pit_py = pit_raw.groupby(["player_id","year"], sort=False).agg(
    bf=("bf","sum"), outs=("outs","sum"),
    g=("g","sum"), gs=("gs","sum"),
    k=("k","sum"), bb=("bb","sum"), hp=("hp","sum"),
    ha_n=("ha_n","sum"), hra_n=("hra_n","sum"),
    gb=("gb","sum"), fb=("fb","sum"),
).reset_index()
pit_py["ip"] = pit_py["outs"] / 3
pit_py = pit_py[pit_py["bf"] >= MIN_BF].copy()
print(f"  Pitching player-years (bf>={MIN_BF}): {len(pit_py)}")

# 1c. Defense (from fielding_raw, aggregate to player-year with positional adjustment)
fld_raw = pd.read_csv(
    os.path.join(INT, "fielding_raw.csv"),
    usecols=["player_id","year","position","ip","zr","arm","framing"]
)
fld_raw = fld_raw[fld_raw["ip"] >= 1].copy()

fld_29 = fld_raw[fld_raw["position"].between(2, 9)].copy()
ip_by_pos = fld_29.groupby(["player_id","year","position"])["ip"].sum().reset_index()
prim_idx  = ip_by_pos.groupby(["player_id","year"])["ip"].idxmax()
prim_fld  = (ip_by_pos.loc[prim_idx, ["player_id","year","position"]]
             .rename(columns={"position":"primary_pos"}))

def_py = fld_raw.groupby(["player_id","year"], sort=False).agg(
    total_ip=("ip","sum"), def_zr=("zr","sum"),
    def_arm=("arm","sum"), def_framing=("framing","sum"),
).reset_index()
def_py = def_py.merge(prim_fld, on=["player_id","year"], how="left")
def_py["primary_pos"] = def_py["primary_pos"].fillna(0).astype(int)
def_py["pos_adj"] = (def_py["primary_pos"].map(POS_ADJ).fillna(0)
                     * (def_py["total_ip"] / (162 * 8.8)))
def_py["def_runs"] = (def_py["def_zr"] + def_py["def_arm"]
                      + def_py["def_framing"] + def_py["pos_adj"])
print(f"  Defense player-years: {len(def_py)}")

# Build fast lookup dicts
bat_by_pid = {pid: grp for pid, grp in bat_py.groupby("player_id")}
pit_by_pid = {pid: grp for pid, grp in pit_py.groupby("player_id")}
def_by_pid = {pid: grp for pid, grp in def_py.groupby("player_id")}

# ── 2. LEAGUE AVERAGES (pooled 21 years) ──────────────────────────────────────
print("\n[2] Computing league averages...")

tb_files = [f for f in glob.glob(os.path.join(DATA, "team_batting_*.csv"))
            if "vsL" not in f and "vsR" not in f]
team_bat = pd.concat(
    [pd.read_csv(f)[(pd.read_csv(f)["tid"].isin(VALID_TIDS)) &
                    (pd.read_csv(f)["split_id"] == 1)]
     for f in sorted(tb_files)],
    ignore_index=True
)
team_bat = team_bat[team_bat["pa"] > 0]
lg_tot = team_bat[["s","d","t","hr","bb","hp","k","pa"]].sum()
LG = {
    "s":  lg_tot["s"]  / lg_tot["pa"],
    "d":  lg_tot["d"]  / lg_tot["pa"],
    "t":  lg_tot["t"]  / lg_tot["pa"],
    "hr": lg_tot["hr"] / lg_tot["pa"],
    "bb": lg_tot["bb"] / lg_tot["pa"],
    "hp": lg_tot["hp"] / lg_tot["pa"],
    "k":  lg_tot["k"]  / lg_tot["pa"],
}
print(f"  Batting: s={LG['s']:.4f} d={LG['d']:.4f} hr={LG['hr']:.4f} "
      f"bb={LG['bb']:.4f} k={LG['k']:.4f}")

tp_files = [f for f in glob.glob(os.path.join(DATA, "team_pitching_*.csv"))
            if "vsL" not in f and "vsR" not in f]
team_pit = pd.concat(
    [pd.read_csv(f)[(pd.read_csv(f)["tid"].isin(VALID_TIDS)) &
                    (pd.read_csv(f)["split_id"] == 1)]
     for f in sorted(tp_files)],
    ignore_index=True
)
team_pit = team_pit[team_pit["bf"] > 0]
lg_pit = team_pit[["ha","hra","bb","hp","k","bf"]].sum()
LG_PIT = {
    "ha":  lg_pit["ha"]  / lg_pit["bf"],
    "hra": lg_pit["hra"] / lg_pit["bf"],
    "bb":  lg_pit["bb"]  / lg_pit["bf"],
    "hp":  lg_pit["hp"]  / lg_pit["bf"],
    "k":   lg_pit["k"]   / lg_pit["bf"],
}
print(f"  Pitching: ha={LG_PIT['ha']:.4f} hra={LG_PIT['hra']:.4f} "
      f"bb={LG_PIT['bb']:.4f} k={LG_PIT['k']:.4f}")

# ── 3. LINEAR WEIGHTS ─────────────────────────────────────────────────────────
print("\n[3] Loading linear weights...")
lw_df = pd.read_csv(os.path.join(INT, "linear_weights.csv"))
lw_bat = dict(zip(lw_df[lw_df["side"]=="batting"]["component"],
                  lw_df[lw_df["side"]=="batting"]["weight"]))
lw_pit = dict(zip(lw_df[lw_df["side"]=="pitching"]["component"],
                  lw_df[lw_df["side"]=="pitching"]["weight"]))

# Precompute league-average RA/BF for pitching RAA
lg_ra_per_bf = (lw_pit["ha"]        * LG_PIT["ha"]  +
                lw_pit["hra_extra"]  * LG_PIT["hra"] +
                lw_pit["bb"]         * LG_PIT["bb"]  +
                lw_pit["hbp"]        * LG_PIT["hp"]  +
                lw_pit["k"]          * LG_PIT["k"]   +
                lw_pit["intercept"])
print(f"  lg_ra_per_bf = {lg_ra_per_bf:.5f}")
print(f"  Bat weights: {lw_bat}")
print(f"  Pit weights: {lw_pit}")

# ── 4. FA SIGNING TRAINING SET ────────────────────────────────────────────────
# Sourced from the in-game transaction log (fa_signings_log.csv), not the
# survivorship-biased contracts.csv snapshot -- see step8_transaction_log_rework.md.
print("\n[4] Loading FA signing training set (transaction log)...")

players = pd.read_csv(os.path.join(DATA, "players.csv"))
fa_log  = pd.read_csv(os.path.join(INT, "fa_signings_log.csv"))

players_sub = players[[
    "ID","date_of_birth","mlb_service_years","Pos","Role","bats"
]].rename(columns={"ID":"player_id"})
players_sub["birth_year"] = pd.to_datetime(
    players_sub["date_of_birth"], errors="coerce"
).dt.year

# Bio position lookup (fallback for players with no fielding data)
bio_pos = players_sub.set_index("player_id")["Pos"].to_dict()

# FA signings only (drop extensions), human-managed era only (signing_year >= 2031;
# pre-2031 was AI-controlled, confirmed with owner).
real_signings = fa_log[(~fa_log["is_extension"]) & (fa_log["human_era"])].copy()
real_signings = real_signings.merge(players_sub, on="player_id", how="left")

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
fa_contracts = real_signings[
    real_signings["service_at_signing"] >= 6  # calibration uses 6+ (established market; see CLAUDE.md)
].copy()
fa_contracts["age_at_signing"] = (
    fa_contracts["signing_year"] - fa_contracts["birth_year"]
)
# aav = total_value/years is the salary0 proxy -- fa_signings_log has no
# year-by-year salary breakdown (flat-AAV approximation; see step 3 notes).
fa_contracts["salary0"] = fa_contracts["aav"]
fa_contracts = fa_contracts.rename(columns={"signing_year": "season_year"})
for col in ("allstar_bonus", "mvp_bonus", "cyyoung_bonus"):
    fa_contracts[col] = 0  # not present in transaction log; unused in feature lists

print(f"  FA signings: {len(fa_contracts)}")
print(f"  By season_year:\n{fa_contracts['season_year'].value_counts().sort_index().to_string()}")
print(f"  By Role: {fa_contracts['Role'].value_counts().to_dict()}")

# ── 5. RETROSPECTIVE FEATURE FUNCTIONS ───────────────────────────────────────
print("\n[5] Computing retrospective features...")

def retro_bat(pid, signing_year):
    """
    3-year gamma-weighted retrospective batting + defense RAR.
    Returns dict or None if insufficient data.
    """
    cutoff = signing_year - 1
    lo     = cutoff - (L_RETRO - 1)

    bat_seasons = bat_by_pid.get(pid)
    if bat_seasons is None:
        return None
    s = bat_seasons[
        (bat_seasons["year"] >= lo) & (bat_seasons["year"] <= cutoff)
    ].copy()
    if len(s) == 0:
        return None

    s["gap"] = cutoff - s["year"]
    s["w"]   = GAMMA ** s["gap"]

    W       = s["w"].sum()
    total_wpa = (s["pa"] * s["w"]).sum()
    if total_wpa == 0:
        return None

    proj_pa = total_wpa / W   # gamma-weighted average PA

    # Volume-and-recency weighted average rates
    def rate(col):
        return (s[col] * s["w"]).sum() / total_wpa

    bat_raa = (
        lw_bat["single"] * (rate("singles_n") - LG["s"]) +
        lw_bat["double"] * (rate("d_n")        - LG["d"]) +
        lw_bat["triple"] * (rate("t_n")        - LG["t"]) +
        lw_bat["hr"]     * (rate("hr_n")       - LG["hr"]) +
        lw_bat["bb"]     * (rate("bb")          - LG["bb"]) +
        lw_bat["hbp"]    * (rate("hp")          - LG["hp"]) +
        lw_bat["k"]      * (rate("k")           - LG["k"])
    ) * proj_pa

    total_wg = (s["g"] * s["w"]).sum()
    proj_g   = total_wg / W
    ubr_rate = (s["ubr"] * s["w"]).sum() / max(total_wg, 1e-9)
    proj_ubr = ubr_rate * proj_g

    # Defense
    proj_def    = 0.0
    primary_pos = int(bio_pos.get(pid, 0)) or 0
    def_s_all   = def_by_pid.get(pid)
    if def_s_all is not None:
        def_s = def_s_all[
            (def_s_all["year"] >= lo) & (def_s_all["year"] <= cutoff)
        ].copy()
        if len(def_s) > 0:
            def_s["w"] = GAMMA ** (cutoff - def_s["year"])
            proj_def = (def_s["def_runs"] * def_s["w"]).sum() / def_s["w"].sum()
            most_recent = def_s.loc[def_s["year"].idxmax()]
            pp = most_recent["primary_pos"]
            if pd.notna(pp) and int(pp) > 0:
                primary_pos = int(pp)

    rep_runs = REPLACEMENT_PER_600PA * proj_pa / 600
    proj_rar = bat_raa + proj_ubr + proj_def + rep_runs

    return {
        "proj_rar":    proj_rar,
        "proj_pa":     proj_pa,
        "primary_pos": primary_pos,
        "n_seasons":   len(s),
        "bat_raa":     bat_raa,
        "proj_ubr":    proj_ubr,
        "proj_def":    proj_def,
    }


def retro_pit(pid, signing_year):
    """
    3-year gamma-weighted retrospective pitching RAR.
    Returns dict or None if insufficient data.
    """
    cutoff = signing_year - 1
    lo     = cutoff - (L_RETRO - 1)

    pit_seasons = pit_by_pid.get(pid)
    if pit_seasons is None:
        return None
    s = pit_seasons[
        (pit_seasons["year"] >= lo) & (pit_seasons["year"] <= cutoff)
    ].copy()
    if len(s) == 0:
        return None

    s["gap"] = cutoff - s["year"]
    s["w"]   = GAMMA ** s["gap"]

    W        = s["w"].sum()
    total_wbf = (s["bf"] * s["w"]).sum()
    if total_wbf == 0:
        return None

    proj_ip  = (s["ip"] * s["w"]).sum() / W
    proj_bf  = total_wbf / W

    def rate(col):
        return (s[col] * s["w"]).sum() / total_wbf

    pit_ra_per_bf = (
        lw_pit["ha"]        * rate("ha_n")  +
        lw_pit["hra_extra"] * rate("hra_n") +
        lw_pit["bb"]        * rate("bb")    +
        lw_pit["hbp"]       * rate("hp")    +
        lw_pit["k"]         * rate("k")     +
        lw_pit["intercept"]
    )

    pit_raa  = (lg_ra_per_bf - pit_ra_per_bf) * proj_bf
    rep_runs = REPLACEMENT_PER_162IP * proj_ip / 162
    proj_rar = pit_raa + rep_runs

    total_wg  = (s["g"]  * s["w"]).sum()
    total_wgs = (s["gs"] * s["w"]).sum()
    sp_flag   = (total_wgs / max(total_wg, 1e-9)) >= 0.5

    total_wgb = (s["gb"] * s["w"]).sum()
    total_wfb = (s["fb"] * s["w"]).sum()
    gb_rate = total_wgb / max(total_wgb + total_wfb, 1e-9)

    return {
        "proj_rar":    proj_rar,
        "proj_ip":     proj_ip,
        "sp_flag":     int(sp_flag),
        "n_seasons":   len(s),
        "pit_raa":     pit_raa,
        "k_rate":      rate("k"),
        "bb_hbp_rate": rate("bb") + rate("hp"),
        "hra_rate":    rate("hra_n"),
        "gb_rate":     gb_rate,
    }


# ── 6. BUILD TRAINING ROWS ────────────────────────────────────────────────────
print("\n[6] Building training rows...")

rows = []
skipped = {"no_bat_data": 0, "no_pit_data": 0, "bad_age": 0}

for _, c in fa_contracts.iterrows():
    pid  = int(c["player_id"])
    yr   = int(c["season_year"])
    role = int(c["Role"]) if pd.notna(c["Role"]) else -1
    age  = c["age_at_signing"]

    if pd.isna(age) or age < 20 or age > 50:
        skipped["bad_age"] += 1
        continue

    row = {
        "player_id":       pid,
        "season_year":     yr,
        "salary0":         c["salary0"],
        "years":           int(c["years"]),
        "log_aav":         np.log(float(c["salary0"])),
        "age":             float(age),
        "Role":            role,
        "allstar_bonus":   int(bool(c.get("allstar_bonus", 0))),
        "mvp_bonus":       int(bool(c.get("mvp_bonus", 0))),
        "cyyoung_bonus":   int(bool(c.get("cyyoung_bonus", 0))),
    }

    if role == 0:
        feat = retro_bat(pid, yr)
        if feat is None:
            skipped["no_bat_data"] += 1
            continue
        row.update(feat)
        row["player_type"] = "batter"
        row["is_premium_def"] = int(feat["primary_pos"] in PREMIUM_DEF_POS)

    elif role in (11, 12, 13):
        feat = retro_pit(pid, yr)
        if feat is None:
            skipped["no_pit_data"] += 1
            continue
        row.update(feat)
        row["player_type"] = "pitcher"
        row["is_premium_def"] = 0

    else:
        # Unknown role; skip
        continue

    rows.append(row)

train_df = pd.DataFrame(rows)
print(f"  Total training rows: {len(train_df)}")
print(f"  Batters: {(train_df['player_type']=='batter').sum()}  "
      f"Pitchers: {(train_df['player_type']=='pitcher').sum()}")
print(f"  By season_year:\n{train_df['season_year'].value_counts().sort_index().to_string()}")
print(f"  Skipped: {skipped}")
print(f"\n  proj_rar range (bat): "
      f"{train_df[train_df['player_type']=='batter']['proj_rar'].min():.1f} to "
      f"{train_df[train_df['player_type']=='batter']['proj_rar'].max():.1f}")
print(f"  proj_rar range (pit): "
      f"{train_df[train_df['player_type']=='pitcher']['proj_rar'].min():.1f} to "
      f"{train_df[train_df['player_type']=='pitcher']['proj_rar'].max():.1f}")

# ── 6b. SALARY FLOOR FILTER ──────────────────────────────────────────────────
# Players with salary0 at the league minimum despite decent stats are
# almost always rating-limited (OOTP ratings drive FA market offers, not
# visible to this model). Including them biases the regression toward
# over-predicting replacement-level contracts and under-predicting stars.
# Apply the same floor as step 3: salary0 > 1.5 × replacement ($1.125M).
REPLACEMENT_SALARY = 750_000
SALARY_FLOOR = REPLACEMENT_SALARY * 1.5   # = $1,125,000

n_total = len(train_df)
train_df = train_df[train_df["salary0"] > SALARY_FLOOR].copy()
n_above  = len(train_df)
print(f"\n  Salary floor ${SALARY_FLOOR:,.0f}: {n_total - n_above} rows removed "
      f"({n_above} remain)")
print(f"  Batters: {(train_df['player_type']=='batter').sum()}  "
      f"Pitchers: {(train_df['player_type']=='pitcher').sum()}")

# ── 6c. DERIVED FEATURES ──────────────────────────────────────────────────────
# proj_rar² — adds quadratic curvature so the model saturates at high RAR
# instead of projecting linearly into never-observed salary territory.
train_df["proj_rar_sq"] = train_df["proj_rar"] ** 2

# log(proj_ip) for pitchers: rewards durability with diminishing returns rather
# than the raw IP value (which was negative at fixed RAR due to efficiency
# collinearity with RAR).  Clipped at 10 IP to avoid log(0).
train_df["log_proj_ip"] = np.log(train_df["proj_ip"].clip(lower=10).fillna(
    train_df["proj_ip"].median()))

# age² for pitchers: LOO-CV ablation confirmed it reduces combined loss by −0.042
# on its own, and combines with k_rate for the best 7-feature pitcher model.
# age² for batters: re-tested 2026-06-17 (research/step9_bat_feature_ablation.py) on the
# rebuilt transaction-log set after step 9 found batters 34+ under-predicted
# by +52% to +87%. Full-sample LOO-CV preferred adding it (-0.061 combined
# loss), but it produced NO improvement in the nested by-signing-year
# validation (step 8: 16.7% -> 16.2% within ±15%, age-bucket bias unchanged).
# Conclusion: the age bias is a small-sample extrapolation problem (only 19
# age-37+ signings total across 5 years), not a missing-curvature problem --
# no feature fixes that with this little data. Reverted; kept batters at 7
# features per the original (pre-rework) ablation finding, now re-confirmed.
pit_mask = train_df["player_type"] == "pitcher"
train_df["age_sq"] = train_df["age"] ** 2

# Interaction terms identified by comprehensive LOO-CV ablation (research/step5_pitcher_ablation3.py):
#   age_x_krate: best single addition (delta=-0.012, alpha stays low at 0.26 -- not collinear)
#   rar_x_hra:   greedy step 2 (delta=-0.003 on top of age_x_krate)
train_df["age_x_krate"] = train_df["age"] * train_df["k_rate"].fillna(0)
train_df["rar_x_hra"]   = train_df["proj_rar"] * train_df["hra_rate"].fillna(0)

print(f"\n  proj_rar² range: {train_df['proj_rar_sq'].min():.0f} – "
      f"{train_df['proj_rar_sq'].max():.0f}")
print(f"  log(proj_ip) range (pitchers): "
      f"{train_df.loc[pit_mask, 'log_proj_ip'].min():.2f} – "
      f"{train_df.loc[pit_mask, 'log_proj_ip'].max():.2f}")
print(f"  k_rate range (pitchers): "
      f"{train_df.loc[pit_mask, 'k_rate'].min():.3f} – "
      f"{train_df.loc[pit_mask, 'k_rate'].max():.3f}")
print(f"  age_x_krate range (pitchers): "
      f"{train_df.loc[pit_mask, 'age_x_krate'].min():.2f} – "
      f"{train_df.loc[pit_mask, 'age_x_krate'].max():.2f}")

# ── 7. RIDGE MODEL FITTING ────────────────────────────────────────────────────
print("\n[7] Fitting ridge models...")

# Batters (7 features): LOO-CV ablation confirmed no benefit from age_sq or
# further additions beyond the decomposed RAR components (re-confirmed
# 2026-06-17 after step 9 -- see age_sq note above for why it was re-tested
# and reverted).
BAT_FEATURES = ["bat_raa", "proj_ubr", "proj_def", "age", "proj_pa",
                "is_premium_def", "proj_rar_sq"]

# Pitchers (9 features): comprehensive LOO-CV ablation (research/step5_pitcher_ablation3.py)
# tested 33 candidates with 40-pt alpha grid + greedy forward selection.
# age_x_krate: best individual addition (delta=-0.012, alpha=0.26 -- not collinear).
#   Captures that the market prices K-rate differently at different ages.
# rar_x_hra: greedy step 2 addition (delta=-0.003): HR suppression premium
#   scales with total RAR -- elite pitchers who also suppress HR get extra value.
# All other candidates (bb_hbp_rate, gb_rate, sp interactions, RAR/age cross-terms,
#   polynomials) either hurt LOO-CV or produced neutral results.
PIT_FEATURES = [
    "proj_rar", "age", "age_sq", "log_proj_ip", "sp_flag",
    "proj_rar_sq", "k_rate", "age_x_krate", "rar_x_hra",
]


def joint_loo_alpha(X_raw, y_aav, y_yrs, alpha_grid):
    """
    Select alpha minimizing sum of normalized LOO-MSE for log(AAV) and years.
    Returns best_alpha and per-alpha results dict.
    """
    loo = LeaveOneOut()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    var_aav = np.var(y_aav)
    var_yrs = np.var(y_yrs)

    results = {}
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        errs_aav, errs_yrs = [], []
        for tr, te in loo.split(X):
            model.fit(X[tr], y_aav[tr])
            errs_aav.append((y_aav[te[0]] - model.predict(X[te])[0]) ** 2)
            model.fit(X[tr], y_yrs[tr])
            errs_yrs.append((y_yrs[te[0]] - model.predict(X[te])[0]) ** 2)
        mse_aav = np.mean(errs_aav) / max(var_aav, 1e-12)
        mse_yrs = np.mean(errs_yrs) / max(var_yrs, 1e-12)
        results[alpha] = {"mse_aav": mse_aav, "mse_yrs": mse_yrs,
                          "combined": mse_aav + mse_yrs}

    best = min(results, key=lambda a: results[a]["combined"])
    return best, results


def fit_and_report(label, df_sub, feature_cols):
    """Fit ridge models for log_aav and years; return coefficient tables + fitted models."""
    X_raw = df_sub[feature_cols].values.astype(float)
    y_aav = df_sub["log_aav"].values.astype(float)
    y_yrs = df_sub["years"].values.astype(float)
    n     = len(df_sub)

    print(f"\n  --- {label} (n={n}) ---")

    best_alpha, cv_results = joint_loo_alpha(X_raw, y_aav, y_yrs, ALPHA_GRID)
    print(f"  Joint LOO-CV alpha selection:")
    print(f"  {'alpha':>8s}  {'MSE_aav':>10s}  {'MSE_yrs':>10s}  {'combined':>10s}")
    for alpha in ALPHA_GRID:
        r = cv_results[alpha]
        mark = " <<" if alpha == best_alpha else ""
        print(f"  {alpha:8.2f}  {r['mse_aav']:10.4f}  {r['mse_yrs']:10.4f}  "
              f"{r['combined']:10.4f}{mark}")

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)

    model_aav = Ridge(alpha=best_alpha).fit(X_sc, y_aav)
    model_yrs = Ridge(alpha=best_alpha).fit(X_sc, y_yrs)

    # In-sample diagnostics
    pred_aav   = model_aav.predict(X_sc)
    pred_yrs   = model_yrs.predict(X_sc)
    r2_aav     = 1 - np.var(y_aav - pred_aav) / np.var(y_aav)
    r2_yrs     = 1 - np.var(y_yrs - pred_yrs) / np.var(y_yrs)
    corr_resid = np.corrcoef(y_aav - pred_aav, y_yrs - pred_yrs)[0, 1]

    print(f"\n  In-sample:  log(AAV) R^2={r2_aav:.3f}  years R^2={r2_yrs:.3f}")
    rmse_log = np.sqrt(np.mean((y_aav - pred_aav)**2))
    print(f"  RMSE log(AAV) = {rmse_log:.3f}  "
          f"(~${np.exp(y_aav.mean()) * rmse_log / 1e6:.2f}M at mean salary)")
    print(f"  Residual correlation (aav_resid vs yrs_resid): {corr_resid:.3f}")
    print(f"\n  log(AAV) coefficients (on standardized features):")
    for f, c in zip(feature_cols, model_aav.coef_):
        print(f"    {f:20s}: {c:+.4f}")
    print(f"  intercept: {model_aav.intercept_:.4f}")
    print(f"\n  years coefficients (on standardized features):")
    for f, c in zip(feature_cols, model_yrs.coef_):
        print(f"    {f:20s}: {c:+.4f}")
    print(f"  intercept: {model_yrs.intercept_:.4f}")

    # Build coefficient save table
    coef_rows = []
    for i, feat in enumerate(feature_cols):
        coef_rows.append({
            "player_type": label,
            "target":      "log_aav",
            "feature":     feat,
            "coef_scaled": model_aav.coef_[i],
            "scaler_mean": scaler.mean_[i],
            "scaler_std":  scaler.scale_[i],
        })
    for i, feat in enumerate(feature_cols):
        coef_rows.append({
            "player_type": label,
            "target":      "years",
            "feature":     feat,
            "coef_scaled": model_yrs.coef_[i],
            "scaler_mean": scaler.mean_[i],
            "scaler_std":  scaler.scale_[i],
        })
    # intercepts
    coef_rows.append({"player_type":label,"target":"log_aav",
                      "feature":"intercept","coef_scaled":model_aav.intercept_,
                      "scaler_mean":np.nan,"scaler_std":np.nan})
    coef_rows.append({"player_type":label,"target":"years",
                      "feature":"intercept","coef_scaled":model_yrs.intercept_,
                      "scaler_mean":np.nan,"scaler_std":np.nan})
    # meta
    coef_rows.append({"player_type":label,"target":"meta",
                      "feature":"alpha","coef_scaled":best_alpha,
                      "scaler_mean":np.nan,"scaler_std":np.nan})
    coef_rows.append({"player_type":label,"target":"meta",
                      "feature":"r2_log_aav","coef_scaled":r2_aav,
                      "scaler_mean":np.nan,"scaler_std":np.nan})
    coef_rows.append({"player_type":label,"target":"meta",
                      "feature":"r2_years","coef_scaled":r2_yrs,
                      "scaler_mean":np.nan,"scaler_std":np.nan})
    coef_rows.append({"player_type":label,"target":"meta",
                      "feature":"n_train","coef_scaled":n,
                      "scaler_mean":np.nan,"scaler_std":np.nan})

    return pd.DataFrame(coef_rows), scaler, model_aav, model_yrs, pred_aav, pred_yrs


bat_df = train_df[train_df["player_type"] == "batter"].copy()
pit_df = train_df[train_df["player_type"] == "pitcher"].copy()

bat_coef, bat_sc, bat_aav_model, bat_yrs_model, bat_pred_aav, bat_pred_yrs = \
    fit_and_report("batter", bat_df, BAT_FEATURES)
pit_coef, pit_sc, pit_aav_model, pit_yrs_model, pit_pred_aav, pit_pred_yrs = \
    fit_and_report("pitcher", pit_df, PIT_FEATURES)

# ── 8. RESIDUAL ANALYSIS ──────────────────────────────────────────────────────
print("\n[8] Residual analysis...")

bat_df = bat_df.copy()
pit_df = pit_df.copy()
bat_df["pred_log_aav"] = bat_pred_aav
bat_df["pred_years"]   = bat_pred_yrs
bat_df["resid_aav"]    = bat_df["log_aav"] - bat_pred_aav
bat_df["pred_aav_M"]   = np.exp(bat_pred_aav) / 1e6
bat_df["actual_aav_M"] = bat_df["salary0"] / 1e6

pit_df["pred_log_aav"] = pit_pred_aav
pit_df["pred_years"]   = pit_pred_yrs
pit_df["resid_aav"]    = pit_df["log_aav"] - pit_pred_aav
pit_df["pred_aav_M"]   = np.exp(pit_pred_aav) / 1e6
pit_df["actual_aav_M"] = pit_df["salary0"] / 1e6

combined = pd.concat([bat_df, pit_df], ignore_index=True, sort=False)

print("\n  Residuals by season_year:")
for yr, grp in combined.groupby("season_year"):
    mean_r = grp["resid_aav"].mean()
    std_r  = grp["resid_aav"].std()
    print(f"    {yr}: n={len(grp):3d}  mean_resid={mean_r:+.3f}  std={std_r:.3f}")

print("\n  Residuals by player_type:")
for pt, grp in combined.groupby("player_type"):
    mean_r = grp["resid_aav"].mean()
    print(f"    {pt}: n={len(grp):3d}  mean_resid={mean_r:+.3f}")

print("\n  Within ±15% of predicted AAV (in $, not log):")
combined["pct_err"] = (combined["actual_aav_M"] - combined["pred_aav_M"]).abs() / combined["pred_aav_M"]
within_15 = (combined["pct_err"] <= 0.15).mean()
within_25 = (combined["pct_err"] <= 0.25).mean()
print(f"    Within ±15%: {within_15:.1%}  Within ±25%: {within_25:.1%}")
print(f"    Median abs % error: {combined['pct_err'].median():.1%}")

# Comparison to step 3
print("\n  Step 3 comparison (crude single-year matching, R^2=0.514):")
from sklearn.linear_model import Ridge as _R
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
if "proj_rar" in combined.columns:
    Xc = pf.fit_transform(combined[["proj_rar"]])
    yc = combined["log_aav"].values
    r3 = _R(alpha=0.01).fit(Xc, yc)
    r2_step3 = 1 - np.var(yc - r3.predict(Xc)) / np.var(yc)
    print(f"    proj_rar-only log(AAV) R^2 on training set: {r2_step3:.3f}")

# ── 9. SAVE OUTPUTS ──────────────────────────────────────────────────────────
print("\n[9] Saving outputs...")

coef_all = pd.concat([bat_coef, pit_coef], ignore_index=True)
coef_all.to_csv(os.path.join(INT, "market_model_coefficients.csv"), index=False)
print(f"  Saved market_model_coefficients.csv ({len(coef_all)} rows)")

# Save training data with predictions
train_out = combined[[
    "player_id","season_year","salary0","years","log_aav","age",
    "player_type","proj_rar","primary_pos",
    "pred_log_aav","pred_years","resid_aav","pred_aav_M","actual_aav_M","pct_err",
]].copy()
# Add component columns where available
for col in ["proj_pa","proj_ip","log_proj_ip","sp_flag","is_premium_def","n_seasons",
            "bat_raa","proj_ubr","proj_def","pit_raa","proj_rar_sq",
            "k_rate","bb_hbp_rate","hra_rate","gb_rate","age_sq",
            "age_x_krate","rar_x_hra"]:
    if col in combined.columns:
        train_out[col] = combined[col].values
train_out.to_csv(os.path.join(INT, "market_training_data.csv"), index=False)
print(f"  Saved market_training_data.csv ({len(train_out)} rows)")

# Residuals summary by position
print("\n  Top over-predicted (actual < predicted):")
miss = train_out.sort_values("resid_aav").head(8)
print(miss[["player_id","season_year","actual_aav_M","pred_aav_M",
             "proj_rar","age","player_type"]].to_string(index=False))

print("\n  Top under-predicted (actual > predicted):")
miss2 = train_out.sort_values("resid_aav", ascending=False).head(8)
print(miss2[["player_id","season_year","actual_aav_M","pred_aav_M",
              "proj_rar","age","player_type"]].to_string(index=False))

# ── 10. SAVE MODEL PREDICTION FUNCTION (for use in step 7) ──────────────────
# The model is fully described by coef + scaler params in the CSV.
# Step 7 will reload this CSV and reconstruct predictions.
#
# Batter application recipe:
#   X_raw = [bat_raa, proj_ubr, proj_def, age, proj_pa, is_premium_def, proj_rar_sq]
#   proj_rar_sq = (bat_raa + proj_ubr + proj_def + rep_runs) ** 2
#
# Pitcher application recipe:
#   X_raw = [proj_rar, age, age_sq, log_proj_ip, sp_flag,
#            proj_rar_sq, k_rate, age_x_krate, rar_x_hra]
#   log_proj_ip = log(max(proj_ip, 10))
#   age_sq      = age ** 2
#   proj_rar_sq = proj_rar ** 2
#   age_x_krate = age * k_rate          (interaction: K premium scales with age)
#   rar_x_hra   = proj_rar * hra_rate   (interaction: HR suppression scales with RAR)
#
# For both:
#   X_sc    = (X_raw - scaler_mean) / scaler_std
#   log_aav = X_sc @ coef_scaled + intercept
#   years   = X_sc @ coef_scaled + intercept  (from years target rows)
#   aav_M   = exp(log_aav) / 1e6   (floor at $750K)

print("\n" + "=" * 60)
print("STEP 5 COMPLETE")
bat_r2 = float(bat_coef[bat_coef["feature"]=="r2_log_aav"]["coef_scaled"].iloc[0])
pit_r2 = float(pit_coef[pit_coef["feature"]=="r2_log_aav"]["coef_scaled"].iloc[0])
bat_r2y = float(bat_coef[bat_coef["feature"]=="r2_years"]["coef_scaled"].iloc[0])
pit_r2y = float(pit_coef[pit_coef["feature"]=="r2_years"]["coef_scaled"].iloc[0])
print(f"  Batter log(AAV) R^2={bat_r2:.3f}  years R^2={bat_r2y:.3f}")
print(f"  Pitcher log(AAV) R^2={pit_r2:.3f}  years R^2={pit_r2y:.3f}")
print(f"  Within ±15% of actual AAV: {within_15:.1%}")
print(f"  Outputs: market_model_coefficients.csv, market_training_data.csv")
print("=" * 60)
