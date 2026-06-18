"""
Step 8: Nested k-fold validation (by signing year)

Outer loop: hold out one signing year (2031-2035) at a time.
Inner loop: project each held-out player's market AAV using a curve-3 and
regression-5 fit trained on every OTHER signing year, then compare the
predicted AAV to the actual signed AAV from fa_signings_log.csv.

Refit-scope decision (confirmed with owner, see step8_transaction_log_rework.md):
  - Aging curves (step 2) and Marcel hyperparameters (step 4) are fit on 21
    years of performance data only -- they never see contracts/signings, so
    there is no leakage path through them. NOT refit per fold.
  - Linear weights (step 3, part 1) are also fit on team-level performance
    data only (no contracts). NOT refit per fold.
  - The $/RAR quadratic curve (step 3, part 2) and the market ridge
    regression (step 5) are both fit directly on signing data. These ARE
    refit per fold, training on every year except the held-out one.
  - The retrospective Marcel-style features themselves (bat_raa, proj_def,
    k_rate, etc.) depend only on each player's OWN performance history before
    their signing year, not on other players' contracts -- so they are
    computed once and reused across folds (recomputing them per fold would
    give identical numbers; refitting only matters for the ridge coefficients).

Success criteria (per spec): 85% of signings within +/-15% of actual AAV.

Inputs:
  intermediate/fa_signings_log.csv, batting_neutral.csv, pitching_neutral.csv,
  fielding_raw.csv, linear_weights.csv, player_values.csv
  frostfire_data/players.csv, team_batting_*.csv, team_pitching_*.csv

Outputs:
  intermediate/step8_validation_results.csv  -- per-signing predictions vs actual
  intermediate/step8_fold_summary.csv        -- per-fold and overall metrics
"""

import os, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA = r"C:\Users\Felto\Downloads\ootp_analysis\frostfire_data"
INT  = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"

CURRENT_YEAR = 2035
VALID_TIDS   = {1,3,4,6,7,8,9,13,16,17,18,20,21,23,24,25,26,27,29,30,301,302}
GAMMA   = 0.9
L_RETRO = 3
MIN_PA, MIN_BF = 100, 100
REPLACEMENT_PER_600PA = 20.0
REPLACEMENT_PER_162IP = 27.0
POS_ADJ = {2:12.5, 3:-12.5, 4:2.5, 5:2.5, 6:7.5, 7:-7.5, 8:2.5, 9:-7.5, 10:-17.5}
PREMIUM_DEF_POS = {2, 6}
ALPHA_GRID = list(np.logspace(-2, 3, 50))
CURVE_ALPHA_GRID = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 50, 100]
SALARY_FLOOR = 1_125_000   # 1.5x replacement, matches steps 3/5
FOLD_YEARS = [2031, 2032, 2033, 2034, 2035]

print("=" * 60)
print("STEP 8: Nested k-fold validation by signing year")
print("=" * 60)

# ── 1. LOAD PANELS (identical to steps 3/5, computed once, fold-independent) ──
print("\n[1] Loading performance panels...")

bat_raw = pd.read_csv(os.path.join(INT, "batting_neutral.csv"),
    usecols=["player_id","year","split_id","pa","g","gs","singles_n","d_n","t_n",
             "hr_n","bb","hp","k","ubr"])
bat_raw = bat_raw[bat_raw["split_id"] == 1].copy()
bat_py = bat_raw.groupby(["player_id","year"], sort=False).agg(
    pa=("pa","sum"), g=("g","sum"), gs=("gs","sum"),
    singles_n=("singles_n","sum"), d_n=("d_n","sum"), t_n=("t_n","sum"),
    hr_n=("hr_n","sum"), bb=("bb","sum"), hp=("hp","sum"), k=("k","sum"), ubr=("ubr","sum"),
).reset_index()
bat_py = bat_py[bat_py["pa"] >= MIN_PA].copy()

pit_raw = pd.read_csv(os.path.join(INT, "pitching_neutral.csv"),
    usecols=["player_id","year","team_id","split_id","bf","outs","g","gs","k","bb",
             "hp","ha_n","hra_n","gb","fb"])
pit_raw = pit_raw[(pit_raw["split_id"] == 1) & (pit_raw["team_id"].isin(VALID_TIDS))].copy()
pit_py = pit_raw.groupby(["player_id","year"], sort=False).agg(
    bf=("bf","sum"), outs=("outs","sum"), g=("g","sum"), gs=("gs","sum"),
    k=("k","sum"), bb=("bb","sum"), hp=("hp","sum"), ha_n=("ha_n","sum"), hra_n=("hra_n","sum"),
    gb=("gb","sum"), fb=("fb","sum"),
).reset_index()
pit_py["ip"] = pit_py["outs"] / 3
pit_py = pit_py[pit_py["bf"] >= MIN_BF].copy()

fld_raw = pd.read_csv(os.path.join(INT, "fielding_raw.csv"),
    usecols=["player_id","year","position","ip","zr","arm","framing"])
fld_raw = fld_raw[fld_raw["ip"] >= 1].copy()
fld_29 = fld_raw[fld_raw["position"].between(2, 9)].copy()
ip_by_pos = fld_29.groupby(["player_id","year","position"])["ip"].sum().reset_index()
prim_idx = ip_by_pos.groupby(["player_id","year"])["ip"].idxmax()
prim_fld = ip_by_pos.loc[prim_idx, ["player_id","year","position"]].rename(columns={"position":"primary_pos"})
def_py = fld_raw.groupby(["player_id","year"], sort=False).agg(
    total_ip=("ip","sum"), def_zr=("zr","sum"), def_arm=("arm","sum"), def_framing=("framing","sum"),
).reset_index()
def_py = def_py.merge(prim_fld, on=["player_id","year"], how="left")
def_py["primary_pos"] = def_py["primary_pos"].fillna(0).astype(int)
def_py["pos_adj"] = def_py["primary_pos"].map(POS_ADJ).fillna(0) * (def_py["total_ip"] / (162 * 8.8))
def_py["def_runs"] = def_py["def_zr"] + def_py["def_arm"] + def_py["def_framing"] + def_py["pos_adj"]

bat_by_pid = {pid: grp for pid, grp in bat_py.groupby("player_id")}
pit_by_pid = {pid: grp for pid, grp in pit_py.groupby("player_id")}
def_by_pid = {pid: grp for pid, grp in def_py.groupby("player_id")}

print(f"  Batting player-years: {len(bat_py)}, pitching: {len(pit_py)}, fielding: {len(def_py)}")

# ── 2. LEAGUE AVERAGES + LINEAR WEIGHTS (fold-independent, stat-only) ─────────
print("\n[2] Loading league averages + linear weights...")

tb_files = [f for f in glob.glob(os.path.join(DATA, "team_batting_*.csv")) if "vsL" not in f and "vsR" not in f]
team_bat = pd.concat([pd.read_csv(f) for f in sorted(tb_files)], ignore_index=True)
team_bat = team_bat[(team_bat["tid"].isin(VALID_TIDS)) & (team_bat["split_id"] == 1) & (team_bat["pa"] > 0)]
lg_tot = team_bat[["s","d","t","hr","bb","hp","k","pa"]].sum()
LG = {c: lg_tot[c] / lg_tot["pa"] for c in ["s","d","t","hr","bb","hp","k"]}

tp_files = [f for f in glob.glob(os.path.join(DATA, "team_pitching_*.csv")) if "vsL" not in f and "vsR" not in f]
team_pit = pd.concat([pd.read_csv(f) for f in sorted(tp_files)], ignore_index=True)
team_pit = team_pit[(team_pit["tid"].isin(VALID_TIDS)) & (team_pit["split_id"] == 1) & (team_pit["bf"] > 0)]
lg_pit_tot = team_pit[["ha","hra","bb","hp","k","bf"]].sum()
LG_PIT = {c: lg_pit_tot[c] / lg_pit_tot["bf"] for c in ["ha","hra","bb","hp","k"]}

lw_df = pd.read_csv(os.path.join(INT, "linear_weights.csv"))
lw_bat = dict(zip(lw_df[lw_df["side"]=="batting"]["component"], lw_df[lw_df["side"]=="batting"]["weight"]))
lw_pit = dict(zip(lw_df[lw_df["side"]=="pitching"]["component"], lw_df[lw_df["side"]=="pitching"]["weight"]))
lg_ra_per_bf = (lw_pit["ha"]*LG_PIT["ha"] + lw_pit["hra_extra"]*LG_PIT["hra"] +
                lw_pit["bb"]*LG_PIT["bb"] + lw_pit["hbp"]*LG_PIT["hp"] +
                lw_pit["k"]*LG_PIT["k"] + lw_pit["intercept"])

# ── 3. RETROSPECTIVE FEATURE FUNCTIONS (identical to step 5) ──────────────────
def retro_bat(pid, signing_year, bio_pos):
    """3-year gamma-weighted retrospective batting + defense RAR for a player as of signing_year. Returns dict or None if insufficient data."""
    cutoff = signing_year - 1
    lo = cutoff - (L_RETRO - 1)
    bat_seasons = bat_by_pid.get(pid)
    if bat_seasons is None:
        return None
    s = bat_seasons[(bat_seasons["year"] >= lo) & (bat_seasons["year"] <= cutoff)].copy()
    if len(s) == 0:
        return None
    s["w"] = GAMMA ** (cutoff - s["year"])
    W = s["w"].sum()
    total_wpa = (s["pa"] * s["w"]).sum()
    if total_wpa == 0:
        return None
    proj_pa = total_wpa / W
    def rate(col):
        return (s[col] * s["w"]).sum() / total_wpa
    bat_raa = (lw_bat["single"]*(rate("singles_n")-LG["s"]) + lw_bat["double"]*(rate("d_n")-LG["d"]) +
               lw_bat["triple"]*(rate("t_n")-LG["t"]) + lw_bat["hr"]*(rate("hr_n")-LG["hr"]) +
               lw_bat["bb"]*(rate("bb")-LG["bb"]) + lw_bat["hbp"]*(rate("hp")-LG["hp"]) +
               lw_bat["k"]*(rate("k")-LG["k"])) * proj_pa
    total_wg = (s["g"] * s["w"]).sum()
    proj_g = total_wg / W
    ubr_rate = (s["ubr"] * s["w"]).sum() / max(total_wg, 1e-9)
    proj_ubr = ubr_rate * proj_g
    proj_def = 0.0
    primary_pos = int(bio_pos.get(pid, 0)) or 0
    def_s_all = def_by_pid.get(pid)
    if def_s_all is not None:
        def_s = def_s_all[(def_s_all["year"] >= lo) & (def_s_all["year"] <= cutoff)].copy()
        if len(def_s) > 0:
            def_s["w"] = GAMMA ** (cutoff - def_s["year"])
            proj_def = (def_s["def_runs"] * def_s["w"]).sum() / def_s["w"].sum()
            most_recent = def_s.loc[def_s["year"].idxmax()]
            pp = most_recent["primary_pos"]
            if pd.notna(pp) and int(pp) > 0:
                primary_pos = int(pp)
    rep_runs = REPLACEMENT_PER_600PA * proj_pa / 600
    proj_rar = bat_raa + proj_ubr + proj_def + rep_runs
    return {"proj_rar": proj_rar, "proj_pa": proj_pa, "primary_pos": primary_pos,
            "bat_raa": bat_raa, "proj_ubr": proj_ubr, "proj_def": proj_def}

def retro_pit(pid, signing_year):
    """3-year gamma-weighted retrospective pitching RAR for a player as of signing_year. Returns dict or None if insufficient data."""
    cutoff = signing_year - 1
    lo = cutoff - (L_RETRO - 1)
    pit_seasons = pit_by_pid.get(pid)
    if pit_seasons is None:
        return None
    s = pit_seasons[(pit_seasons["year"] >= lo) & (pit_seasons["year"] <= cutoff)].copy()
    if len(s) == 0:
        return None
    s["w"] = GAMMA ** (cutoff - s["year"])
    W = s["w"].sum()
    total_wbf = (s["bf"] * s["w"]).sum()
    if total_wbf == 0:
        return None
    proj_ip = (s["ip"] * s["w"]).sum() / W
    proj_bf = total_wbf / W
    def rate(col):
        return (s[col] * s["w"]).sum() / total_wbf
    pit_ra_per_bf = (lw_pit["ha"]*rate("ha_n") + lw_pit["hra_extra"]*rate("hra_n") +
                     lw_pit["bb"]*rate("bb") + lw_pit["hbp"]*rate("hp") +
                     lw_pit["k"]*rate("k") + lw_pit["intercept"])
    pit_raa = (lg_ra_per_bf - pit_ra_per_bf) * proj_bf
    rep_runs = REPLACEMENT_PER_162IP * proj_ip / 162
    proj_rar = pit_raa + rep_runs
    total_wg = (s["g"] * s["w"]).sum()
    total_wgs = (s["gs"] * s["w"]).sum()
    sp_flag = (total_wgs / max(total_wg, 1e-9)) >= 0.5
    return {"proj_rar": proj_rar, "proj_ip": proj_ip, "sp_flag": int(sp_flag),
            "pit_raa": pit_raa, "k_rate": rate("k"), "hra_rate": rate("hra_n")}

# ── 4. BUILD MASTER FEATURE TABLE (all human-era FA signings, once) ───────────
print("\n[3] Building master feature table from fa_signings_log...")

players = pd.read_csv(os.path.join(DATA, "players.csv"))
fa_log  = pd.read_csv(os.path.join(INT, "fa_signings_log.csv"))
player_values = pd.read_csv(os.path.join(INT, "player_values.csv"))

players_sub = players[["ID","date_of_birth","mlb_service_years","Pos","Role"]].rename(columns={"ID":"player_id"})
players_sub["birth_year"] = pd.to_datetime(players_sub["date_of_birth"], errors="coerce").dt.year
bio_pos = players_sub.set_index("player_id")["Pos"].to_dict()

real_signings = fa_log[(~fa_log["is_extension"]) & (fa_log["human_era"])].copy()
real_signings = real_signings.merge(players_sub, on="player_id", how="left")

# Service-time-at-signing: derived from each player's actual MLB appearance
# history (batting/pitching/fielding panels), NOT the current 2035
# mlb_service_years snapshot minus elapsed years -- see CLAUDE.md step 10 for
# why the snapshot-based formula silently breaks for retired players.
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
fa_contracts = real_signings[real_signings["service_at_signing"] >= 6].copy()
fa_contracts["age_at_signing"] = fa_contracts["signing_year"] - fa_contracts["birth_year"]
fa_contracts["salary0"] = fa_contracts["aav"]
fa_contracts = fa_contracts.rename(columns={"signing_year": "season_year"})
fa_contracts = fa_contracts[fa_contracts["season_year"].isin(FOLD_YEARS)].copy()

rows = []
for _, c in fa_contracts.iterrows():
    pid, yr, role = int(c["player_id"]), int(c["season_year"]), (int(c["Role"]) if pd.notna(c["Role"]) else -1)
    age = c["age_at_signing"]
    if pd.isna(age) or age < 20 or age > 50:
        continue
    row = {"player_id": pid, "season_year": yr, "salary0": float(c["salary0"]),
           "log_aav": np.log(float(c["salary0"])), "age": float(age), "Role": role}
    if role == 0:
        feat = retro_bat(pid, yr, bio_pos)
        if feat is None:
            continue
        row.update(feat); row["player_type"] = "batter"
        row["is_premium_def"] = int(feat["primary_pos"] in PREMIUM_DEF_POS)
    elif role in (11, 12, 13):
        feat = retro_pit(pid, yr)
        if feat is None:
            continue
        row.update(feat); row["player_type"] = "pitcher"
        row["is_premium_def"] = 0
    else:
        continue
    rows.append(row)

master = pd.DataFrame(rows)
master = master[master["salary0"] > SALARY_FLOOR].copy()
master["proj_rar_sq"] = master["proj_rar"] ** 2
master["log_proj_ip"] = np.log(master["proj_ip"].clip(lower=10)) if "proj_ip" in master.columns else np.nan
master["age_sq"] = master["age"] ** 2
master["age_x_krate"] = master["age"] * master.get("k_rate", 0).fillna(0)
master["rar_x_hra"] = master["proj_rar"] * master.get("hra_rate", 0).fillna(0)
print(f"  Master table: {len(master)} signings ({(master['player_type']=='batter').sum()} batters, "
      f"{(master['player_type']=='pitcher').sum()} pitchers) above salary floor")
print(f"  By season_year:\n{master['season_year'].value_counts().sort_index().to_string()}")

# Curve-3 style RAR matching (single prior-year RAR from player_values.csv)
master_curve3 = fa_contracts.merge(
    player_values.rename(columns={"year": "perf_year"})[["player_id","perf_year","rar"]],
    left_on=["player_id"], right_on=["player_id"], how="inner"
)
master_curve3["perf_year_target"] = master_curve3["season_year"] - 1
master_curve3 = master_curve3[master_curve3["perf_year"] == master_curve3["perf_year_target"]].copy()
master_curve3 = master_curve3[master_curve3["salary0"] > SALARY_FLOOR].copy()
print(f"  Curve-3 RAR-matched rows: {len(master_curve3)}")

BAT_FEATURES = ["bat_raa", "proj_ubr", "proj_def", "age", "proj_pa", "is_premium_def", "proj_rar_sq"]
PIT_FEATURES = ["proj_rar", "age", "age_sq", "log_proj_ip", "sp_flag", "proj_rar_sq",
                 "k_rate", "age_x_krate", "rar_x_hra"]

def joint_loo_alpha(X_raw, y_aav, alpha_grid):
    """Select the ridge alpha minimizing leave-one-out MSE (normalized by target variance) for log(AAV)."""
    loo = LeaveOneOut()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    var_aav = np.var(y_aav)
    results = {}
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        errs = []
        for tr, te in loo.split(X):
            model.fit(X[tr], y_aav[tr])
            errs.append((y_aav[te[0]] - model.predict(X[te])[0]) ** 2)
        results[alpha] = np.mean(errs) / max(var_aav, 1e-12)
    best = min(results, key=results.get)
    return best

def fit_curve3(train_df):
    """Refit the step 3 quadratic salary~RAR ridge curve on train_df, selecting alpha by 5-fold CV. Returns (intercept, b, c)."""
    X_rar = train_df["rar"].values
    y_sal = train_df["salary0"].values / 1e6
    X_poly = np.column_stack([X_rar, X_rar**2])
    from sklearn.model_selection import KFold, cross_val_score
    kf = KFold(n_splits=min(5, len(train_df)), shuffle=True, random_state=42)
    cv_mse = {a: -cross_val_score(Ridge(alpha=a), X_poly, y_sal, cv=kf,
                                   scoring="neg_mean_squared_error").mean() for a in CURVE_ALPHA_GRID}
    best_alpha = min(cv_mse, key=cv_mse.get)
    ridge = Ridge(alpha=best_alpha).fit(X_poly, y_sal)
    return ridge.intercept_, ridge.coef_[0], ridge.coef_[1]

def fit_regression5(train_df, feature_cols):
    """Refit the step 5 market ridge model (log_aav target) on train_df's feature_cols, selecting alpha via joint_loo_alpha. Returns (scaler, model, best_alpha)."""
    X_raw = train_df[feature_cols].values.astype(float)
    y_aav = train_df["log_aav"].values.astype(float)
    best_alpha = joint_loo_alpha(X_raw, y_aav, ALPHA_GRID)
    scaler = StandardScaler().fit(X_raw)
    model = Ridge(alpha=best_alpha).fit(scaler.transform(X_raw), y_aav)
    return scaler, model, best_alpha

# ── 5. NESTED K-FOLD LOOP ──────────────────────────────────────────────────────
print("\n[4] Running nested k-fold validation...")

all_results = []
fold_summaries = []

for test_year in FOLD_YEARS:
    train_m = master[master["season_year"] != test_year]
    test_m  = master[master["season_year"] == test_year]
    train_c3 = master_curve3[master_curve3["season_year"] != test_year]

    if len(test_m) == 0:
        continue

    # Refit curve 3 (diagnostic only -- not an input to the AAV prediction below,
    # since regression 5 never consumes curve_coefficients.csv; kept for
    # completeness per the refit-scope decision)
    if len(train_c3) >= 5:
        a3, b3, c3 = fit_curve3(train_c3)
    else:
        a3, b3, c3 = np.nan, np.nan, np.nan

    fold_preds = []
    for ptype, feats in [("batter", BAT_FEATURES), ("pitcher", PIT_FEATURES)]:
        tr = train_m[train_m["player_type"] == ptype]
        te = test_m[test_m["player_type"] == ptype]
        if len(tr) < 10 or len(te) == 0:
            continue
        scaler, model, best_alpha = fit_regression5(tr, feats)
        X_te = scaler.transform(te[feats].values.astype(float))
        pred_log_aav = model.predict(X_te)
        for i, (_, row) in enumerate(te.iterrows()):
            fold_preds.append({
                "player_id": row["player_id"], "season_year": row["season_year"],
                "player_type": ptype, "actual_aav_M": row["salary0"] / 1e6,
                "pred_aav_M": np.exp(pred_log_aav[i]) / 1e6, "alpha": best_alpha,
            })

    fold_df = pd.DataFrame(fold_preds)
    if len(fold_df) == 0:
        continue
    fold_df["pred_aav_M"] = fold_df["pred_aav_M"].clip(lower=0.75)  # floor at replacement
    fold_df["pct_err"] = (fold_df["actual_aav_M"] - fold_df["pred_aav_M"]).abs() / fold_df["pred_aav_M"]
    fold_df["within_15"] = fold_df["pct_err"] <= 0.15
    fold_df["within_25"] = fold_df["pct_err"] <= 0.25
    fold_df["test_year"] = test_year
    fold_df["curve3_a"], fold_df["curve3_b"], fold_df["curve3_c"] = a3, b3, c3
    all_results.append(fold_df)

    within15 = fold_df["within_15"].mean()
    within25 = fold_df["within_25"].mean()
    med_err = fold_df["pct_err"].median()
    ss_res = np.sum((fold_df["actual_aav_M"] - fold_df["pred_aav_M"])**2)
    ss_tot = np.sum((fold_df["actual_aav_M"] - fold_df["actual_aav_M"].mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    fold_summaries.append({
        "test_year": test_year, "n": len(fold_df),
        "within_15pct": within15, "within_25pct": within25,
        "median_abs_pct_err": med_err, "r2": r2,
    })
    print(f"  Fold {test_year} (n={len(fold_df):3d}): within±15%={within15:.1%}  "
          f"within±25%={within25:.1%}  median_err={med_err:.1%}  R^2={r2:.3f}")

results = pd.concat(all_results, ignore_index=True)
results.to_csv(os.path.join(INT, "step8_validation_results.csv"), index=False)

# Overall metrics (pooled across folds)
overall_within15 = results["within_15"].mean()
overall_within25 = results["within_25"].mean()
overall_med_err = results["pct_err"].median()
ss_res = np.sum((results["actual_aav_M"] - results["pred_aav_M"])**2)
ss_tot = np.sum((results["actual_aav_M"] - results["actual_aav_M"].mean())**2)
overall_r2 = 1 - ss_res / ss_tot

fold_summary_df = pd.DataFrame(fold_summaries)
fold_summary_df.loc[len(fold_summary_df)] = {
    "test_year": "OVERALL", "n": len(results),
    "within_15pct": overall_within15, "within_25pct": overall_within25,
    "median_abs_pct_err": overall_med_err, "r2": overall_r2,
}
fold_summary_df.to_csv(os.path.join(INT, "step8_fold_summary.csv"), index=False)

print("\n" + "=" * 60)
print("STEP 8 COMPLETE")
print(f"  Total held-out signings evaluated: {len(results)}")
print(f"  OVERALL within ±15% of actual AAV: {overall_within15:.1%}  (success criteria: 85%)")
print(f"  OVERALL within ±25% of actual AAV: {overall_within25:.1%}")
print(f"  OVERALL median abs %% error: {overall_med_err:.1%}")
print(f"  OVERALL R^2 (predicted vs actual AAV): {overall_r2:.3f}")
print(f"  {'MEETS' if overall_within15 >= 0.85 else 'DOES NOT MEET'} the 85%% within ±15%% success criteria.")
print(f"  Outputs: step8_validation_results.csv, step8_fold_summary.csv")
print("=" * 60)
