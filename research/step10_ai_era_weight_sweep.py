"""
Step 10: sweep a fixed downweight factor for pre-2031 ("AI days") FA signings
added to the ridge training pool, validating ONLY on the human-era window
(2031-2035) via the same nested-by-signing-year harness as step 8.

Design (confirmed with owner 2026-06-17):
  - Training pool: ALL FA signings 2015-2035 (the full transaction-log
    history, now that src/data/transactions_parser.py has been re-run
    against the complete Mar2015-Oct2035 file set -- see CLAUDE.md step 10).
    Human-era (signing_year >= 2031) rows get weight 1.0; AI-era rows get a
    fixed weight W < 1.0, swept over a grid to find what (if any) W helps.
  - Validation: step 8's FOLD_YEARS (2031-2035) is UNCHANGED -- every test
    fold is still a real human-era signing year. Only the training pool
    feeding each fold grows (it now also includes all AI-era rows at weight
    W, in addition to the other human-era years it already used). This
    keeps the success-criteria check measuring against real human market
    behavior while letting AI-era data inform the fit.
  - W=0.0 reproduces the step 8 baseline exactly (AI-era rows contribute
    nothing) and is included as the control row.
  - W=1.0 is equal weighting (no downweight at all) and is included for
    comparison, though the prior is that some downweight is appropriate
    since AI-era pricing logic may differ from human GM behavior.

Reuses the identical data-loading / retro-feature / ridge-fitting code from
src/pipeline/step5_market_regression.py and src/pipeline/step8_validation.py, with the
human_era filter removed from the FA-signing selection (kept only as a
column for weighting).

Outputs:
  intermediate/step10_ai_era_weight_sweep.csv   -- per-W, per-fold metrics
  prints best W vs baseline to console
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
GAMMA, L_RETRO = 0.9, 3
MIN_PA, MIN_BF = 100, 100
REPLACEMENT_PER_600PA, REPLACEMENT_PER_162IP = 20.0, 27.0
POS_ADJ = {2:12.5, 3:-12.5, 4:2.5, 5:2.5, 6:7.5, 7:-7.5, 8:2.5, 9:-7.5, 10:-17.5}
PREMIUM_DEF_POS = {2, 6}
ALPHA_GRID = list(np.logspace(-2, 3, 50))
SALARY_FLOOR = 1_125_000
FOLD_YEARS = [2031, 2032, 2033, 2034, 2035]   # validation window UNCHANGED -- human era only

W_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]   # AI-era row weight, swept

print("=" * 60)
print("STEP 10: AI-era training-row weight sweep (human-era-only validation)")
print("=" * 60)

# ── 1. LOAD PANELS (identical to step 5/8, full 21-year range) ────────────────
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

def retro_bat(pid, signing_year, bio_pos):
    """Build a gamma-decayed retrospective batting feature set (proj_rar/proj_pa/proj_def/etc.) for player pid as of signing_year."""
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
    bio_pos_v = bio_pos.get(pid, 0)
    primary_pos = int(bio_pos_v) if pd.notna(bio_pos_v) else 0
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
    """Build a gamma-decayed retrospective pitching feature set (proj_rar/proj_ip/sp_flag/etc.) for player pid as of signing_year."""
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

# ── 2. MASTER TABLE: ALL FA signings 2015-2035 (no human_era filter) ──────────
print("\n[1] Building master feature table from fa_signings_log (FULL history)...")

players = pd.read_csv(os.path.join(DATA, "players.csv"))
fa_log  = pd.read_csv(os.path.join(INT, "fa_signings_log.csv"))

players_sub = players[["ID","date_of_birth","mlb_service_years","Pos","Role"]].rename(columns={"ID":"player_id"})
players_sub["birth_year"] = pd.to_datetime(players_sub["date_of_birth"], errors="coerce").dt.year
bio_pos = players_sub.set_index("player_id")["Pos"].to_dict()

# Service-time-at-signing derivation: NOT the current 2035 mlb_service_years
# snapshot minus elapsed years -- that approximation assumes the player kept
# accruing service continuously through 2035, which silently fails for any
# player who has since retired (their mlb_service_years counter is frozen at
# whatever it was when they left, so subtracting years elapsed since an old
# signing overcounts and pushes service_at_signing deeply negative). Confirmed
# empirically: under the snapshot method only 37 of 787 AI-era (pre-2031) FA
# signings survived the >=6 service filter, almost all from 2024-2030 --
# everything earlier was being silently dropped by this artifact.
# Fix: count each player's actual distinct prior MLB seasons directly from
# the performance panels (any appearance at the majors level, batting,
# pitching, or fielding -- not gated by the model's >=100 PA/BF qualifying
# threshold, since service time accrues from being on an active MLB roster,
# not from meeting a performance-sample-size bar). This is robust to
# retirement because it never depends on the player's CURRENT-day state.
bat_years_raw = pd.read_csv(os.path.join(INT, "batting_raw.csv"), usecols=["player_id","year","team_id"])
pit_years_raw = pd.read_csv(os.path.join(INT, "pitching_raw.csv"), usecols=["player_id","year","team_id"])
fld_years_raw = pd.read_csv(os.path.join(INT, "fielding_raw.csv"), usecols=["player_id","year","team_id"])
mlb_years = pd.concat([
    bat_years_raw[bat_years_raw["team_id"].isin(VALID_TIDS)][["player_id","year"]],
    pit_years_raw[pit_years_raw["team_id"].isin(VALID_TIDS)][["player_id","year"]],
    fld_years_raw[fld_years_raw["team_id"].isin(VALID_TIDS)][["player_id","year"]],
], ignore_index=True).drop_duplicates()
mlb_years_by_pid = {pid: set(grp["year"]) for pid, grp in mlb_years.groupby("player_id")}

def service_at_signing(pid, signing_year):
    """Count distinct prior MLB seasons (from real appearance history, not the mlb_service_years snapshot) before signing_year."""
    yrs = mlb_years_by_pid.get(pid)
    if yrs is None:
        return 0
    return sum(1 for y in yrs if y < signing_year)

# NOTE: human_era filter removed from selection -- kept only as a column for weighting.
real_signings = fa_log[~fa_log["is_extension"]].copy()
real_signings = real_signings.merge(players_sub, on="player_id", how="left")
real_signings["service_at_signing"] = real_signings.apply(
    lambda r: service_at_signing(int(r["player_id"]), int(r["signing_year"])), axis=1
)
fa_contracts = real_signings[real_signings["service_at_signing"] >= 6].copy()
fa_contracts["age_at_signing"] = fa_contracts["signing_year"] - fa_contracts["birth_year"]
fa_contracts["salary0"] = fa_contracts["aav"]
fa_contracts = fa_contracts.rename(columns={"signing_year": "season_year"})
fa_contracts = fa_contracts[fa_contracts["season_year"] <= 2035].copy()   # drop pre-signed 2036 rows

rows = []
for _, c in fa_contracts.iterrows():
    pid, yr, role = int(c["player_id"]), int(c["season_year"]), (int(c["Role"]) if pd.notna(c["Role"]) else -1)
    age = c["age_at_signing"]
    if pd.isna(age) or age < 20 or age > 50:
        continue
    row = {"player_id": pid, "season_year": yr, "salary0": float(c["salary0"]),
           "log_aav": np.log(float(c["salary0"])), "age": float(age), "Role": role,
           "human_era": bool(c["human_era"])}
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

n_human = master["human_era"].sum()
n_ai = (~master["human_era"]).sum()
print(f"  Master table: {len(master)} signings above salary floor "
      f"({n_human} human-era 2031-2035, {n_ai} AI-era 2015-2030)")
print(f"  By season_year:\n{master['season_year'].value_counts().sort_index().to_string()}")
print(f"  Batters: {(master['player_type']=='batter').sum()}  Pitchers: {(master['player_type']=='pitcher').sum()}")

BAT_FEATURES = ["bat_raa", "proj_ubr", "proj_def", "age", "proj_pa", "is_premium_def", "proj_rar_sq"]
PIT_FEATURES = ["proj_rar", "age", "age_sq", "log_proj_ip", "sp_flag", "proj_rar_sq",
                 "k_rate", "age_x_krate", "rar_x_hra"]

def joint_loo_alpha_weighted(X_raw, y_aav, w, alpha_grid):
    """Select the ridge alpha minimizing leave-one-out normalized MSE for log(AAV), fitting with per-row sample weights w."""
    loo = LeaveOneOut()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    var_aav = np.var(y_aav)
    results = {}
    for alpha in alpha_grid:
        errs = []
        for tr, te in loo.split(X):
            model = Ridge(alpha=alpha)
            model.fit(X[tr], y_aav[tr], sample_weight=w[tr])
            errs.append((y_aav[te[0]] - model.predict(X[te])[0]) ** 2)
        results[alpha] = np.mean(errs) / max(var_aav, 1e-12)
    return min(results, key=results.get)

def fit_ridge_weighted(train_df, feature_cols):
    """Fit a sample-weighted ridge regression (LOO-selected alpha) on train_df, returning the fitted scaler and model."""
    X_raw = train_df[feature_cols].values.astype(float)
    y_aav = train_df["log_aav"].values.astype(float)
    w = train_df["weight"].values.astype(float)
    best_alpha = joint_loo_alpha_weighted(X_raw, y_aav, w, ALPHA_GRID)
    scaler = StandardScaler().fit(X_raw)
    model = Ridge(alpha=best_alpha).fit(scaler.transform(X_raw), y_aav, sample_weight=w)
    return scaler, model

# ── 3. NESTED-BY-YEAR LOOP (human-era folds only), SWEEPING AI-ERA WEIGHT W ────
print("\n[2] Nested-by-signing-year validation (folds = 2031-2035 only), sweeping W...")

all_rows = []
for W in W_GRID:
    master_w = master.copy()
    master_w["weight"] = np.where(master_w["human_era"], 1.0, W)
    # Drop AI-era rows entirely from the pool when W=0 (faster, and avoids
    # them ever entering a LOO split with zero weight, which Ridge handles
    # fine but is wasted compute).
    pool = master_w if W > 0 else master_w[master_w["human_era"]].copy()

    fold_dfs = []
    for test_year in FOLD_YEARS:
        train_m = pool[pool["season_year"] != test_year]
        test_m  = pool[(pool["season_year"] == test_year) & (pool["human_era"])]
        if len(test_m) == 0:
            continue
        fold_preds = []
        for ptype, feats in [("batter", BAT_FEATURES), ("pitcher", PIT_FEATURES)]:
            tr = train_m[train_m["player_type"] == ptype]
            te = test_m[test_m["player_type"] == ptype]
            if len(tr) < 10 or len(te) == 0:
                continue
            scaler, model = fit_ridge_weighted(tr, feats)
            X_te = scaler.transform(te[feats].values.astype(float))
            pred_log_aav = model.predict(X_te)
            for i, (_, row) in enumerate(te.iterrows()):
                fold_preds.append({
                    "player_id": row["player_id"], "season_year": row["season_year"],
                    "player_type": ptype, "actual_aav_M": row["salary0"] / 1e6,
                    "pred_aav_M": np.exp(pred_log_aav[i]) / 1e6, "W": W,
                    "n_train": len(tr),
                })
        fold_df = pd.DataFrame(fold_preds)
        if len(fold_df) == 0:
            continue
        fold_df["pred_aav_M"] = fold_df["pred_aav_M"].clip(lower=0.75)
        fold_df["pct_err"] = (fold_df["actual_aav_M"] - fold_df["pred_aav_M"]).abs() / fold_df["pred_aav_M"]
        fold_df["within_15"] = fold_df["pct_err"] <= 0.15
        fold_df["within_25"] = fold_df["pct_err"] <= 0.25
        fold_df["test_year"] = test_year
        fold_dfs.append(fold_df)

    fold_all = pd.concat(fold_dfs, ignore_index=True)
    all_rows.append(fold_all)

    within15 = fold_all["within_15"].mean()
    within25 = fold_all["within_25"].mean()
    med_err = fold_all["pct_err"].median()
    ss_res = np.sum((fold_all["actual_aav_M"] - fold_all["pred_aav_M"])**2)
    ss_tot = np.sum((fold_all["actual_aav_M"] - fold_all["actual_aav_M"].mean())**2)
    r2 = 1 - ss_res / ss_tot
    avg_n_train = fold_all["n_train"].mean()

    print(f"  W={W:.1f}  n_test={len(fold_all):3d}  avg_train_pool={avg_n_train:5.0f}  "
          f"within±15%={within15:.1%}  within±25%={within25:.1%}  "
          f"median_err={med_err:.1%}  R^2={r2:.3f}")

results_all = pd.concat(all_rows, ignore_index=True)
results_all.to_csv(os.path.join(INT, "step10_ai_era_weight_sweep.csv"), index=False)

print("\n" + "=" * 60)
print("STEP 10 WEIGHT SWEEP COMPLETE")
print("  Outputs: intermediate/step10_ai_era_weight_sweep.csv")
print("  W=0.0 reproduces step 8's 16.7% within±15% baseline (AI-era rows excluded).")
print("  Pick the best W (if any beats baseline) before propagating into steps 3/5/6/7.")
print("=" * 60)
