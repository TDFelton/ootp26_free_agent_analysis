"""
Step 5/8 hyperparameter sweep: GAMMA (gap-year decay) x L_RETRO (lookback window)

These two constants in src/pipeline/step5_market_regression.py / src/pipeline/step8_validation.py were
borrowed wholesale from step 4's Marcel CV-fit (GAMMA=0.9, chosen to minimize
performance-projection RMSE) and L_RETRO=3 (a spec default), but never
independently tuned against the thing step 8 actually measures: AAV prediction
accuracy. This script sweeps both against the real nested-by-signing-year
validation harness (identical logic to src/pipeline/step8_validation.py) and reports
whether any combination beats the current production baseline:

    GAMMA=0.9, L_RETRO=3 -> within±15%=18.6%, within±25%=29.6%, R^2=0.568, n=280

Does NOT touch production files. Only reports results.
"""

import os, glob, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA = r"C:\Users\Felto\Downloads\ootp_analysis\frostfire_data"
INT  = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"

VALID_TIDS = {1,3,4,6,7,8,9,13,16,17,18,20,21,23,24,25,26,27,29,30,301,302}
MIN_PA, MIN_BF = 100, 100
REPLACEMENT_PER_600PA = 20.0
REPLACEMENT_PER_162IP = 27.0
POS_ADJ = {2:12.5, 3:-12.5, 4:2.5, 5:2.5, 6:7.5, 7:-7.5, 8:2.5, 9:-7.5, 10:-17.5}
PREMIUM_DEF_POS = {2, 6}
ALPHA_GRID = list(np.logspace(-2, 3, 50))
SALARY_FLOOR = 1_125_000
FOLD_YEARS = [2031, 2032, 2033, 2034, 2035]

BAT_FEATURES = ["bat_raa", "proj_ubr", "proj_def", "age", "proj_pa", "is_premium_def", "proj_rar_sq"]
PIT_FEATURES = ["proj_rar", "age", "age_sq", "log_proj_ip", "sp_flag", "proj_rar_sq",
                 "k_rate", "age_x_krate", "rar_x_hra"]

print("=" * 70)
print("STEP 5/8 HYPERPARAMETER SWEEP: GAMMA x L_RETRO")
print("=" * 70)

# ── 1. LOAD PANELS (fold- and hyperparam-independent) ───────────────────────
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

players = pd.read_csv(os.path.join(DATA, "players.csv"))
fa_log  = pd.read_csv(os.path.join(INT, "fa_signings_log.csv"))
players_sub = players[["ID","date_of_birth","mlb_service_years","Pos","Role"]].rename(columns={"ID":"player_id"})
players_sub["birth_year"] = pd.to_datetime(players_sub["date_of_birth"], errors="coerce").dt.year
bio_pos = players_sub.set_index("player_id")["Pos"].to_dict()

real_signings = fa_log[(~fa_log["is_extension"]) & (fa_log["human_era"])].copy()
real_signings = real_signings.merge(players_sub, on="player_id", how="left")

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
    """Count distinct prior MLB seasons (from real appearance history, not the mlb_service_years snapshot) before signing_year."""
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
fa_contracts = fa_contracts[
    fa_contracts["age_at_signing"].notna() &
    (fa_contracts["age_at_signing"] >= 20) & (fa_contracts["age_at_signing"] <= 50)
].copy()
print(f"  FA contracts eligible for feature building: {len(fa_contracts)}")

# ── 2. PARAMETRIZED RETRO FEATURE FUNCTIONS ──────────────────────────────────

def retro_bat(pid, signing_year, gamma, l_retro):
    """Build a gamma-decayed, l_retro-season-lookback retrospective batting feature set (proj_rar/proj_pa/proj_def/etc.) for player pid as of signing_year."""
    cutoff = signing_year - 1
    lo = cutoff - (l_retro - 1)
    bat_seasons = bat_by_pid.get(pid)
    if bat_seasons is None:
        return None
    s = bat_seasons[(bat_seasons["year"] >= lo) & (bat_seasons["year"] <= cutoff)].copy()
    if len(s) == 0:
        return None
    s["w"] = gamma ** (cutoff - s["year"])
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
            def_s["w"] = gamma ** (cutoff - def_s["year"])
            proj_def = (def_s["def_runs"] * def_s["w"]).sum() / def_s["w"].sum()
            most_recent = def_s.loc[def_s["year"].idxmax()]
            pp = most_recent["primary_pos"]
            if pd.notna(pp) and int(pp) > 0:
                primary_pos = int(pp)
    rep_runs = REPLACEMENT_PER_600PA * proj_pa / 600
    proj_rar = bat_raa + proj_ubr + proj_def + rep_runs
    return {"proj_rar": proj_rar, "proj_pa": proj_pa, "primary_pos": primary_pos,
            "bat_raa": bat_raa, "proj_ubr": proj_ubr, "proj_def": proj_def}

def retro_pit(pid, signing_year, gamma, l_retro):
    """Build a gamma-decayed, l_retro-season-lookback retrospective pitching feature set (proj_rar/proj_ip/sp_flag/etc.) for player pid as of signing_year."""
    cutoff = signing_year - 1
    lo = cutoff - (l_retro - 1)
    pit_seasons = pit_by_pid.get(pid)
    if pit_seasons is None:
        return None
    s = pit_seasons[(pit_seasons["year"] >= lo) & (pit_seasons["year"] <= cutoff)].copy()
    if len(s) == 0:
        return None
    s["w"] = gamma ** (cutoff - s["year"])
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

def build_master(gamma, l_retro):
    """Build the full training table (one row per eligible FA signing) by running retro_bat/retro_pit at the given gamma/l_retro and adding derived features."""
    rows = []
    for _, c in fa_contracts.iterrows():
        pid, yr, role = int(c["player_id"]), int(c["season_year"]), (int(c["Role"]) if pd.notna(c["Role"]) else -1)
        age = c["age_at_signing"]
        row = {"player_id": pid, "season_year": yr, "salary0": float(c["salary0"]),
               "log_aav": np.log(float(c["salary0"])), "age": float(age), "Role": role}
        if role == 0:
            feat = retro_bat(pid, yr, gamma, l_retro)
            if feat is None:
                continue
            row.update(feat); row["player_type"] = "batter"
            row["is_premium_def"] = int(feat["primary_pos"] in PREMIUM_DEF_POS)
        elif role in (11, 12, 13):
            feat = retro_pit(pid, yr, gamma, l_retro)
            if feat is None:
                continue
            row.update(feat); row["player_type"] = "pitcher"
            row["is_premium_def"] = 0
        else:
            continue
        rows.append(row)
    master = pd.DataFrame(rows)
    if len(master) == 0:
        return master
    master = master[master["salary0"] > SALARY_FLOOR].copy()
    master["proj_rar_sq"] = master["proj_rar"] ** 2
    master["log_proj_ip"] = np.log(master["proj_ip"].clip(lower=10)) if "proj_ip" in master.columns else np.nan
    master["age_sq"] = master["age"] ** 2
    master["age_x_krate"] = master["age"] * master.get("k_rate", 0).fillna(0)
    master["rar_x_hra"] = master["proj_rar"] * master.get("hra_rate", 0).fillna(0)
    return master

def joint_loo_alpha(X_raw, y_aav, alpha_grid):
    """Select the ridge alpha minimizing leave-one-out normalized MSE for log(AAV)."""
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
    return min(results, key=results.get)

def fit_regression5(train_df, feature_cols):
    """Fit the step-5-style ridge regression (LOO-selected alpha) on train_df, returning the fitted scaler and model."""
    X_raw = train_df[feature_cols].values.astype(float)
    y_aav = train_df["log_aav"].values.astype(float)
    best_alpha = joint_loo_alpha(X_raw, y_aav, ALPHA_GRID)
    scaler = StandardScaler().fit(X_raw)
    model = Ridge(alpha=best_alpha).fit(scaler.transform(X_raw), y_aav)
    return scaler, model

def run_nested_cv(master):
    """Run step-8-style nested-by-signing-year cross-validation over master, refitting per fold and returning held-out predictions."""
    all_results = []
    for test_year in FOLD_YEARS:
        train_m = master[master["season_year"] != test_year]
        test_m  = master[master["season_year"] == test_year]
        if len(test_m) == 0:
            continue
        fold_preds = []
        for ptype, feats in [("batter", BAT_FEATURES), ("pitcher", PIT_FEATURES)]:
            tr = train_m[train_m["player_type"] == ptype]
            te = test_m[test_m["player_type"] == ptype]
            if len(tr) < 10 or len(te) == 0:
                continue
            scaler, model = fit_regression5(tr, feats)
            X_te = scaler.transform(te[feats].values.astype(float))
            pred_log_aav = model.predict(X_te)
            for i, (_, row) in enumerate(te.iterrows()):
                fold_preds.append({
                    "actual_aav_M": row["salary0"] / 1e6,
                    "pred_aav_M": np.exp(pred_log_aav[i]) / 1e6,
                })
        fold_df = pd.DataFrame(fold_preds)
        if len(fold_df) == 0:
            continue
        fold_df["pred_aav_M"] = fold_df["pred_aav_M"].clip(lower=0.75)
        all_results.append(fold_df)
    if not all_results:
        return None
    results = pd.concat(all_results, ignore_index=True)
    results["pct_err"] = (results["actual_aav_M"] - results["pred_aav_M"]).abs() / results["pred_aav_M"]
    within15 = (results["pct_err"] <= 0.15).mean()
    within25 = (results["pct_err"] <= 0.25).mean()
    med_err = results["pct_err"].median()
    ss_res = np.sum((results["actual_aav_M"] - results["pred_aav_M"])**2)
    ss_tot = np.sum((results["actual_aav_M"] - results["actual_aav_M"].mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"n": len(results), "within_15": within15, "within_25": within25,
            "median_err": med_err, "r2": r2}

# ── 3. SWEEP ───────────────────────────────────────────────────────────────
GAMMA_GRID   = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
LRETRO_GRID  = [2, 3, 4, 5]

print(f"\n[2] Sweeping GAMMA x L_RETRO ({len(GAMMA_GRID)}x{len(LRETRO_GRID)}={len(GAMMA_GRID)*len(LRETRO_GRID)} combos)...")
print(f"{'GAMMA':>6} {'L_RETRO':>8} {'n':>5} {'within15':>10} {'within25':>10} {'med_err':>9} {'R2':>7}")

sweep_rows = []
for gamma, l_retro in itertools.product(GAMMA_GRID, LRETRO_GRID):
    master = build_master(gamma, l_retro)
    if len(master) == 0:
        continue
    metrics = run_nested_cv(master)
    if metrics is None:
        continue
    mark = " <<< BASELINE" if (gamma == 0.9 and l_retro == 3) else ""
    print(f"{gamma:6.2f} {l_retro:8d} {metrics['n']:5d} {metrics['within_15']:10.1%} "
          f"{metrics['within_25']:10.1%} {metrics['median_err']:9.1%} {metrics['r2']:7.3f}{mark}")
    sweep_rows.append({"gamma": gamma, "l_retro": l_retro, **metrics})

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(os.path.join(INT, "step5_gamma_lretro_sweep.csv"), index=False)

baseline = sweep_df[(sweep_df["gamma"] == 0.9) & (sweep_df["l_retro"] == 3)]
best_w15 = sweep_df.sort_values("within_15", ascending=False).iloc[0]
best_r2  = sweep_df.sort_values("r2", ascending=False).iloc[0]

print("\n" + "=" * 70)
print("SWEEP COMPLETE")
print(f"  Baseline (gamma=0.9, l_retro=3): within15={baseline['within_15'].iloc[0]:.1%}  R2={baseline['r2'].iloc[0]:.3f}")
print(f"  Best within±15%: gamma={best_w15['gamma']}, l_retro={int(best_w15['l_retro'])}  "
      f"-> within15={best_w15['within_15']:.1%}, within25={best_w15['within_25']:.1%}, "
      f"med_err={best_w15['median_err']:.1%}, R2={best_w15['r2']:.3f}, n={int(best_w15['n'])}")
print(f"  Best R^2:        gamma={best_r2['gamma']}, l_retro={int(best_r2['l_retro'])}  "
      f"-> within15={best_r2['within_15']:.1%}, R2={best_r2['r2']:.3f}")
print(f"  Saved: intermediate/step5_gamma_lretro_sweep.csv")
print("=" * 70)
