"""
Step 9 follow-up #2: leave-one-age-bucket-out diagnostic.

Step 8's nested-by-signing-year validation showed a strong age-related
under-prediction bias (step 9), but that fold structure confounds two
possible causes: (a) genuine age extrapolation (thin age-37+ sample) and
(b) year-to-year market drift. This script isolates (a) directly: hold out
one age bucket at a time (pooling ALL signing years for training), refit,
and measure how badly the model does on the held-out age range. If the
held-out error is large and grows with how thin the bucket is, that
confirms sample-size/extrapolation as the driver -- not something fixable
by adding features.

Reuses the identical data-loading / retro-feature code from
src/pipeline/step5_market_regression.py so training rows match step 5 exactly.

Outputs: prints per-bucket held-out metrics to console; saves
intermediate/step9b_age_holdout_results.csv
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

print("=" * 60)
print("STEP 9b: leave-one-age-bucket-out diagnostic")
print("=" * 60)

# ── 1. LOAD PANELS (identical to step 5/8) ──
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

def retro_bat(pid, signing_year):
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

players = pd.read_csv(os.path.join(DATA, "players.csv"))
fa_log  = pd.read_csv(os.path.join(INT, "fa_signings_log.csv"))
players_sub = players[["ID","date_of_birth","mlb_service_years","Pos","Role"]].rename(columns={"ID":"player_id"})
players_sub["birth_year"] = pd.to_datetime(players_sub["date_of_birth"], errors="coerce").dt.year
bio_pos = players_sub.set_index("player_id")["Pos"].to_dict()

real_signings = fa_log[(~fa_log["is_extension"]) & (fa_log["human_era"])].copy()
real_signings = real_signings.merge(players_sub, on="player_id", how="left")
real_signings["service_at_signing"] = (
    real_signings["mlb_service_years"] - (CURRENT_YEAR - real_signings["signing_year"])
).clip(lower=0)
fa_contracts = real_signings[real_signings["service_at_signing"] >= 6].copy()
fa_contracts["age_at_signing"] = fa_contracts["signing_year"] - fa_contracts["birth_year"]
fa_contracts["salary0"] = fa_contracts["aav"]
fa_contracts = fa_contracts.rename(columns={"signing_year": "season_year"})

rows = []
for _, c in fa_contracts.iterrows():
    pid, yr, role = int(c["player_id"]), int(c["season_year"]), (int(c["Role"]) if pd.notna(c["Role"]) else -1)
    age = c["age_at_signing"]
    if pd.isna(age) or age < 20 or age > 50:
        continue
    row = {"player_id": pid, "season_year": yr, "salary0": float(c["salary0"]),
           "years": int(c["years"]), "log_aav": np.log(float(c["salary0"])), "age": float(age)}
    if role == 0:
        feat = retro_bat(pid, yr)
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
master["age_bucket"] = pd.cut(master["age"], bins=[19, 27, 30, 33, 36, 50],
                                labels=["<=27", "28-30", "31-33", "34-36", "37+"])

print(f"\nMaster table: {len(master)} signings "
      f"({(master['player_type']=='batter').sum()} batters, {(master['player_type']=='pitcher').sum()} pitchers)")
print(master["age_bucket"].value_counts().sort_index().to_string())

BAT_FEATURES = ["bat_raa", "proj_ubr", "proj_def", "age", "proj_pa", "is_premium_def", "proj_rar_sq"]
PIT_FEATURES = ["proj_rar", "age", "age_sq", "log_proj_ip", "sp_flag", "proj_rar_sq",
                 "k_rate", "age_x_krate", "rar_x_hra"]

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

print("\n[Leave-one-age-bucket-out, pooling ALL signing years for training]")
all_results = []
for bucket in ["<=27", "28-30", "31-33", "34-36", "37+"]:
    train_m = master[master["age_bucket"] != bucket]
    test_m  = master[master["age_bucket"] == bucket]
    preds = []
    for ptype, feats in [("batter", BAT_FEATURES), ("pitcher", PIT_FEATURES)]:
        tr = train_m[train_m["player_type"] == ptype]
        te = test_m[test_m["player_type"] == ptype]
        if len(tr) < 10 or len(te) == 0:
            continue
        X_raw = tr[feats].values.astype(float)
        y_aav = tr["log_aav"].values.astype(float)
        best_alpha = joint_loo_alpha(X_raw, y_aav, ALPHA_GRID)
        scaler = StandardScaler().fit(X_raw)
        model = Ridge(alpha=best_alpha).fit(scaler.transform(X_raw), y_aav)
        X_te = scaler.transform(te[feats].values.astype(float))
        pred_log_aav = model.predict(X_te)
        for i, (_, row) in enumerate(te.iterrows()):
            preds.append({"player_id": row["player_id"], "season_year": row["season_year"],
                           "player_type": ptype, "age_bucket": bucket,
                           "actual_aav_M": row["salary0"] / 1e6,
                           "pred_aav_M": max(np.exp(pred_log_aav[i]) / 1e6, 0.75)})
    pdf = pd.DataFrame(preds)
    if len(pdf) == 0:
        continue
    pdf["pct_err"] = (pdf["actual_aav_M"] - pdf["pred_aav_M"]).abs() / pdf["pred_aav_M"]
    pdf["signed_pct_err"] = (pdf["actual_aav_M"] - pdf["pred_aav_M"]) / pdf["pred_aav_M"]
    pdf["within_15"] = pdf["pct_err"] <= 0.15
    within15 = pdf["within_15"].mean()
    med_err = pdf["pct_err"].median()
    mean_signed = pdf["signed_pct_err"].mean()
    ss_res = np.sum((pdf["actual_aav_M"] - pdf["pred_aav_M"])**2)
    ss_tot = np.sum((pdf["actual_aav_M"] - pdf["actual_aav_M"].mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f"  {bucket:6s} (n={len(pdf):3d}, train_n={len(train_m):3d}): "
          f"within±15%={within15:.1%}  median_err={med_err:.1%}  "
          f"mean_signed_err={mean_signed:+.1%}  R^2={r2:.3f}")
    all_results.append(pdf.assign(test_bucket=bucket))

results = pd.concat(all_results, ignore_index=True)
results.to_csv(os.path.join(INT, "step9b_age_holdout_results.csv"), index=False)

print("\n[Comparison] same metric but with bucket KEPT in training (in-distribution baseline):")
for bucket in ["<=27", "28-30", "31-33", "34-36", "37+"]:
    sub_master = master[master["age_bucket"] == bucket]
    print(f"  {bucket:6s}: n={len(sub_master):3d} (this many rows of 'self' data were excluded from training above)")

print("\nSTEP 9b COMPLETE -- intermediate/step9b_age_holdout_results.csv")
