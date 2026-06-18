"""
Step 9 follow-up: re-test whether batters benefit from age_sq / age x proj_rar
interaction now that step 5 trains on the unbiased transaction-log signing set
(216 rows) instead of the old survivorship-biased contracts.csv (134 rows).

The original "batters don't benefit from age_sq" finding (noted in
src/pipeline/step5_market_regression.py) was established on the OLD training set, before the
2026-06-17 rework. Step 9's residual analysis on the REBUILT model found a
strong, monotonic age-related under-prediction bias for batters 34+ that
warrants re-testing this decision on the new data rather than assuming the old
conclusion still holds.

Reuses the identical data-loading / retro-feature code from
src/pipeline/step5_market_regression.py (steps 1-6b) so the training rows are byte-identical
to what step 5 actually fits on. Only the ablation step (testing extra
candidate features for batters via LOO-CV) is new.
"""

import os, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA  = r"C:\Users\Felto\Downloads\ootp_analysis\frostfire_data"
INT   = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"

CURRENT_YEAR  = 2035
VALID_TIDS    = {1,3,4,6,7,8,9,13,16,17,18,20,21,23,24,25,26,27,29,30,301,302}
GAMMA         = 0.9
L_RETRO       = 3
MIN_PA        = 100
MIN_BF        = 100
REPLACEMENT_PER_600PA = 20.0
REPLACEMENT_PER_162IP = 27.0
POS_ADJ = {2:12.5, 3:-12.5, 4:2.5, 5:2.5, 6:7.5, 7:-7.5, 8:2.5, 9:-7.5, 10:-17.5}
PREMIUM_DEF_POS = {2, 6}
ALPHA_GRID = list(np.logspace(-2, 3, 50))
REPLACEMENT_SALARY = 750_000
SALARY_FLOOR = REPLACEMENT_SALARY * 1.5

print("=" * 60)
print("BATTER FEATURE ABLATION (re-test age_sq / age x proj_rar on rebuilt data)")
print("=" * 60)

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
def_by_pid = {pid: grp for pid, grp in def_py.groupby("player_id")}

tb_files = [f for f in glob.glob(os.path.join(DATA, "team_batting_*.csv")) if "vsL" not in f and "vsR" not in f]
team_bat = pd.concat([pd.read_csv(f) for f in sorted(tb_files)], ignore_index=True)
team_bat = team_bat[(team_bat["tid"].isin(VALID_TIDS)) & (team_bat["split_id"] == 1) & (team_bat["pa"] > 0)]
lg_tot = team_bat[["s","d","t","hr","bb","hp","k","pa"]].sum()
LG = {c: lg_tot[c] / lg_tot["pa"] for c in ["s","d","t","hr","bb","hp","k"]}

lw_df = pd.read_csv(os.path.join(INT, "linear_weights.csv"))
lw_bat = dict(zip(lw_df[lw_df["side"]=="batting"]["component"], lw_df[lw_df["side"]=="batting"]["weight"]))

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

rows = []
for _, c in fa_contracts.iterrows():
    pid, yr, role = int(c["player_id"]), int(c["season_year"]), (int(c["Role"]) if pd.notna(c["Role"]) else -1)
    age = c["age_at_signing"]
    if pd.isna(age) or age < 20 or age > 50 or role != 0:
        continue
    feat = retro_bat(pid, yr)
    if feat is None:
        continue
    row = {"player_id": pid, "season_year": yr, "salary0": float(c["salary0"]),
           "log_aav": np.log(float(c["salary0"])), "age": float(age)}
    row.update(feat)
    row["is_premium_def"] = int(feat["primary_pos"] in PREMIUM_DEF_POS)
    rows.append(row)

bat_df = pd.DataFrame(rows)
bat_df = bat_df[bat_df["salary0"] > SALARY_FLOOR].copy()
bat_df["proj_rar_sq"] = bat_df["proj_rar"] ** 2
bat_df["age_sq"] = bat_df["age"] ** 2
bat_df["age_x_rar"] = bat_df["age"] * bat_df["proj_rar"]
print(f"\nBatter training rows: {len(bat_df)}")

BASE = ["bat_raa", "proj_ubr", "proj_def", "age", "proj_pa", "is_premium_def", "proj_rar_sq"]

def joint_loo_combined(X_raw, y_aav, y_yrs, alpha_grid):
    """Select the ridge alpha minimizing combined leave-one-out normalized MSE across log(AAV) and years."""
    loo = LeaveOneOut()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    var_aav, var_yrs = np.var(y_aav), np.var(y_yrs)
    best = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        errs_aav, errs_yrs = [], []
        for tr, te in loo.split(X):
            model.fit(X[tr], y_aav[tr])
            errs_aav.append((y_aav[te[0]] - model.predict(X[te])[0]) ** 2)
            model.fit(X[tr], y_yrs[tr])
            errs_yrs.append((y_yrs[te[0]] - model.predict(X[te])[0]) ** 2)
        combined = np.mean(errs_aav)/max(var_aav,1e-12) + np.mean(errs_yrs)/max(var_yrs,1e-12)
        if best is None or combined < best[1]:
            best = (alpha, combined)
    return best  # (alpha, combined_loss)

y_aav = bat_df["log_aav"].values.astype(float)
y_yrs = bat_df["years"].values.astype(float) if "years" in bat_df.columns else None
if y_yrs is None:
    # years not carried through in this standalone script's row dict; refetch from fa_contracts
    yrs_map = fa_contracts.set_index(["player_id","season_year"])["years"].to_dict()
    bat_df["years"] = [yrs_map.get((r.player_id, r.season_year), np.nan) for r in bat_df.itertuples()]
    y_yrs = bat_df["years"].values.astype(float)

candidates = {
    "baseline (7 feat)": BASE,
    "+ age_sq": BASE + ["age_sq"],
    "+ age_x_rar": BASE + ["age_x_rar"],
    "+ age_sq + age_x_rar": BASE + ["age_sq", "age_x_rar"],
}

print(f"\n{'feature set':28s} {'alpha':>8s} {'combined_loo_loss':>20s}")
results = {}
for label, feats in candidates.items():
    X = bat_df[feats].values.astype(float)
    alpha, combined = joint_loo_combined(X, y_aav, y_yrs, ALPHA_GRID)
    results[label] = combined
    print(f"{label:28s} {alpha:8.3f} {combined:20.4f}")

base_loss = results["baseline (7 feat)"]
print("\nDelta vs baseline (negative = improvement):")
for label, loss in results.items():
    print(f"  {label:28s} {loss - base_loss:+.4f}")
