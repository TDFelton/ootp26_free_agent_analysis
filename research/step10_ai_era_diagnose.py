"""
Step 10, round 2: diagnose WHY blending in AI-era rows hurt held-out human-era
validation (research/step10_ai_era_weight_sweep.py showed every W>0 underperforming
W=0.0, even after fixing the service-time-at-signing eligibility bug).

Two suspected mechanisms, tested independently so we know which one (if
either) is the actual cause:

  (A) Alpha-selection scope bug: the LOO-CV used to pick ridge's alpha was
      leaving out AI-era points too and scoring against them. That tunes
      regularization to minimize error on a mix of human+AI rows, not on
      what we actually deploy against (predicting human-era prices). Fix:
      restrict the leave-one-out evaluation to human-era rows only -- AI-era
      rows are always in the training partition (never held out), since at
      deployment they're training augmentation, never a prediction target.

  (B) Domain-level mismatch: AI-era and human-era FA pricing may follow a
      systematically different price level (and possibly slope) for the same
      underlying performance, because AI GMs and human GMs don't price
      identically. If true, pooling them with a single shared intercept
      drags predictions toward a blended level that fits neither population
      well. Fix: add an `is_ai_era` dummy feature (always 0 at deployment
      time, since future signings are always human-era) so ridge can let
      AI-era rows have their own intercept offset while still sharing the
      RAR/age/etc. slope information. A second variant also lets proj_rar's
      slope differ (`proj_rar_x_ai`) in case the AI/human gap isn't just a
      level shift.

Grid: {loo_scope: all, human_only} x {features: base, +dummy, +dummy+slope}
x {W: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}, all validated on the SAME human-era
nested-by-signing-year folds (2031-2035) as the rest of this round.

Outputs: intermediate/step10_diagnose_v3_results.csv
"""

import os, glob, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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
ALPHA_GRID = list(np.logspace(-2, 3, 30))   # trimmed from 50 -> 30, this grid is run a LOT of times
SALARY_FLOOR = 1_125_000
FOLD_YEARS = [2031, 2032, 2033, 2034, 2035]
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
LOO_SCOPES = ["all", "human_only"]
FEATURE_VARIANTS = ["base", "+dummy", "+dummy+slope"]

print("=" * 60)
print("STEP 10 ROUND 2: diagnosing the AI-era blending regression")
print("=" * 60)

# ── 1. LOAD PANELS (identical to prior scripts) ────────────────────────────────
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

print("\n[1] Building master feature table (full history, corrected service-at-signing)...")
players = pd.read_csv(os.path.join(DATA, "players.csv"))
fa_log  = pd.read_csv(os.path.join(INT, "fa_signings_log.csv"))
players_sub = players[["ID","date_of_birth","mlb_service_years","Pos","Role"]].rename(columns={"ID":"player_id"})
players_sub["birth_year"] = pd.to_datetime(players_sub["date_of_birth"], errors="coerce").dt.year
bio_pos = players_sub.set_index("player_id")["Pos"].to_dict()

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

real_signings = fa_log[~fa_log["is_extension"]].copy()
real_signings = real_signings.merge(players_sub, on="player_id", how="left")
real_signings["service_at_signing"] = real_signings.apply(
    lambda r: service_at_signing(int(r["player_id"]), int(r["signing_year"])), axis=1
)
fa_contracts = real_signings[real_signings["service_at_signing"] >= 6].copy()
fa_contracts["age_at_signing"] = fa_contracts["signing_year"] - fa_contracts["birth_year"]
fa_contracts["salary0"] = fa_contracts["aav"]
fa_contracts = fa_contracts.rename(columns={"signing_year": "season_year"})
fa_contracts = fa_contracts[fa_contracts["season_year"] <= 2035].copy()

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
master["is_ai_era"] = (~master["human_era"]).astype(float)
master["proj_rar_x_ai"] = master["proj_rar"] * master["is_ai_era"]

n_human = master["human_era"].sum()
n_ai = (~master["human_era"]).sum()
print(f"  Master table: {len(master)} signings above salary floor "
      f"({n_human} human-era 2031-2035, {n_ai} AI-era 2015-2030)")

BASE_BAT = ["bat_raa", "proj_ubr", "proj_def", "age", "proj_pa", "is_premium_def", "proj_rar_sq"]
BASE_PIT = ["proj_rar", "age", "age_sq", "log_proj_ip", "sp_flag", "proj_rar_sq",
            "k_rate", "age_x_krate", "rar_x_hra"]

def feats_for(base, variant):
    """Return the feature list for a given ablation variant: the base set, base+is_ai_era dummy, or base+dummy+AI-era RAR slope interaction."""
    if variant == "base":
        return base
    elif variant == "+dummy":
        return base + ["is_ai_era"]
    elif variant == "+dummy+slope":
        return base + ["is_ai_era", "proj_rar_x_ai"]
    raise ValueError(variant)

def fit_weighted_ridge_custom_loo(train_df, feature_cols, loo_scope):
    """Select alpha via LOO restricted to loo_scope, then fit final model on
    the full weighted training set."""
    X_raw = train_df[feature_cols].values.astype(float)
    y_aav = train_df["log_aav"].values.astype(float)
    w = train_df["weight"].values.astype(float)
    is_human = train_df["human_era"].values.astype(bool)

    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    if loo_scope == "all":
        eval_idx = np.arange(len(train_df))
    else:  # human_only
        eval_idx = np.where(is_human)[0]

    var_aav = np.var(y_aav[eval_idx]) if len(eval_idx) > 1 else np.var(y_aav)
    n = len(train_df)
    best_alpha, best_score = None, np.inf
    for alpha in ALPHA_GRID:
        errs = []
        for idx in eval_idx:
            tr_mask = np.ones(n, dtype=bool)
            tr_mask[idx] = False
            m = Ridge(alpha=alpha)
            m.fit(X[tr_mask], y_aav[tr_mask], sample_weight=w[tr_mask])
            pred = m.predict(X[idx:idx+1])[0]
            errs.append((y_aav[idx] - pred) ** 2)
        score = np.mean(errs) / max(var_aav, 1e-12)
        if score < best_score:
            best_score, best_alpha = score, alpha

    model = Ridge(alpha=best_alpha).fit(X, y_aav, sample_weight=w)
    return scaler, model

print("\n[2] Grid: loo_scope x feature_variant x W, validated on human-era folds only...")
all_summaries = []
detail_rows = []

for loo_scope, variant in itertools.product(LOO_SCOPES, FEATURE_VARIANTS):
    for W in W_GRID:
        master_w = master.copy()
        master_w["weight"] = np.where(master_w["human_era"], 1.0, W)
        pool = master_w if W > 0 else master_w[master_w["human_era"]].copy()

        fold_dfs = []
        for test_year in FOLD_YEARS:
            train_m = pool[pool["season_year"] != test_year]
            test_m  = pool[(pool["season_year"] == test_year) & (pool["human_era"])]
            if len(test_m) == 0:
                continue
            fold_preds = []
            for ptype, base in [("batter", BASE_BAT), ("pitcher", BASE_PIT)]:
                feats = feats_for(base, variant)
                tr = train_m[train_m["player_type"] == ptype]
                te = test_m[test_m["player_type"] == ptype]
                if len(tr) < 10 or len(te) == 0:
                    continue
                # is_ai_era / proj_rar_x_ai are constant-zero for human-only pools
                # (W=0) -- ridge handles a zero-variance column fine (coef -> 0),
                # so no special-casing needed.
                scaler, model = fit_weighted_ridge_custom_loo(tr, feats, loo_scope)
                X_te = scaler.transform(te[feats].values.astype(float))
                pred_log_aav = model.predict(X_te)
                for i, (_, row) in enumerate(te.iterrows()):
                    fold_preds.append({
                        "player_id": row["player_id"], "season_year": row["season_year"],
                        "player_type": ptype, "actual_aav_M": row["salary0"] / 1e6,
                        "pred_aav_M": np.exp(pred_log_aav[i]) / 1e6,
                    })
            fold_df = pd.DataFrame(fold_preds)
            if len(fold_df) == 0:
                continue
            fold_df["pred_aav_M"] = fold_df["pred_aav_M"].clip(lower=0.75)
            fold_df["pct_err"] = (fold_df["actual_aav_M"] - fold_df["pred_aav_M"]).abs() / fold_df["pred_aav_M"]
            fold_df["within_15"] = fold_df["pct_err"] <= 0.15
            fold_df["within_25"] = fold_df["pct_err"] <= 0.25
            fold_dfs.append(fold_df)

        fold_all = pd.concat(fold_dfs, ignore_index=True)
        within15 = fold_all["within_15"].mean()
        within25 = fold_all["within_25"].mean()
        med_err = fold_all["pct_err"].median()
        ss_res = np.sum((fold_all["actual_aav_M"] - fold_all["pred_aav_M"])**2)
        ss_tot = np.sum((fold_all["actual_aav_M"] - fold_all["actual_aav_M"].mean())**2)
        r2 = 1 - ss_res / ss_tot

        all_summaries.append({
            "loo_scope": loo_scope, "feature_variant": variant, "W": W,
            "n": len(fold_all), "within_15pct": within15, "within_25pct": within25,
            "median_abs_pct_err": med_err, "r2": r2,
        })
        print(f"  loo={loo_scope:11s} feat={variant:13s} W={W:.1f}  n={len(fold_all):3d}  "
              f"within±15%={within15:.1%}  within±25%={within25:.1%}  "
              f"median_err={med_err:.1%}  R^2={r2:.3f}")

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(os.path.join(INT, "step10_diagnose_v3_results.csv"), index=False)

print("\n" + "=" * 60)
best = summary_df.sort_values("within_15pct", ascending=False).head(5)
print("Top 5 by within±15%:")
print(best.to_string(index=False))
print("\nBaseline to beat (loo=all, feat=base, W=0.0): "
      f"{summary_df[(summary_df.loo_scope=='all') & (summary_df.feature_variant=='base') & (summary_df.W==0.0)].iloc[0].to_dict()}")
print("STEP 10 ROUND 2 COMPLETE -- intermediate/step10_diagnose_v3_results.csv")
print("=" * 60)
