"""
step4_marcel_hyperparam_diag.py — Step 4 diagnostics + extended hyperparameter search

Reads the existing marcel_cv_scores.csv and intermediate panels, then:
  1. Flags grid-boundary issues (best K at ceiling or floor)
  2. Runs extended K/gamma grids for at-boundary components
  3. Adds volume (PA, IP) CV that is currently missing
  4. Reports CV surface flatness (how sensitive are results to hyperparam choice)
  5. Reports projection distribution sanity stats
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("frostfire_data")
INT_DIR  = Path("intermediate")

PROJ_YEAR    = 2036
CURRENT_YEAR = 2035
SP_THRESHOLD = 0.5

MIN_PA_QUAL  = 100
MIN_BF_QUAL  = 100
MIN_IP_QUAL  = 50
MIN_BAT_VOL  = 50
MIN_PIT_VOL  = 50

VALID_TIDS = {1,3,4,6,7,8,9,13,16,17,18,20,21,23,24,25,26,27,29,30,301,302}
POS_LABELS = {2:"C",3:"1B",4:"2B",5:"3B",6:"SS",7:"LF",8:"CF",9:"RF",10:"DH"}

BAT_COMPONENTS = ["hr_pa", "xbh_pa", "single_pa", "bb_pa", "k_pa", "ubr_g", "hp_pa"]
PIT_COMPONENTS = ["k_bf", "bb_hbp_bf", "hra_bf", "ha_n_bf"]
FLD_COMPONENTS = ["zr_rate", "arm_rate", "framing_rate"]

# Extended grids — regular plus higher K values and broader gamma
K_GRID_BAT_EXT  = [100, 200, 400, 700, 1200, 2000, 3500]
K_GRID_PIT_EXT  = [50, 100, 200, 400, 700, 1200]
K_GRID_FLD_EXT  = [30, 70, 150, 300, 600, 1000, 1500]
K_GRID_VOL_BAT  = [0.5, 1.0, 2.0, 3.0, 5.0]   # in "seasons" units for PA volume
K_GRID_VOL_PIT  = [0.5, 1.0, 2.0, 3.0, 5.0]   # in "seasons" units for IP volume
GAMMA_GRID_EXT  = [0.1, 0.3, 0.5, 0.7, 0.9]
L_GRID          = [2, 3, 4, 5]


def sep(title=""):
    """Prints a section divider, optionally with a centered title line."""
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print("=" * 60)


# ─── LOAD PANELS (same logic as main script) ─────────────────────────────────

def load_players():
    """Loads players.csv bio columns and derives birth_year from date_of_birth."""
    cols = ["ID","date_of_birth","bats","Pos","Role",
            "mlb_service_years","is_active","Level","Retired",
            "First Name","Last Name"]
    p = pd.read_csv(DATA_DIR / "players.csv", usecols=cols)
    p.rename(columns={"ID":"player_id","Pos":"primary_pos",
                       "First Name":"first_name","Last Name":"last_name"}, inplace=True)
    p["birth_year"] = pd.to_datetime(p["date_of_birth"], errors="coerce").dt.year
    return p


def build_position_lookup(players):
    """Builds per-player-year primary position lookups from fielding innings, plus a bio-position fallback.

    Returns:
        Tuple of (primary position by innings-leader per player-year, bio position Series from players.csv).
    """
    fld = pd.read_csv(INT_DIR / "fielding_raw.csv", usecols=["player_id","year","position","ip"])
    fld = fld[fld["position"].between(2, 9)].copy()
    fld_agg = fld.groupby(["player_id","year","position"], sort=False)["ip"].sum().reset_index()
    idx_max = fld_agg.groupby(["player_id","year"])["ip"].idxmax()
    primary = fld_agg.loc[idx_max, ["player_id","year","position"]].copy()
    primary = primary.set_index(["player_id","year"])["position"]
    bio_pos = players.set_index("player_id")["primary_pos"].where(lambda x: x.between(2, 10))
    return primary, bio_pos


def build_batting_panel(pos_lookup):
    """Aggregates park-neutral batting to player-year grain and derives per-PA/per-game rate components plus position."""
    usecols = ["player_id","year","split_id","pa","g",
               "hr_n","d_n","t_n","singles_n","bb","hp","k","ubr"]
    bat = pd.read_csv(INT_DIR / "batting_neutral.csv", usecols=usecols)
    bat = bat[bat["split_id"] == 1].copy()
    agg = bat.groupby(["player_id","year"], sort=False).agg(
        pa=("pa","sum"), g=("g","sum"),
        hr_n=("hr_n","sum"), d_n=("d_n","sum"), t_n=("t_n","sum"),
        singles_n=("singles_n","sum"), bb=("bb","sum"),
        hp=("hp","sum"), k=("k","sum"), ubr=("ubr","sum"),
    ).reset_index()
    agg = agg[agg["pa"] >= MIN_PA_QUAL].copy()
    agg["hr_pa"]     = agg["hr_n"]               / agg["pa"]
    agg["xbh_pa"]    = (agg["d_n"] + agg["t_n"]) / agg["pa"]
    agg["single_pa"] = agg["singles_n"]           / agg["pa"]
    agg["bb_pa"]     = agg["bb"]                  / agg["pa"]
    agg["k_pa"]      = agg["k"]                   / agg["pa"]
    agg["hp_pa"]     = agg["hp"]                  / agg["pa"]
    agg["ubr_g"]     = agg["ubr"]                 / agg["g"].clip(lower=1)
    primary, bio_pos = pos_lookup
    idx = pd.MultiIndex.from_frame(agg[["player_id","year"]])
    pos_fld  = primary.reindex(idx).values.astype(float)
    pos_bio  = agg["player_id"].map(bio_pos).values.astype(float)
    pos_mrg  = np.where(np.isnan(pos_fld), pos_bio, pos_fld)
    agg["position"] = np.where(np.isnan(pos_mrg), 0, pos_mrg).astype(int)
    agg["position"] = agg["position"].replace({10: 3})
    agg["pos_label"] = agg["position"].map(POS_LABELS).fillna("unknown")
    return agg


def build_pitching_panel():
    """Aggregates park-neutral pitching to player-year grain, derives per-BF rate components, and flags SP vs RP."""
    usecols = ["player_id","year","team_id","split_id",
               "bf","outs","g","gs","k","bb","hp","ha_n","hra_n"]
    pit = pd.read_csv(INT_DIR / "pitching_neutral.csv", usecols=usecols)
    pit = pit[(pit["split_id"] == 1) & (pit["team_id"].isin(VALID_TIDS))].copy()
    agg = pit.groupby(["player_id","year"], sort=False).agg(
        bf=("bf","sum"), outs=("outs","sum"),
        g=("g","sum"), gs=("gs","sum"),
        k=("k","sum"), bb=("bb","sum"), hp=("hp","sum"),
        ha_n=("ha_n","sum"), hra_n=("hra_n","sum"),
    ).reset_index()
    agg["ip"]  = agg["outs"] / 3
    agg = agg[agg["bf"] >= MIN_BF_QUAL].copy()
    agg["k_bf"]       = agg["k"]                / agg["bf"]
    agg["bb_hbp_bf"]  = (agg["bb"] + agg["hp"]) / agg["bf"]
    agg["hra_bf"]     = agg["hra_n"]             / agg["bf"]
    agg["ha_n_bf"]    = agg["ha_n"]              / agg["bf"]
    agg["sp_flag"] = (agg["gs"] / agg["g"].clip(lower=1)) >= SP_THRESHOLD
    agg["role"]    = agg["sp_flag"].map({True:"SP", False:"RP"})
    return agg


def build_fielding_panel():
    """Aggregates fielding to player-year grain and derives per-1000-IP zone/arm/framing rates plus primary position."""
    usecols = ["player_id","year","position","ip","zr","arm","framing"]
    fld = pd.read_csv(INT_DIR / "fielding_raw.csv", usecols=usecols)
    fld = fld[fld["ip"] >= 1].copy()
    fld_29 = fld[fld["position"].between(2, 9)].copy()
    fld_ip = (fld_29.groupby(["player_id","year","position"])["ip"]
              .sum().reset_index())
    idx_max = fld_ip.groupby(["player_id","year"])["ip"].idxmax()
    prim_fld = (fld_ip.loc[idx_max, ["player_id","year","position"]]
                .set_index(["player_id","year"])["position"])
    agg = fld.groupby(["player_id","year"], sort=False).agg(
        ip=("ip","sum"), zr=("zr","sum"),
        arm=("arm","sum"), framing=("framing","sum"),
    ).reset_index()
    agg = agg[agg["ip"] >= MIN_IP_QUAL].copy()
    agg["zr_rate"]      = agg["zr"]      / agg["ip"] * 1000
    agg["arm_rate"]     = agg["arm"]     / agg["ip"] * 1000
    agg["framing_rate"] = agg["framing"] / agg["ip"] * 1000
    idx = pd.MultiIndex.from_frame(agg[["player_id","year"]])
    agg["fld_pos"]       = prim_fld.reindex(idx).values
    agg["fld_pos"]       = pd.to_numeric(agg["fld_pos"], errors="coerce")
    agg["fld_pos_label"] = agg["fld_pos"].map(
        lambda x: POS_LABELS.get(int(x), "unknown") if pd.notna(x) else "unknown"
    )
    return agg


def build_league_means(bat, pit, fld, players):
    """Builds league-average lookup tables (overall, by position/role, and by age within group) for every component, used as Marcel regression priors."""
    bio = players[["player_id","birth_year"]].dropna().set_index("player_id")["birth_year"]
    MIN_AGE_N = 10
    def weighted_mean(df, rc, vc):
        w = df[vc]
        return float((df[rc]*w).sum()/w.sum()) if w.sum()>0 else float(df[rc].mean())
    def age_means(grp, rc, vc):
        res={}
        for age, gg in grp.groupby("age"):
            if 20<=int(age)<=40 and len(gg)>=MIN_AGE_N:
                res[int(age)]=weighted_mean(gg,rc,vc)
        return res
    means={}
    bat_a=bat.copy(); bat_a["age"]=bat_a["player_id"].map(bio).rsub(bat_a["year"])
    overall_bat={}
    for comp in BAT_COMPONENTS:
        vol="g" if comp=="ubr_g" else "pa"
        overall_bat[comp]=weighted_mean(bat_a,comp,vol) if comp in bat_a.columns else 0.0
    for pos, grp in bat_a[bat_a["position"]>0].groupby("position"):
        lbl=POS_LABELS.get(int(pos),str(pos))
        for comp in BAT_COMPONENTS:
            if comp not in bat_a.columns: continue
            vol="g" if comp=="ubr_g" else "pa"
            pm=weighted_mean(grp,comp,vol) if len(grp)>=5 else overall_bat[comp]
            means[(lbl,comp)]={"age_means":age_means(grp,comp,vol),"pos_mean":pm,"overall_mean":overall_bat[comp]}
    for comp in BAT_COMPONENTS:
        if comp not in bat_a.columns: continue
        vol="g" if comp=="ubr_g" else "pa"
        means[("overall",comp)]={"age_means":{},"pos_mean":overall_bat[comp],"overall_mean":overall_bat[comp]}
    # Volume (PA)
    for pos, grp in bat_a[bat_a["position"]>0].groupby("position"):
        lbl=POS_LABELS.get(int(pos),str(pos))
        ov_pa=float(grp["pa"].mean()) if len(grp) else 400.0
        am_pa={}
        for age, gg in grp.groupby("age"):
            if 20<=int(age)<=40 and len(gg)>=MIN_AGE_N:
                am_pa[int(age)]=float(gg["pa"].mean())
        means[(lbl,"pa")]={"age_means":am_pa,"pos_mean":ov_pa,"overall_mean":ov_pa}
    means[("overall","pa")]={"age_means":{},"pos_mean":float(bat_a["pa"].mean()),"overall_mean":float(bat_a["pa"].mean())}
    # Pitching
    pit_a=pit.copy(); pit_a["age"]=pit_a["player_id"].map(bio).rsub(pit_a["year"])
    for role, grp in pit_a.groupby("role"):
        for comp in PIT_COMPONENTS:
            ov=weighted_mean(grp,comp,"bf") if len(grp)>=5 else 0.0
            means[(role,comp)]={"age_means":age_means(grp,comp,"bf"),"pos_mean":ov,"overall_mean":ov}
        ov_ip=float(grp["ip"].mean()) if len(grp) else 100.0
        am_ip={}
        for age, gg in grp.groupby("age"):
            if 20<=int(age)<=40 and len(gg)>=MIN_AGE_N:
                am_ip[int(age)]=float(gg["ip"].mean())
        means[(role,"ip")]={"age_means":am_ip,"pos_mean":ov_ip,"overall_mean":ov_ip}
    # Fielding
    fld_a=fld.copy(); fld_a["age"]=fld_a["player_id"].map(bio).rsub(fld_a["year"])
    for pos_lbl, grp in fld_a[fld_a["fld_pos_label"]!="unknown"].groupby("fld_pos_label"):
        for comp in FLD_COMPONENTS:
            ov=weighted_mean(grp,comp,"ip") if len(grp)>=5 else 0.0
            means[(pos_lbl,comp)]={"age_means":age_means(grp,comp,"ip"),"pos_mean":ov,"overall_mean":ov}
    return means


def lookup_mean(means, group_label, comp, age):
    """Looks up the league-mean prior for a component, preferring age-specific mean, then group mean, then overall mean."""
    for key in [(group_label, comp), ("overall", comp)]:
        if key in means:
            e=means[key]
            v=e["age_means"].get(age)
            if v is not None: return float(v)
            pm=e.get("pos_mean")
            if pm is not None: return float(pm)
            return float(e.get("overall_mean", 0.0))
    return 0.0


def marcel_weights(qualifying_years, target_year, L, gamma):
    """Computes per-year Marcel weights over the lookback window, decaying by recency and by gamma**cumulative_gap for missing years."""
    window = sorted([y for y in qualifying_years if target_year-L<=y<target_year], reverse=True)
    if not window: return {}
    weights={}; cum_gap=0; prev=target_year
    for yr in window:
        gap=prev-yr-1; cum_gap+=gap
        base=float(max(0, L-(target_year-yr)+1))
        weights[yr]=base*(gamma**cum_gap); prev=yr
    return weights


def marcel_predict(history, target_year, league_mean, L, K, gamma):
    """Computes the Marcel projection for one player-year: weighted history rate regressed toward the league mean by K.

    Returns:
        Tuple of (projected rate, total weighted volume).
    """
    w=marcel_weights(set(history), target_year, L, gamma)
    if not w: return float(league_mean), 0.0
    total_wvol=sum(w[yr]*history[yr][1] for yr in w)
    total_wcount=sum(w[yr]*history[yr][1]*history[yr][0] for yr in w)
    if total_wvol<=0: return float(league_mean), 0.0
    raw=total_wcount/total_wvol
    proj=(raw*total_wvol+K*league_mean)/(total_wvol+K)
    return float(proj), float(total_wvol)


def cv_fit_extended(panel, comp, vol_col, group_col, bio, means, K_grid,
                    gamma_grid=None, min_vol=MIN_BAT_VOL, label=""):
    """Runs a leave-future-out CV grid search over L/K/gamma for one rate component and returns the full grid plus the best combo."""
    if gamma_grid is None:
        gamma_grid = GAMMA_GRID_EXT
    player_hist={}
    for pid, grp in panel.groupby("player_id"):
        player_hist[pid]={row["year"]:(row[comp],row[vol_col]) for _,row in grp.iterrows()}
    test_panel=panel[(panel["year"]>=2017)&(panel["year"]<=CURRENT_YEAR)].copy()
    test_panel["birth_year"]=test_panel["player_id"].map(bio)
    test_panel=test_panel.dropna(subset=["birth_year"])
    test_panel["age"]=(test_panel["year"]-test_panel["birth_year"]).astype(int)
    test_panel=test_panel[test_panel[vol_col]>=min_vol]
    records=[
        {"pid":r.player_id,"year":int(r.year),"actual_rate":float(getattr(r,comp)),
         "actual_vol":float(getattr(r,vol_col)),"group_label":str(getattr(r,group_col)),"age":int(r.age)}
        for r in test_panel.itertuples()
    ]
    if not records:
        return pd.DataFrame(), None
    cv_rows=[]
    for L in L_GRID:
        for K in K_grid:
            for gamma in gamma_grid:
                wss=0.0; wt=0.0
                for rec in records:
                    hist={y:v for y,v in player_hist[rec["pid"]].items() if y<rec["year"]}
                    lm=lookup_mean(means,rec["group_label"],comp,rec["age"])
                    proj,_=marcel_predict(hist,rec["year"],lm,L,K,gamma)
                    wss+=(proj-rec["actual_rate"])**2*rec["actual_vol"]
                    wt+=rec["actual_vol"]
                rmse=float(np.sqrt(wss/wt)) if wt>0 else np.inf
                cv_rows.append({"component":comp,"L":L,"K":K,"gamma":gamma,"rmse":rmse})
    cv_df=pd.DataFrame(cv_rows)
    best=cv_df.loc[cv_df["rmse"].idxmin()]
    return cv_df, {"L":int(best.L),"K":float(best.K),"gamma":float(best.gamma),"rmse":float(best.rmse)}


def cv_fit_volume(panel, rate_col, vol_col, group_col, bio, means, K_grid_seasons, gamma_grid=None, min_vol=1):
    """CV for volume (PA or IP) where the 'rate' IS the volume per season."""
    if gamma_grid is None:
        gamma_grid = GAMMA_GRID_EXT
    # Build per-player history: each year maps (volume, 1.0) — volume treated as rate against unit denominator
    player_hist={}
    for pid, grp in panel.groupby("player_id"):
        player_hist[pid]={row["year"]:(row[vol_col], 1.0) for _,row in grp.iterrows()}
    test_panel=panel[(panel["year"]>=2017)&(panel["year"]<=CURRENT_YEAR)].copy()
    test_panel["birth_year"]=test_panel["player_id"].map(bio)
    test_panel=test_panel.dropna(subset=["birth_year"])
    test_panel["age"]=(test_panel["year"]-test_panel["birth_year"]).astype(int)
    # For volume, min_vol is in seasons (≥1 actual season)
    records=[
        {"pid":r.player_id,"year":int(r.year),"actual_rate":float(getattr(r,vol_col)),
         "actual_vol":1.0,"group_label":str(getattr(r,group_col)),"age":int(r.age)}
        for r in test_panel.itertuples()
    ]
    if not records:
        return pd.DataFrame(), None
    # For volume means, we need a league mean (average PA/IP at that age/position)
    # Borrow from means[(group, vol_col)] which was built in build_league_means
    cv_rows=[]
    for L in L_GRID:
        for K in K_grid_seasons:
            for gamma in gamma_grid:
                wss=0.0; wt=0.0
                for rec in records:
                    hist={y:v for y,v in player_hist[rec["pid"]].items() if y<rec["year"]}
                    lm=lookup_mean(means,rec["group_label"],vol_col,rec["age"])
                    proj,_=marcel_predict(hist,rec["year"],lm,L,K,gamma)
                    proj=float(np.clip(proj, 0, 700))
                    wss+=(proj-rec["actual_rate"])**2
                    wt+=1.0
                rmse=float(np.sqrt(wss/wt)) if wt>0 else np.inf
                cv_rows.append({"component":vol_col,"L":L,"K":K,"gamma":gamma,"rmse":rmse})
    cv_df=pd.DataFrame(cv_rows)
    best=cv_df.loc[cv_df["rmse"].idxmin()]
    return cv_df, {"L":int(best.L),"K":float(best.K),"gamma":float(best.gamma),"rmse":float(best.rmse)}


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    """Runs the full step 4 diagnostic suite: grid-boundary checks, CV surface flatness, extended hyperparameter search, volume CV, and projection sanity stats."""
    sep("STEP 4 DIAGNOSTICS")

    # ── 1. GRID BOUNDARY ANALYSIS on existing results ──────────────────────
    sep("1. GRID BOUNDARY ANALYSIS")
    orig_hp = pd.read_csv(INT_DIR / "marcel_hyperparams.csv")
    orig_cv = pd.read_csv(INT_DIR / "marcel_cv_scores.csv")

    bat_K_max = 1200; pit_K_max = 700; fld_K_max = 600
    bat_K_min = 100;  pit_K_min = 50;  fld_K_min = 30
    gamma_max = 0.7;  gamma_min = 0.3

    boundary_flags = []
    for _, row in orig_hp.iterrows():
        comp  = row["component"]
        flags = []
        if comp in BAT_COMPONENTS:
            if row["K"] >= bat_K_max: flags.append("K_AT_CEILING")
            if row["K"] <= bat_K_min: flags.append("K_AT_FLOOR")
        elif comp in PIT_COMPONENTS:
            if row["K"] >= pit_K_max: flags.append("K_AT_CEILING")
            if row["K"] <= pit_K_min: flags.append("K_AT_FLOOR")
        elif comp in FLD_COMPONENTS:
            if row["K"] >= fld_K_max: flags.append("K_AT_CEILING")
            if row["K"] <= fld_K_min: flags.append("K_AT_FLOOR")
        if row["gamma"] >= gamma_max: flags.append("GAMMA_AT_CEILING")
        if row["gamma"] <= gamma_min: flags.append("GAMMA_AT_FLOOR")
        boundary_flags.append({"component": comp, "L": int(row["L"]),
                                "K": row["K"], "gamma": row["gamma"],
                                "rmse": row["rmse"], "flags": "|".join(flags) if flags else "OK"})

    bf_df = pd.DataFrame(boundary_flags)
    print(bf_df.to_string(index=False))

    at_K_ceil  = bf_df[bf_df["flags"].str.contains("K_AT_CEILING")]["component"].tolist()
    at_K_floor = bf_df[bf_df["flags"].str.contains("K_AT_FLOOR")]["component"].tolist()
    at_G_ceil  = bf_df[bf_df["flags"].str.contains("GAMMA_AT_CEILING")]["component"].tolist()
    at_G_floor = bf_df[bf_df["flags"].str.contains("GAMMA_AT_FLOOR")]["component"].tolist()
    print(f"\nComponents with K at ceiling ({len(at_K_ceil)}): {at_K_ceil}")
    print(f"Components with K at floor   ({len(at_K_floor)}): {at_K_floor}")
    print(f"Components with gamma at ceiling ({len(at_G_ceil)}): {at_G_ceil}")
    print(f"Components with gamma at floor   ({len(at_G_floor)}): {at_G_floor}")

    # ── 2. CV SURFACE FLATNESS ─────────────────────────────────────────────
    sep("2. CV SURFACE FLATNESS (existing grid)")
    flat_rows = []
    for comp in orig_cv["component"].unique():
        sub = orig_cv[orig_cv["component"] == comp]
        best_rmse = sub["rmse"].min()
        worst_rmse = sub["rmse"].max()
        p90_rmse = sub["rmse"].quantile(0.9)
        flat_rows.append({
            "component": comp,
            "best_rmse":  round(best_rmse, 6),
            "worst_rmse": round(worst_rmse, 6),
            "range":      round(worst_rmse - best_rmse, 6),
            "pct_range":  round((worst_rmse - best_rmse) / best_rmse * 100, 2),
            "p90_vs_best":round((p90_rmse - best_rmse) / best_rmse * 100, 2),
            "n_configs":  len(sub),
        })
    flat_df = pd.DataFrame(flat_rows).sort_values("pct_range", ascending=False)
    print(flat_df.to_string(index=False))
    print("\nLow pct_range = flat surface (choice of hyperparams doesn't matter much)")
    print("High pct_range = sensitive (getting the params right is important)")

    # ── 3. LOAD PANELS FOR EXTENDED GRID SEARCH ───────────────────────────
    sep("3. EXTENDED GRID SEARCH")
    print("Loading panels...")
    players = load_players()
    bio = players[["player_id","birth_year"]].dropna().set_index("player_id")["birth_year"]
    pos_lkp = build_position_lookup(players)
    bat = build_batting_panel(pos_lkp)
    pit = build_pitching_panel()
    fld = build_fielding_panel()
    means = build_league_means(bat, pit, fld, players)
    print(f"  Panels loaded: bat={len(bat)}, pit={len(pit)}, fld={len(fld)}")

    ext_results = []

    # Batting — at-ceiling components
    bat_ceil_comps = [c for c in at_K_ceil if c in BAT_COMPONENTS]
    if bat_ceil_comps:
        print(f"\nExtended K search for batting: {bat_ceil_comps}")
        for comp in bat_ceil_comps:
            vol = "g" if comp == "ubr_g" else "pa"
            cv_df, best = cv_fit_extended(bat, comp, vol, "pos_label", bio, means,
                                          K_GRID_BAT_EXT, GAMMA_GRID_EXT, label=comp)
            if best:
                old = orig_hp[orig_hp["component"]==comp].iloc[0]
                improvement = (old["rmse"] - best["rmse"]) / old["rmse"] * 100
                print(f"  {comp:15s}: OLD K={old['K']:6.0f} g={old['gamma']:.1f} RMSE={old['rmse']:.6f} | "
                      f"NEW K={best['K']:6.0f} g={best['gamma']:.1f} RMSE={best['rmse']:.6f} | "
                      f"d={improvement:+.3f}%")
                ext_results.append({"component":comp,"domain":"batting",
                                    "old_K":old["K"],"new_K":best["K"],
                                    "old_gamma":old["gamma"],"new_gamma":best["gamma"],
                                    "old_rmse":old["rmse"],"new_rmse":best["rmse"],
                                    "pct_improvement":improvement,
                                    "new_L":best["L"]})

    # Fielding — at-ceiling components
    fld_ceil_comps = [c for c in at_K_ceil if c in FLD_COMPONENTS]
    if fld_ceil_comps:
        print(f"\nExtended K search for fielding: {fld_ceil_comps}")
        for comp in fld_ceil_comps:
            cv_df, best = cv_fit_extended(fld, comp, "ip", "fld_pos_label", bio, means,
                                          K_GRID_FLD_EXT, GAMMA_GRID_EXT, label=comp)
            if best:
                old = orig_hp[orig_hp["component"]==comp].iloc[0]
                improvement = (old["rmse"] - best["rmse"]) / old["rmse"] * 100
                print(f"  {comp:15s}: OLD K={old['K']:6.0f} g={old['gamma']:.1f} RMSE={old['rmse']:.6f} | "
                      f"NEW K={best['K']:6.0f} g={best['gamma']:.1f} RMSE={best['rmse']:.6f} | "
                      f"d={improvement:+.3f}%")
                ext_results.append({"component":comp,"domain":"fielding",
                                    "old_K":old["K"],"new_K":best["K"],
                                    "old_gamma":old["gamma"],"new_gamma":best["gamma"],
                                    "old_rmse":old["rmse"],"new_rmse":best["rmse"],
                                    "pct_improvement":improvement,
                                    "new_L":best["L"]})

    # Pitching — at-ceiling or floor gamma
    pit_ext_comps = [c for c in PIT_COMPONENTS if c in at_G_floor or c in at_G_ceil]
    if pit_ext_comps:
        print(f"\nExtended gamma search for pitching: {pit_ext_comps}")
        for comp in pit_ext_comps:
            cv_df, best = cv_fit_extended(pit, comp, "bf", "role", bio, means,
                                          K_GRID_PIT_EXT, GAMMA_GRID_EXT,
                                          min_vol=MIN_PIT_VOL, label=comp)
            if best:
                old = orig_hp[orig_hp["component"]==comp].iloc[0]
                improvement = (old["rmse"] - best["rmse"]) / old["rmse"] * 100
                print(f"  {comp:15s}: OLD K={old['K']:6.0f} g={old['gamma']:.1f} RMSE={old['rmse']:.6f} | "
                      f"NEW K={best['K']:6.0f} g={best['gamma']:.1f} RMSE={best['rmse']:.6f} | "
                      f"d={improvement:+.3f}%")
                ext_results.append({"component":comp,"domain":"pitching",
                                    "old_K":old["K"],"new_K":best["K"],
                                    "old_gamma":old["gamma"],"new_gamma":best["gamma"],
                                    "old_rmse":old["rmse"],"new_rmse":best["rmse"],
                                    "pct_improvement":improvement,
                                    "new_L":best["L"]})

    # ── 4. VOLUME CV (PA + IP — missing from original) ────────────────────
    sep("4. VOLUME CV (PA and IP — not in original script)")
    print("CV for PA volume (batting)...")
    pa_cv, pa_best = cv_fit_volume(bat, "pa", "pa", "pos_label", bio, means,
                                    K_GRID_VOL_BAT, GAMMA_GRID_EXT, min_vol=1)
    if pa_best:
        print(f"  PA volume: best L={pa_best['L']}, K={pa_best['K']:.1f} seasons, "
              f"gamma={pa_best['gamma']:.1f}, RMSE={pa_best['rmse']:.2f} PA")
        ext_results.append({"component":"pa","domain":"volume",
                            "old_K":"(borrowed)","new_K":pa_best["K"],
                            "old_gamma":"(borrowed)","new_gamma":pa_best["gamma"],
                            "old_rmse":None,"new_rmse":pa_best["rmse"],
                            "pct_improvement":None,"new_L":pa_best["L"]})

    print("CV for IP volume (SP)...")
    pit_sp = pit[pit["role"]=="SP"].copy()
    ip_cv_sp, ip_best_sp = cv_fit_volume(pit_sp, "ip", "ip", "role", bio, means,
                                          K_GRID_VOL_PIT, GAMMA_GRID_EXT, min_vol=1)
    if ip_best_sp:
        print(f"  IP volume (SP): best L={ip_best_sp['L']}, K={ip_best_sp['K']:.1f} seasons, "
              f"gamma={ip_best_sp['gamma']:.1f}, RMSE={ip_best_sp['rmse']:.2f} IP")

    print("CV for IP volume (RP)...")
    pit_rp = pit[pit["role"]=="RP"].copy()
    ip_cv_rp, ip_best_rp = cv_fit_volume(pit_rp, "ip", "ip", "role", bio, means,
                                          K_GRID_VOL_PIT, GAMMA_GRID_EXT, min_vol=1)
    if ip_best_rp:
        print(f"  IP volume (RP): best L={ip_best_rp['L']}, K={ip_best_rp['K']:.1f} seasons, "
              f"gamma={ip_best_rp['gamma']:.1f}, RMSE={ip_best_rp['rmse']:.2f} IP")

    # Baseline comparison: what RMSE does the current (borrowed) approach give?
    # Current: L=hr_pa_L, K=hr_pa_K/200, gamma=hr_pa_gamma
    hr_pa_row = orig_hp[orig_hp["component"]=="hr_pa"].iloc[0]
    print(f"\n  Current PA volume params (borrowed from hr_pa): "
          f"L={int(hr_pa_row['L'])}, K={hr_pa_row['K']/200:.1f} seasons, gamma={hr_pa_row['gamma']:.1f}")

    if pa_cv is not None and not pa_cv.empty:
        old_L = int(hr_pa_row["L"])
        old_K = hr_pa_row["K"] / 200.0
        old_g = hr_pa_row["gamma"]
        match = pa_cv[(pa_cv["L"]==old_L) &
                      (pa_cv["K"].apply(lambda k: abs(k-old_K)<0.01)) &
                      (pa_cv["gamma"]==old_g)]
        if not match.empty:
            old_pa_rmse = match.iloc[0]["rmse"]
            print(f"  Current PA volume RMSE (borrowed params): {old_pa_rmse:.2f} PA")
            if pa_best:
                print(f"  New PA volume RMSE (own CV params):        {pa_best['rmse']:.2f} PA  "
                      f"(d={(old_pa_rmse-pa_best['rmse'])/old_pa_rmse*100:+.2f}%)")

    # ── 5. PROJECTION SANITY STATS ─────────────────────────────────────────
    sep("5. PROJECTION SANITY STATS")
    proj = pd.read_csv(INT_DIR / "marcel_projections.csv")

    bat_proj = proj[~proj["refused_bat"].fillna(True)].copy()
    pit_proj = proj[~proj["refused_pit"].fillna(True)].copy()

    print(f"\nBatting projections (n={len(bat_proj)}):")
    bat_rate_cols = [f"{c}_proj" for c in BAT_COMPONENTS if f"{c}_proj" in bat_proj.columns]
    bat_stats = bat_proj[bat_rate_cols + ["pa_proj"]].describe().loc[["mean","std","min","25%","50%","75%","max"]]
    print(bat_stats.round(5).to_string())

    # Check for extreme outliers
    print("\nBatting outliers (values >3 std from mean):")
    any_outlier = False
    for col in bat_rate_cols:
        mu = bat_proj[col].mean(); sd = bat_proj[col].std()
        outliers = bat_proj[(bat_proj[col] - mu).abs() > 3*sd][["player_id","pos_label",col]]
        if len(outliers):
            any_outlier = True
            print(f"  {col}: {len(outliers)} outliers | min={bat_proj[col].min():.5f} max={bat_proj[col].max():.5f}")
    if not any_outlier:
        print("  None found (>3 std)")

    print(f"\nPitching projections (n={len(pit_proj)}):")
    pit_rate_cols = [f"{c}_proj" for c in PIT_COMPONENTS if f"{c}_proj" in pit_proj.columns]
    pit_stats = pit_proj[pit_rate_cols + ["ip_proj"]].describe().loc[["mean","std","min","25%","50%","75%","max"]]
    print(pit_stats.round(5).to_string())

    print("\nPitching outliers (>3 std from mean):")
    any_outlier = False
    for col in pit_rate_cols:
        mu = pit_proj[col].mean(); sd = pit_proj[col].std()
        outliers = pit_proj[(pit_proj[col] - mu).abs() > 3*sd][["player_id","role",col]]
        if len(outliers):
            any_outlier = True
            print(f"  {col}: {len(outliers)} outliers | min={pit_proj[col].min():.5f} max={pit_proj[col].max():.5f}")
    if not any_outlier:
        print("  None found (>3 std)")

    # PA distribution sanity: compare projected PA distribution to actual PA distribution
    actual_pa_2035 = bat[bat["year"]==2035]["pa"]
    print(f"\nPA sanity check:")
    print(f"  Actual 2035 PA: mean={actual_pa_2035.mean():.0f}, "
          f"median={actual_pa_2035.median():.0f}, "
          f"p25={actual_pa_2035.quantile(0.25):.0f}, "
          f"p75={actual_pa_2035.quantile(0.75):.0f}")
    print(f"  Projected 2036 PA: mean={bat_proj['pa_proj'].mean():.0f}, "
          f"median={bat_proj['pa_proj'].median():.0f}, "
          f"p25={bat_proj['pa_proj'].quantile(0.25):.0f}, "
          f"p75={bat_proj['pa_proj'].quantile(0.75):.0f}")

    # IP sanity
    actual_ip_sp_2035 = pit[(pit["year"]==2035) & (pit["role"]=="SP")]["ip"]
    actual_ip_rp_2035 = pit[(pit["year"]==2035) & (pit["role"]=="RP")]["ip"]
    proj_sp = pit_proj[pit_proj["role"]=="SP"]["ip_proj"]
    proj_rp = pit_proj[pit_proj["role"]=="RP"]["ip_proj"]
    print(f"\nIP sanity check (SP):")
    print(f"  Actual 2035 SP IP: mean={actual_ip_sp_2035.mean():.1f}, median={actual_ip_sp_2035.median():.1f}")
    print(f"  Projected 2036 SP IP: mean={proj_sp.mean():.1f}, median={proj_sp.median():.1f}")
    print(f"IP sanity check (RP):")
    print(f"  Actual 2035 RP IP: mean={actual_ip_rp_2035.mean():.1f}, median={actual_ip_rp_2035.median():.1f}")
    print(f"  Projected 2036 RP IP: mean={proj_rp.mean():.1f}, median={proj_rp.median():.1f}")

    # Age distribution of projections
    print(f"\nAge distribution of batting projections:")
    print(bat_proj["age_proj"].describe().round(1).to_string())

    # ── 6. SUMMARY AND RECOMMENDATIONS ───────────────────────────────────
    sep("6. SUMMARY AND RECOMMENDATIONS")
    if ext_results:
        ext_df = pd.DataFrame(ext_results)
        print("\nExtended-grid improvements found:")
        print(ext_df.to_string(index=False))
        ext_df.to_csv(INT_DIR / "diag_extended_hp_results.csv", index=False)
        print(f"\nSaved to intermediate/diag_extended_hp_results.csv")

    print("\nRECOMMENDATIONS:")
    print("  1. Expand K grids for at-ceiling components (see above)")
    print("  2. Add separate PA-volume and IP-volume CV to main script")
    print("  3. Expand gamma grid to [0.1, 0.3, 0.5, 0.7, 0.9]")
    print("  4. Update src/pipeline/step1a_foundation.py K grids based on extended results")

    sep("DIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    main()
