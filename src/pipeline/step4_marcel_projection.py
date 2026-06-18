"""
step4_marcel_projection.py — Step 4: Marcel projections

For every player who has any qualifying major-league data in the last 5 seasons,
projects per-component rates + volume for PROJ_YEAR using:
  1. CV-fit Marcel (weighted average of recent seasons, linear-decay weights)
  2. Regression toward position-age league mean (shrinkage constant K fit by CV)
  3. Gap-year decay (weight penalty γ per missing year, fit by CV)
  4. One-year aging adjustment from step 2 curves (loess50, fallback poly3)

Reads:
  intermediate/batting_neutral.csv
  intermediate/pitching_neutral.csv
  intermediate/fielding_raw.csv
  intermediate/aging_curves_smooth.csv
  frostfire_data/players.csv

Writes:
  intermediate/marcel_projections.csv    — per-player projected rates + volume
  intermediate/marcel_hyperparams.csv    — CV-fit (L, K, gamma) per component
  intermediate/marcel_cv_scores.csv      — full grid of CV RMSE per component
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("frostfire_data")
INT_DIR  = Path("intermediate")

# ─── Config ──────────────────────────────────────────────────────────────────
PROJ_YEAR    = 2036          # season we're projecting into
CURRENT_YEAR = 2035          # last complete season in the data
SP_THRESHOLD = 0.5           # gs/g >= threshold → SP, else RP

MIN_PA_QUAL  = 100           # min PA to include a batting season
MIN_BF_QUAL  = 100           # min BF to include a pitching season
MIN_IP_QUAL  = 50            # min total IP to include a fielding player-year

THIN_BAT_PA  = 150           # flag thin if weighted PA below this
THIN_PIT_BF  = 150           # flag thin if weighted BF below this

MIN_BAT_VOL  = 50            # refuse batting projection if weighted PA < this
MIN_PIT_VOL  = 50            # refuse pitching projection if weighted BF < this

L_GRID      = [2, 3, 4, 5]
K_GRID_BAT  = [100, 200, 400, 700, 1200, 2000, 3500]
K_GRID_PIT  = [50,  100, 200, 400, 700,  1200]
K_GRID_FLD  = [30,  70,  150, 300, 600,  1000, 1500]
K_GRID_VOL  = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]  # in "seasons" units for PA/IP
GAMMA_GRID  = [0.1, 0.3, 0.5, 0.7, 0.9]

# Match step 2 naming; hp_pa added (HBP/PA — small aging signal, projects with strong K)
BAT_COMPONENTS = ["hr_pa", "xbh_pa", "single_pa", "bb_pa", "k_pa", "ubr_g", "hp_pa"]
# ha_n_bf: hits-allowed rate, no aging curve (BABIP is mostly luck)
PIT_COMPONENTS = ["k_bf", "bb_hbp_bf", "hra_bf", "ha_n_bf"]
FLD_COMPONENTS = ["zr_rate", "arm_rate", "framing_rate"]

# Which components have aging deltas from step 2
AGING_BAT_SET = {"hr_pa", "xbh_pa", "single_pa", "bb_pa", "k_pa", "ubr_g"}
AGING_PIT_SET = {"k_bf", "bb_hbp_bf", "hra_bf"}
AGING_FLD_SET = {"zr_rate", "arm_rate", "framing_rate"}

VALID_TIDS = {1,3,4,6,7,8,9,13,16,17,18,20,21,23,24,25,26,27,29,30,301,302}
POS_LABELS = {2:"C",3:"1B",4:"2B",5:"3B",6:"SS",7:"LF",8:"CF",9:"RF",10:"DH"}

_issues: list[str] = []
def _ok(m):
    """Print a passing-check message with an [OK] tag."""
    print(f"  [OK]   {m}")
def _info(m):
    """Print an informational message with an [INFO] tag."""
    print(f"  [INFO] {m}")
def _warn(m):
    """Record a warning message in _issues and print it with a [WARN] tag."""
    _issues.append(m); print(f"  [WARN] {m}")


# ─── 1. LOAD PLAYERS ─────────────────────────────────────────────────────────

def load_players() -> pd.DataFrame:
    """Load player bio columns from players.csv, computing birth_year and ages at the current and projection years."""
    print("\n=== PLAYERS ===")
    cols = ["ID","date_of_birth","bats","Pos","Role",
            "mlb_service_years","is_active","Level","Retired",
            "First Name","Last Name"]
    p = pd.read_csv(DATA_DIR / "players.csv", usecols=cols)
    p.rename(columns={"ID":"player_id","Pos":"primary_pos",
                       "First Name":"first_name","Last Name":"last_name"}, inplace=True)
    p["birth_year"] = pd.to_datetime(p["date_of_birth"], errors="coerce").dt.year
    p["age_2035"]   = CURRENT_YEAR - p["birth_year"]
    p["age_2036"]   = PROJ_YEAR    - p["birth_year"]
    n_ml = int(((p["is_active"]==1) & (p["Level"]==1) & (p["Retired"]==0)).sum())
    _ok(f"{len(p):,} total players; {n_ml:,} active ML (is_active=1, Level=1, Retired=0)")
    return p


# ─── 2. POSITION LOOKUP ──────────────────────────────────────────────────────

def build_position_lookup(players: pd.DataFrame):
    """Returns (primary_series, bio_series) exactly as in step 2."""
    fld = pd.read_csv(INT_DIR / "fielding_raw.csv", usecols=["player_id","year","position","ip"])
    fld = fld[fld["position"].between(2, 9)].copy()
    fld_agg = fld.groupby(["player_id","year","position"], sort=False)["ip"].sum().reset_index()
    idx_max = fld_agg.groupby(["player_id","year"])["ip"].idxmax()
    primary = fld_agg.loc[idx_max, ["player_id","year","position"]].copy()
    primary = primary.set_index(["player_id","year"])["position"]
    bio_pos = (players.set_index("player_id")["primary_pos"]
               .where(lambda x: x.between(2, 10)))
    _ok(f"Position lookup: {len(primary):,} player-year entries from fielding")
    return primary, bio_pos


# ─── 3. COMPONENT PANELS ─────────────────────────────────────────────────────

def build_batting_panel(pos_lookup) -> pd.DataFrame:
    """One row per (player_id, year); split_id=1; stints aggregated."""
    print("\n=== BATTING PANEL ===")
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

    agg["hr_pa"]     = agg["hr_n"]                / agg["pa"]
    agg["xbh_pa"]    = (agg["d_n"] + agg["t_n"])  / agg["pa"]
    agg["single_pa"] = agg["singles_n"]            / agg["pa"]
    agg["bb_pa"]     = agg["bb"]                   / agg["pa"]
    agg["k_pa"]      = agg["k"]                    / agg["pa"]
    agg["hp_pa"]     = agg["hp"]                   / agg["pa"]
    agg["ubr_g"]     = agg["ubr"]                  / agg["g"].clip(lower=1)
    agg["d_frac"]    = agg["d_n"] / (agg["d_n"] + agg["t_n"]).clip(lower=1e-9)

    # Position from fielding; DH→1B remap (same as step 2)
    primary, bio_pos = pos_lookup
    idx = pd.MultiIndex.from_frame(agg[["player_id","year"]])
    pos_from_fld  = primary.reindex(idx).values.astype(float)    # NaN where no fielding data
    pos_from_bio  = agg["player_id"].map(bio_pos).values.astype(float)  # fallback
    pos_merged    = np.where(np.isnan(pos_from_fld), pos_from_bio, pos_from_fld)
    # Assign directly from numpy array (avoids index-alignment mismatch after filtering)
    agg["position"] = np.where(np.isnan(pos_merged), 0, pos_merged).astype(int)
    agg["position"] = agg["position"].replace({10: 3})
    agg["pos_label"] = agg["position"].map(POS_LABELS).fillna("unknown")

    _ok(f"Batting panel: {len(agg):,} player-years (pa>={MIN_PA_QUAL})")
    return agg


def build_pitching_panel() -> pd.DataFrame:
    """One row per (player_id, year); split_id=1; stints aggregated by team."""
    print("\n=== PITCHING PANEL ===")
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

    agg["k_bf"]       = agg["k"]                  / agg["bf"]
    agg["bb_hbp_bf"]  = (agg["bb"] + agg["hp"])   / agg["bf"]
    agg["hra_bf"]     = agg["hra_n"]               / agg["bf"]
    agg["ha_n_bf"]    = agg["ha_n"]                / agg["bf"]
    # bb fraction of bb+hbp (for downstream decomposition)
    agg["bb_frac"]    = agg["bb"] / (agg["bb"] + agg["hp"]).clip(lower=1e-9)

    agg["sp_flag"] = (agg["gs"] / agg["g"].clip(lower=1)) >= SP_THRESHOLD
    agg["role"]    = agg["sp_flag"].map({True:"SP", False:"RP"})

    _ok(f"Pitching panel: {len(agg):,} player-years (bf>={MIN_BF_QUAL})")
    return agg


def build_fielding_panel() -> pd.DataFrame:
    """One row per (player_id, year): total ip, zr, arm, framing summed across positions."""
    print("\n=== FIELDING PANEL ===")
    usecols = ["player_id","year","position","ip","zr","arm","framing"]
    fld = pd.read_csv(INT_DIR / "fielding_raw.csv", usecols=usecols)
    fld = fld[fld["ip"] >= 1].copy()

    # Primary fielding position (most IP among 2-9)
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

    _ok(f"Fielding panel: {len(agg):,} player-years (ip>={MIN_IP_QUAL})")
    return agg


# ─── 4. POSITION-AGE LEAGUE MEANS ────────────────────────────────────────────

def build_league_means(bat, pit, fld, players) -> dict:
    """
    Returns nested dict:
      means[(group_label, component)] = {"age_means": {age: val}, "pos_mean": val, "overall_mean": val}
    """
    print("\n=== LEAGUE MEANS ===")
    bio = players[["player_id","birth_year"]].dropna().set_index("player_id")["birth_year"]
    MIN_AGE_N = 10

    def weighted_mean(df, rate_col, vol_col):
        w = df[vol_col]
        return float((df[rate_col] * w).sum() / w.sum()) if w.sum() > 0 else float(df[rate_col].mean())

    def age_means_for_group(grp, rate_col, vol_col):
        res = {}
        for age, gg in grp.groupby("age"):
            if 20 <= int(age) <= 40 and len(gg) >= MIN_AGE_N:
                res[int(age)] = weighted_mean(gg, rate_col, vol_col)
        return res

    means: dict = {}

    # Batting
    bat_a = bat.copy()
    bat_a["age"] = bat_a["player_id"].map(bio).rsub(bat_a["year"])
    overall_bat = {}
    for comp in BAT_COMPONENTS:
        vol = "g" if comp == "ubr_g" else "pa"
        if comp in bat_a.columns:
            overall_bat[comp] = weighted_mean(bat_a, comp, vol)
        else:
            overall_bat[comp] = 0.0

    for pos, grp in bat_a[bat_a["position"] > 0].groupby("position"):
        lbl = POS_LABELS.get(int(pos), str(pos))
        for comp in BAT_COMPONENTS:
            if comp not in bat_a.columns:
                continue
            vol = "g" if comp == "ubr_g" else "pa"
            pos_mean = weighted_mean(grp, comp, vol) if len(grp) >= 5 else overall_bat[comp]
            am = age_means_for_group(grp, comp, vol)
            means[(lbl, comp)] = {
                "age_means": am,
                "pos_mean":  pos_mean,
                "overall_mean": overall_bat[comp],
            }

    for comp in BAT_COMPONENTS:
        if comp not in bat_a.columns:
            continue
        vol = "g" if comp == "ubr_g" else "pa"
        means[("overall", comp)] = {
            "age_means": {},
            "pos_mean":  overall_bat[comp],
            "overall_mean": overall_bat[comp],
        }

    # Pitching
    pit_a = pit.copy()
    pit_a["age"] = pit_a["player_id"].map(bio).rsub(pit_a["year"])
    for role, grp in pit_a.groupby("role"):
        for comp in PIT_COMPONENTS:
            ov_mean = weighted_mean(grp, comp, "bf") if len(grp) >= 5 else 0.0
            am = age_means_for_group(grp, comp, "bf")
            means[(role, comp)] = {
                "age_means": am,
                "pos_mean":  ov_mean,
                "overall_mean": ov_mean,
            }
    # Volume means for pitchers (IP)
    for role, grp in pit_a.groupby("role"):
        ov_ip = float(grp["ip"].mean()) if len(grp) else 100.0
        am_ip = {}
        for age, gg in grp.groupby("age"):
            if 20 <= int(age) <= 40 and len(gg) >= MIN_AGE_N:
                am_ip[int(age)] = float(gg["ip"].mean())
        means[(role, "ip")] = {
            "age_means": am_ip,
            "pos_mean":  ov_ip,
            "overall_mean": ov_ip,
        }

    # Fielding
    fld_a = fld.copy()
    fld_a["age"] = fld_a["player_id"].map(bio).rsub(fld_a["year"])
    for pos_lbl, grp in fld_a[fld_a["fld_pos_label"] != "unknown"].groupby("fld_pos_label"):
        for comp in FLD_COMPONENTS:
            ov_mean = weighted_mean(grp, comp, "ip") if len(grp) >= 5 else 0.0
            am = age_means_for_group(grp, comp, "ip")
            means[(pos_lbl, comp)] = {
                "age_means": am,
                "pos_mean":  ov_mean,
                "overall_mean": ov_mean,
            }

    # Volume (PA) means for batters
    bat_a["pa_season"] = bat_a["pa"]
    for pos, grp in bat_a[bat_a["position"] > 0].groupby("position"):
        lbl = POS_LABELS.get(int(pos), str(pos))
        ov_pa = float(grp["pa"].mean()) if len(grp) else 400.0
        am_pa = {}
        for age, gg in grp.groupby("age"):
            if 20 <= int(age) <= 40 and len(gg) >= MIN_AGE_N:
                am_pa[int(age)] = float(gg["pa"].mean())
        means[(lbl, "pa")] = {"age_means": am_pa, "pos_mean": ov_pa, "overall_mean": ov_pa}
    means[("overall","pa")] = {"age_means": {}, "pos_mean": float(bat_a["pa"].mean()),
                                "overall_mean": float(bat_a["pa"].mean())}

    _ok(f"Built league means for {len(means)} (group, component) pairs")
    return means


def lookup_mean(means: dict, group_label: str, comp: str, age: int) -> float:
    """Lookup position-age mean; fallback to position mean, then overall mean."""
    for key in [(group_label, comp), ("overall", comp)]:
        if key in means:
            entry = means[key]
            v = entry["age_means"].get(age)
            if v is not None:
                return float(v)
            pm = entry.get("pos_mean")
            if pm is not None:
                return float(pm)
            return float(entry.get("overall_mean", 0.0))
    return 0.0


# ─── 5. AGING CURVES ─────────────────────────────────────────────────────────

def load_aging_lookup() -> dict:
    """Returns {(group_label, component, age): one_year_delta}."""
    print("\n=== AGING CURVES ===")
    cols = ["group_label","component","age","loess50_delta","poly3_delta"]
    c = pd.read_csv(INT_DIR / "aging_curves_smooth.csv", usecols=cols)
    c["delta"] = c["loess50_delta"].fillna(c["poly3_delta"]).fillna(0.0)
    lookup = {
        (str(r.group_label), str(r.component), int(r.age)): float(r.delta)
        for r in c.itertuples()
    }
    _ok(f"Loaded {len(lookup):,} aging curve deltas")
    return lookup


def aging_delta(lookup: dict, group_label: str, comp: str, age_current: int) -> float:
    """One-year delta from age_current → age_current+1."""
    return lookup.get((str(group_label), str(comp), int(age_current)), 0.0)


# ─── 6. MARCEL CORE ──────────────────────────────────────────────────────────

def marcel_weights(qualifying_years: set, target_year: int, L: int, gamma: float) -> dict:
    """
    Returns {year: weight} for years in [target-L, target-1] ∩ qualifying_years.
    Weights: base = (L - years_back + 1); decayed by gamma^(cumulative_gap).
    """
    window = sorted(
        [y for y in qualifying_years if target_year - L <= y < target_year],
        reverse=True,
    )
    if not window:
        return {}
    weights: dict = {}
    cum_gap = 0
    prev = target_year
    for yr in window:
        gap = prev - yr - 1
        cum_gap += gap
        base = float(max(0, L - (target_year - yr) + 1))
        weights[yr] = base * (gamma ** cum_gap)
        prev = yr
    return weights


def marcel_predict(
    history: dict,        # {year: (rate, volume)}
    target_year: int,
    league_mean: float,
    L: int,
    K: float,
    gamma: float,
) -> tuple[float, float]:
    """
    Returns (projected_rate, total_weighted_volume).
    projected_rate = (Σ w·vol·rate + K·mean) / (Σ w·vol + K)
    """
    w = marcel_weights(set(history), target_year, L, gamma)
    if not w:
        return float(league_mean), 0.0

    total_wvol  = sum(w[yr] * history[yr][1] for yr in w)
    total_wcount= sum(w[yr] * history[yr][1] * history[yr][0] for yr in w)

    if total_wvol <= 0:
        return float(league_mean), 0.0

    raw  = total_wcount / total_wvol
    proj = (raw * total_wvol + K * league_mean) / (total_wvol + K)
    return float(proj), float(total_wvol)


# ─── 7. CV FITTING ───────────────────────────────────────────────────────────

def cv_fit(
    panel: pd.DataFrame,
    comp: str,
    vol_col: str,
    group_col: str,
    bio: pd.Series,           # player_id → birth_year
    means: dict,
    K_grid: list,
    min_vol: float = MIN_BAT_VOL,
) -> tuple[dict, pd.DataFrame]:
    """
    Leave-one-season-out CV over all valid player-years (year 2017+).
    Returns best {L, K, gamma, rmse} and full grid DataFrame.
    """
    # Build per-player history for fast lookup
    player_hist: dict = {}
    for pid, grp in panel.groupby("player_id"):
        player_hist[pid] = {
            row["year"]: (row[comp], row[vol_col])
            for _, row in grp.iterrows()
        }

    # Build test records
    test_panel = panel[(panel["year"] >= 2017) & (panel["year"] <= CURRENT_YEAR)].copy()
    test_panel["birth_year"] = test_panel["player_id"].map(bio)
    test_panel = test_panel.dropna(subset=["birth_year"])
    test_panel["age"] = (test_panel["year"] - test_panel["birth_year"]).astype(int)
    test_panel = test_panel[test_panel[vol_col] >= min_vol]

    records = [
        {
            "pid":          r.player_id,
            "year":         int(r.year),
            "actual_rate":  float(getattr(r, comp)),
            "actual_vol":   float(getattr(r, vol_col)),
            "group_label":  str(getattr(r, group_col)),
            "age":          int(r.age),
        }
        for r in test_panel.itertuples()
    ]

    if not records:
        _warn(f"CV: no test records for {comp}")
        return {"L":3,"K":K_grid[0],"gamma":0.7,"rmse":np.inf}, pd.DataFrame()

    cv_rows = []
    for L in L_GRID:
        for K in K_grid:
            for gamma in GAMMA_GRID:
                wss = 0.0
                wt  = 0.0
                for rec in records:
                    hist = {y: v for y, v in player_hist[rec["pid"]].items() if y < rec["year"]}
                    lm   = lookup_mean(means, rec["group_label"], comp, rec["age"])
                    proj, _ = marcel_predict(hist, rec["year"], lm, L, K, gamma)
                    err  = (proj - rec["actual_rate"]) ** 2
                    wss += err * rec["actual_vol"]
                    wt  += rec["actual_vol"]
                rmse = float(np.sqrt(wss / wt)) if wt > 0 else np.inf
                cv_rows.append({"component":comp,"L":L,"K":K,"gamma":gamma,"rmse":rmse})

    cv_df = pd.DataFrame(cv_rows)
    best  = cv_df.loc[cv_df["rmse"].idxmin()]
    bp = {"L":int(best.L),"K":float(best.K),"gamma":float(best.gamma),"rmse":float(best.rmse)}
    _ok(f"  {comp:15s}: best L={bp['L']}, K={bp['K']:6.0f}, gamma={bp['gamma']:.1f}, CV-RMSE={bp['rmse']:.5f}")
    return bp, cv_df


def cv_fit_volume(
    panel: pd.DataFrame,
    vol_col: str,           # "pa" or "ip"
    group_col: str,         # "pos_label" or "role"
    bio: pd.Series,
    means: dict,
    comp_label: str,        # key to store in best_hp, e.g. "pa_vol" or "ip_vol_SP"
    min_seasons: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """
    Leave-one-season-out CV for volume (PA or IP) as a first-class component.
    K is in 'seasons' units (not PA/BF). Unweighted RMSE since volume IS the target.
    """
    player_hist: dict = {}
    for pid, grp in panel.groupby("player_id"):
        player_hist[pid] = {row["year"]: (row[vol_col], 1.0) for _, row in grp.iterrows()}

    test_panel = panel[(panel["year"] >= 2017) & (panel["year"] <= CURRENT_YEAR)].copy()
    test_panel["birth_year"] = test_panel["player_id"].map(bio)
    test_panel = test_panel.dropna(subset=["birth_year"])
    test_panel["age"] = (test_panel["year"] - test_panel["birth_year"]).astype(int)

    records = [
        {
            "pid":         r.player_id,
            "year":        int(r.year),
            "actual_vol":  float(getattr(r, vol_col)),
            "group_label": str(getattr(r, group_col)),
            "age":         int(r.age),
        }
        for r in test_panel.itertuples()
    ]
    if not records:
        _warn(f"CV-volume: no test records for {comp_label}")
        return {"L":3,"K":3.0,"gamma":0.7,"rmse":np.inf}, pd.DataFrame()

    cv_rows = []
    for L in L_GRID:
        for K in K_GRID_VOL:
            for gamma in GAMMA_GRID:
                ss = 0.0; n = 0
                for rec in records:
                    hist = {y: v for y, v in player_hist[rec["pid"]].items() if y < rec["year"]}
                    lm   = lookup_mean(means, rec["group_label"], vol_col, rec["age"])
                    proj, _ = marcel_predict(hist, rec["year"], lm, L, K, gamma)
                    proj = float(np.clip(proj, 0, 700))
                    ss += (proj - rec["actual_vol"]) ** 2
                    n  += 1
                rmse = float(np.sqrt(ss / n)) if n > 0 else np.inf
                cv_rows.append({"component":comp_label,"L":L,"K":K,"gamma":gamma,"rmse":rmse})

    cv_df = pd.DataFrame(cv_rows)
    best  = cv_df.loc[cv_df["rmse"].idxmin()]
    bp = {"L":int(best.L),"K":float(best.K),"gamma":float(best.gamma),"rmse":float(best.rmse)}
    _ok(f"  {comp_label:15s}: best L={bp['L']}, K={bp['K']:5.1f} seasons, gamma={bp['gamma']:.1f}, CV-RMSE={bp['rmse']:.2f}")
    return bp, cv_df


# ─── 8. PROJECT ALL PLAYERS ──────────────────────────────────────────────────

def project_batting(bat: pd.DataFrame, players: pd.DataFrame,
                    means: dict, aging_lkp: dict, hp: dict) -> pd.DataFrame:
    """Build per-player Marcel projections for all batting components plus PA volume, applying aging deltas and flagging thin/refused players.

    Args:
        bat: batting panel (one row per player-year).
        players: player bio table with birth_year.
        means: position-age league mean lookup from build_league_means().
        aging_lkp: one-year aging delta lookup from load_aging_lookup().
        hp: best-fit Marcel hyperparameters per component.

    Returns:
        One row per player with projected rates, weighted volumes, and refused/thin flags.
    """
    print("\n=== PROJECT BATTING ===")
    bio = players[["player_id","birth_year"]].dropna().set_index("player_id")["birth_year"]

    # Pre-build per-player histories
    player_data: dict = {}
    for pid, grp in bat.groupby("player_id"):
        grp = grp.sort_values("year")
        latest = grp.iloc[-1]
        pos_val = latest["position"]
        pos_int = int(pos_val) if pd.notna(pos_val) and pos_val != 0 else 0
        player_data[pid] = {
            "pos":      pos_int,
            "pos_lbl":  POS_LABELS.get(pos_int, "unknown"),
            "d_frac":   float((grp["d_n"].sum() /
                              (grp["d_n"] + grp["t_n"]).sum().clip(1e-9))),
            "comp_hist": {
                c: {row["year"]: (row[c], row["pa"] if c != "ubr_g" else row["g"])
                    for _, row in grp.iterrows()}
                for c in BAT_COMPONENTS
            },
            "pa_hist":  {row["year"]: (row["pa"], 1.0) for _, row in grp.iterrows()},
        }

    rows = []
    for pid, pd_ in player_data.items():
        byr = bio.get(pid)
        if pd.isna(byr):
            continue
        age35 = int(CURRENT_YEAR - byr)
        age36 = int(PROJ_YEAR    - byr)
        pos   = pd_["pos"]
        lbl   = pd_["pos_lbl"]

        rec = {
            "player_id":   pid,
            "birth_year":  int(byr),
            "age_proj":    age36,
            "primary_pos": pos,
            "pos_label":   lbl,
            "d_frac":      pd_["d_frac"],
        }
        refused = False
        thin    = False
        max_wvol = 0.0

        for comp in BAT_COMPONENTS:
            h = pd_["comp_hist"].get(comp, {})
            lm = lookup_mean(means, lbl, comp, age35)
            p_hp = hp.get(comp, {"L":3,"K":400,"gamma":0.7})
            proj, wvol = marcel_predict(h, PROJ_YEAR, lm,
                                        p_hp["L"], p_hp["K"], p_hp["gamma"])
            # Aging delta (only for components with step 2 curves)
            if comp in AGING_BAT_SET:
                proj += aging_delta(aging_lkp, lbl, comp, age35)

            rec[f"{comp}_proj"] = proj
            rec[f"{comp}_wvol"] = wvol
            if wvol > max_wvol:
                max_wvol = wvol

        if max_wvol < MIN_BAT_VOL:
            refused = True
        elif max_wvol < THIN_BAT_PA:
            thin = True

        # Project PA (volume): own CV-fit params in seasons units
        p_pa   = hp.get("pa_vol", {"L":2,"K":2.0,"gamma":0.9})
        pa_lm  = lookup_mean(means, lbl, "pa", age35)
        pa_proj, _ = marcel_predict(pd_["pa_hist"], PROJ_YEAR, pa_lm,
                                     p_pa["L"], p_pa["K"], p_pa["gamma"])
        rec["pa_proj"] = float(np.clip(pa_proj, 0, 650))

        rec["refused_bat"] = refused
        rec["thin_bat"]    = thin
        rows.append(rec)

    df = pd.DataFrame(rows)
    _ok(f"Batting: {int((~df['refused_bat']).sum())} projected, "
        f"{int(df['thin_bat'].sum())} thin, {int(df['refused_bat'].sum())} refused")
    return df


def project_pitching(pit: pd.DataFrame, players: pd.DataFrame,
                     means: dict, aging_lkp: dict, hp: dict) -> pd.DataFrame:
    """Build per-player Marcel projections for all pitching components plus IP volume (role-specific), applying aging deltas and flagging thin/refused players."""
    print("\n=== PROJECT PITCHING ===")
    bio = players[["player_id","birth_year"]].dropna().set_index("player_id")["birth_year"]

    player_data: dict = {}
    for pid, grp in pit.groupby("player_id"):
        grp = grp.sort_values("year")
        latest = grp.iloc[-1]
        player_data[pid] = {
            "role":    str(latest["role"]),
            "bb_frac": float((grp["bb"].sum() /
                              (grp["bb"] + grp["hp"]).sum().clip(1e-9))),
            "comp_hist": {
                c: {row["year"]: (row[c], row["bf"]) for _, row in grp.iterrows()}
                for c in PIT_COMPONENTS
            },
            "ip_hist": {row["year"]: (row["ip"], 1.0) for _, row in grp.iterrows()},
        }

    rows = []
    for pid, pd_ in player_data.items():
        byr = bio.get(pid)
        if pd.isna(byr):
            continue
        age35 = int(CURRENT_YEAR - byr)
        age36 = int(PROJ_YEAR    - byr)
        role  = pd_["role"]

        rec = {
            "player_id":  pid,
            "birth_year": int(byr),
            "age_proj":   age36,
            "role":       role,
            "bb_frac":    pd_["bb_frac"],
        }
        refused = False
        thin    = False
        max_wvol = 0.0

        for comp in PIT_COMPONENTS:
            h  = pd_["comp_hist"].get(comp, {})
            lm = lookup_mean(means, role, comp, age35)
            p_hp = hp.get(comp, {"L":3,"K":200,"gamma":0.7})
            proj, wvol = marcel_predict(h, PROJ_YEAR, lm,
                                        p_hp["L"], p_hp["K"], p_hp["gamma"])
            if comp in AGING_PIT_SET:
                proj += aging_delta(aging_lkp, role, comp, age35)

            rec[f"{comp}_proj"] = proj
            rec[f"{comp}_wvol"] = wvol
            if wvol > max_wvol:
                max_wvol = wvol

        if max_wvol < MIN_PIT_VOL:
            refused = True
        elif max_wvol < THIN_PIT_BF:
            thin = True

        # Project IP (volume): own CV-fit params per role, in seasons units
        ip_comp = f"ip_vol_{role}"
        p_ip   = hp.get(ip_comp, {"L":3,"K":5.0,"gamma":0.8})
        ip_lm  = lookup_mean(means, role, "ip", age35)
        ip_proj, _ = marcel_predict(pd_["ip_hist"], PROJ_YEAR, ip_lm,
                                     p_ip["L"], p_ip["K"], p_ip["gamma"])
        ip_cap = 220.0 if role == "SP" else 90.0
        rec["ip_proj"] = float(np.clip(ip_proj, 0, ip_cap))
        rec["bf_proj"] = rec["ip_proj"] * 4.25  # ~4.25 BF per IP (rough average)

        rec["refused_pit"] = refused
        rec["thin_pit"]    = thin
        rows.append(rec)

    df = pd.DataFrame(rows)
    _ok(f"Pitching: {int((~df['refused_pit']).sum())} projected, "
        f"{int(df['thin_pit'].sum())} thin, {int(df['refused_pit'].sum())} refused")
    return df


def project_fielding(fld: pd.DataFrame, players: pd.DataFrame,
                     means: dict, aging_lkp: dict, hp: dict) -> pd.DataFrame:
    """Build per-player Marcel projections for all fielding rate components, applying aging deltas where available."""
    print("\n=== PROJECT FIELDING ===")
    bio = players[["player_id","birth_year"]].dropna().set_index("player_id")["birth_year"]

    player_data: dict = {}
    for pid, grp in fld.groupby("player_id"):
        grp = grp.sort_values("year")
        latest = grp.iloc[-1]
        lbl = str(latest["fld_pos_label"])
        player_data[pid] = {
            "fld_pos_label": lbl,
            "fld_pos":       latest["fld_pos"],
            "comp_hist": {
                c: {row["year"]: (row[c], row["ip"]) for _, row in grp.iterrows()}
                for c in FLD_COMPONENTS
            },
            "ip_hist": {row["year"]: (row["ip"], 1.0) for _, row in grp.iterrows()},
        }

    rows = []
    for pid, pd_ in player_data.items():
        byr = bio.get(pid)
        if pd.isna(byr):
            continue
        age35 = int(CURRENT_YEAR - byr)
        age36 = int(PROJ_YEAR    - byr)
        lbl   = pd_["fld_pos_label"]

        rec = {
            "player_id":     pid,
            "birth_year_f":  int(byr),
            "age_proj_f":    age36,
            "fld_pos":       pd_["fld_pos"],
            "fld_pos_label": lbl,
        }

        for comp in FLD_COMPONENTS:
            h  = pd_["comp_hist"].get(comp, {})
            lm = lookup_mean(means, lbl, comp, age35)
            p_hp = hp.get(comp, {"L":3,"K":150,"gamma":0.7})
            proj, wvol = marcel_predict(h, PROJ_YEAR, lm,
                                        p_hp["L"], p_hp["K"], p_hp["gamma"])
            if comp in AGING_FLD_SET:
                proj += aging_delta(aging_lkp, lbl, comp, age35)
            rec[f"{comp}_proj"] = proj
            rec[f"{comp}_wvol"] = wvol

        rows.append(rec)

    df = pd.DataFrame(rows)
    _ok(f"Fielding: {len(df)} players projected")
    return df


# ─── 9. MAIN ─────────────────────────────────────────────────────────────────

def main():
    """Run the full step 4 pipeline: build component panels, league means, and aging lookups, CV-fit Marcel hyperparameters, project all players, merge, and write marcel_projections.csv."""
    print("=" * 60)
    print("STEP 4: Marcel projections")
    print("=" * 60)

    players  = load_players()
    pos_lkp  = build_position_lookup(players)
    bat      = build_batting_panel(pos_lkp)
    pit      = build_pitching_panel()
    fld      = build_fielding_panel()

    means    = build_league_means(bat, pit, fld, players)
    aging_lkp = load_aging_lookup()

    bio      = players[["player_id","birth_year"]].dropna().set_index("player_id")["birth_year"]

    # ── CV FITTING ──────────────────────────────────────────────────────
    print("\n=== CV FITTING ===")
    best_hp: dict = {}
    all_cv:  list = []

    print("\n  Batting components:")
    for comp in BAT_COMPONENTS:
        vol = "g" if comp == "ubr_g" else "pa"
        bp, cv_df = cv_fit(bat, comp, vol, "pos_label", bio, means, K_GRID_BAT)
        best_hp[comp] = bp
        if not cv_df.empty:
            all_cv.append(cv_df)

    print("\n  Pitching components:")
    for comp in PIT_COMPONENTS:
        bp, cv_df = cv_fit(pit, comp, "bf", "role", bio, means, K_GRID_PIT,
                           min_vol=MIN_PIT_VOL)
        best_hp[comp] = bp
        if not cv_df.empty:
            all_cv.append(cv_df)

    print("\n  Fielding components:")
    for comp in FLD_COMPONENTS:
        bp, cv_df = cv_fit(fld, comp, "ip", "fld_pos_label", bio, means, K_GRID_FLD)
        best_hp[comp] = bp
        if not cv_df.empty:
            all_cv.append(cv_df)

    print("\n  Volume components (PA, IP — own CV in seasons units):")
    bp_pa, cv_pa = cv_fit_volume(bat, "pa", "pos_label", bio, means, "pa_vol")
    best_hp["pa_vol"] = bp_pa
    if not cv_pa.empty:
        all_cv.append(cv_pa)

    for role in ["SP", "RP"]:
        pit_role = pit[pit["role"] == role].copy()
        comp_lbl = f"ip_vol_{role}"
        bp_ip, cv_ip = cv_fit_volume(pit_role, "ip", "role", bio, means, comp_lbl)
        best_hp[comp_lbl] = bp_ip
        if not cv_ip.empty:
            all_cv.append(cv_ip)

    if all_cv:
        pd.concat(all_cv, ignore_index=True).to_csv(INT_DIR / "marcel_cv_scores.csv", index=False)
        _ok("Saved marcel_cv_scores.csv")

    hp_rows = [{"component": c, **v} for c, v in best_hp.items()]
    pd.DataFrame(hp_rows).to_csv(INT_DIR / "marcel_hyperparams.csv", index=False)
    _ok("Saved marcel_hyperparams.csv")

    # ── PROJECTIONS ─────────────────────────────────────────────────────
    bat_proj = project_batting(bat, players, means, aging_lkp, best_hp)
    pit_proj = project_pitching(pit, players, means, aging_lkp, best_hp)
    fld_proj = project_fielding(fld, players, means, aging_lkp, best_hp)

    # ── MERGE ───────────────────────────────────────────────────────────
    print("\n=== MERGE + SAVE ===")

    # Start with batting (position players)
    proj = bat_proj.merge(
        pit_proj.drop(columns=["birth_year"], errors="ignore"),
        on="player_id", how="outer", suffixes=("","_pit"),
    )
    proj = proj.merge(
        fld_proj.drop(columns=["birth_year_f","age_proj_f"], errors="ignore"),
        on="player_id", how="outer",
    )

    # Consolidate birth_year and age_proj from batting/pitching
    if "birth_year_pit" in proj.columns:
        proj["birth_year"] = proj["birth_year"].fillna(proj["birth_year_pit"])
    if "age_proj_pit" in proj.columns:
        proj["age_proj"]   = proj["age_proj"].fillna(proj["age_proj_pit"])

    # Add player metadata
    meta = players[["player_id","first_name","last_name",
                     "mlb_service_years","Role","is_active","Level","Retired","bats"]].copy()
    proj = proj.merge(meta, on="player_id", how="left")
    proj["year_proj"] = PROJ_YEAR

    # Players absent from bat_proj/pit_proj get refused=True (they have no data at all).
    # Players present but thin stay as thin=False-with-refused=False.
    # Cast to numpy bool to avoid Python bool bitwise issue (~True == -2).
    for col, default in [("refused_bat", True), ("thin_bat", False),
                          ("refused_pit", True), ("thin_pit", False)]:
        if col in proj.columns:
            proj[col] = proj[col].fillna(default).astype(bool)
        else:
            proj[col] = bool(default)

    # Drop redundant suffix columns if any
    for col in list(proj.columns):
        if col.endswith("_pit") and col not in ["refused_pit","thin_pit","bb_frac"]:
            proj.drop(columns=[col], inplace=True, errors="ignore")

    proj.to_csv(INT_DIR / "marcel_projections.csv", index=False)
    _ok(f"Saved marcel_projections.csv ({len(proj):,} rows)")

    # ── SUMMARY ─────────────────────────────────────────────────────────
    n_bat = int((~proj["refused_bat"]).sum())
    n_pit = int((~proj["refused_pit"]).sum())

    print(f"\n{'='*60}")
    print(f"Step 4 complete - {PROJ_YEAR} projections")
    print(f"  Batting projections:  {n_bat}")
    print(f"  Pitching projections: {n_pit}")
    print(f"  Total rows in output: {len(proj)}")
    if _issues:
        print(f"  Warnings ({len(_issues)}):")
        for iss in _issues:
            print(f"    - {iss}")
    else:
        print("  No warnings.")


if __name__ == "__main__":
    main()
