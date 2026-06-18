"""
step2_aging_curves.py — Step 2: Delta-method aging curves

Reads:
  intermediate/batting_neutral.csv
  intermediate/pitching_neutral.csv
  intermediate/fielding_raw.csv
  frostfire_data/players.csv

Writes (all to intermediate/):
  aging_deltas.csv          — raw consecutive-pair deltas for every component
  aging_cell_stats.csv      — per (group, component, age): n, mean, median, SE, weighted mean
  aging_curves_smooth.csv   — raw cell stats + poly2/3/4 and LOESS 30/50/70 predictions
                              + cumulative adjustment from age 20 for each smoother
  aging_fit_stats.csv       — R², RMSE, AIC/BIC, LOOA-CV RMSE per (group, component, model)
  aging_threshold_sens.csv  — stability check: key metrics at PA/BF/IP thresholds 50/100/150/200

Components tracked (split_id=1 overall only):

  Batting — per PA, park-neutral counts; group = primary position at highest-PA stint:
    hr_pa     = hr_n / pa
    xbh_pa    = (d_n + t_n) / pa
    single_pa = singles_n / pa
    bb_pa     = bb / pa
    k_pa      = k / pa           (higher = worse for batter)
    ubr_g     = ubr / g

  Pitching — per BF, park-neutral hra; group = SP (gs/g >= SP_THRESHOLD) or RP:
    k_bf       = k / bf
    bb_hbp_bf  = (bb + hp) / bf
    hra_bf     = hra_n / bf      (higher = worse for pitcher)

  Fielding — per 1000 innings; group = fielding position 2–9 (pitchers excluded):
    zr_rate      = zr / ip * 1000
    arm_rate     = arm / ip * 1000
    framing_rate = framing / ip * 1000  (non-null only for catchers, position 2)

Position group labels:
  Batting/baserunning: C(2), 1B(3), 2B(4), 3B(5), SS(6), LF(7), CF(8), RF(9), DH(10)
  Pitching: SP, RP
  Fielding/defense: same label as batting but based on where they actually fielded

Hyperparameters (vary these to tune):
  MIN_PA = 100   → batting threshold: also tested at 50 / 150 / 200
  MIN_BF = 100   → pitching threshold: also tested at 50 / 150 / 200
  MIN_IP = 50    → fielding threshold: also tested at 20 / 100 / 200
  SP_THRESHOLD = 0.5  (gs/g cutoff)
  Poly degrees: 2, 3, 4; LOESS bandwidths: 0.3, 0.5, 0.7

Delta method: for each player with consecutive qualifying seasons Y and Y+1:
  delta = component(Y+1) - component(Y)
  assigned to age = year_Y - birth_year   (start-year age)
  group assigned from year Y (start year), regardless of role/position in year Y+1
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

DATA_DIR = Path("frostfire_data")
INT_DIR  = Path("intermediate")

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

MIN_PA        = 100
MIN_BF        = 100
MIN_IP_FLD    = 50
SP_THRESHOLD  = 0.5    # gs/g >= threshold → SP, else RP
AGE_MIN       = 20
AGE_MAX       = 40
MIN_PAIRS_AGE = 5      # flag (but retain) age cells below this count

POLY_DEGREES  = [2, 3, 4]
LOESS_FRACS   = [0.3, 0.5, 0.7]

# ── Survivorship-bias correction ─────────────────────────────────────────────
# Woolner-style exit correction: for each player-group-age A that has a
# qualifying season but NO qualifying next-year season, add a synthetic delta
#   delta_synthetic = group_mean_at_age_(A+1) – player_value_at_age_A
# This pulls the average down at older ages where below-average players exit,
# counteracting the upward bias from only observing survivors.
APPLY_EXIT_CORRECTION  = True
EXIT_CORRECTION_WEIGHT = 0.5   # how much to weight synthetic deltas vs real ones

PA_THRESHOLDS  = [50, 100, 150, 200]
BF_THRESHOLDS  = [50, 100, 150, 200]
IP_THRESHOLDS  = [20,  50, 100, 200]

POS_LABELS = {
    2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS",
    7: "LF", 8: "CF", 9: "RF", 10: "DH",
}

BAT_COMPONENTS = ["hr_pa", "xbh_pa", "single_pa", "bb_pa", "k_pa", "ubr_g"]
PIT_COMPONENTS = ["k_bf", "bb_hbp_bf", "hra_bf"]
FLD_COMPONENTS = ["zr_rate", "arm_rate", "framing_rate"]

FULL_AGES = np.arange(AGE_MIN, AGE_MAX + 1, dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

_issues: list[str] = []

def _warn(msg: str) -> None:
    """Record a warning message in _issues and print it with a [WARN] tag."""
    _issues.append(msg)
    print(f"  [WARN] {msg}")

def _ok(msg: str) -> None:
    """Print a passing-check message with an [OK] tag."""
    print(f"  [OK]   {msg}")

def _info(msg: str) -> None:
    """Print an informational message with an [INFO] tag."""
    print(f"  [INFO] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _group_label(x: str | int) -> str:
    """Map a group value to a human-readable label."""
    try:
        return POS_LABELS.get(int(x), str(x))
    except (ValueError, TypeError):
        return str(x)

def _cumulative(delta_preds: np.ndarray) -> np.ndarray:
    """
    Cumulative sum of smoothed per-age deltas.
    cum[i] = total change from AGE_MIN to (AGE_MIN + i).
    cum[0] = 0 by definition (anchor at the youngest age).
    """
    return np.concatenate([[0.0], np.nancumsum(delta_preds[:-1])])

# ─────────────────────────────────────────────────────────────────────────────
# Players: birth year
# ─────────────────────────────────────────────────────────────────────────────

def load_players() -> pd.DataFrame:
    """Load player_id and birth_year from players.csv, dropping rows with unparseable dates of birth."""
    print("\n=== PLAYERS ===")
    p = pd.read_csv(DATA_DIR / "players.csv", usecols=["ID", "date_of_birth"])
    p.rename(columns={"ID": "player_id"}, inplace=True)
    p["birth_year"] = pd.to_datetime(p["date_of_birth"], errors="coerce").dt.year
    bad = int(p["birth_year"].isna().sum())
    if bad:
        _warn(f"{bad:,} players with unparseable date_of_birth — excluded from aging curves")
    else:
        _ok(f"All {len(p):,} players have parseable birth years")
    bmin = int(p["birth_year"].min())
    bmax = int(p["birth_year"].max())
    _info(f"birth_year range: {bmin} – {bmax}")
    return p[["player_id", "birth_year"]].dropna()

# ─────────────────────────────────────────────────────────────────────────────
# Season-component builders
# ─────────────────────────────────────────────────────────────────────────────

def build_position_lookup(players: pd.DataFrame) -> pd.Series:
    """
    For each (player_id, year), derive primary defensive position from
    fielding_raw (position with most innings among 2–9).  Falls back to
    players.csv Pos for player-years with no fielding record.

    NOTE: position column in the batting stat files is always 0 in this
    dataset — OOTP stores the defensive position in the fielding file only.
    """
    fld = pd.read_csv(INT_DIR / "fielding_raw.csv", usecols=["player_id", "year", "position", "ip"])
    fld = fld[fld["position"].between(2, 9)].copy()

    # Per (player_id, year): position with the most innings
    fld_agg = fld.groupby(["player_id", "year", "position"], sort=False)["ip"].sum().reset_index()
    idx_max = fld_agg.groupby(["player_id", "year"])["ip"].idxmax()
    primary = fld_agg.loc[idx_max, ["player_id", "year", "position"]].copy()
    primary = primary.set_index(["player_id", "year"])["position"]

    # Bio fallback: players.csv Pos (current primary position), only for positions 2–9
    bio_pos = (
        players.merge(
            pd.read_csv(DATA_DIR / "players.csv", usecols=["ID", "Pos"])
            .rename(columns={"ID": "player_id", "Pos": "pos_bio"}),
            on="player_id", how="left",
        )
        .set_index("player_id")["pos_bio"]
    )
    # Keep only defensive positions 2–9 in bio; remap DH (10) to 10, ignore pitchers (1)
    bio_pos = bio_pos[bio_pos.between(2, 10)]

    _info(f"Position lookup: {len(primary):,} player-year entries from fielding; "
          f"bio fallback covers {len(bio_pos):,} players")
    return primary, bio_pos


def batting_seasons(
    min_pa: int = MIN_PA,
    pos_lookup: tuple | None = None,
) -> pd.DataFrame:
    """
    One row per (player_id, year) for split_id=1 seasons with >= min_pa PA.
    Multi-stint seasons are aggregated by summing counting stats.

    Position is derived from the fielding file (innings-primary) via
    pos_lookup, not from the batting file (which stores 0 for every row).

    pos_lookup = (primary_series, bio_series) as returned by build_position_lookup().
    If not supplied the function loads the fielding data directly.
    """
    usecols = ["player_id", "year", "split_id",
               "pa", "g", "hr_n", "d_n", "t_n", "singles_n", "bb", "k", "ubr"]
    bat = pd.read_csv(INT_DIR / "batting_neutral.csv", usecols=usecols)
    bat = bat[bat["split_id"] == 1].copy()

    agg = bat.groupby(["player_id", "year"], sort=False).agg(
        pa        = ("pa",        "sum"),
        g         = ("g",         "sum"),
        hr_n      = ("hr_n",      "sum"),
        d_n       = ("d_n",       "sum"),
        t_n       = ("t_n",       "sum"),
        singles_n = ("singles_n", "sum"),
        bb        = ("bb",        "sum"),
        k         = ("k",         "sum"),
        ubr       = ("ubr",       "sum"),
    ).reset_index()
    agg = agg[agg["pa"] >= min_pa].copy()

    # Attach position from fielding lookup
    if pos_lookup is None:
        # Minimal players df just for the fallback
        _p = pd.read_csv(DATA_DIR / "players.csv", usecols=["ID"])\
               .rename(columns={"ID": "player_id"})
        primary_s, bio_s = build_position_lookup(_p)
    else:
        primary_s, bio_s = pos_lookup

    def _get_pos(pid: int, yr: int) -> int:
        pos = primary_s.get((pid, yr))
        if pos is not None:
            return int(pos)
        # bio fallback: players.csv primary position
        pos_b = bio_s.get(pid)
        if pos_b is not None:
            return int(pos_b)
        return -1  # unknown — will be excluded from position-specific curves

    agg["position"] = [_get_pos(r.player_id, r.year)
                       for r in agg[["player_id", "year"]].itertuples()]

    # Pool DH (position 10) into 1B (position 3) — only 4 qualifying seasons in 21 years,
    # far too thin for its own aging curve.
    agg["position"] = agg["position"].replace({10: 3})

    n_unknown = (agg["position"] == -1).sum()
    if n_unknown:
        _warn(f"Batting: {n_unknown:,} player-years with no position assignment -> group=-1")

    agg["hr_pa"]     = agg["hr_n"]               / agg["pa"]
    agg["xbh_pa"]    = (agg["d_n"] + agg["t_n"]) / agg["pa"]
    agg["single_pa"] = agg["singles_n"]           / agg["pa"]
    agg["bb_pa"]     = agg["bb"]                  / agg["pa"]
    agg["k_pa"]      = agg["k"]                   / agg["pa"]
    agg["ubr_g"]     = agg["ubr"]                 / agg["g"].clip(lower=1)

    pos_dist = agg["position"].value_counts().sort_index().to_dict()
    _info(f"Batting seasons (pa>={min_pa}): {len(agg):,} player-years, "
          f"{agg['player_id'].nunique():,} unique players")
    _info(f"  Position distribution: {pos_dist}")
    return agg[["player_id", "year", "pa", "g", "position"] + BAT_COMPONENTS]


def pitching_seasons(min_bf: int = MIN_BF) -> pd.DataFrame:
    """
    One row per (player_id, year) for split_id=1 seasons with >= min_bf BF.
    Role = SP if aggregate gs/g >= SP_THRESHOLD, else RP.
    """
    usecols = ["player_id", "year", "split_id", "bf", "k", "bb", "hp", "hra_n", "g", "gs"]
    pit = pd.read_csv(INT_DIR / "pitching_neutral.csv", usecols=usecols)
    pit = pit[pit["split_id"] == 1].copy()

    agg = pit.groupby(["player_id", "year"], sort=False).agg(
        bf    = ("bf",    "sum"),
        k     = ("k",     "sum"),
        bb    = ("bb",    "sum"),
        hp    = ("hp",    "sum"),
        hra_n = ("hra_n", "sum"),
        g     = ("g",     "sum"),
        gs    = ("gs",    "sum"),
    ).reset_index()

    agg["role"] = np.where(agg["gs"] / agg["g"].clip(lower=1) >= SP_THRESHOLD, "SP", "RP")
    agg = agg[agg["bf"] >= min_bf].copy()

    agg["k_bf"]      = agg["k"]                  / agg["bf"]
    agg["bb_hbp_bf"] = (agg["bb"] + agg["hp"])   / agg["bf"]
    agg["hra_bf"]    = agg["hra_n"]               / agg["bf"]

    sp_n = (agg["role"] == "SP").sum()
    rp_n = (agg["role"] == "RP").sum()
    _info(f"Pitching seasons (bf>={min_bf}): {len(agg):,} player-years "
          f"[SP={sp_n:,}, RP={rp_n:,}]")
    return agg[["player_id", "year", "bf", "role"] + PIT_COMPONENTS]


def fielding_seasons(min_ip: float = MIN_IP_FLD) -> pd.DataFrame:
    """
    One row per (player_id, year, position) for positions 2–9 with >= min_ip innings.
    Multi-team seasons are summed. Framing rate is non-null only for catchers (position 2).
    """
    usecols = ["player_id", "year", "position", "ip", "zr", "framing", "arm"]
    fld = pd.read_csv(INT_DIR / "fielding_raw.csv", usecols=usecols)
    fld = fld[fld["position"] >= 2].copy()

    agg = fld.groupby(["player_id", "year", "position"], sort=False).agg(
        ip      = ("ip",      "sum"),
        zr      = ("zr",      "sum"),
        framing = ("framing", "sum"),
        arm     = ("arm",     "sum"),
    ).reset_index()
    agg = agg[agg["ip"] >= min_ip].copy()

    agg["zr_rate"]      = agg["zr"]      / agg["ip"] * 1000
    agg["arm_rate"]     = agg["arm"]     / agg["ip"] * 1000
    agg["framing_rate"] = np.where(
        agg["position"] == 2,
        agg["framing"] / agg["ip"] * 1000,
        np.nan,
    )

    _info(f"Fielding seasons (ip>={min_ip}): {len(agg):,} player-position-years, "
          f"positions: {sorted(agg['position'].unique().tolist())}")
    return agg[["player_id", "year", "position", "ip"] + FLD_COMPONENTS]

# ─────────────────────────────────────────────────────────────────────────────
# Delta-method pair formation
# ─────────────────────────────────────────────────────────────────────────────

def make_pairs(
    seasons: pd.DataFrame,
    group_col: str,
    component_cols: list[str],
    weight_col: str,
    players: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """
    For each player with consecutive qualifying seasons (year Y, year Y+1),
    compute delta = component(Y+1) – component(Y).

    Assigns each delta to:
      age   = year_Y – birth_year
      group = group_col value from year Y (start-year assignment)

    Weight = average of the weight_col (PA, BF, or IP) across both seasons.
    """
    df = seasons.merge(players, on="player_id", how="inner")

    # Build "end year" lookup: year → year-1 so it aligns with start year on join
    end_cols = {c: f"{c}_end" for c in component_cols + [weight_col]}
    nxt = (
        df[["player_id", "year"] + component_cols + [weight_col]]
        .copy()
        .rename(columns=end_cols)
    )
    nxt["year_start"] = nxt["year"] - 1
    nxt = nxt.drop(columns=["year"])

    # Build "start year" table (group_col and birth_year retained unmodified)
    start_rename = {c: f"{c}_start" for c in component_cols + [weight_col]}
    start = (
        df[["player_id", "year", "birth_year", group_col] + component_cols + [weight_col]]
        .copy()
        .rename(columns={**start_rename, "year": "year_start"})
    )

    merged = start.merge(nxt, on=["player_id", "year_start"], how="inner")
    merged["age"] = merged["year_start"] - merged["birth_year"]
    merged = merged[
        merged["age"].between(AGE_MIN, AGE_MAX)
    ].copy()

    records: list[pd.DataFrame] = []
    for comp in component_cols:
        s = merged.dropna(subset=[f"{comp}_start", f"{comp}_end"]).copy()
        if s.empty:
            continue
        s["delta"]     = s[f"{comp}_end"] - s[f"{comp}_start"]
        s["weight"]    = (s[f"{weight_col}_start"] + s[f"{weight_col}_end"]) / 2.0
        s["component"] = comp
        s["group"]     = s[group_col].astype(str)
        s["data_type"] = label
        records.append(
            s[["player_id", "year_start", "age", "data_type", "group", "component",
               "delta", "weight"]]
        )

    if not records:
        _warn(f"make_pairs ({label}): no valid pairs found")
        return pd.DataFrame(
            columns=["player_id", "year_start", "age", "data_type",
                     "group", "component", "delta", "weight"]
        )
    out = pd.concat(records, ignore_index=True)
    _info(f"{label} pairs: {len(out):,} rows, "
          f"{out['player_id'].nunique():,} unique players")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Survivorship-bias correction (Woolner exit correction)
# ─────────────────────────────────────────────────────────────────────────────

def exit_correction_pairs(
    seasons: pd.DataFrame,
    group_col: str,
    component_cols: list[str],
    weight_col: str,
    players: pd.DataFrame,
    label: str,
    exit_weight: float = EXIT_CORRECTION_WEIGHT,
) -> pd.DataFrame:
    """
    Woolner-style exit correction for survivorship bias.

    For each player-group who has a qualifying season at age A but NOT at age
    A+1, we add a synthetic delta:

        delta_synthetic = group_mean_component(age A+1) – player_value(age A)

    The group mean at age A+1 is computed from the raw seasons DataFrame
    (all qualifying players at that position-age, pooled across all years).
    Synthetic deltas get weight = player's real weight × exit_weight.

    Why this helps on defense: without correction, only players who hold onto
    premium defensive positions (SS, CF) into their late 30s end up in those
    age-35+ cells — these are the elite fielders, so the raw delta looks flat
    or positive.  Adding synthetic exits for the below-average fielders who
    left those positions pulls the average back down.

    Returned data_type is "{label}_exit" so it can be distinguished from
    real pairs in the output CSVs.
    """
    df = seasons.merge(players, on="player_id", how="inner")
    df["age"] = df["year"] - df["birth_year"]
    df = df[df["age"].between(AGE_MIN, AGE_MAX)].copy()

    # Normalise the group column to string throughout to avoid int/float
    # type-mismatch when building the age_means lookup keys.
    df = df.copy()
    df[group_col] = df[group_col].astype(str)

    # ── Group-age-component means from all qualifying seasons ────────────────
    age_means: dict[tuple, dict[str, float]] = {}
    for (grp, age), g in df.groupby([group_col, "age"], sort=False):
        key = (str(grp), int(age))
        age_means[key] = {}
        for comp in component_cols:
            vals = g[comp].dropna()
            if len(vals) >= 5:
                age_means[key][comp] = float(vals.mean())

    # ── Find player-group-years that have NO next-year season ────────────────
    # Build next_keys explicitly to avoid pandas index-alignment surprises.
    next_keys = df[[group_col, "player_id", "year"]].copy()
    next_keys["year_start"] = next_keys["year"] - 1
    next_keys = next_keys.drop(columns=["year"])

    start = df[[group_col, "player_id", "year", "age"] + component_cols + [weight_col]].copy()
    start = start.rename(columns={"year": "year_start"})

    merged = start.merge(
        next_keys.assign(_has_next=1),
        on=["player_id", "year_start", group_col],
        how="left",
    )
    exiting = merged[merged["_has_next"].isna()].drop(columns=["_has_next"]).copy()
    # Don't try to correct the last age — no A+1 reference exists
    exiting = exiting[exiting["age"] < AGE_MAX].copy()

    # ── Build synthetic deltas ────────────────────────────────────────────────
    records: list[pd.DataFrame] = []
    for comp in component_cols:
        rows: list[dict] = []
        for _, row in exiting.iterrows():
            grp_str = str(row[group_col])
            age_A   = int(row["age"])
            key_A1  = (grp_str, age_A + 1)
            if key_A1 not in age_means or comp not in age_means[key_A1]:
                continue
            val_A = row[comp]
            if pd.isna(val_A):
                continue
            rows.append({
                "player_id":  int(row["player_id"]),
                "year_start": int(row["year_start"]),
                "age":        age_A,
                "data_type":  f"{label}_exit",
                "group":      grp_str,
                "component":  comp,
                "delta":      age_means[key_A1][comp] - val_A,
                "weight":     float(row[weight_col]) * exit_weight,
            })
        if rows:
            records.append(pd.DataFrame(rows))

    if not records:
        _warn(f"exit_correction ({label}): no synthetic deltas produced")
        return pd.DataFrame(
            columns=["player_id", "year_start", "age", "data_type",
                     "group", "component", "delta", "weight"]
        )
    out = pd.concat(records, ignore_index=True)
    _info(f"{label} exit-correction: {len(exiting):,} exiting player-years -> "
          f"{len(out):,} synthetic deltas "
          f"(weight={exit_weight})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cell-level statistics
# ─────────────────────────────────────────────────────────────────────────────

def cell_stats(deltas: pd.DataFrame) -> pd.DataFrame:
    """
    Per (group, component, age) rich statistics for hyperparameter evaluation.

    The `data_type` column distinguishes real consecutive-year pairs from
    synthetic exit-correction pairs (data_type ends with "_exit").  Both
    are included in the aggregate statistics; counts of each are reported
    separately for transparency.
    """
    rows: list[dict] = []
    for (group, comp, age), grp in deltas.groupby(
        ["group", "component", "age"], sort=True
    ):
        is_exit = grp["data_type"].str.endswith("_exit")
        n_nat   = int((~is_exit).sum())
        n_exit  = int(is_exit.sum())
        n       = n_nat + n_exit

        vals = grp["delta"].values
        wts  = grp["weight"].values
        wts  = np.where(np.isfinite(wts) & (wts > 0), wts, 1.0)

        # flag_thin is based on natural pairs only (exit synthetics don't
        # represent independent observations)
        rows.append({
            "group":              group,
            "component":          comp,
            "age":                int(age),
            "n_pairs":            n,
            "n_natural":          n_nat,
            "n_exit_correction":  n_exit,
            "flag_thin":          int(n_nat < MIN_PAIRS_AGE),
            "mean_delta":         float(np.mean(vals)),
            "median_delta":       float(np.median(vals)),
            "std_delta":          float(np.std(vals, ddof=1)) if n > 1 else np.nan,
            "se_delta":           float(sp_stats.sem(vals))   if n > 1 else np.nan,
            "q25_delta":          float(np.percentile(vals, 25)),
            "q75_delta":          float(np.percentile(vals, 75)),
            "weighted_mean":      float(np.average(vals, weights=wts)),
            "bias_raw_vs_wt":     float(np.mean(vals) - np.average(vals, weights=wts)),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Polynomial + LOESS fitting
# ─────────────────────────────────────────────────────────────────────────────

def _poly_fit(
    ages: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
    deg: int,
) -> tuple:
    """
    Weighted polynomial fit of given degree.
    Returns (coeffs, r2, rmse, aic, bic, cv_rmse).
    All metrics are NaN (and coeffs is None) if the sample is too small.
    """
    n, k = len(ages), deg + 1
    if n <= deg:
        return None, np.nan, np.nan, np.nan, np.nan, np.nan

    coeffs   = np.polyfit(ages, means, deg, w=np.sqrt(weights))
    pred     = np.polyval(coeffs, ages)
    residuals = means - pred

    # Weighted R²
    wt_mean  = np.average(means, weights=weights)
    rss      = float(np.sum(weights * residuals**2))
    tss      = float(np.sum(weights * (means - wt_mean)**2))
    r2       = float(1.0 - rss / tss) if tss > 0 else np.nan

    rmse = float(np.sqrt(np.mean(residuals**2)))
    eps  = 1e-14
    aic  = float(n * np.log(rss / n + eps) + 2 * k)
    bic  = float(n * np.log(rss / n + eps) + k * np.log(max(n, 2)))

    # Leave-one-age-out cross-validation
    cv_errs: list[float] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if mask.sum() <= deg:
            continue
        c_cv = np.polyfit(ages[mask], means[mask], deg, w=np.sqrt(weights[mask]))
        cv_errs.append(float(means[i] - np.polyval(c_cv, ages[i])))
    cv_rmse = float(np.sqrt(np.mean(np.array(cv_errs)**2))) if cv_errs else np.nan

    return coeffs, r2, rmse, aic, bic, cv_rmse


def fit_all_curves(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (group, component) pair, fit poly2/3/4 and LOESS 30/50/70 on the
    age-level mean deltas.

    Returns:
      smooth_df — one row per (group, component, age) in AGE_MIN..AGE_MAX
      fit_df    — one row per (group, component, model) with goodness-of-fit metrics
    """
    smooth_rows: list[dict] = []
    fit_rows:    list[dict] = []

    gc_pairs = (
        cells[["group", "component"]]
        .drop_duplicates()
        .sort_values(["group", "component"])
        .values
    )

    for group, comp in gc_pairs:
        gc = (
            cells[(cells["group"] == group) & (cells["component"] == comp)]
            .sort_values("age")
            .dropna(subset=["mean_delta"])
        )
        if len(gc) < 4:
            _warn(f"Skipping {group}/{comp}: only {len(gc)} age cells with data")
            continue

        ages    = gc["age"].values.astype(float)
        means   = gc["mean_delta"].values.astype(float)
        weights = gc["n_pairs"].values.astype(float)

        # Lookup dicts for cell stats by age
        cs_by_age = gc.set_index("age").to_dict(orient="index")

        # ── Polynomial fits ──────────────────────────────────────────────────
        poly_preds:  dict[int, np.ndarray | None] = {}
        poly_cumsums: dict[int, np.ndarray | None] = {}

        for deg in POLY_DEGREES:
            coeffs, r2, rmse, aic, bic, cv_rmse = _poly_fit(ages, means, weights, deg)
            if coeffs is None:
                poly_preds[deg]   = None
                poly_cumsums[deg] = None
            else:
                preds = np.polyval(coeffs, FULL_AGES)
                cum   = _cumulative(preds)
                poly_preds[deg]   = preds
                poly_cumsums[deg] = cum
                peak_age = int(FULL_AGES[np.argmax(cum)])

                fit_rows.append({
                    "group":       group,
                    "component":   comp,
                    "model":       f"poly{deg}",
                    "r2":          r2,
                    "rmse":        rmse,
                    "aic":         aic,
                    "bic":         bic,
                    "cv_rmse":     cv_rmse,
                    "n_ages":      len(ages),
                    "n_ages_good": int((weights >= MIN_PAIRS_AGE).sum()),
                    "n_pairs":     int(weights.sum()),
                    "n_thin_ages": int(gc["flag_thin"].sum()),
                    "age_data_min": int(ages.min()),
                    "age_data_max": int(ages.max()),
                    "peak_age":    peak_age,
                    **{f"cum_age{a}": float(cum[a - AGE_MIN])
                       for a in [25, 28, 30, 32, 35, 40]},
                })

        # ── LOESS fits ────────────────────────────────────────────────────────
        loess_preds:   dict[str, np.ndarray | None] = {}
        loess_cumsums: dict[str, np.ndarray | None] = {}

        for frac in LOESS_FRACS:
            key = f"loess{int(frac * 100)}"
            try:
                sm     = sm_lowess(means, ages, frac=frac, it=3, return_sorted=True)
                preds  = np.interp(FULL_AGES, sm[:, 0], sm[:, 1],
                                   left=np.nan, right=np.nan)
                cum    = _cumulative(preds)
                loess_preds[key]   = preds
                loess_cumsums[key] = cum

                valid = ~np.isnan(preds)
                peak_age_l = (
                    int(FULL_AGES[valid][np.argmax(cum[valid])])
                    if valid.any() else np.nan
                )
                fit_rows.append({
                    "group":       group,
                    "component":   comp,
                    "model":       key,
                    "r2":          np.nan,
                    "rmse":        np.nan,
                    "aic":         np.nan,
                    "bic":         np.nan,
                    "cv_rmse":     np.nan,
                    "n_ages":      len(ages),
                    "n_ages_good": int((weights >= MIN_PAIRS_AGE).sum()),
                    "n_pairs":     int(weights.sum()),
                    "n_thin_ages": int(gc["flag_thin"].sum()),
                    "age_data_min": int(ages.min()),
                    "age_data_max": int(ages.max()),
                    "peak_age":    peak_age_l,
                    **{f"cum_age{a}": (
                        float(cum[a - AGE_MIN])
                        if not np.isnan(preds[a - AGE_MIN]) else np.nan
                    ) for a in [25, 28, 30, 32, 35, 40]},
                })
            except Exception as e:
                _warn(f"LOESS failed {group}/{comp}/frac={frac}: {e}")
                loess_preds[key]   = None
                loess_cumsums[key] = None

        # ── Assemble per-age smooth rows ─────────────────────────────────────
        for ia, age in enumerate(FULL_AGES):
            cs = cs_by_age.get(int(age), {})
            row: dict = {
                "group":         group,
                "component":     comp,
                "age":           int(age),
                "n_pairs":       cs.get("n_pairs",        0),
                "flag_thin":     cs.get("flag_thin",      1),
                "mean_delta":    cs.get("mean_delta",     np.nan),
                "median_delta":  cs.get("median_delta",   np.nan),
                "se_delta":      cs.get("se_delta",       np.nan),
                "weighted_mean": cs.get("weighted_mean",  np.nan),
            }
            for deg in POLY_DEGREES:
                p = poly_preds.get(deg)
                c = poly_cumsums.get(deg)
                row[f"poly{deg}_delta"]      = float(p[ia]) if p is not None else np.nan
                row[f"poly{deg}_cumulative"] = float(c[ia]) if c is not None else np.nan
            for frac in LOESS_FRACS:
                key = f"loess{int(frac * 100)}"
                lp  = loess_preds.get(key)
                lc  = loess_cumsums.get(key)
                row[f"{key}_delta"]      = float(lp[ia]) if lp is not None else np.nan
                row[f"{key}_cumulative"] = float(lc[ia]) if lc is not None else np.nan

            smooth_rows.append(row)

    smooth_df = pd.DataFrame(smooth_rows)
    fit_df    = pd.DataFrame(fit_rows)

    for df in (smooth_df, fit_df):
        if not df.empty:
            df.insert(1, "group_label", df["group"].map(_group_label))

    return smooth_df, fit_df

# ─────────────────────────────────────────────────────────────────────────────
# Threshold sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def _run_sensitivity_block(
    seasons_fn,
    thresh_values: list[int | float],
    thresh_kwarg: str,
    group_col: str,
    comp_cols: list[str],
    weight_col: str,
    label: str,
    players: pd.DataFrame,
    extra_kwargs: dict | None = None,
) -> list[dict]:
    """Sweep a qualification threshold (PA/BF/IP) through thresh_values, refitting poly3 aging curves at each level to assess hyperparameter sensitivity."""
    rows: list[dict] = []
    kw = extra_kwargs or {}
    for thresh in thresh_values:
        ss = seasons_fn(**{thresh_kwarg: thresh, **kw})
        deltas = make_pairs(ss, group_col, comp_cols, weight_col, players, label)
        if deltas.empty:
            continue
        cells = cell_stats(deltas)
        for (grp, comp), gc in cells.groupby(["group", "component"]):
            gc = gc.sort_values("age").dropna(subset=["mean_delta"])
            if len(gc) < 4:
                continue
            ages  = gc["age"].values.astype(float)
            means = gc["mean_delta"].values.astype(float)
            wts   = gc["n_pairs"].values.astype(float)

            _, r2, _, _, _, cv = _poly_fit(ages, means, wts, 3)

            if len(ages) > 3:
                preds = np.polyval(np.polyfit(ages, means, 3, w=np.sqrt(wts)), FULL_AGES)
                cum   = _cumulative(preds)
                peak  = int(FULL_AGES[np.argmax(cum)])
            else:
                peak = np.nan

            rows.append({
                "label":            label,
                "group":            grp,
                "group_label":      _group_label(grp),
                "component":        comp,
                f"{thresh_kwarg}":  thresh,
                "n_pairs":          int(gc["n_pairs"].sum()),
                "n_ages_covered":   len(gc),
                "n_thin_ages":      int(gc["flag_thin"].sum()),
                "poly3_r2":         r2,
                "poly3_cv_rmse":    cv,
                "peak_age_poly3":   peak,
            })
    return rows


def threshold_sensitivity(players: pd.DataFrame, pos_lookup: tuple) -> pd.DataFrame:
    """Run the threshold sensitivity sweep for batting, pitching, and fielding qualification thresholds and combine the results."""
    print("\n  Running batting threshold sensitivity...")
    bat_rows = _run_sensitivity_block(
        batting_seasons, PA_THRESHOLDS, "min_pa",
        "position", BAT_COMPONENTS, "pa", "batting", players,
        extra_kwargs={"pos_lookup": pos_lookup},
    )
    print("  Running pitching threshold sensitivity...")
    pit_rows = _run_sensitivity_block(
        pitching_seasons, BF_THRESHOLDS, "min_bf",
        "role", PIT_COMPONENTS, "bf", "pitching", players,
    )
    print("  Running fielding threshold sensitivity...")
    fld_rows = _run_sensitivity_block(
        fielding_seasons, IP_THRESHOLDS, "min_ip",
        "position", FLD_COMPONENTS, "ip", "fielding", players,
    )
    return pd.DataFrame(bat_rows + pit_rows + fld_rows)

# ─────────────────────────────────────────────────────────────────────────────
# Console preview helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_fit_table(fit_df: pd.DataFrame, model: str, n: int, ascending: bool, title: str) -> None:
    """Print the top/bottom n rows of fit_df for the given model, sorted by R², under a title heading."""
    sub = fit_df[fit_df["model"] == model].sort_values("r2", ascending=ascending)
    cols = ["group_label", "component", "r2", "cv_rmse", "peak_age",
            "n_pairs", "n_ages", "n_ages_good", "n_thin_ages",
            "age_data_min", "age_data_max"]
    cols = [c for c in cols if c in sub.columns]
    print(f"\n{title}:")
    print(sub[cols].head(n).to_string(index=False))


def _print_cumulative_at_reference_ages(smooth_df: pd.DataFrame) -> None:
    """Print cumulative adjustment at ages 28, 33, 38 for each (group, component) using poly3."""
    col_28 = "poly3_cumulative"
    if col_28 not in smooth_df.columns:
        return
    sub = smooth_df[smooth_df["age"].isin([20, 25, 28, 30, 32, 35, 38, 40])].copy()
    sub = sub[["group_label", "component", "age", col_28]].rename(columns={col_28: "cum_poly3"})
    piv = sub.pivot_table(index=["group_label", "component"],
                          columns="age", values="cum_poly3")
    piv.columns = [f"age{c}" for c in piv.columns]
    piv = piv.reset_index()
    print("\nCumulative poly3 adjustment (relative to age 20):")
    print(piv.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full step 2 pipeline: build season components, form delta pairs, apply exit correction, fit aging curves, run threshold sensitivity, and write all output CSVs."""
    print("Frostfire Aging Curves — Step 2: Delta Method")
    print("=" * 60)
    print(f"Hyperparameters: MIN_PA={MIN_PA}, MIN_BF={MIN_BF}, "
          f"MIN_IP_FLD={MIN_IP_FLD}, SP_THRESHOLD={SP_THRESHOLD}")
    print(f"Age range: {AGE_MIN}–{AGE_MAX} | Min pairs/age flagged: {MIN_PAIRS_AGE}")

    players = load_players()

    # ── Build position lookup once (shared by batting seasons + sensitivity) ──
    print("\n=== BUILDING POSITION LOOKUP ===")
    pos_lookup = build_position_lookup(players)

    # ── Season components at default thresholds ──────────────────────────────
    print("\n=== COMPUTING SEASON COMPONENTS (default thresholds) ===")
    bat_ss = batting_seasons(MIN_PA, pos_lookup=pos_lookup)
    pit_ss = pitching_seasons(MIN_BF)
    fld_ss = fielding_seasons(MIN_IP_FLD)

    # ── Consecutive-year delta pairs ─────────────────────────────────────────
    print("\n=== FORMING CONSECUTIVE PAIRS ===")
    bat_d  = make_pairs(bat_ss, "position", BAT_COMPONENTS, "pa",  players, "batting")
    pit_d  = make_pairs(pit_ss, "role",     PIT_COMPONENTS, "bf",  players, "pitching")
    fld_d  = make_pairs(fld_ss, "position", FLD_COMPONENTS, "ip",  players, "fielding")

    pieces = [bat_d, pit_d, fld_d]

    # ── Survivorship-bias correction ─────────────────────────────────────────
    if APPLY_EXIT_CORRECTION:
        print("\n=== APPLYING EXIT CORRECTION (survivorship bias) ===")
        bat_exit = exit_correction_pairs(bat_ss, "position", BAT_COMPONENTS, "pa",
                                          players, "batting")
        pit_exit = exit_correction_pairs(pit_ss, "role",     PIT_COMPONENTS, "bf",
                                          players, "pitching")
        fld_exit = exit_correction_pairs(fld_ss, "position", FLD_COMPONENTS, "ip",
                                          players, "fielding")
        pieces += [bat_exit, pit_exit, fld_exit]

    all_deltas = pd.concat(pieces, ignore_index=True)
    all_deltas["group_label"] = all_deltas["group"].map(_group_label)
    _info(f"Total delta pairs: {len(all_deltas):,} "
          f"({all_deltas['player_id'].nunique():,} unique players)")

    # Summary: pairs per data type
    for dtype, n in all_deltas.groupby("data_type")["delta"].count().items():
        _info(f"  {dtype}: {n:,} pairs")

    # ── Cell statistics ──────────────────────────────────────────────────────
    print("\n=== COMPUTING CELL STATISTICS ===")
    cells = cell_stats(all_deltas)
    cells["group_label"] = cells["group"].map(_group_label)
    _info(f"Cell stats: {len(cells):,} rows | "
          f"{cells['group'].nunique()} groups | "
          f"{cells['component'].nunique()} components")

    # Filter unknown-position rows (group="-1") before curve fitting;
    # they remain in aging_deltas.csv for diagnostic inspection
    n_unknown_cells = (cells["group"] == "-1").sum()
    if n_unknown_cells:
        _warn(f"{n_unknown_cells:,} cell rows for group=-1 (no position assignment) — excluded from curve fitting")
    cells_fit = cells[cells["group"] != "-1"].copy()

    n_thin = int((cells_fit["flag_thin"] == 1).sum())
    _info(f"Age cells with < {MIN_PAIRS_AGE} pairs (flagged): {n_thin:,} "
          f"({100*n_thin/len(cells_fit):.1f}%)")

    # ── Curve fitting ────────────────────────────────────────────────────────
    print("\n=== FITTING POLYNOMIAL + LOESS CURVES ===")
    smooth_df, fit_df = fit_all_curves(cells_fit)
    _info(f"Smooth table: {len(smooth_df):,} rows "
          f"({smooth_df['group'].nunique()} groups, "
          f"{smooth_df['component'].nunique()} components)")
    _info(f"Fit stats: {len(fit_df):,} rows "
          f"(= {fit_df['group'].nunique()} groups × "
          f"{fit_df['component'].nunique()} components × "
          f"{fit_df['model'].nunique()} models)")

    # ── Threshold sensitivity ────────────────────────────────────────────────
    print("\n=== THRESHOLD SENSITIVITY ANALYSIS ===")
    sens_df = threshold_sensitivity(players, pos_lookup)
    _info(f"Sensitivity table: {len(sens_df):,} rows")

    # ── Write outputs ────────────────────────────────────────────────────────
    print("\n=== WRITING OUTPUTS ===")
    INT_DIR.mkdir(exist_ok=True)

    outputs = [
        ("aging_deltas.csv",         all_deltas),
        ("aging_cell_stats.csv",     cells),
        ("aging_curves_smooth.csv",  smooth_df),
        ("aging_fit_stats.csv",      fit_df),
        ("aging_threshold_sens.csv", sens_df),
    ]
    for fname, df in outputs:
        df.to_csv(INT_DIR / fname, index=False)
        print(f"  {fname}: {len(df):,} rows, {len(df.columns)} columns")

    # ── Console preview ──────────────────────────────────────────────────────
    if not fit_df.empty:
        _print_fit_table(fit_df, "poly3", 20, False,
                         "Top 20 fits (poly3, best R²)")
        _print_fit_table(fit_df, "poly3", 10, True,
                         "Bottom 10 fits (poly3, worst R² = thinnest data)")

    if not smooth_df.empty:
        _print_cumulative_at_reference_ages(smooth_df)

    # Model comparison: poly3 vs loess50 cv_rmse
    if not fit_df.empty:
        p3 = fit_df[fit_df["model"] == "poly3"][["group_label", "component", "cv_rmse"]]\
            .rename(columns={"cv_rmse": "poly3_cv"})
        p4 = fit_df[fit_df["model"] == "poly4"][["group_label", "component", "cv_rmse"]]\
            .rename(columns={"cv_rmse": "poly4_cv"})
        cmp = p3.merge(p4, on=["group_label", "component"], how="outer")
        cmp["poly4_vs_poly3_cv"] = cmp["poly4_cv"] - cmp["poly3_cv"]
        print("\nModel comparison — poly4 vs poly3 CV RMSE (negative = poly4 better):")
        print(cmp.sort_values("poly4_vs_poly3_cv").head(20).to_string(index=False))

    # ── Audit summary ────────────────────────────────────────────────────────
    print("\n=== AUDIT SUMMARY ===")
    if _issues:
        print(f"  {len(_issues)} issue(s):")
        for issue in _issues:
            print(f"    [WARN] {issue}")
    else:
        print("  No issues. All checks passed.")

    print("\nStep 2 complete.")
    print("Next: review aging_fit_stats.csv and aging_threshold_sens.csv to")
    print("select hyperparameters, then proceed to step 3 ($/component curve).")


if __name__ == "__main__":
    main()
