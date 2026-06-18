"""
step1a_foundation.py — Step 1a: Audit + clean load

Validates real columns/grain against the data catalog, applies the 22-team
ML allowlist filter, and writes cleaned stint-level tables to intermediate/
as CSV files (batting_raw.csv, pitching_raw.csv, fielding_raw.csv).

Key finding: player_batting_YYYY.csv and player_pitching_YYYY.csv each
contain ALL three splits (split_id 1/2/3) in one file. The _vsLHP/_vsRHP
and _vsLHB/_vsRHB files are redundant subsets — load only the main file.

Run from the repo root:
    python src/pipeline/step1a_foundation.py
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("frostfire_data")
OUT_DIR  = Path("intermediate")
YEARS    = range(2015, 2036)

# Allowlist of the 22 real ML team_ids (from ballparks.json cross-reference).
# Never use a blocklist of All-Star IDs — use this explicit allowlist.
REAL_ML_TEAMS = frozenset({
    1, 3, 4, 6, 7, 8, 9, 13, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 29, 30,
    301, 302,
})

# Expected columns from frostfire_data_summary.md.
# Reference by header name only — never by positional index.
BATTING_COLS = frozenset({
    "id", "player_id", "year", "team_id", "game_id", "league_id", "level_id",
    "split_id", "position", "ab", "h", "k", "pa", "pitches_seen", "g", "gs",
    "d", "t", "hr", "r", "rbi", "sb", "cs", "bb", "ibb", "gdp", "sh", "sf",
    "hp", "ci", "wpa", "stint", "ubr", "war",
})

PITCHING_COLS = frozenset({
    "id", "player_id", "year", "team_id", "game_id", "league_id", "level_id",
    "split_id", "ip", "ab", "tb", "ha", "k", "bf", "rs", "bb", "r", "er",
    "gb", "fb", "pi", "ipf", "g", "gs", "w", "l", "s", "sa", "da", "sh",
    "sf", "ta", "hra", "bk", "ci", "iw", "wp", "hp", "gf", "dp", "qs",
    "svo", "bs", "ra", "cg", "sho", "sb", "cs", "hld", "ir", "irs", "wpa",
    "li", "stint", "outs", "sd", "md", "war", "ra9war",
})

FIELDING_COLS = frozenset({
    "id", "player_id", "year", "team_id", "league_id", "level_id", "split_id",
    "position", "tc", "a", "po", "er", "ip", "g", "gs", "e", "dp", "tp",
    "pb", "sba", "rto", "ipf", "plays", "plays_base", "roe",
    "opps_0", "opps_made_0", "opps_1", "opps_made_1",
    "opps_2", "opps_made_2", "opps_3", "opps_made_3",
    "opps_4", "opps_made_4", "opps_5", "opps_made_5",
    "framing", "arm", "zr",
})

# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

_issues = []

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


def audit_columns(df: pd.DataFrame, expected: frozenset, label: str) -> None:
    """Compare df's columns against the expected set, warning on missing columns and noting any extras."""
    actual  = set(df.columns)
    missing = expected - actual
    extra   = actual - expected
    if missing:
        _warn(f"{label}: missing expected columns: {sorted(missing)}")
    else:
        _ok(f"{label}: all {len(expected)} expected columns present")
    if extra:
        _info(f"{label}: extra columns not in catalog (schema growth OK): {sorted(extra)}")


def audit_grain(df: pd.DataFrame, key_cols: list[str], label: str) -> None:
    """Warn if any rows duplicate the given key_cols, otherwise confirm the grain is unique."""
    dupes = df.duplicated(subset=key_cols, keep=False)
    if dupes.any():
        _warn(f"{label}: {dupes.sum():,} rows violate grain {key_cols}")
    else:
        _ok(f"{label}: grain ({', '.join(key_cols)}) is unique")


def audit_split_ids(df: pd.DataFrame, expected: set, label: str) -> None:
    """Warn if df contains split_id values outside the expected set, otherwise confirm the found values."""
    found = set(df["split_id"].unique())
    unexpected = found - expected
    if unexpected:
        _warn(f"{label}: unexpected split_id values: {unexpected}")
    else:
        _ok(f"{label}: split_id values are {sorted(found)}")


def audit_nonneg(df: pd.DataFrame, cols: list[str], label: str) -> None:
    """Warn for each column in cols that contains any negative values."""
    for col in cols:
        if col in df.columns and (df[col] < 0).any():
            _warn(f"{label}: negative values in '{col}'")

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_batting() -> pd.DataFrame:
    """Load all per-year batting files, audit columns/grain/splits, drop undocumented split_id rows, and apply the majors-team filter."""
    print("\n=== BATTING ===")
    frames, missing = [], []

    for yr in YEARS:
        path = DATA_DIR / f"player_batting_{yr}.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        frames.append(pd.read_csv(path))

    if missing:
        _warn(f"Batting: {len(missing)} year file(s) not found")

    bat = pd.concat(frames, ignore_index=True)
    _info(f"Loaded {len(bat):,} rows across {len(frames)} year files")

    audit_columns(bat, BATTING_COLS, "batting")

    # split_id=21 rows appear across all years with real ML team_ids and genuine PA
    # data. Nature unknown (not in the public API docs). Excluded — the foundation
    # only uses splits 1 (overall), 2 (vs LHP), 3 (vs RHP).
    unknown_splits = bat[~bat["split_id"].isin({1, 2, 3})]
    if len(unknown_splits):
        split_counts = unknown_splits["split_id"].value_counts().to_dict()
        _info(f"Dropping {len(unknown_splits):,} rows with undocumented split_id(s): {split_counts}")
        bat = bat[bat["split_id"].isin({1, 2, 3})].copy()

    audit_split_ids(bat, {1, 2, 3}, "batting")
    audit_grain(bat, ["player_id", "year", "stint", "split_id"], "batting")
    audit_nonneg(bat, ["pa", "ab", "h", "hr", "bb", "k"], "batting")

    before = len(bat)
    bat = bat[bat["team_id"].isin(REAL_ML_TEAMS)].copy()
    _info(f"Majors filter: {before:,} -> {len(bat):,} rows ({before - len(bat):,} dropped)")
    _info(f"Unique players: {bat['player_id'].nunique():,} | "
          f"Year range: {bat['year'].min()}-{bat['year'].max()}")
    return bat


def load_pitching() -> pd.DataFrame:
    """Load all per-year pitching files, audit columns/grain/splits, drop undocumented split_id rows, and apply the majors-team filter."""
    print("\n=== PITCHING ===")
    frames, missing = [], []

    for yr in YEARS:
        path = DATA_DIR / f"player_pitching_{yr}.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        frames.append(pd.read_csv(path))

    if missing:
        _warn(f"Pitching: {len(missing)} year file(s) not found")

    pit = pd.concat(frames, ignore_index=True)
    _info(f"Loaded {len(pit):,} rows across {len(frames)} year files")

    audit_columns(pit, PITCHING_COLS, "pitching")

    unknown_splits = pit[~pit["split_id"].isin({1, 2, 3})]
    if len(unknown_splits):
        split_counts = unknown_splits["split_id"].value_counts().to_dict()
        _info(f"Dropping {len(unknown_splits):,} rows with undocumented split_id(s): {split_counts}")
        pit = pit[pit["split_id"].isin({1, 2, 3})].copy()

    audit_split_ids(pit, {1, 2, 3}, "pitching")
    # Pitching stint column is always 0 (does not increment on mid-season trades).
    # Actual grain uses team_id to differentiate stints, same as fielding.
    audit_grain(pit, ["player_id", "year", "team_id", "split_id"], "pitching")
    audit_nonneg(pit, ["bf", "outs", "hra", "k", "bb"], "pitching")

    before = len(pit)
    pit = pit[pit["team_id"].isin(REAL_ML_TEAMS)].copy()
    _info(f"Majors filter: {before:,} -> {len(pit):,} rows ({before - len(pit):,} dropped)")
    _info(f"Unique players: {pit['player_id'].nunique():,} | "
          f"Year range: {pit['year'].min()}-{pit['year'].max()}")
    return pit


def load_fielding() -> pd.DataFrame:
    """Load all per-year fielding files, audit columns/grain/split_id, and apply the majors-team filter."""
    print("\n=== FIELDING ===")
    frames, missing = [], []

    for yr in YEARS:
        path = DATA_DIR / f"player_fielding_{yr}.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        frames.append(pd.read_csv(path))

    if missing:
        _warn(f"Fielding: {len(missing)} year file(s) not found")

    fld = pd.concat(frames, ignore_index=True)
    _info(f"Loaded {len(fld):,} rows across {len(frames)} year files")

    audit_columns(fld, FIELDING_COLS, "fielding")

    split_vals = set(fld["split_id"].unique())
    if split_vals != {0}:
        _warn(f"Fielding: expected split_id={{0}} only, found {split_vals}")
    else:
        _ok("fielding: split_id is uniformly 0 (no splits, as expected)")

    # Grain: one row per player-position-team per season
    audit_grain(fld, ["player_id", "year", "position", "team_id"], "fielding")

    before = len(fld)
    fld = fld[fld["team_id"].isin(REAL_ML_TEAMS)].copy()
    _info(f"Majors filter: {before:,} -> {len(fld):,} rows ({before - len(fld):,} dropped)")
    _info(f"Unique players: {fld['player_id'].nunique():,} | "
          f"Year range: {fld['year'].min()}-{fld['year'].max()}")
    return fld

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full step 1a load: batting/pitching/fielding loaders, then write cleaned tables to intermediate/."""
    print("Frostfire Foundation — Step 1a: Audit + clean load")
    print("=" * 60)

    OUT_DIR.mkdir(exist_ok=True)

    bat = load_batting()
    pit = load_pitching()
    fld = load_fielding()

    print("\n=== WRITING INTERMEDIATE TABLES ===")
    bat.to_csv(OUT_DIR / "batting_raw.csv", index=False)
    print(f"  batting_raw.csv   ({len(bat):,} rows)")
    pit.to_csv(OUT_DIR / "pitching_raw.csv", index=False)
    print(f"  pitching_raw.csv  ({len(pit):,} rows)")
    fld.to_csv(OUT_DIR / "fielding_raw.csv", index=False)
    print(f"  fielding_raw.csv  ({len(fld):,} rows)")

    print("\n=== AUDIT SUMMARY ===")
    if _issues:
        print(f"  {len(_issues)} issue(s) found:")
        for issue in _issues:
            print(f"    - {issue}")
    else:
        print("  No issues found. All checks passed.")

    print("\nStep 1a complete. Next: 1b (park factors + neutralization).")


if __name__ == "__main__":
    main()
