"""
step1b_park_factors.py — Step 1b: Park factors + neutralization

Reads batting_raw.csv and pitching_raw.csv from intermediate/.
Loads ballparks.json to get per-park component factors.
Computes effective park factor = (home_pf + 1.0) / 2 per component.
Adds park-neutral stat columns to each table.
Writes batting_neutral.csv and pitching_neutral.csv.

Park factor selection:
  Batting — by BATTER handedness (from players.csv `bats` column):
      bats=1 (R): avg_r, hr_r   (d and t have no handedness split)
      bats=2 (L): avg_l, hr_l
      bats=3 (S): avg,   hr  (blended)
  Pitching — by split_id (encodes BATTER handedness for pitcher splits):
      split_id=1 (overall):  avg,   hr
      split_id=2 (vs LHB):   avg_l, hr_l
      split_id=3 (vs RHB):   avg_r, hr_r

Adjusted columns added (suffix _n = park-neutral):
  Batting:  singles, singles_n, d_n, t_n, hr_n, h_n
            h_n = singles_n + d_n + t_n + hr_n
  Pitching: ha_n, hra_n

Effective park factors retained for auditability (prefix eff_, suffix _pf):
  Batting:  eff_avg_pf, eff_d_pf, eff_t_pf, eff_hr_pf
  Pitching: eff_avg_pf, eff_hr_pf

Walks, K, HBP, baserunning: not adjusted (not meaningfully park-affected).
"""

from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path("frostfire_data")
INT_DIR  = Path("intermediate")

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


# ---------------------------------------------------------------------------
# Park factor loader
# ---------------------------------------------------------------------------

def load_park_factors() -> pd.DataFrame:
    """
    Parse ballparks.json into a DataFrame indexed by team_id.
    All factor columns are prefixed pf_ to avoid conflicts with batting columns
    that share names (d = doubles, t = triples, hr = home runs).
    """
    with open(DATA_DIR / "ballparks.json") as f:
        raw = json.load(f)
    parks = pd.DataFrame(raw["ballparks"])
    parks = parks[["team_id", "avg_r", "avg_l", "avg", "d", "t", "hr_r", "hr_l", "hr"]].copy()
    parks.rename(columns={
        "avg_r": "pf_avg_r", "avg_l": "pf_avg_l", "avg": "pf_avg",
        "d": "pf_d", "t": "pf_t",
        "hr_r": "pf_hr_r", "hr_l": "pf_hr_l", "hr": "pf_hr",
    }, inplace=True)
    return parks.set_index("team_id")


def eff(series: pd.Series) -> pd.Series:
    """Effective park factor = (home_pf + 1.0) / 2."""
    return (series + 1.0) / 2


# ---------------------------------------------------------------------------
# Batting neutralization
# ---------------------------------------------------------------------------

def neutralize_batting(bat: pd.DataFrame, parks: pd.DataFrame,
                       bats_map: pd.Series) -> pd.DataFrame:
    """Apply batter-handedness-aware park factors to batting rows, adding park-neutral singles/d/t/hr/h columns and effective-PF audit columns.

    Args:
        bat: stint-level batting rows.
        parks: per-team park factor table indexed by team_id.
        bats_map: player_id -> bats handedness code lookup.

    Returns:
        bat with park-neutral columns added and working/raw PF columns dropped.
    """
    print("\n=== BATTING PARK NEUTRALIZATION ===")

    before_cols = set(bat.columns)
    bat = bat.join(parks, on="team_id")

    # Join batter handedness (1=R, 2=L, 3=Switch)
    bat["_bats"] = bat["player_id"].map(bats_map)
    n_missing = bat["_bats"].isna().sum()
    if n_missing:
        _warn(f"Batting: {n_missing:,} rows with unknown batter handedness → using blended park factors")
    else:
        _ok(f"All {len(bat):,} batting rows have known batter handedness")

    # Select handedness-appropriate avg and hr factors; d and t are always blended
    bat["_sel_avg"] = bat["pf_avg"]   # default: switch or unknown
    bat["_sel_hr"]  = bat["pf_hr"]

    rhb = bat["_bats"] == 1
    lhb = bat["_bats"] == 2
    bat.loc[rhb, "_sel_avg"] = bat.loc[rhb, "pf_avg_r"]
    bat.loc[rhb, "_sel_hr"]  = bat.loc[rhb, "pf_hr_r"]
    bat.loc[lhb, "_sel_avg"] = bat.loc[lhb, "pf_avg_l"]
    bat.loc[lhb, "_sel_hr"]  = bat.loc[lhb, "pf_hr_l"]

    # Effective factors: (raw_pf + 1.0) / 2
    bat["eff_avg_pf"] = eff(bat["_sel_avg"])
    bat["eff_d_pf"]   = eff(bat["pf_d"])
    bat["eff_t_pf"]   = eff(bat["pf_t"])
    bat["eff_hr_pf"]  = eff(bat["_sel_hr"])

    # Park-neutral component columns
    bat["singles"]   = bat["h"] - bat["d"] - bat["t"] - bat["hr"]
    bat["singles_n"] = bat["singles"] / bat["eff_avg_pf"]
    bat["d_n"]       = bat["d"]       / bat["eff_d_pf"]
    bat["t_n"]       = bat["t"]       / bat["eff_t_pf"]
    bat["hr_n"]      = bat["hr"]      / bat["eff_hr_pf"]
    bat["h_n"]       = bat["singles_n"] + bat["d_n"] + bat["t_n"] + bat["hr_n"]

    # Sanity checks
    n_neg_singles = (bat["singles"] < 0).sum()
    if n_neg_singles:
        _warn(f"Batting: {n_neg_singles:,} rows where h < d+t+hr (singles < 0)")
    else:
        _ok("No negative singles values")

    # Threshold [0.88, 1.20] accommodates known league extremes:
    # Colorado t=1.35 (eff 1.175) and Cincinnati LHB hr=1.24 (eff 1.12).
    out_of_range = ((bat["eff_avg_pf"] < 0.88) | (bat["eff_avg_pf"] > 1.20) |
                    (bat["eff_d_pf"]   < 0.88) | (bat["eff_d_pf"]   > 1.20) |
                    (bat["eff_t_pf"]   < 0.88) | (bat["eff_t_pf"]   > 1.20) |
                    (bat["eff_hr_pf"]  < 0.88) | (bat["eff_hr_pf"]  > 1.20)).sum()
    if out_of_range:
        _warn(f"Batting: {out_of_range:,} rows with effective PF outside [0.88, 1.20] — investigate")
    else:
        _ok("All effective park factors in [0.88, 1.20] (expected range)")

    _info(f"eff_avg_pf range: {bat['eff_avg_pf'].min():.4f} – {bat['eff_avg_pf'].max():.4f}")
    _info(f"eff_d_pf range:   {bat['eff_d_pf'].min():.4f} – {bat['eff_d_pf'].max():.4f}")
    _info(f"eff_t_pf range:   {bat['eff_t_pf'].min():.4f} – {bat['eff_t_pf'].max():.4f}")
    _info(f"eff_hr_pf range:  {bat['eff_hr_pf'].min():.4f} – {bat['eff_hr_pf'].max():.4f}")

    # Drop all pf_* (raw park factor columns) and working columns
    drop_cols = [c for c in bat.columns
                 if c.startswith("pf_") or c.startswith("_")]
    bat = bat.drop(columns=drop_cols)

    new_cols = set(bat.columns) - before_cols
    _info(f"New columns added: {sorted(new_cols)}")
    return bat


# ---------------------------------------------------------------------------
# Pitching neutralization
# ---------------------------------------------------------------------------

def neutralize_pitching(pit: pd.DataFrame, parks: pd.DataFrame) -> pd.DataFrame:
    """Apply split_id-based (batter handedness) park factors to pitching rows, adding park-neutral ha_n/hra_n columns and effective-PF audit columns."""
    print("\n=== PITCHING PARK NEUTRALIZATION ===")

    before_cols = set(pit.columns)
    pit = pit.join(parks, on="team_id")

    # Pitching splits encode batter handedness:
    #   split_id=1 → overall (blended)
    #   split_id=2 → vs LHB  → use LHB park factors (avg_l, hr_l)
    #   split_id=3 → vs RHB  → use RHB park factors (avg_r, hr_r)
    pit["_sel_avg"] = pit["pf_avg"]
    pit["_sel_hr"]  = pit["pf_hr"]

    vs_lhb = pit["split_id"] == 2
    vs_rhb = pit["split_id"] == 3
    pit.loc[vs_lhb, "_sel_avg"] = pit.loc[vs_lhb, "pf_avg_l"]
    pit.loc[vs_lhb, "_sel_hr"]  = pit.loc[vs_lhb, "pf_hr_l"]
    pit.loc[vs_rhb, "_sel_avg"] = pit.loc[vs_rhb, "pf_avg_r"]
    pit.loc[vs_rhb, "_sel_hr"]  = pit.loc[vs_rhb, "pf_hr_r"]

    # Effective factors
    pit["eff_avg_pf"] = eff(pit["_sel_avg"])
    pit["eff_hr_pf"]  = eff(pit["_sel_hr"])

    # Park-neutral hits allowed and HR allowed
    pit["ha_n"]  = pit["ha"]  / pit["eff_avg_pf"]
    pit["hra_n"] = pit["hra"] / pit["eff_hr_pf"]

    # Sanity checks
    out_of_range = ((pit["eff_avg_pf"] < 0.88) | (pit["eff_avg_pf"] > 1.20) |
                    (pit["eff_hr_pf"]  < 0.88) | (pit["eff_hr_pf"]  > 1.20)).sum()
    if out_of_range:
        _warn(f"Pitching: {out_of_range:,} rows with effective PF outside [0.88, 1.20] — investigate")
    else:
        _ok("All effective park factors in [0.88, 1.20] (expected range)")

    _info(f"eff_avg_pf range: {pit['eff_avg_pf'].min():.4f} – {pit['eff_avg_pf'].max():.4f}")
    _info(f"eff_hr_pf range:  {pit['eff_hr_pf'].min():.4f} – {pit['eff_hr_pf'].max():.4f}")

    drop_cols = [c for c in pit.columns
                 if c.startswith("pf_") or c.startswith("_")]
    pit = pit.drop(columns=drop_cols)

    new_cols = set(pit.columns) - before_cols
    _info(f"New columns added: {sorted(new_cols)}")
    return pit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load park factors and batter handedness, neutralize batting/pitching tables, and write the _neutral CSVs."""
    print("Frostfire Foundation — Step 1b: Park factors + neutralization")
    print("=" * 60)

    parks = load_park_factors()
    _info(f"Loaded park factors for {len(parks)} teams: {sorted(parks.index.tolist())}")

    # Load batter handedness (1=R, 2=L, 3=Switch) from the full player bio file
    players = pd.read_csv(DATA_DIR / "players.csv", usecols=["ID", "bats"])
    bats_map = players.set_index("ID")["bats"]
    _info(f"players.csv: {len(bats_map):,} entries (bats distribution: "
          f"R={( bats_map==1).sum():,}, L={(bats_map==2).sum():,}, S={(bats_map==3).sum():,})")

    bat = pd.read_csv(INT_DIR / "batting_raw.csv")
    _info(f"batting_raw.csv: {len(bat):,} rows, {len(bat.columns)} columns")
    bat = neutralize_batting(bat, parks, bats_map)

    pit = pd.read_csv(INT_DIR / "pitching_raw.csv")
    _info(f"pitching_raw.csv: {len(pit):,} rows, {len(pit.columns)} columns")
    pit = neutralize_pitching(pit, parks)

    print("\n=== WRITING INTERMEDIATE TABLES ===")
    bat.to_csv(INT_DIR / "batting_neutral.csv", index=False)
    print(f"  batting_neutral.csv  ({len(bat):,} rows, {len(bat.columns)} columns)")
    pit.to_csv(INT_DIR / "pitching_neutral.csv", index=False)
    print(f"  pitching_neutral.csv ({len(pit):,} rows, {len(pit.columns)} columns)")

    print("\n=== AUDIT SUMMARY ===")
    if _issues:
        print(f"  {len(_issues)} issue(s):")
        for issue in _issues:
            print(f"    - {issue}")
    else:
        print("  No issues. All checks passed.")

    print("\nStep 1b complete. Next: step 2 (aging curves).")


if __name__ == "__main__":
    main()
