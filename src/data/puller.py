"""
Frostfire StatsPlus data puller -- production version.

Built against endpoints and parameters confirmed empirically by
probes/api_probe_round1.py and probes/api_probe_round2.py. No guesses.

What it pulls:
  Snapshots (called once, no params):
    teams, players, date, exports, draftv2, contract, contractextension,
    ballparks

  Historical stats (looped over years; splits where supported):
    playerbatstatsv2     year x (overall, vs LHP, vs RHP)
    playerpitchstatsv2   year x (overall, vs LHB, vs RHB)
    playerfieldstatsv2   year (no splits on fielding)
    teambatstats         year
    teampitchstats       year

  Skipped:
    gamehistory          requires login (HTTP 401)
    ratings              requires token (set TOKEN below if you get one)
    splits 4-10          require login (HTTP 401); likely situational splits
                         like home/away, RISP, by month

File naming:
    Overall files use the existing naming convention you already have:
      player_batting_2033.csv, player_pitching_2033.csv, etc.
    Split files get a clear suffix:
      player_batting_2033_vsLHP.csv, player_pitching_2033_vsLHB.csv, etc.

Polite: 1s between requests. Idempotent: skips files already on disk
(set SKIP_EXISTING = False to force re-pull).

Total runtime: ~3.5 minutes for a full 21-year fresh pull (~200 requests).
Incremental updates: seconds, once cache is warm.
"""
import time
from pathlib import Path

import requests

# --- Configuration ------------------------------------------------------------
BASE = "https://atl-02.statsplus.net/frostfire/api"
OUT = Path("frostfire_data")
OUT.mkdir(exist_ok=True)

YEARS = list(range(2015, 2036))   # 2015..2035 inclusive; edit to extend
SKIP_EXISTING = False           # set False to overwrite existing files
RATE_LIMIT_SECONDS = 1.0          # delay between requests, be polite

# Optional ratings token. Get it from your S+ Preferences page if available.
# Leave None if you don't have one -- ratings phase will be skipped.
TOKEN = None

# Split definitions: API split number -> file suffix
# Batting splits: 1 = overall, 2 = vs LHP, 3 = vs RHP
# Pitching splits: 1 = overall, 2 = vs LHB, 3 = vs RHB
BAT_SPLITS = {1: "overall", 2: "vsLHP", 3: "vsRHP"}
PITCH_SPLITS = {1: "overall", 2: "vsLHB", 3: "vsRHB"}


# --- HTTP --------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Frostfire-Puller/1.0"})


def fetch(endpoint, params, dest):
    """Fetch one endpoint to dest. Returns short status string."""
    if SKIP_EXISTING and dest.exists() and dest.stat().st_size > 100:
        return "skipped (cached)"
    try:
        r = SESSION.get(f"{BASE}/{endpoint}/", params=params, timeout=60)
    except Exception as e:
        return f"ERROR: {e}"
    if r.status_code == 204:
        return "no content (204)"
    if r.status_code != 200:
        return f"HTTP {r.status_code}"
    if len(r.content) < 100:
        # 11-byte placeholder = endpoint exists but no data for this query
        return f"empty placeholder ({len(r.content)}b)"
    dest.write_bytes(r.content)
    return f"ok ({len(r.content):,}b)"


def pace():
    """Sleep for RATE_LIMIT_SECONDS between requests."""
    time.sleep(RATE_LIMIT_SECONDS)


# --- Phase 1: snapshots ------------------------------------------------------
print("=" * 70)
print("Phase 1: snapshots")
print("=" * 70)

SNAPSHOTS = [
    ("teams",             "teams.csv"),
    ("players",           "players.csv"),
    ("date",              "date.json"),
    ("exports",           "exports.json"),
    ("draftv2",           "draft.json"),
    ("contract",          "contracts.csv"),
    ("contractextension", "contract_extensions.csv"),
    ("ballparks",         "ballparks.json"),
]
for endpoint, filename in SNAPSHOTS:
    result = fetch(endpoint, None, OUT / filename)
    print(f"  {endpoint:<22} -> {filename:<28} {result}")
    pace()


# --- Phase 2: historical stats ------------------------------------------------
print()
print("=" * 70)
print(f"Phase 2: historical stats ({YEARS[0]}-{YEARS[-1]})")
print("=" * 70)


def pull_stat(endpoint, file_prefix, year, split=None, suffix=None):
    """One historical stat call. split=None or 1 means overall (no split param)."""
    params = {"year": year}
    filename = f"{file_prefix}_{year}"
    if split is not None and split != 1:
        params["split"] = split
        filename += f"_{suffix}"
    filename += ".csv"
    dest = OUT / filename
    result = fetch(endpoint, params, dest)
    print(f"    {filename:<44} {result}")
    pace()


for year in YEARS:
    print(f"\n  year {year}")

    # Batting: overall + 2 splits
    for split, label in BAT_SPLITS.items():
        pull_stat("playerbatstatsv2", "player_batting", year, split, label)

    # Pitching: overall + 2 splits
    for split, label in PITCH_SPLITS.items():
        pull_stat("playerpitchstatsv2", "player_pitching", year, split, label)

    # Fielding: overall only
    pull_stat("playerfieldstatsv2", "player_fielding", year)

    # Team aggregates
    pull_stat("teambatstats", "team_batting", year)
    pull_stat("teampitchstats", "team_pitching", year)


# --- Phase 3: ratings (only if token provided) -------------------------------
if TOKEN:
    print()
    print("=" * 70)
    print("Phase 3: ratings (async, may take a few minutes)")
    print("=" * 70)
    for osa in (0, 1):
        params = {"token": TOKEN}
        if osa:
            params["osa"] = 1
        try:
            r = SESSION.get(f"{BASE}/ratings/", params=params, timeout=60)
            followup = r.text.strip().split()[-1]
            print(f"  ratings (osa={osa}): waiting on {followup}")
            for attempt in range(20):
                time.sleep(20)
                rr = SESSION.get(followup, timeout=60)
                if rr.status_code == 200 and len(rr.content) > 100:
                    name = "ratings_osa.csv" if osa else "ratings.csv"
                    (OUT / name).write_bytes(rr.content)
                    print(f"    saved {name} ({len(rr.content):,}b)")
                    break
            else:
                print(f"    timed out after 400s")
        except Exception as e:
            print(f"  ratings (osa={osa}): ERROR {e}")


# --- Summary -----------------------------------------------------------------
files = sorted(OUT.glob("*"))
total_mb = sum(f.stat().st_size for f in files) / 1e6
print()
print("=" * 70)
print(f"Done. {len(files)} files, {total_mb:.1f} MB total in {OUT}/")
print("=" * 70)
print()
print("Files NOT pulled (require auth):")
print("  - gamehistory (HTTP 401)")
print("  - splits 4-10 on batting/pitching (HTTP 401)")
print("  - ratings (no TOKEN set)")
print()
print("Categories not exposed by this API (confirmed empirically):")
print("  - award histories")
print("  - transaction / signing logs")
print("  - injury logs (beyond point-in-time fields in players.csv)")
print("  - minor-league stats")