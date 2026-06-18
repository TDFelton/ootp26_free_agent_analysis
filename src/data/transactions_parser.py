"""
Step 8 rework, stage 1: parse OOTP's in-game transaction-log HTML files into
an unbiased free-agent signing log (intermediate/fa_signings_log.csv).

Why this exists: contracts.csv (pulled via the API) is a live snapshot only —
the API ignores all date filters, so a contract row is visible only if it
hasn't expired as of the pull date. That survivorship-biases early signing
cohorts toward long deals. The in-game transaction log has no such bias: it
records every signing event as it happened. See step8_transaction_log_rework.md
for full background.

Source files:
    <OOTP saves>/news/html/leagues/league_203_all_transactions_{M}_{YYYY}.html
    M = month (1-12, no leading zero), YYYY = year.

Scope: all signings found in the available transaction-log HTML files are
kept in this output (no rows dropped here). A `human_era` flag marks rows
with signing_year >= 2031 (the owner's earlier guidance was that pre-2031
signings were AI-controlled, not human-managed market behavior) so that
downstream steps (3, 5) can decide whether to include or exclude them —
filtering happens there, not in this parser.

Season-year mapping: a contract signed in month <= 6 of calendar year Y is for
the Y season (spring signing, e.g. a player signed in Feb 2032 plays the 2032
season). A contract signed in month > 6 of calendar year Y is for the Y+1
season (offseason signing after the Y season ends, e.g. a player signed in
Dec 2030 plays the 2031 season). This matches contracts.csv's season_year
semantics (verified against the empirical signing-event counts in
frostfire_data_summary.md: the heavy Jan/Feb signing cluster in calendar year
2030 belongs to season_year=2030, and the post-2030-season Dec 2030 cluster
belongs to season_year=2031).

Applying the owner's cutoff ("offseason after the 2030 season onward") in
season_year terms means: season_year >= 2031.
"""

import csv
import glob
import os
import re

TRANSACTIONS_DIR = (
    r"C:\Users\Felto\Documents\Out of the Park Developments\OOTP Baseball 26"
    r"\saved_games\Frostfire Baseball League.lg\news\html\leagues"
)
OUTPUT_PATH = os.path.join("intermediate", "fa_signings_log.csv")

HUMAN_ERA_MIN_SEASON_YEAR = 2031  # flag only, not a drop filter -- see module docstring

FILENAME_RE = re.compile(r"league_203_all_transactions_(\d{1,2})_(\d{4})\.html$")

# FA signing (non-extension): "Signed free agent {POS} <player link> to a
# {N}-year contract worth a total of ${TOTAL}."
FA_SIGNING_RE = re.compile(
    r'<a href="\.\./teams/team_(?P<team_id>\d+)\.html">(?P<team_name>[^<]*)</a>:'
    r' Signed free agent (?P<position>[A-Za-z0-9]+)'
    r' <a href="\.\./players/player_(?P<player_id>\d+)\.html">(?P<player_name>[^<]*)</a>'
    r' to a (?P<years>\d+)-year contract worth a total of \$(?P<total>[\d,]+)\.'
)

# Extension (non-FA): "Signed {POS} <player link> to a {N}-year contract
# extension worth a total of ${TOTAL}." Same team-prefix structure but no
# "free agent" token.
EXTENSION_RE = re.compile(
    r'<a href="\.\./teams/team_(?P<team_id>\d+)\.html">(?P<team_name>[^<]*)</a>:'
    r' Signed (?P<position>[A-Za-z0-9]+)'
    r' <a href="\.\./players/player_(?P<player_id>\d+)\.html">(?P<player_name>[^<]*)</a>'
    r' to a (?P<years>\d+)-year contract extension worth a total of \$(?P<total>[\d,]+)\.'
)


def season_year_for(file_year, file_month):
    """Map a transaction-log file's calendar year/month to the season_year it belongs to.

    Month <= 6 maps to the same calendar year (spring signing); month > 6 maps to
    the following year (offseason signing for next season).
    """
    return file_year if file_month <= 6 else file_year + 1


def parse_file(path, file_year, file_month):
    """Parse one transaction-log HTML file and return a list of signing/extension row dicts."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    rows = []
    season_year = season_year_for(file_year, file_month)

    for m in FA_SIGNING_RE.finditer(text):
        rows.append(_row_from_match(m, file_year, file_month, season_year, is_extension=False))

    for m in EXTENSION_RE.finditer(text):
        rows.append(_row_from_match(m, file_year, file_month, season_year, is_extension=True))

    return rows


def _row_from_match(m, file_year, file_month, season_year, is_extension):
    """Build one output row dict from a regex match (FA signing or extension)."""
    years = int(m.group("years"))
    total_value = int(m.group("total").replace(",", ""))
    return {
        "player_id": int(m.group("player_id")),
        "player_name": m.group("player_name"),
        "team_id": int(m.group("team_id")),
        "team_name": m.group("team_name"),
        "position": m.group("position"),
        "file_year": file_year,
        "file_month": file_month,
        "signing_year": season_year,
        "years": years,
        "total_value": total_value,
        "aav": round(total_value / years, 2),
        "is_extension": is_extension,
        "human_era": season_year >= HUMAN_ERA_MIN_SEASON_YEAR,
    }


def main():
    """Parse all transaction-log HTML files in TRANSACTIONS_DIR, print summary stats, and write fa_signings_log.csv."""
    paths = glob.glob(os.path.join(TRANSACTIONS_DIR, "league_203_all_transactions_*.html"))

    all_rows = []
    files_parsed = 0
    for path in sorted(paths):
        fname = os.path.basename(path)
        match = FILENAME_RE.match(fname)
        if not match:
            continue
        file_month, file_year = int(match.group(1)), int(match.group(2))
        rows = parse_file(path, file_year, file_month)
        all_rows.extend(rows)
        files_parsed += 1

    print(f"Parsed {files_parsed} transaction-log files, {len(all_rows)} signing rows total "
          f"(all rows kept -- no season_year filtering in this parser).")

    n_fa = sum(1 for r in all_rows if not r["is_extension"])
    n_ext = sum(1 for r in all_rows if r["is_extension"])
    n_human = sum(1 for r in all_rows if r["human_era"])
    print(f"{n_fa} FA signings, {n_ext} extensions. {n_human} rows flagged human_era=True "
          f"(signing_year >= {HUMAN_ERA_MIN_SEASON_YEAR}); {len(all_rows) - n_human} flagged False.")

    by_year = {}
    for r in all_rows:
        if not r["is_extension"]:
            by_year[r["signing_year"]] = by_year.get(r["signing_year"], 0) + 1
    print("FA signings by season_year:", dict(sorted(by_year.items())))

    os.makedirs("intermediate", exist_ok=True)
    fieldnames = [
        "player_id", "player_name", "team_id", "team_name", "position",
        "file_year", "file_month", "signing_year", "years", "total_value",
        "aav", "is_extension", "human_era",
    ]
    all_rows.sort(key=lambda r: (r["signing_year"], r["file_month"], r["player_id"]))
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {OUTPUT_PATH} ({len(all_rows)} rows).")


if __name__ == "__main__":
    main()
