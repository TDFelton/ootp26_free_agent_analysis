"""
Frostfire StatsPlus API discovery probe.

The fact-checking step before extending your data puller. Does NOT modify or
read your existing data files. Writes to ./probe_results/.
Run this script from the probes/ directory (or otherwise account for the
script's working directory) -- output lands at probes/probe_results/.

What it does:
  Phase A — Baseline each endpoint you already know works.
  Phase B — On stat / time-bound endpoints, test common parameter names
            (year, season, split_id, level_id, etc.) and compare the response
            fingerprint against the baseline. Detects which params actually
            change the response vs. which are silently ignored.
  Phase C — Probe plausible unknown endpoint names focused on data useful for
            the contract-valuation model (awards, transactions, injuries,
            minor-league stats, standings).
  Output  — probe_results/probe_report.txt with the verdict for every request,
            plus saved response samples in probe_results/samples/ for
            inspection.

Polite: 1s between requests, single shared session, User-Agent set.
Runtime: ~2-3 minutes (~120 requests).
"""
import hashlib
import time
from pathlib import Path

import requests

BASE = "https://atl-02.statsplus.net/frostfire/api"
OUT = Path("probe_results")
SAMPLES = OUT / "samples"
OUT.mkdir(exist_ok=True)
SAMPLES.mkdir(exist_ok=True)

# --- Known endpoints from your existing puller --------------------------------
KNOWN_ENDPOINTS = [
    "teams", "players", "date", "contract", "contractextension",
    "exports", "gamehistory", "draftv2",
    "teambatstats", "teampitchstats",
    "playerbatstatsv2", "playerpitchstatsv2", "playerfieldstatsv2",
]

# --- Parameter tests, per endpoint --------------------------------------------
# Only tested where it makes sense; the goal is to find which param name (if any)
# unlocks historical years, platoon splits, and minor leagues.
PARAM_TESTS_BY_ENDPOINT = {
    "playerbatstatsv2": [
        {"year": 2032}, {"year": 2025}, {"year": 2015},
        {"season": 2032}, {"yr": 2032},
        {"split_id": 2}, {"split_id": 3}, {"split": 2},
        {"level_id": 2}, {"level_id": 3}, {"level": 2},
        {"league_id": 203},
    ],
    "playerpitchstatsv2": [
        {"year": 2032}, {"year": 2015},
        {"split_id": 2}, {"split_id": 3},
        {"level_id": 2}, {"level_id": 3},
    ],
    "playerfieldstatsv2": [
        {"year": 2032}, {"year": 2015},
        {"split_id": 2},
        {"level_id": 2},
    ],
    "teambatstats": [{"year": 2032}, {"split_id": 2}],
    "teampitchstats": [{"year": 2032}, {"split_id": 2}],
    "contract": [
        {"year": 2032}, {"season": 2032},
        {"season_year": 2032}, {"current_year": 1},
    ],
    "contractextension": [{"year": 2032}, {"season_year": 2032}],
    "gamehistory": [{"year": 2032}],
    "draftv2": [{"year": 2032}, {"year": 2030}],
    "players": [{"is_active": 1}, {"team_id": 1}, {"include_retired": 1}],
}

# --- Unknown endpoint guesses, ordered by relevance to the contract model -----
# Naming follows the existing convention: lowercase concatenated, optional v2.
UNKNOWN_ENDPOINTS = [
    # Awards / honors -- features for #3 (contract regression)
    "awards", "awardsv2", "playerawards", "teamawards",
    "allstar", "allstars", "mvp", "cyyoung",
    "hof", "halloffame",
    # Transactions / signings -- precise FA signing events for #3
    "transactions", "transactionsv2", "trade", "trades",
    "signings", "signing", "freeagents", "freeagency",
    "waivers", "waiver", "rule5", "arbitration", "releases",
    # Injuries -- durability distribution for #8 Monte Carlo
    "injuries", "injury", "playerinjuries", "playerinjury",
    "il", "dl", "disabledlist",
    # Minor leagues -- prospect priors substituting for unavailable ratings
    "minorbatstats", "minorpitchstats", "minorfieldstats",
    "milbbatstats", "milbpitchstats", "milbfieldstats",
    "minorleaguebatstats", "playerminorbatstats",
    "prospects", "topprospects", "minorroster",
    # Standings / playoffs -- win-curve / contention context (future use)
    "standings", "standingsv2", "playoffs", "postseason",
    "schedule", "boxscore", "boxscores", "games",
    # Splits, in case they live at a sibling endpoint rather than as a param
    "splits", "playerbatsplits", "playerpitchsplits",
    # Versioned variants of known stat endpoints
    "playerbatstats", "playerpitchstats", "playerfieldstats",
    "playerbatstatsv3", "playerpitchstatsv3", "playerfieldstatsv3",
    "contractsv2", "contractv2",
    # Other plausible
    "league", "leagues", "season", "seasons", "history",
    "finances", "salaries", "payroll",
    "park", "parks", "ballparks",
    "coaches", "managers",
    "news", "leaders", "leaderboard",
]

# --- Helpers ------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Frostfire-Probe/1.0 (research)"})


def fingerprint(content: bytes) -> str:
    """Return a short hash of the first 4096 bytes of content, used to compare responses."""
    return hashlib.sha256(content[:4096]).hexdigest()[:12]


def classify(r):
    """Classify an HTTP response into a short verdict string (HTTP error, empty, HTML/login, JSON, or data)."""
    if r.status_code != 200:
        return f"HTTP_{r.status_code}"
    n = len(r.content)
    if n == 0:
        return "EMPTY"
    head = r.content[:300].decode("utf-8", errors="replace").lower()
    if "<html" in head or "<!doctype" in head:
        return f"HTML_{n}b"
    if "login" in head or "log in" in head or "not logged" in head:
        return f"LOGIN_{n}b"
    if r.content[:1] in (b"[", b"{"):
        return f"JSON_{n}b"
    return f"DATA_{n}b"


results = []         # (label, verdict, fingerprint, note)
baselines = {}       # endpoint -> (fingerprint, n_bytes)


def probe(endpoint, params=None, label_extra="", save_as=None):
    """Request one endpoint/params combo, classify and record the result, optionally save the response body.

    Compares the response fingerprint against any stored baseline for the endpoint to
    flag whether the params actually changed the response. Returns (fingerprint, n_bytes)
    on a successful (HTTP 200) request, otherwise None.
    """
    url = f"{BASE}/{endpoint}/"
    label = endpoint + (f" {label_extra}" if label_extra else "")
    try:
        r = SESSION.get(url, params=params, timeout=30)
    except Exception as e:
        results.append((label, "ERROR", "", str(e)[:80]))
        time.sleep(1)
        return None

    verdict = classify(r)
    fp = fingerprint(r.content) if r.status_code == 200 else ""
    note = ""
    if endpoint in baselines and label_extra:
        base_fp, base_n = baselines[endpoint]
        if fp and fp == base_fp:
            note = "same as baseline (param ignored)"
        elif fp:
            note = f"DIFFERS from baseline ({base_n}b -> {len(r.content)}b)"
    results.append((label, verdict, fp, note))
    if save_as and r.status_code == 200 and len(r.content) > 0:
        (SAMPLES / save_as).write_bytes(r.content[:50_000])
    time.sleep(1)
    return (fp, len(r.content)) if r.status_code == 200 else None


# --- Phase A: baseline known endpoints ----------------------------------------
print("Phase A: baseline known endpoints")
for ep in KNOWN_ENDPOINTS:
    result = probe(ep, save_as=f"A_{ep}_baseline.bin")
    if result:
        baselines[ep] = result

# --- Phase B: parameter probes ------------------------------------------------
print("\nPhase B: parameter probes")
for ep, params_list in PARAM_TESTS_BY_ENDPOINT.items():
    for params in params_list:
        plabel = "&".join(f"{k}={v}" for k, v in params.items())
        safe = plabel.replace("=", "-").replace("&", "_")
        probe(ep, params=params, label_extra=f"?{plabel}",
              save_as=f"B_{ep}__{safe}.bin")

# --- Phase C: unknown endpoint discovery --------------------------------------
print("\nPhase C: unknown endpoint discovery")
for ep in UNKNOWN_ENDPOINTS:
    probe(ep, save_as=f"C_{ep}.bin")

# --- Report -------------------------------------------------------------------
report_path = OUT / "probe_report.txt"
with report_path.open("w") as f:
    f.write(f"Frostfire API probe -- {len(results)} requests\n")
    f.write(f"Base: {BASE}\n\n")
    f.write(f"{'LABEL':<58} {'VERDICT':<14} {'FP':<14} NOTE\n")
    f.write("-" * 130 + "\n")
    for lbl, v, fp, note in results:
        f.write(f"{lbl:<58} {v:<14} {fp:<14} {note}\n")

# Console summary
def has_prefix(v, *prefixes):
    """Return True if string v starts with any of the given prefixes."""
    return any(v.startswith(p) for p in prefixes)

ok       = sum(1 for _, v, _, _ in results if has_prefix(v, "DATA", "JSON"))
empty    = sum(1 for _, v, _, _ in results if v in ("EMPTY", "HTTP_204"))
notfound = sum(1 for _, v, _, _ in results if v == "HTTP_404")
html     = sum(1 for _, v, _, _ in results if has_prefix(v, "HTML", "LOGIN"))
errs     = sum(1 for _, v, _, _ in results if v == "ERROR")
diffs    = sum(1 for _, _, _, n in results if n.startswith("DIFFERS"))

print(f"\n--- Summary ({len(results)} requests) ---")
print(f"  data returned: {ok}")
print(f"  empty / 204:   {empty}")
print(f"  404 not found: {notfound}")
print(f"  html / login:  {html}")
print(f"  errors:        {errs}")
print(f"  params that DIFFER from baseline (likely accepted): {diffs}")
print(f"\nFull report:    {report_path}")
print(f"Sample bodies:  {SAMPLES}/")
print("\nNext step: send the contents of probe_report.txt and we'll decide")
print("what to add to your puller based on what actually responded.")