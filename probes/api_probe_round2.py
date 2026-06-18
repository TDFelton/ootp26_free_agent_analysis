"""
Frostfire StatsPlus API probe -- round 2.

Based on round 1:
  - ?year=YYYY works on all stat endpoints (history available)
  - ?split=2 works on playerbatstatsv2 (note: split, not split_id)
  - Awards/transactions/injuries/minor-league endpoints all dead

Round 2 goals:
  1. Confirm ?split= works on pitching and fielding (round 1 used wrong param)
  2. Scan split values 2-10 to see if OOTP exposes situational splits
  3. Test year + split combinations
  4. A few more endpoint-name variations for the data we still want
     (especially minor-league stats)

Polite: 1s between requests. ~70 requests, ~70 seconds.

Run this script from the probes/ directory -- output lands at
probes/probe_results_2/.
"""
import hashlib
import time
from pathlib import Path

import requests

BASE = "https://atl-02.statsplus.net/frostfire/api"
OUT = Path("probe_results_2")
SAMPLES = OUT / "samples"
OUT.mkdir(exist_ok=True)
SAMPLES.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Frostfire-Probe/2.0 (research)"})

# Known placeholder fingerprint from round 1 (11-byte "no data" response)
DEAD_FP = "588b5586d557"


def fp(content: bytes) -> str:
    """Return a short hash of the first 4096 bytes of content, used to compare responses."""
    return hashlib.sha256(content[:4096]).hexdigest()[:12]


def classify(r, content_fp):
    """Classify an HTTP response into a short verdict string, treating the known dead-placeholder fingerprint specially."""
    if r.status_code != 200:
        return f"HTTP_{r.status_code}"
    n = len(r.content)
    if content_fp == DEAD_FP:
        return "DEAD_PLACEHOLDER"
    if n == 0:
        return "EMPTY"
    head = r.content[:300].decode("utf-8", errors="replace").lower()
    if "<html" in head or "<!doctype" in head:
        return f"HTML_{n}b"
    if "login" in head or "log in" in head:
        return f"LOGIN_{n}b"
    if r.content[:1] in (b"[", b"{"):
        return f"JSON_{n}b"
    return f"DATA_{n}b"


results = []
baselines = {}


def probe(endpoint, params=None, label_extra="", save_as=None, baseline_key=None):
    """Request one endpoint/params combo, classify and record the result, optionally save the response body.

    Compares the response fingerprint against the named baseline (if any) to flag
    whether the params actually changed the response. Returns (fingerprint, n_bytes)
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

    content_fp = fp(r.content) if r.status_code == 200 else ""
    verdict = classify(r, content_fp)
    note = ""
    if baseline_key and baseline_key in baselines:
        base_fp, base_n = baselines[baseline_key]
        if content_fp and content_fp == base_fp:
            note = f"same as {baseline_key} baseline"
        elif content_fp:
            note = f"DIFFERS ({base_n}b -> {len(r.content)}b)"
    results.append((label, verdict, content_fp, note))
    if save_as and r.status_code == 200 and len(r.content) > 0:
        (SAMPLES / save_as).write_bytes(r.content[:50_000])
    time.sleep(1)
    return (content_fp, len(r.content)) if r.status_code == 200 else None


# --- Baselines ----------------------------------------------------------------
print("Baselines")
for ep in ["playerbatstatsv2", "playerpitchstatsv2", "playerfieldstatsv2"]:
    result = probe(ep, save_as=f"base_{ep}.bin")
    if result:
        baselines[ep] = result

# --- Split scan: confirm split= on pitching/fielding, explore higher values ---
print("\nSplit value scan (2-10) on all three stat endpoints")
for ep in ["playerbatstatsv2", "playerpitchstatsv2", "playerfieldstatsv2"]:
    for s in range(2, 11):
        probe(ep, params={"split": s}, label_extra=f"?split={s}",
              save_as=f"split_{ep}_{s}.bin", baseline_key=ep)

# --- Year + split combinations ------------------------------------------------
print("\nYear + split combinations")
for ep in ["playerbatstatsv2", "playerpitchstatsv2"]:
    for y in (2015, 2032):
        for s in (2, 3):
            probe(ep, params={"year": y, "split": s},
                  label_extra=f"?year={y}&split={s}",
                  save_as=f"ys_{ep}_{y}_{s}.bin", baseline_key=ep)

# --- Additional endpoint guesses, focused on still-missing data ---------------
print("\nMore endpoint variations")
MORE_ENDPOINTS = [
    # Minor-league stats (the column exists in player files; endpoint TBD)
    "playerbatstatsv2minor", "playerbatstatsminor", "playerbatstatsfarm",
    "playerpitchstatsv2minor", "playerfieldstatsv2minor",
    "minorstats", "milbstats", "farmstats",
    # Service / FA / arb status
    "servicetime", "service", "freeagent", "fa", "freeagentsv2",
    "arbitrationeligible", "qualifyingoffer", "options",
    # Awards / honors with more variations
    "awardhistory", "honors", "playerhonors", "playeraward",
    "seasonawards", "annual",
    # Transactions / signings
    "transaction", "tradehistory", "signinghistory", "movement",
    "fasignings", "freeagentsignings",
    # Injuries
    "injuryhistory", "injurylog", "ilhistory", "dlhistory",
    # Career / season summaries
    "playercareer", "careerstats", "playerstats",
    "seasonstats", "yearstats",
    # Roster
    "roster", "rosters", "fortyman", "40man",
    # Scouting
    "scouting", "scoutingreport", "scoutingv2",
]
for ep in MORE_ENDPOINTS:
    probe(ep, save_as=f"more_{ep}.bin")

# --- Report -------------------------------------------------------------------
report_path = OUT / "probe_report_2.txt"
with report_path.open("w") as f:
    f.write(f"Round 2 probe -- {len(results)} requests\n\n")
    f.write(f"{'LABEL':<60} {'VERDICT':<22} {'FP':<14} NOTE\n")
    f.write("-" * 140 + "\n")
    for lbl, v, content_fp, note in results:
        f.write(f"{lbl:<60} {v:<22} {content_fp:<14} {note}\n")

# Summary
ok          = sum(1 for _, v, _, _ in results if v.startswith(("DATA", "JSON")))
dead        = sum(1 for _, v, _, _ in results if v == "DEAD_PLACEHOLDER")
diffs       = sum(1 for _, _, _, n in results if n.startswith("DIFFERS"))
same        = sum(1 for _, _, _, n in results if n.startswith("same as"))

print(f"\n--- Summary ({len(results)} requests) ---")
print(f"  data returned:                                {ok}")
print(f"  dead/placeholder endpoints:                   {dead}")
print(f"  params DIFFER from baseline (likely working): {diffs}")
print(f"  params same as baseline (ignored):            {same}")
print(f"\nFull report: {report_path}")
print(f"\nSend the report back and we'll write the production puller.")