# Transaction-log rework — steps 3, 5, 8 (pending, start here)

This documents a mid-build discovery that changes how steps 3 and 5 should be
trained, and how step 8 (nested k-fold validation) should be built. Read this
before resuming work; it supersedes the "Step 3/5 complete" status in
`CLAUDE.md` until the rework lands.

## Why this exists

`contracts.csv` (pulled via the `contract` API endpoint) is a **live snapshot
only** — confirmed empirically that the endpoint ignores `?year=`,
`?season=`, `?season_year=`, `?current_year=` (all return byte-identical
output; see `probe_results/probe_report.txt` lines 45–48). A contract row
only exists if it's still active as of the pull date (2035). This means
expired contracts are invisible, and early signing cohorts are severely
survivorship-biased toward long deals:

| `season_year` (signing year) | Visible only if `years ≥` | Visible count |
|---|---|---|
| 2031 | 5 | 1 |
| 2032 | 4 | 9 |
| 2033 | 3 | 20 |
| 2034 | 2 | 44 |
| 2035 | 1 | 852 |

This is why step 3/5's training set (~134–142 rows) is too thin and biased
for a real nested k-fold validation (step 8) — the 2031–2033 "cohorts" aren't
real cohorts, they're whichever long contracts happened to survive.

No other API endpoint fills this gap — `transactions`, `signings`,
`freeagency`, `history`, `finances`, `salaries`, `payroll` are all dead
placeholder endpoints (confirmed in `probe_results/probe_report.txt`).

## The fix: in-game transaction log (already partially on disk)

OOTP itself retains a full transaction history and can render it to static
HTML via the game client (League → History → Transactions, or similar). The
owner generated these and they land at:

```
C:\Users\Felto\Documents\Out of the Park Developments\OOTP Baseball 26\saved_games\Frostfire Baseball League.lg\news\html\leagues\league_203_all_transactions_{M}_{YYYY}.html
```

`M` is month (1–12, no leading zero), `YYYY` is year. Once generated in the
game client, the file is written to disk permanently — no need to keep any
window open, and no browser/API access required. Reading the file directly
from this local path is just as good as reading it now or in any future
session.

### Coverage confirmed on disk (as of this writing)

- **2030-01 through 2034-12: complete** (all 60 months present)
- **2035-01 through 2035-10: complete**
- **2035-11, 2035-12: not yet available** — the in-game calendar hasn't
  reached those months yet. Re-check/regenerate once the season advances.

### Confirmed line formats (from manual inspection, e.g.
`league_203_all_transactions_12_2032.html`)

**Major-league FA signing (the row type we want):**
```html
<a href="../teams/team_{TID}.html">{Team Name}</a>: Signed free agent {POS} <a href="../players/player_{PID}.html">{Name}</a> to a {N}-year contract worth a total of ${TOTAL}.
```
(`N`=1 case reads "to a 1-year contract worth a total of $X" — same pattern,
singular "year" not pluralized differently, verify when parsing.)

**Minor-league FA deal — exclude:**
```html
... to a minor league contract with a ${X} signing bonus.
```

**Contract extension — exclude from FA market training (per spec, pre-FA discount bias):**
```html
<a href="../teams/team_{TID}.html">{Team}</a>: Signed {POS} <a href="../players/player_{PID}.html">{Name}</a> to a {N}-year contract extension worth a total of ${TOTAL}.
```
or for minor-league extensions: `... to a minor league contract extension.`

Both `team_id` and `player_id` are recoverable from the href attributes —
no fuzzy name matching needed to join back to `players.csv` / `teams.csv`.

### Scope decision (confirmed with owner)

**Only use signings from the offseason after the 2030 season onward.** Before
that point, contracts were AI-signed (not human-managed market behavior), so
they shouldn't be pooled into the market regression training set even though
the files exist back further under the older `league_203_transactions_*`
naming (without `all_`).

**Open question to resolve before parsing:** the exact month cutoff for
"offseason after the 2030 season." We have full-year 2030 data so this is a
within-2030 boundary, not a missing-data problem — need to determine which
month the 2030 season ends / offseason begins (likely Oct–Dec 2030 based on
typical OOTP season structure) and exclude pre-cutoff 2030 signings. Check
`league_203_schedule_8.html` or in-game season-end date, or just ask the
owner directly, before finalizing the parser's date filter.

## Plan for next session

1. **Write `src/data/transactions_parser.py`.** Parse every
   `league_203_all_transactions_{M}_{YYYY}.html` file for 2030–2035 (apply
   the offseason-2030-onward cutoff once resolved). Extract major-league FA
   signings only (skip minor-league and extension rows) into
   `intermediate/fa_signings_log.csv` with columns: `player_id`,
   `player_name`, `team_id`, `team_name`, `position`, `signing_year`,
   `signing_month`, `years`, `total_value`, `aav` (`total_value/years`),
   `is_extension` (bool, kept but flagged — don't silently drop, in case
   useful for a future extension-specific track).

2. **Recompute service-time-at-signing per player.** `players.csv`
   `mlb_service_years` is the *current* (2035) value, not the value at
   signing. Back-calculate as
   `mlb_service_years - (2035 - signing_year)`, same approach already
   documented in `docs/data_summary.md`'s SQL pseudocode for step 5.
   Filter to `>= 6` for calibration (steps 3/5), consistent with the existing
   FA-threshold-split rule — now on a much larger, unbiased base population
   instead of the snapshot survivors.

3. **Rebuild step 3** (`src/pipeline/step3_dollar_curve.py`) — refit the $/RAR quadratic
   curve on the expanded `fa_signings_log.csv`-derived training set. Expect
   n far above the current 134 rows, and no survivorship skew toward long
   deals in early cohorts. Re-verify the convexity finding (`c > 0`) holds
   and re-check the pitcher/batter pooled-curve residual pattern noted in
   the existing step 3 notes.

4. **Rebuild step 5** (`src/pipeline/step5_market_regression.py`) — same expanded training
   set, same feature set (batter: `bat_raa`, `proj_ubr`, `proj_def`, `age`,
   `proj_pa`, `is_premium_def`, `proj_rar_sq`; pitcher: the 9-feature set
   from the ablation work), but re-run alpha selection via LOO-CV since n
   has changed substantially — the existing alpha=0.17/0.34 may no longer be
   optimal.

5. **Re-run step 6 (Monte Carlo) and step 7 (optimizer)** since they consume
   step 3/5 outputs (`curve_coefficients.csv`, `market_model_coefficients.csv`).
   Recommendations will likely shift now that pricing is calibrated on a
   much larger, unbiased sample.

6. **Then do step 8** (nested k-fold validation) on the rebuilt pipeline:
   - Outer loop: hold out one signing year 2031–2035 (now real, not
     survivor-biased, cohorts).
   - Inner loop: project each held-out player using only data dated before
     their signing year.
   - **Still open / needs a decision before building:** how much of the
     pipeline to refit per fold. Earlier discussion in this conversation
     proposed three options (full refit of aging curves + Marcel + curve 3 +
     regression 5 per fold vs. refit only curve 3 + regression 5 vs. no
     refit at all) — this question was asked but not yet answered before the
     transaction-log discovery interrupted it. Re-ask once steps 3/5 rework
     is done and we're ready to actually build step 8.
   - Success criteria unchanged: 85% of signings within ±15% of actual AAV.

7. **Update `CLAUDE.md` and `docs/data_summary.md`** once the rework is
   done: mark steps 3/5 as rebuilt (not their original snapshot-based
   versions), document the new `fa_signings_log.csv` intermediate file and
   the transaction-log data source, and retire `contracts.csv` as the
   training-set source (it remains useful for other things — e.g. current
   active-roster payroll bookkeeping — just not for market-regression
   training).

## Files touched / to be created

| File | Status |
|---|---|
| `src/data/transactions_parser.py` | Not yet written |
| `intermediate/fa_signings_log.csv` | Not yet written |
| `src/pipeline/step3_dollar_curve.py` | Needs rework (currently trained on snapshot-biased data) |
| `src/pipeline/step5_market_regression.py` | Needs rework (same) |
| `src/pipeline/step6_monte_carlo.py` | Needs re-run after 3/5 rework (no code changes expected) |
| `src/pipeline/step7_optimizer.py` | Needs re-run after 6 (no code changes expected) |
| Step 8 script (not yet named) | Not started — blocked on 3/5 rework + the refit-scope decision above |
