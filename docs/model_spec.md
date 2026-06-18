# Frostfire free-agent contract valuation — project spec

Canonical reference for the model build. Self-contained so a fresh Claude conversation can pick this up without prior context.

## Project context

**League:** Frostfire, an OOTP Baseball online league hosted on StatsPlus at `https://atl-02.statsplus.net/frostfire/`.

**Owner's team:** team_id `30`, plays at park_id `32` (Nationals Stadium).

**Goal:** Build a model that recommends a single free-agent contract (years + AAV) for any free agent the owner is considering. The model should produce a defensible recommendation grounded in 21 years of league performance data and 5 years of contract data, with a calibrated probability that the deal underperforms its market price.

**Scope:** Free agents only (not extensions, not arbitration deals, not minor-league signings). Two-way players are out of scope for v1. Players with thin major-league samples (rookies, returning prospects below a fit threshold) are refused projection rather than projected with wide bands.

**Status:** Spec locked. No code written yet beyond data-pull scripts. Next step is the foundation-table builder.

## Data access

### API endpoints (empirically confirmed)

Base URL: `https://atl-02.statsplus.net/frostfire/api/`

**Snapshot endpoints (no params, point-in-time):**
- `teams`, `players`, `date`, `exports`, `draftv2`, `contract`, `contractextension`, `ballparks`

**Historical stat endpoints (accept `?year=YYYY`):**
- `playerbatstatsv2` — also accepts `?split=2` (vs LHP) and `?split=3` (vs RHP); `split=1` is default overall
- `playerpitchstatsv2` — also accepts `?split=2` (vs LHB) and `?split=3` (vs RHB)
- `playerfieldstatsv2` — year only, no splits
- `teambatstats`, `teampitchstats` — year only

**Confirmed unavailable (HTTP 401 or no endpoint):**
- `gamehistory` — requires auth
- Splits 4–10 on bat/pitch — require auth (likely situational splits like home/away, RISP)
- Awards, transactions, injury logs, minor-league stats, prospects, scouting reports — no endpoint exists in the public API

**Optional with token:** `ratings` (async, two-step pattern). Not currently used; owner does not have a token.

### Data files

The puller (`src/data/puller.py`) writes to `frostfire_data/`. Year range 2015–2035.

**Snapshot files:** `teams.csv`, `players.csv`, `date.json`, `exports.json`, `draft.json`, `contracts.csv`, `contract_extensions.csv`, `ballparks.json`

**Historical files per year YYYY:**
- `player_batting_YYYY.csv` (overall)
- `player_batting_YYYY_vsLHP.csv`, `player_batting_YYYY_vsRHP.csv`
- `player_pitching_YYYY.csv` (overall)
- `player_pitching_YYYY_vsLHB.csv`, `player_pitching_YYYY_vsRHB.csv`
- `player_fielding_YYYY.csv` (one row per player-position-season)
- `team_batting_YYYY.csv`, `team_pitching_YYYY.csv`

Roughly 200 files, ~50 MB total for a fresh pull.

### Key schema notes

- StatsPlus uses schema versioning; always reference columns by header name, never by positional index.
- `player_batting_YYYY.csv` includes `war`, `ubr`, plus standard counting stats; `split_id` column will be 1 (overall), 2, or 3.
- `player_fielding_YYYY.csv` is per-player-per-position-per-season; aggregate by `player_id` weighted by innings at each position.
- `contracts.csv` has `salary0` through `salary14` (year-by-year salary structure for up to 15 years), `years`, `current_year` (which year of the contract this row represents), plus `allstar_bonus` and other bonus fields. Filter to `current_year=1, is_major=1` for signing events.
- `contract_extensions.csv` has identical schema — keep separate, do not pool.
- `players.csv` has bio, draft info, `mlb_service_years` (use for FA eligibility filter).

## Data inventory: what we have and what we don't

### Available (21 years, 2015–2035)
- Per-player batting overall + vs LHP + vs RHP, with WAR
- Per-player pitching overall + vs LHB + vs RHB, with WAR and RA9-WAR
- Per-player fielding per position, with zone rating, framing, arm runs
- Team-level batting and pitching aggregates
- Park factors

### Available (current snapshot only)
- All active contracts with full year-by-year salary structure
- All extensions
- Bio data for all players ever (alive, retired, draftees)
- Current draft year
- Park factors

### Confirmed unavailable
- Award histories (no MVP/Cy Young/All-Star feature data beyond `allstar_bonus` in contracts)
- Transaction logs (no precise FA signing dates or trade history)
- Injury logs (only point-in-time `dl_days_this_year` in `players.csv`)
- Minor-league stats (no prospect priors)
- Past contracts no longer in force (contracts.csv is a current snapshot)

### Contract data depth
Meaningful contract data starts at the 2031 offseason. With 5-year max contracts, signings 2031-and-later that remain in force are captured. Earlier signings that have already expired are not retrievable. Effective training set for the contract regression: ~50–100 FA signings per cohort × 5 cohorts = 250–500 rows.

### League economics
**No inflation.** Confirmed by owner. Means one pooled $/component fit across all 21 years, no year fixed effects, no per-season recalibration.

## Architecture

Two parallel tracks meeting at a length optimizer.

**Track A — Value (what the player is worth to your team):** Component-level projection → Monte Carlo → dollar conversion → park-adjusted to Nationals Stadium → produces a distribution of contract value.

**Track B — Price (what the market will pay):** Regression on historical FA signings with the projection as a feature → predicts AAV and years jointly.

**Convergence:** Length optimizer evaluates candidate lengths 1–5 using `mean(surplus) − 1.15 · stdev(surplus)` where surplus = value − predicted market price. Picks the length that maximizes the objective. Outputs single recommended contract.

## Components

### Hitters (4 buckets + framing)

1. **Power** — extra-base runs from doubles, triples, HR above average (ISO scaled to runs)
2. **Contact/eye** — runs from singles, walks, K avoidance (on-base-skills bucket)
3. **Baserunning** — `ubr` plus stolen base runs
4. **Defense** — innings-weighted runs across all positions played (zone rating + arm + non-framing catcher defense)

**Framing** is a fifth component, modeled separately:
- Tracked as a column for every player but zero for non-catchers
- Aging curve fit only on catcher-seasons
- Has its own $/unit price in the dollar conversion

**Defense aggregation rule:** For multi-position players, weight by innings at each position. A SS/2B who played 60% SS, 40% 2B gets a position-weighted defense value.

### Pitchers (parallel SP/RP tracks)

Pitchers split into **starter** and **reliever** tracks based on usage:
- Starter components: K rate, BB+HBP rate, HR-allowed rate (per batter faced) → FIP-based runs prevented; innings as playing-time component
- Reliever components: same skill components but fit on RP-only seasons (different aging, different variance)
- Mixed-role pitchers (swingmen) get prorated across both tracks by usage in the season being evaluated

### Skipped for v1
Two-way players. Edge case, not worth the modeling complexity now.

## Modeling layers

### Foundation table

One row per player-season-component. Built by joining:
- All 21 years of batting/pitching/fielding files
- Bio fields from `players.csv` (age, position eligibility, handedness, service time)
- Park factors from `ballparks.json`
- Contract status flags from `contracts.csv` / `contract_extensions.csv`

**Park adjustment happens at this layer.** Every offensive component is divided by the player's home park factor for that year, producing park-neutral values that flow downstream into aging curves, projections, and pricing. Defense and pitching components are adjusted analogously.

### Aging curves

- **Method:** Delta method on consecutive-season pairs
- **Scope:** Per position, per component (each component has its own curve at each position)
- **Age range:** 20–40
- **Survivorship bias:** Acknowledged but not corrected in v1; the Monte Carlo's variance terms partially compensate
- **Framing curve:** Fit only on catcher-seasons
- **Pitcher curves:** Separate for SP and RP since they age differently

Pooling across positions is a fallback only if a position's sample is too thin for a component (e.g., catcher framing in very early years).

### Marcel projection

Weighted average of recent seasons → regressed toward position-and-age league baseline → aged into the projection year. Run per player, per component.

- **Weights:** Fit per component via cross-validation rather than using the canonical 5/4/3. Power and defense are noisier and tend to need different weighting than contact and walks.
- **Lookback length:** Also fit per component (test 2, 3, 4, 5 years)
- **Regression constant K:** Fit per component
- **Prior to regress toward:** League-average at the player's position and age (player-specific prior, not a global mean)

**Returning players (gap years):** Treated as new players for the projection, but their pre-gap data gets a weight that decays exponentially with gap length. The decay rate is fit empirically against players whose post-gap performance can be observed.

**Rookies and thin samples:** Below a fit-during-validation PA/IP threshold, the model returns "insufficient data" rather than projecting. This is a deliberate choice to avoid noisy rookie valuations.

### Monte Carlo

- **Simulations per player:** 30–50k careers across candidate contract length
- **Variance split:** Persistent (one draw per simulated career, applied to all years) vs transient (one draw per year)
  - Persistent captures "we might be wrong about true talent"
  - Transient captures within-season luck
- **Correlation across components within a player:** Fitted correlation matrix from 21 years of residuals after Marcel projection. Empirically, power and defense are slightly anti-correlated, contact and walks correlate positively, etc.
- **Playing-time model:** Games-played-by-age distributions fit from foundation data; injury-rate model layered on top for catastrophic-injury tail risk (lower priority — fit only if time permits)
- **Common random numbers:** Use the same random stream when comparing players head-to-head so noise doesn't muddy the comparison

### Dollar conversion + replacement

- **Curve shape:** Convex (quadratic in component-runs). Linear slope plus positive curvature so each additional run is worth more than the last — captures star premium.
- **Calibration:** Fit once across all 21 years pooled. No inflation in the league means no year fixed effects needed.
- **Positional scarcity:** Baked in at the foundation layer via standard positional adjustments to defense values (catcher and SS get positive adjustments, 1B and DH get negative). The $/component curve itself doesn't need position interaction terms.
- **Replacement baseline:** League minimum salary, identified as the modal low value in `contracts.csv`. Every roster spot costs at least this much.
- **Per-career total value:** Sum across contract years of (replacement floor + priced components above replacement, with positional adjustments applied).

### Contract regression (#3) — the market track

- **Training set:** First-year-of-contract rows from `contracts.csv` filtered to FA-eligible (6+ years service time) and excluding players in `contract_extensions.csv`. Effective n ≈ 250–500 across 5 cohorts.
- **Targets:** AAV (modeled as `log(salary0)`) and years (ordinal 1–5), modeled **jointly** so their correlation is respected
- **Features:**
  - The projection output itself (player's projected production for the upcoming season, computed using only data from before the signing)
  - Age at signing
  - Primary position
  - Durability summary (recent games played, dl days)
  - Bonus structure present in the contract (`allstar_bonus` etc.)
- **Regularization:** Ridge — non-negotiable given the thin sample
- **Extensions:** Strictly excluded. Pooling biases FA predictions downward because of the pre-FA discount.

### Park adjustment treatment

- **Foundation-layer adjustment** (park-neutral): Normalizes everyone for aging curves, projections, league $/component fit
- **Value-side re-adjustment** (Nationals Stadium specific): When computing value for the owner's team, re-apply Nationals Stadium's park factors so the projection reflects how that player would actually produce there. A pull-power hitter gets a boost or penalty based on the park's HR factor; a fly-ball pitcher gets dinged or helped similarly.
- **Market-side stays neutral:** #3 predicts what the market would pay generically. The market doesn't price players for one specific team's park.

### Length optimization

For each candidate length 1–5:
1. Compute full surplus distribution: per simulated career, sum value across years − (predicted market AAV × length)
2. Compute `mean(surplus) − 1.15 · stdev(surplus)` — the risk-adjusted objective
3. Pick the length that maximizes the objective

If the best length has negative `mean − 1.15·stdev`, the model recommends **not signing** — the player isn't worth what the market will charge.

The 1.15 risk coefficient leans slightly risk-averse — a small premium for downside protection without crippling sensitivity to upside.

## Output

Single recommended contract per evaluated player:
- **Years** (1–5, or "do not sign")
- **AAV** (the predicted market AAV at the recommended length; this is the minimum bid that wins)
- **Expected value** (mean of value distribution)
- **Expected surplus** (mean of value − cost distribution)
- **Surplus stdev**
- **P(value below predicted market price)**

Per-year breakdowns are not output in v1. Totals only.

## Validation

**Nested k-fold by season.**

- Outer loop: hold out one season's FA class as test set. Five folds (2031–2035).
- Inner loop: for each held-out player, project using only data from before that player's signing season. So 2031 signings get projected with 2015–2030 data, 2032 signings with 2015–2031 data, etc.
- For each fold, compute the model's recommended AAV against the actual signed AAV.

**Success criteria:** 85% of signings within ±15% of actual AAV.

**Note:** This bar is genuinely tight. MLB-equivalent contract models typically hit closer to 70% within ±20%. If we miss it on the first validation pass, the response is to examine residuals (which signings does the model miss on?) rather than relax the criteria silently.

**Aggregate validation metrics to report:**
- Median absolute error
- % within ±15%
- R² between predicted and actual AAV
- Residual analysis by position, age, and signing year

## Build order

1. **Foundation table** — joins, park-neutral adjustment, component aggregation. This is the substrate everything else uses.
2. **Aging curves** — pure regressions on the foundation, fit per position per component
3. **League $/component curve** — pure regression on the foundation + active contracts, in parallel with step 2
4. **Marcel projection layer** — applies aging curves to weighted-average inputs, regresses to position-age priors
5. **Contract regression #3** — uses projection output as a feature; this is where the projection layer must be done first
6. **Monte Carlo** — wraps the projection with the correlation matrix and variance splits
7. **Dollar conversion + length optimization** — closes the loop into a recommendation
8. **Nested k-fold validation** — measures performance against success criteria
9. **Residual analysis** — examines where the model misses and why

Steps 1–4 deliver a working but un-validated point-estimate model. Everything from step 5 onward adds rigor and the distributional output.

## Known limitations to accept

- **No award histories** → can't detect All-Star/MVP/Cy Young feature effects beyond what's literally in `allstar_bonus`
- **No transaction logs** → FA signing identification has to be reconstructed from contract structure plus service time; some misclassification expected
- **No injury logs** → durability modeling relies on games-played history and `dl_days_this_year` point-in-time fields; catastrophic-injury tail risk is approximated rather than measured
- **No minor-league stats** → no prospect priors substituting for the unavailable ratings; rookies are refused projection
- **5-season contract dataset** → market-track predictions have wide error bars; this is the model's main fragility
- **No ratings token** → talent estimation is stats-only; would meaningfully improve if a token becomes available
- **Survivorship bias in aging curves** → standard delta-method limitation; Monte Carlo variance partly compensates

## Quick reference

| Parameter | Value | Source |
|---|---|---|
| Owner's team_id | 30 | (Nationals) |
| Home park_id | 32 | (Nationals Stadium) |
| Year range | 2015–2035 | 21 seasons |
| Splits per season | 1 (overall), 2, 3 | per batting + pitching |
| Risk-aversion λ | 1.15 | length optimization |
| Monte Carlo sims | 30k–50k | per player |
| Aging age range | 20–40 | delta method |
| Marcel weights/lookback/K | per component (CV-fit) | not 5/4/3 default |
| Contract regression target | AAV + years jointly | log(AAV), ordinal years |
| Validation | Nested 5-fold by season | 2031–2035 |
| Success target | 85% within ±15% of AAV | tight; expect to examine residuals |
| Two-way players | Skipped | v1 scope |
| Rookies / thin samples | Refused projection | not wide-banded |
| Returning-from-gap | New player + decay-weighted pre-gap | decay rate fit empirically |
| Inflation handling | None — league has no inflation | one pooled $/component fit |
| Position scarcity | Baked into foundation via positional adjustments | not interaction terms in price curve |
| Extensions | Strictly excluded from #3 | pre-FA discount bias |

## Conversation continuity

When picking this up in a new conversation:
1. The data is in `frostfire_data/` after running `src/data/puller.py`
2. The probe scripts (`probes/api_probe_round1.py`, `probes/api_probe_round2.py`) document the API empirically — re-run only if endpoints might have changed
3. Build order step 1 (foundation table) is the natural next thing to write
4. All design decisions in this spec are locked unless explicitly revisited
