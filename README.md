# Frostfire Free-Agent Contract Valuation Model

A statistical model that recommends a single free-agent contract (years + AAV)
for any free agent in **Frostfire**, an OOTP Baseball online league played on
[StatsPlus](https://atl-02.statsplus.net/frostfire/). The recommendation is
grounded in 21 years of league performance data (2015–2035) and 5 years of
real free-agent signing data, with no externally-borrowed MLB constants —
every run value, aging curve, and price curve is derived from Frostfire's own
data.

Built for the owner of the Washington Nationals (`team_id=30`) to decide who
to sign and at what price.

## What this project actually is

This whole repo is **vibe-coded** — built end-to-end through conversation
with Claude rather than by hand-writing the pipeline myself. Two things
motivated it:

1. **I wanted to see if I could predict free-agent valuations in my online
   OOTP league well enough to gain a real competitive advantage.** Frostfire
   is a human-managed league, and other GMs are pricing free agents using
   information I don't have automated access to — most importantly, OOTP's
   internal player ratings. The StatsPlus API *can* expose ratings, but only
   through a special access token, and using one felt like it crossed the
   line from "build a smarter model" into "get information other GMs in my
   own league can't get" — which is cheating, not analytics. So I deliberately
   built this without that token, stats-only, and tried to get as close to
   real market pricing as a stats-only model honestly can. As detailed below,
   that turned out to be a real, diagnosable ceiling on accuracy — not a
   failure of the modeling, but a confirmation of exactly the limitation I
   expected going in.
2. **I wanted to try fully vibe-coding a project** — to see how far and how
   efficiently Claude could carry a multi-week, statistically rigorous build
   if I supplied the domain knowledge and judgment and let it handle
   implementation, iteration, and validation. The `research/` folder and the
   long decision trail in [`CLAUDE.md`](CLAUDE.md) are largely a record of
   that experiment: propose something, validate it honestly against held-out
   data, revert it if it doesn't survive contact with reality, write down
   why. It went well enough that I plan to do more projects this way.

## Why this exists

OOTP doesn't tell you what a free agent is actually worth, or what the market
will pay for them. This project builds both halves of that question from
scratch:

- **Track A — Value**: project a player's future performance, convert it to
  runs above replacement, price those runs in dollars, and re-adjust for the
  Nationals' home park.
- **Track B — Price**: a regression trained on real free-agent signings that
  predicts what the market (other GMs) will actually pay.
- **Recommendation**: a length optimizer compares value against price across
  contract lengths 1–5 years and risk-adjusts for projection uncertainty,
  recommending a contract only when expected value clears expected cost with
  margin to spare.

## The theory behind it

The model is built on a sabermetric idea that's standard in real-world MLB
front offices but doesn't exist anywhere in OOTP's own UI: **separate "what is
this player worth" from "what will it cost to sign him,"** and only sign when
the first number clears the second by enough margin to absorb the risk of
being wrong.

That separation is why the pipeline is two parallel tracks (A and B) instead
of one model that goes straight from stats to a dollar figure:

- **Value (Track A)** has to be built bottom-up, because nothing in the data
  hands you "how many wins is this player worth." It's assembled in layers:
  raw box-score stats → park-neutralized (so a hitter in Colorado isn't
  confused for a better hitter than one in Seattle) → decomposed into
  components a real GM would recognize (power, contact/eye, baserunning,
  defense, with catcher framing split out as its own component) → converted
  to **runs above replacement** using linear weights derived empirically from
  Frostfire's own run-scoring environment (not borrowed MLB constants, since
  a custom league's offense levels don't have to match the majors) → aged
  forward using **delta-method aging curves** fit per position per component
  (a shortstop's defense and a first baseman's power age differently, so one
  global curve would wash out real signal) → projected with a **Marcel-style**
  weighted average regressed toward a position-and-age prior, with the
  weighting scheme and regression strength each *fit by cross-validation per
  component* rather than assumed (`5/4/3` is a famous default, but there's no
  reason a league this small should inherit it unmodified) → run through a
  **40,000-iteration Monte Carlo** per candidate contract length so the output
  is a distribution, not a single number, with variance split into
  persistent ("we might be wrong about this player's true talent") and
  transient ("normal season-to-season luck") components, correlated using a
  matrix fit from 21 years of residuals → priced in dollars using a convex
  $/run curve (each additional run is worth more than the last, capturing the
  real-world "stars get paid a premium" effect) → and finally re-adjusted for
  Nationals Stadium specifically, since the question "what is this player
  worth to *my* team" is different from "what is this player worth in a
  league-average park."
- **Price (Track B)** is comparatively simple by design: a ridge-regularized
  regression of actual free-agent signings against the same projected
  production features used in Track A. It deliberately does *not* try to be
  clever — with only a few hundred real signings to learn from, a simple,
  heavily regularized model generalizes better than a complex one, which
  several rounds of experimentation (below) ended up confirming empirically
  rather than just assuming.
- **The length optimizer** is the only place the two tracks touch. For each
  candidate contract length, it computes `mean(value − price) − 1.15 ×
  stdev(value − price)` — a risk-adjusted expected surplus — and picks the
  length that maximizes it. If no length clears zero, the recommendation is
  "do not sign," which is itself an important output: the model is allowed to
  say a player isn't worth what the market will charge for him, not just rank
  everyone and pick the best AAV.

This structure — independent value and price models, reconciled by an
explicit risk-adjusted objective — is the core idea borrowed from real
front-office analytics (the public-facing version of this is sometimes called
a "surplus value" model). Everything else in the build (park factors, aging
curves, Marcel weights, the Monte Carlo correlation matrix) exists to make
the two halves of that comparison as honest as the available data allows.

## Results

- **280 held-out signings → R² = 0.568**, 18.6% of predictions land within
  ±15% of actual AAV (median absolute error ≈ 42%; 29.6% within ±25%).
- That's short of the project's own pre-committed success criterion: **85% of
  signings within ±15% of actual AAV**, set at the start of the project as a
  genuinely tight bar (real MLB-equivalent contract models typically land
  closer to 70% within ±20%).

### The goal, and why it wasn't reached

The goal was never just "build a model that runs" — it was to hit that 85%
bar with a methodology rigorous enough to trust for real signing decisions.
Missing it was treated as a finding to diagnose, not a target to quietly
lower, and three independent rounds of investigation (steps 9, 10, 11 —
gradient boosting, quantile/robust regression, track-record features, partial
pooling toward the age axis, blending in 16 extra years of pre-2031 "AI era"
signings, sweeping every untested hyperparameter in the pipeline) all
converged on the same conclusion: **the gap is structural, not a tuning
problem**, for two reasons that no amount of further modeling on this dataset
can fix:

1. **OOTP exposes no player-ratings data through its public API.** Real
   front offices (and real OOTP GMs making human decisions) price players
   partly on scouting information — projected ceiling, makeup, tools grades —
   that never shows up in a box score. This model is necessarily stats-only,
   and the largest individual misses in validation are exactly what you'd
   expect from that blind spot: e.g. a player who actually signed for $11.9M
   predicted at $2.5M, or $14.0M predicted at $4.1M. You can't regress your
   way around a feature that was never collected.
2. **Only 5 cohorts (2031–2035) of real, unbiased signing data exist** —
   roughly 280 usable rows after every filter. (`contracts.csv` itself turned
   out to be a live snapshot with severe survivorship bias toward older,
   longer deals — a mid-project discovery described below — which is why the
   real training set comes from a parsed transaction log instead.) A sample
   that size pins down a ridge regression's coefficients only loosely, and
   every attempt to spend more model complexity on it (more features, a
   different model family, partial pooling, more rows from a different era
   of the league) made held-out accuracy worse, not better, once tested
   against the real nested-by-year validation harness rather than in-sample
   fit. Full numbers and the closed investigation log:
   [`docs/step11_accuracy_ceiling.md`](docs/step11_accuracy_ceiling.md).

In other words: the architecture (two-track value/price separation, park
neutralization, per-component aging curves, CV-fit Marcel weights, Monte
Carlo risk adjustment) is doing what it's supposed to do. The ceiling comes
from what the data *can't* tell the model, not from a fixable bug in how the
model uses the data it has. Despite the gap against the 85% target, the
model still meaningfully outperforms the obvious naive baseline (predicting
market-average AAV for every free agent) and surfaces a short, defensible
"sign at this price" list each offseason — see `intermediate/recommendations.csv`
after running the pipeline.

## How it works

The pipeline runs in nine numbered steps, each reading the previous step's
output from `intermediate/` and writing its own:

| Step | Script | What it does |
|---|---|---|
| 1a | [`src/pipeline/step1a_foundation.py`](src/pipeline/step1a_foundation.py) | Load 21 years of raw batting/pitching/fielding files, validate schema against the data catalog, filter to the 22 real major-league teams |
| 1b | [`src/pipeline/step1b_park_factors.py`](src/pipeline/step1b_park_factors.py) | Park-neutralize every offensive/pitching stat using each team's home park factors |
| 2 | [`src/pipeline/step2_aging_curves.py`](src/pipeline/step2_aging_curves.py) | Fit delta-method aging curves per position/role per stat component, ages 20–40 |
| 3 | [`src/pipeline/step3_dollar_curve.py`](src/pipeline/step3_dollar_curve.py) | Derive Frostfire-specific linear run weights, compute runs above replacement (RAR), fit a convex $/RAR curve against real signings |
| 4 | [`src/pipeline/step4_marcel_projection.py`](src/pipeline/step4_marcel_projection.py) | Marcel-style weighted-average + age-regressed projections for every active player, with per-component hyperparameters chosen by cross-validation |
| 5 | [`src/pipeline/step5_market_regression.py`](src/pipeline/step5_market_regression.py) | Ridge regression of `log(AAV)` and contract years on projected production — this is the "what will the market pay" model |
| 6 | [`src/pipeline/step6_monte_carlo.py`](src/pipeline/step6_monte_carlo.py) | 40,000-career Monte Carlo simulation per player per candidate length, with variance/correlation modeling, aging, playing-time, and park adjustment |
| 7 | [`src/pipeline/step7_optimizer.py`](src/pipeline/step7_optimizer.py) | Pick the contract length that maximizes `mean(surplus) − 1.15·std(surplus)`; writes `intermediate/recommendations.csv` |
| 8 | [`src/pipeline/step8_validation.py`](src/pipeline/step8_validation.py) | Nested k-fold validation by signing year (2031–2035), refitting steps 3 and 5 per fold |
| 9 | [`src/pipeline/step9_residual_analysis.py`](src/pipeline/step9_residual_analysis.py) | Breaks down step 8's misses by age/position/signing year to separate fixable bias from the model's known blind spots |

A tenth, unnumbered piece — [`src/data/transactions_parser.py`](src/data/transactions_parser.py)
— exists because of a major mid-project discovery: `contracts.csv` (the
league's contracts API) is a **live snapshot only** and silently
survivorship-biases early free-agent cohorts (a contract is only visible if
it hasn't expired yet). The fix was to parse OOTP's own in-game
transaction-log HTML for every signing event ever recorded, with no
survivorship bias. Steps 3 and 5 train on that parsed log
(`intermediate/fa_signings_log.csv`), not on `contracts.csv` directly. Full
story: [`docs/step8_transaction_log_rework.md`](docs/step8_transaction_log_rework.md).

### Key findings along the way

- **The market underprices old players less than the model expects, or the
  model underprices old players more than the market does** — signings at
  age 37+ are underpredicted by an average of +57.8%. Diagnosed as a
  small-sample extrapolation problem (only 19 such signings exist across all
  5 cohorts), not a missing-feature problem — tested and confirmed not
  fixable with available data ([`docs/step9_followup_plan.md`](docs/step9_followup_plan.md)).
- **OOTP exposes no player-ratings data.** This is the model's single biggest
  blind spot — the real market clearly pays for scouting/ratings information
  this model cannot see, and the largest individual misses in validation are
  all consistent with that (e.g. a $11.9M actual signing predicted at $2.5M).
- **No combination of additional features, model families, or
  hyperparameters beat the ~210–280 row baseline.** Three independent rounds
  of experimentation (gradient boosting, quantile/robust regression,
  track-record features, partial pooling, blending in pre-2031 "AI era" data)
  all failed to durably improve on the ridge baseline once tested against the
  real held-out validation harness rather than in-sample fit. The accuracy
  ceiling is the ~250-row contract sample, not the modeling approach — see
  `research/` and [`docs/step11_accuracy_ceiling.md`](docs/step11_accuracy_ceiling.md).
- **The league has no inflation**, confirmed by the owner — so the model uses
  one pooled $/run fit across all 21 years rather than year fixed effects.

## Repository layout

```
ootp_analysis/
├── README.md                   This file
├── CLAUDE.md                   Authoritative build log + working agreements (for AI-assisted sessions)
├── docs/                        Spec, data catalog, and per-step writeups
│   ├── model_spec.md            Locked architecture/design spec (read this first)
│   ├── data_summary.md          Empirically-verified data catalog (schemas, codes, gotchas)
│   ├── step2_aging_curve_notes.md
│   ├── step8_transaction_log_rework.md
│   ├── step9_followup_plan.md
│   └── step11_accuracy_ceiling.md
├── src/
│   ├── data/
│   │   ├── puller.py             Pulls all raw data from the StatsPlus API → frostfire_data/
│   │   └── transactions_parser.py  Parses OOTP's transaction-log HTML → unbiased FA signing log
│   ├── pipeline/                 The 9 numbered production steps (see table above)
│   └── viz/                      Diagnostic plots (aging-curve fit quality, etc.)
├── research/                     Closed investigations: hyperparameter sweeps, ablations,
│                                  follow-up experiments. Mostly tested and reverted — kept
│                                  for documentation, not part of the production pipeline.
├── probes/                       One-time API discovery scripts + their raw output
├── archive/
│   └── legacy_xgboost_notebook/  An earlier, superseded XGBoost-based approach (not used)
├── frostfire_data/                Raw pulled data (gitignored — regenerate with src/data/puller.py)
└── intermediate/                  Pipeline outputs/artifacts (gitignored — regenerate by running the pipeline)
```

## Running it

All scripts read/write `frostfire_data/` and `intermediate/` as paths
relative to the **current working directory**, so always run them from the
repo root:

```bash
# 1. Pull raw data from the StatsPlus API (writes frostfire_data/)
python src/data/puller.py

# 2. Parse the in-game transaction log for unbiased FA signing history
#    (requires the OOTP saved-game folder; see the script's docstring)
python src/data/transactions_parser.py

# 3. Run the pipeline in order
python src/pipeline/step1a_foundation.py
python src/pipeline/step1b_park_factors.py
python src/pipeline/step2_aging_curves.py
python src/pipeline/step3_dollar_curve.py
python src/pipeline/step4_marcel_projection.py
python src/pipeline/step5_market_regression.py
python src/pipeline/step6_monte_carlo.py
python src/pipeline/step7_optimizer.py     # -> intermediate/recommendations.csv
python src/pipeline/step8_validation.py    # -> validation metrics
python src/pipeline/step9_residual_analysis.py
```

`frostfire_data/` and `intermediate/` are gitignored — they're large
(tens of MB) and fully regenerable from the scripts above, so they're not
committed.

## Caveats this model is upfront about

- **No player-ratings data.** Stats-only talent estimation; would improve
  meaningfully with an OOTP ratings token, which isn't available here.
- **Only 5 years of usable contract history** (2031–2035 signings), ~250–280
  rows after filtering. Every regression in steps 3 and 5 is ridge-regularized
  because of this, and the validation in step 8 treats the resulting error
  bars as real, not as a bug to be tuned away.
- **No award, transaction (pre-rework), injury, or minor-league data** is
  available from the public StatsPlus API. See
  [`docs/model_spec.md`](docs/model_spec.md#known-limitations-to-accept) for
  the full list of accepted limitations.
- **Two-way players are out of scope** for this version.
