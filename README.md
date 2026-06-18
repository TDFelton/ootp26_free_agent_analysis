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

## Results

- **280 held-out signings → R² = 0.568**, 18.6% of predictions land within
  ±15% of actual AAV (median absolute error ≈ 42%).
- That's short of the project's own 85%-within-±15% target — see
  [`docs/step11_accuracy_ceiling.md`](docs/step11_accuracy_ceiling.md) for why
  this is treated as a real, diagnosed ceiling (OOTP exposes no player-rating
  data, and the contract sample is only 5 cohorts deep) rather than a bug to
  keep chasing.
- Despite the gap against that target, the model still meaningfully
  outperforms the obvious baseline (predicting market-average AAV for every
  free agent) and surfaces a short, defensible "sign at this price" list each
  offseason — see `intermediate/recommendations.csv` after running the
  pipeline.

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
