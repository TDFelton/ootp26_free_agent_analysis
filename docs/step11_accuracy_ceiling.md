# Frostfire FA Contract Model — Realistic Accuracy Ceiling

Written 2026-06-17 after step 11 (hyperparameter sweep against step 8/9). Answers
the question: given everything tried so far, what accuracy can this model
*reliably* hit — not the best single run, but the range supported by repeated,
independent search.

## Headline

**~18-20% of predictions land within ±15% of actual AAV, ~28-32% land within
±25%, and the typical (median) prediction is off by ~40-43% of actual AAV.**
R² between predicted and actual AAV sits at ~0.50-0.57 — the model explains
roughly half the variance in what players actually get paid.

This falls well short of the spec's 85%-within-±15% success criterion (see
`docs/model_spec.md`), and that gap is not closeable by further tuning —
see "Why it's capped here" below.

## Why this range, not just the single best run

Across three independent rounds of search — step 9's original follow-up round
(age curvature, empirical-Bayes shrinkage, track-record features, gradient
boosting, robust/quantile regression), step 10's AI-era data-blending
investigation, and step 11's hyperparameter sweep (36 GAMMA×L_RETRO combos, 6
salary-floor levels, 6 MIN_PA/MIN_BF levels) — the metrics cluster tightly:

| Metric | Range observed across all rounds | Current production value |
|---|---|---|
| Within ±15% | 14% – 22% | 18.6% |
| Within ±25% | 24% – 36% | 29.6% |
| Median abs % error | 35% – 50% | 42.4% |
| R² | 0.33 – 0.59 | 0.568 |

No configuration tested — and a lot were tested, independently, across model
families (ridge, GBM, Huber, quantile), feature sets (age², track-record,
interaction terms), and hyperparameters (GAMMA, lookback window, salary floor,
qualification thresholds) — pushed durably past the high end of these bands
without trading away another metric or relying on a single-fold noise
artifact (see step 9's followup plan and step 11 in `CLAUDE.md` for the
full list of rejected attempts and why each was rejected). That convergence
from multiple unrelated angles is the evidence: this isn't a model that's one
good idea away from 85%, it's a model that has found its ceiling for the data
it has.

## What this means practically

- For a player projected at $5M AAV, the typical prediction error is **~$2M**
  (42% of $5M). For a $15M player, typical error is **~$6M**.
- About 1 in 5 predictions will be tight (within 15%) — good enough to anchor
  a real offer. The rest need to be treated as a wide band, not a point
  estimate.
- The model is directionally useful (it correctly separates star talent from
  replacement level — independently-derived batter RAR correlates 0.958 with
  OOTP's own WAR) but not precise enough to be the sole basis for a specific
  dollar figure on a specific contract.

## Why it's capped here, not lower or higher

- **Sample size**: ~280 usable FA signings across 5 cohorts is thin for a
  market-pricing regression, especially at the tails (very young/old players,
  very high/low RAR — see step 9's age-bucket residual analysis).
- **Invisible ratings**: the league's actual price-setters (AI GMs and the
  owner's competitors) see scouting/ratings information this model never
  will. Residual analysis shows the biggest misses are systematic — high
  actual AAV with unremarkable visible stats — consistent with ratings
  driving real price variance with zero footprint in any pullable data.
- **Spec's own admission**: the 85%-within-±15% target was set knowing it's
  tighter than real MLB-equivalent contract models (~70% within ±20%), which
  have vastly more historical signings to train on and still don't hit this
  bar.

## What would actually move the needle (not more tuning)

1. **More contract-year history** — every offseason the league plays adds
   ~50-80 more signings to train on; the sample-size constraint loosens over
   time on its own, no code changes needed.
2. **An OOTP ratings token** — would directly close the dominant blind spot,
   not just smooth around it.

Both are outside this model's control on any given day. Tuning further
within the current data and feature set is very unlikely to help — three
independent rounds of search have now confirmed that (step 9 follow-up round,
step 10 AI-era blending, step 11 hyperparameter sweep).
