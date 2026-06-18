# Step 9 follow-up plan — residual-bias mitigation

Picks up after step 9 (residual analysis) and step 9b (age-bucket holdout
diagnostic). Self-contained — a fresh session can start here without
re-deriving the background. Read `CLAUDE.md`'s step 9 entry first for full
detail on how we got here; this file is the forward-looking plan only.

## Where we left off

**Step 8 baseline (current, restored, verified):** nested-by-signing-year
validation gives 16.7% within ±15% of actual AAV, R²=0.503, median abs %
error 40.1%. Steps 5/6/7 outputs all match this baseline exactly (batter
log(AAV) R²=0.738, pitcher R²=0.517, 22/212 "sign" recommendations).

**Step 9 finding:** a real, monotonic age-related under-prediction bias.
Mean signed % error by age bucket (positive = model under-predicted):
28–30 = +14.5%, 31–33 = +12.0%, 34–36 = +30.8%, 37+ = +57.8%. Holds within
both batters and pitchers and within every proj_rar quartile.

**Tested and reverted:** adding `age_sq` to the batter feature set
(`research/step9_bat_feature_ablation.py`). Won on full-sample LOO-CV but produced zero
improvement in the actual nested-by-year validation (16.7%→16.2%,
R² 0.503→0.498). Reverted; `BAT_FEATURES` is back to the original 7 features.

**Step 9b finding (`research/step9_age_holdout.py`, leave-one-age-bucket-out,
pooling all 5 signing years for training instead of holding out a whole
year):**

| age bucket | n held out | train n | within±15% | mean signed err | R² |
|---|---|---|---|---|---|
| 28–30 | 41 | 175 | 26.8% | +42.1% | 0.338 |
| 31–33 | 106 | 110 | 12.3% | +1.8% | 0.483 |
| 34–36 | 49 | 167 | 20.4% | +33.6% | 0.526 |
| 37+ | 20 | 196 | 15.0% | +29.7% | 0.040 |

Key takeaway: with full temporal training data, the 37+ bias roughly halves
vs. the year-fold number (57.8% → 29.7%), and 28–30 is *also* bad when held
out (+42.1%, worse than 34–36). **This means the age effect is a general
data-density problem across the whole age axis — the model needs examples
spread across ages to interpolate at all — not an "old players are uniquely
unpredictable" wall.** That's encouraging for methods that let the model
borrow statistical strength across age (partial pooling, track-record
features) rather than just adding more polynomial terms in age itself.
Output: `intermediate/step9b_age_holdout_results.csv`.

## Decisions made on the candidate fix list

Out of the 10 ideas proposed after step 9, here's what's in/out and why:

- **Dropped:** #4 (prior-AAV anchoring) — most of the affected early-cohort
  signings were AI-managed, not human market behavior, so anchoring on those
  isn't meaningful. #7 (censored/Tobit regression for the salary floor) — no
  signings actually fall below the floor in the current data, so it's moot.
  #9 (pool batters+pitchers into one model) — rejected, scrap. #10 (just
  document the ceiling and move on) — skipped for now, revisit later if
  nothing else moves the needle.
- **In, in this order:** #5 → #1 → #8 → #6 → #3.

## The plan, in order

### 1. #5 — Hierarchical / partial pooling on age (and position) effects

Replace ridge's single global linear age coefficient with a model that lets
age-bucket (and position) effects shrink toward the global trend by sample
size — thin buckets (37+, n=20) get pulled toward the population trend,
dense buckets (31–33, n=106) keep more of their own signal. Likely
implementation: a mixed-effects / empirical-Bayes shrinkage layer on top of
(or replacing) the current ridge — e.g. `statsmodels` `MixedLM` with age
bucket as a random effect, or a simple manual empirical-Bayes shrinkage of
per-bucket residual means weighted by `n / (n + k)`. Validate using the
existing nested-by-year harness (step 8) AND the new age-bucket-holdout
harness (step 9b) — report both so we know if a fix helps general
interpolation (year fold) vs. specifically the age-density problem
(age-bucket fold).

### 2. #1 — Track-record feature for batters (and maybe pitchers)

Add a feature capturing "established veteran with a track record" distinct
from raw age — candidates: `n_seasons` (already computed in
`market_training_data.csv`, currently unused as a model feature, only
diagnostic), or a cumulative trailing-career RAR sum. Hypothesis: a
37-year-old still performing well isn't "old," the market reads him as
"proven," and age_sq failed precisely because it's still just a function of
age. Test via the same ablation harness as `research/step9_bat_feature_ablation.py`
(LOO-CV on full sample) AND re-run step 8 / step 9b to check it actually
generalizes — age_sq's lesson is that in-sample LOO-CV wins don't guarantee
held-out wins, so don't skip the second check this time.

### 3. #8 — Gradient-boosted trees with monotonic constraints

Try GBM (e.g. `sklearn.ensemble.HistGradientBoostingRegressor` with
monotonic_cst, or `lightgbm` if available in the env) in place of ridge for
log(AAV), letting it find age×position×RAR interactions automatically.
High overfitting risk at n≈216 — validate strictly via the existing nested
year-fold harness before trusting it over ridge. If it doesn't beat ridge
out-of-sample, document that and move on; don't force it in.

### 4. #6 — Conformal prediction intervals

Once the point-estimate work above is settled (whatever combination wins),
shift focus from point accuracy to calibrated uncertainty: build conformal
prediction intervals around the AAV prediction so the model's output is "I
predict $X, and the true value falls in [$Y, $Z] with 90% coverage" instead
of just a point guess. This is arguably more useful for the actual
downstream consumer (step 6's Monte Carlo / step 7's risk-adjusted
objective) than chasing the point-accuracy number further. Validate
coverage empirically on the step 8 held-out folds (does the interval
actually contain the actual AAV ~90% of the time?).

### 5. #3 — Quantile / Huber regression (do last)

Re-fit log(AAV) with a robust loss (Huber) or quantile regression (median
target) instead of ridge's squared error, so the handful of extreme
outliers (e.g. player 41743: actual $11.9M vs predicted $2.5M) don't drag
the fit. Saved for last because steps 1–4 above may already substantially
change the feature set / model family, and robust-loss tuning is most
useful as a final polish once the feature side is settled, not before.

## Results (run 2026-06-17)

All four of #5, #1, #8, #6 have been implemented and validated against the
real held-out nested-by-signing-year harness (not just in-sample LOO-CV --
the lesson from `age_sq`). Baseline to beat: **16.7% within ±15%, R²=0.503**
(reproduced exactly as the `K=inf` / `baseline` / `ridge` row in each script).

- **#5 — empirical-Bayes age-bucket shrinkage (`research/step9_followup1_age_pooling.py`):
  NEGATIVE, reverted.** Swept shrinkage K in {1,2,5,10,20,50,100,inf}. Every
  finite K made within±15% *worse* (13.8%-15.7% vs the 16.7% no-correction
  baseline); R² also monotonically improves as K→inf (more shrinkage =
  worse). The correction direction itself wasn't wrong, but with the
  thinnest buckets (37+, n≈19 across all 5 years) there's too little
  training-fold residual signal to shrink toward without adding noise.
  Confirms step 9b's "general data-density problem" framing — there isn't
  enough contract history to support even disciplined partial pooling yet.

- **#1 — track-record feature (`research/step9_followup2_track_record.py`):
  NEGATIVE, reverted.** Tested `n_career_seasons` (count of distinct prior
  MLB seasons on record) and `career_rar_cum` (full-career, unweighted RAR
  sum, distinct from the existing 3yr-gamma-weighted `proj_rar`), both
  individually and combined. None beat baseline on within±15% (15.7-16.7%
  vs 16.7%); `career_rar_cum` alone was clearly worse (13.3%). R² ticked up
  for the n_career_seasons variants (0.503→0.549) but that's exactly the
  age_sq trap — in-sample-flavored improvement that doesn't survive the
  ±15% bar. "Proven veteran" isn't separable from "old" with only ~210
  signings to learn it from.

- **#8 — HistGradientBoostingRegressor with monotonic constraints
  (`research/step9_followup3_gbm.py`): NEGATIVE, reverted, confirms the
  documented overfitting concern.** GBM underperformed ridge on every
  metric (within±15% 15.7% vs 16.7%, median err 48.0% vs 40.1%, R² 0.421 vs
  0.503) despite shallow trees (max_depth=3, max_leaf_nodes=8), strong L2,
  and early stopping. At n≈150-180 per fold per player type, there simply
  isn't enough data for a tree ensemble to beat a 7-9 feature linear model.
  Per the plan's own instruction ("if it doesn't beat ridge out-of-sample,
  document and move on"), not pursued further.

- **#6 — CV+ (K-fold) conformal prediction intervals
  (`research/step9_followup4_conformal.py`): POSITIVE — keep.** Built calibrated
  $-scale intervals around the existing ridge point estimate using
  out-of-fold residuals as nonconformity scores (calibration set for fold f
  = pooled residuals from the OTHER 4 folds, so no leakage). Empirical
  coverage tracks the target closely: 80% target → 81.4% actual, 90% target
  → 91.0% actual (pooled batters+pitchers; per-player-type breakdown is
  similarly tight). **This is the first follow-up that actually delivers
  something usable** — the point estimate's accuracy hasn't improved, but
  we now have an honest, well-calibrated uncertainty band instead of a
  single number that's silently wrong ~83% of the time. The cost is width:
  median interval is $14.5M wide at 80% coverage and $20M wide at 90%
  (hi/lo ratio 5-10x) — this is the model being honest about how little
  the data actually pins down AAV at this sample size, not a bug.
  **Not yet wired into steps 6/7** — Monte Carlo's Track B currently uses
  the point estimate only. If adopted, the natural integration point is
  widening Track B's cost distribution by this calibrated band rather than
  treating predicted AAV as a fixed cost.

**Net effect on the original goal:** the 85% within ±15% success criteria
is still not met, and per the diagnostics above (steps 9, 9b, and this
follow-up round) it's very unlikely to be met by further feature/model
engineering on this size of contract sample — every lever tried compounds
to the same conclusion. The conformal-interval result is the one piece of
genuinely new capability to come out of this round: it doesn't fix the
point estimate, but it gives a defensible, validated way to quantify how
much to trust it.

- **#3 — robust/quantile regression (`research/step9_followup5_quantile.py`):
  NEGATIVE, reverted.** Tested HuberRegressor (grid over epsilon × L2 alpha)
  and QuantileRegressor(quantile=0.5) (grid over L1 alpha), both with
  hyperparameters selected via the same LOO-CV procedure as ridge's alpha.
  Huber lost on every metric (15.2% within±15%, R²=0.487 vs ridge's
  16.7%/0.503). Quantile regression nudged within±15% up to 17.1% but gave
  up substantial ground elsewhere (within±25% 27.1% vs 31.0%, median error
  44.0% vs 40.1%, R² 0.425 vs 0.503) — a 0.4pp move on one metric at n=210
  (one signing) is noise, not signal, and the rest of the scorecard is
  clearly worse. Not a genuine win; same shape of false positive as
  `age_sq`'s in-sample LOO-CV result. Reverted.

## Final status (all 5 items run, round closed 2026-06-17)

**#5, #1, #8, #3 are all negative and reverted. #6 (conformal intervals) is
the only positive result and is kept as a new, separate capability** — not
integrated into steps 5/6/7's point estimate, but available as
`research/step9_followup4_conformal.py` / `intermediate/step9f_conformal_*.csv`
for whoever wants to widen step 6's Track B cost distribution by the
calibrated band instead of treating predicted AAV as fixed.

**Bottom line: the point estimate (ridge, 7/9 features, 16.7% within ±15%,
R²=0.503) is very likely at or near the ceiling supportable by ~210 rows of
contract data.** Four independently-motivated attempts to beat it — partial
pooling, a new feature class, a fundamentally different model family (trees),
and two different robust-loss objectives — all converged on the same
conclusion. Per the original spec's own guidance, the data-quantity
constraint (5 in-game contract cohorts) is the binding limitation, not the
modeling approach. The only paths left that could plausibly move the needle
are external to this round's scope: more pulled seasons of contract data as
the league continues, or an OOTP ratings token if one becomes available.

## Working notes for whoever picks this up

- Python: `C:\Users\Felto\miniconda3\envs\baseball\python.exe` (call directly,
  not via `conda run`).
- Reuse the data-loading / retro-feature code pattern already duplicated
  across `src/pipeline/step5_market_regression.py`, `src/pipeline/step8_validation.py`,
  `research/step9_bat_feature_ablation.py`, and `research/step9_age_holdout.py` — it's
  intentionally copy-pasted per the project's existing style (each step
  script is self-contained), not refactored into a shared module.
- Any change to `BAT_FEATURES`/`PIT_FEATURES` in step 5 must be mirrored in
  step 8's validation script AND step 6's `_bat_market_features` /
  `_pit_market_features` (Track B inference), per the existing
  "Step 6 Track B feature definitions must match step 5 training exactly"
  rule in `CLAUDE.md`. Forgetting one of the three causes silent mis-pricing,
  not a crash.
- After any model change, re-run in this order to keep everything
  consistent: `src/pipeline/step5_market_regression.py` → `src/pipeline/step8_validation.py`
  (check the held-out metrics) → if keeping the change,
  `src/pipeline/step6_monte_carlo.py` → `src/pipeline/step7_optimizer.py` → re-run
  `src/pipeline/step9_residual_analysis.py` and/or `research/step9_age_holdout.py` to see if
  the bias actually moved.
- Always compare against the documented baseline above before deciding to
  keep a change — `age_sq` is the cautionary example: it looked like a clear
  win until checked against the real held-out harness.
