# Frostfire free-agent contract valuation model

A model that recommends a single free-agent contract (years + AAV) for any FA in
the Frostfire OOTP league, grounded in 21 years of performance data and 5 years
of contract data. See the two spec files below — they are authoritative and
self-contained.

## Authoritative context (read these first, every session)

@docs/model_spec.md
@docs/data_summary.md

**All design decisions in the spec are LOCKED unless I explicitly revisit them.**
If something here conflicts with the spec, the spec wins — except where the data
summary corrects the spec (the data summary was verified empirically).

---

## Keeping docs up to date (standing instruction)

**Update this file and `docs/data_summary.md` whenever:**
- A build step is completed — mark it done in the progress tracker below and note the output files.
- An empirical fact about the data contradicts or refines the spec — add it to "Easy-to-get-wrong rules" and to the data summary.
- A design decision is revisited or a new constraint is discovered.
- A package, environment, or tooling issue is resolved.

The goal is that a fresh conversation can read these files and pick up exactly where the last one left off, without re-deriving anything.

---

## Environment

- **Python:** `C:\Users\Felto\miniconda3\envs\baseball\python.exe`
- **Working directory:** `C:\Users\Felto\Downloads\ootp_analysis\`
- **Do NOT** use `conda run -n baseball` — it fails on multiline `-c` args in PowerShell. Call the python.exe directly.
- **pyarrow is not installed** in this env (conda install fails due to a corrupt pytorch cache entry). Use CSV for intermediate files, not parquet.

---

## Where things are

**Repo layout was reorganized for GitHub (2026-06-17) — paths below are current.**
All pipeline scripts read/write `frostfire_data/` and `intermediate/` as paths
relative to the **current working directory**, not the script location, so
always run them from the repo root (e.g. `python src/pipeline/step1a_foundation.py`).

| Path | Contents |
|---|---|
| `frostfire_data/` | All raw CSV/JSON files pulled by `src/data/puller.py` (gitignored — regenerate, don't commit) |
| `intermediate/` | Cleaned base tables and model artifacts produced by the pipeline scripts (gitignored — regenerate, don't commit) |
| `src/data/puller.py` | API puller (this doc previously called it `frostfire_puller.py`, which never matched the real filename — it was `datapull.py`, now `src/data/puller.py`) |
| `src/data/transactions_parser.py` | Parses OOTP transaction-log HTML → `intermediate/fa_signings_log.csv` (unbiased FA signing data) |
| `src/pipeline/step1a_foundation.py` | Step 1a: audit + clean load |
| `src/pipeline/step1b_park_factors.py` | Step 1b: park factors + neutralization |
| `src/pipeline/step2_aging_curves.py` | Step 2: aging curves |
| `src/pipeline/step3_dollar_curve.py` | Step 3: $/RAR curve (trained on `fa_signings_log.csv`) |
| `src/pipeline/step4_marcel_projection.py` | Step 4: Marcel projections |
| `src/pipeline/step5_market_regression.py` | Step 5: market regression (trained on `fa_signings_log.csv`) |
| `src/pipeline/step6_monte_carlo.py` | Step 6: Monte Carlo career simulation |
| `src/pipeline/step7_optimizer.py` | Step 7: length optimizer + recommendations |
| `src/pipeline/step8_validation.py` | Step 8: nested k-fold validation by signing year |
| `src/pipeline/step9_residual_analysis.py` | Step 9: residual analysis |
| `src/viz/` | Diagnostic plotting scripts (aging curve fit quality, etc.) |
| `research/` | Closed investigations (hyperparameter sweeps, ablations, follow-up experiments) — documented for history, not part of the production pipeline. Most were tested and reverted; see `docs/` for which ones and why. |
| `probes/` | One-time API exploration scripts/output — re-run only if the StatsPlus API changes |
| `archive/legacy_xgboost_notebook/` | An earlier, superseded XGBoost-based notebook approach, kept for history only — **not** the current model |
| `docs/model_spec.md` | Authoritative spec (locked design) |
| `docs/data_summary.md` | Data catalog (empirically verified) |

---

## Build progress

Steps follow the build order in the spec. Check off each step when done; note the
output files and any surprises so the next session starts cold-but-informed.

**✅ Transaction-log rework complete (2026-06-17).** Steps 3, 5, 6, 7 have been
rebuilt on the unbiased transaction-log signing data, and step 8 (nested
k-fold validation) has been built and run. `contracts.csv` was discovered to
be a live snapshot only (the API ignores all date params — confirmed
empirically), causing severe survivorship bias in early FA cohorts. The fix:
`src/data/transactions_parser.py` parses OOTP's in-game transaction-log HTML
(`news/html/leagues/league_203_all_transactions_{M}_{YYYY}.html`) into
`intermediate/fa_signings_log.csv`, which captures every FA signing event
with no survivorship bias. **Full background in
[`docs/step8_transaction_log_rework.md`](docs/step8_transaction_log_rework.md).**
`contracts.csv` is retired as a training-set source but still used for
replacement-salary calibration (a snapshot-appropriate use).

### Step 1 — Foundation table

- [x] **1a: Audit + clean load** (`src/pipeline/step1a_foundation.py`)
  - Outputs: `intermediate/batting_raw.csv` (27,655 rows), `intermediate/pitching_raw.csv` (36,040 rows), `intermediate/fielding_raw.csv` (32,509 rows)
  - All splits 1/2/3 embedded in the main per-year file; split_id=21 rows dropped (undocumented, present across all years)
  - Pitching grain corrected: `(player_id, year, team_id, split_id)` — stint is always 0 in pitching files
  - Audit passes clean with zero warnings

- [x] **1b: Park factors + neutralization** (`src/pipeline/step1b_park_factors.py`)
  - Outputs: `intermediate/batting_neutral.csv` (27,655 rows, 44 cols), `intermediate/pitching_neutral.csv` (36,040 rows, 63 cols)
  - Park factor selection: batting uses **batter handedness** from `players.csv` (`bats`=1→avg_r/hr_r, 2→avg_l/hr_l, 3→blended); pitching uses **split_id** (1→blended, 2→avg_l/hr_l for vs-LHB, 3→avg_r/hr_r for vs-RHB)
  - Batting columns added: `singles`, `singles_n`, `d_n`, `t_n`, `hr_n`, `h_n` (= singles_n+d_n+t_n+hr_n), `eff_avg_pf`, `eff_d_pf`, `eff_t_pf`, `eff_hr_pf`
  - Pitching columns added: `ha_n`, `hra_n`, `eff_avg_pf`, `eff_hr_pf`
  - Walks, K, HBP, baserunning left unadjusted (not meaningfully park-affected)
  - Known extremes verified correct: Colorado `eff_t_pf`=1.175 (raw t=1.35), Cincinnati LHB `eff_hr_pf`=1.12 (raw hr_l=1.24)
  - Identity `h_n == singles_n+d_n+t_n+hr_n` holds to floating-point precision; audit passes clean

### Step 2 — Aging curves

- [x] **2: Delta-method aging curves per position per component** (`src/pipeline/step2_aging_curves.py`)
  - Full implementation notes: `docs/step2_aging_curve_notes.md`
  - Outputs: `intermediate/aging_deltas.csv` (99,923 rows = 66,442 natural + 33,481 synthetic exit-correction deltas), `intermediate/aging_cell_stats.csv` (1,434 rows), `intermediate/aging_curves_smooth.csv` (1,491 rows), `intermediate/aging_fit_stats.csv` (426 rows), `intermediate/aging_threshold_sens.csv` (284 rows)
  - Components tracked: batting (hr_pa, xbh_pa, single_pa, bb_pa, k_pa, ubr_g per PA/game); pitching (k_bf, bb_hbp_bf, hra_bf per BF); fielding (zr_rate, arm_rate, framing_rate per 1000 IP)
  - Age assignment: `year – birth_year` from players.csv; age range 20–40
  - SP/RP split at gs/g ≥ 0.5 (default); gives SP=2,426, RP=4,591 qualifying seasons (bf≥100)
  - Six smoothers per curve: poly2/3/4 (weighted by n_pairs) + LOESS at 0.3/0.5/0.7 bandwidth; LOOA CV RMSE, AIC, BIC in fit stats; `poly3_cumulative` anchored at 0 at age 20
  - Threshold sensitivity table covers PA/BF/IP at 50/100/150/200 for stability evaluation
  - **Key data finding:** `position` column in player_batting_YYYY.csv is always 0 — OOTP does not store a fielding position in batting files. Position assignment derived from fielding_raw.csv (innings-primary per player-year), with players.csv `Pos` as fallback
  - **DH (position 10) pooled with 1B:** only 4 qualifying batting seasons across 21 years. Remapped via `position.replace({10: 3})` in `batting_seasons()`
  - **Woolner exit correction applied (APPLY_EXIT_CORRECTION=True, EXIT_CORRECTION_WEIGHT=0.5):** players who qualify at age A but not A+1 get a synthetic delta = age_A+1_group_mean − their_value, at half weight. Correction counts: batting 2,830 exiting player-years → 16,590 synthetic deltas; pitching 2,641 → 7,905; fielding 4,368 → 8,986
  - **Residual survivorship in CF/SS zr_rate:** even after correction, CF (+45.7) and SS (+52.6) zr_rate cumulative curves still rise through age 40. This is acceptable — the Marcel step (step 4) projects relative to same-age positional mean, not as absolute change from 20
  - **Single_pa fits poorly everywhere (poly3 R² < 0.05 for most positions)** — signal is weak; prefer loess70 or wide uncertainty
  - **Polynomial end-effects:** poly3/poly4 can diverge at ages 38–40 when cells are thin; prefer LOESS loess50 for thin edge-age cells (flag_thin > 0)

### Step 3 — League $/component curve

- [x] **3: Dollar value per run above replacement** (`src/pipeline/step3_dollar_curve.py`) — **rebuilt 2026-06-17 on transaction-log data**
  - Outputs: `intermediate/linear_weights.csv`, `intermediate/player_values.csv` (15,603 rows), `intermediate/curve_coefficients.csv`, `intermediate/curve_training_data.csv`
  - **Batting linear weights** (team-level R²=0.870): single=+0.625, double=+0.894, triple=+1.486, HR=+1.398, BB=+0.470, HBP=+0.327, K=−0.021
  - **Pitching linear weights** (team-level R²=0.825): ha=+0.637, hra_extra=+0.778, BB=+0.442, HBP=+0.335, K=−0.005 (unchanged — these come from team-level stat data, not contracts, so they weren't affected by the rework)
  - **RAR validation**: independently-derived batter RAR correlates 0.958 with OOTP's own WAR (PA≥300) — strong confirmation linear weights are working
  - **Training set now sourced from `intermediate/fa_signings_log.csv`** (parsed from the in-game transaction log — see "Transaction-log rework" note above), not `contracts.csv`. `salary0` is approximated as `total_value/years` (flat-AAV proxy; the log has no year-by-year salary breakdown). Excludes extensions and the pre-2031 AI-managed era (`human_era` flag).
  - **$/RAR quadratic (pooled, rebuilt 2026-06-17 with the step 10 service-time fix)**: salary = 3.428 + 0.2179·RAR + 0.003125·RAR² ($M); R²=0.400 on 257 rows above the salary floor (268 training rows total, 110 batters + 158 pitchers, before the floor — up from 216/192 under the old snapshot-based service-time formula; see step 10 for why). Training uses `service_at_signing >= 6` for calibration (see FA threshold split rule below; `service_at_signing` is now derived from performance-panel appearance history, not the `mlb_service_years` snapshot).
  - **Replacement salary**: $750,000 (mode of 584 contracts at that level, still read from `contracts.csv` — this is a snapshot-appropriate use, not training-set bias-sensitive); replacement offset = 20 runs/600 PA for batters, 27 runs/162 IP for pitchers
  - **Implied $/WAR**: $4.0M/WAR at replacement-level rising with convexity for elite players (star premium confirmed; intercept and slope both increased vs. the old fit since low-RAR signings in the unbiased sample still command real money — partly a ratings-limited-model artifact, see step 5 notes)
  - *Superseded findings from the old (`contracts.csv`-trained) fit — kept for history:* old curve was salary = 1.592 + 0.401·RAR + 0.002037·RAR² ($M), R²=0.523 on 134 rows; an `is_pitcher` dummy was tried and reverted (inflated batter intercept); old residuals showed SPs overpredicted ~$4M, batters underpredicted ~$3.5M (this informed step 6's `PITCHER_TRACK_A_DISCOUNT_M`, still in place — re-validate against the new fit if revisiting).

### Step 4 — Marcel projection layer

- [x] **4: Marcel projections** (`src/pipeline/step4_marcel_projection.py`)
  - Outputs: `intermediate/marcel_projections.csv` (3,090 rows), `intermediate/marcel_hyperparams.csv`, `intermediate/marcel_cv_scores.csv`
  - **Components projected:** batting (hr_pa, xbh_pa, single_pa, bb_pa, k_pa, ubr_g, hp_pa + pa volume); pitching (k_bf, bb_hbp_bf, hra_bf, ha_n_bf + ip/bf volume); fielding (zr_rate, arm_rate, framing_rate)
  - **CV hyperparams** (leave-one-season-out, volume-weighted RMSE): batting K=700–1200 PA (heavy regression), k_pa has lowest K=200 (stickiest skill); pitching hra_bf L=2 γ=0.3 (very volatile, short memory); fielding K=600 IP (very noisy, strong regression); gamma=0.7 wins for almost all components
  - **Active ML coverage:** 300 position players with batting projections (position distribution realistic across C/1B/2B/SS/etc.), 291 pitchers (112 SP + 179 RP); 534 total batting projections (not refused), 499 pitching
  - **Refused threshold:** batting refused if max weighted PA < 50; pitching if max weighted BF < 50; thin-flagged if < 150
  - **Aging delta applied:** loess50_delta from step 2 curves (fallback poly3_delta) for all components that have aging curves; ha_n_bf gets no aging adjustment (BABIP is largely random)
  - **Position lookup bug fixed:** `position` values must be assigned directly from the numpy array (not via `pd.Series()` which introduces index-alignment mismatch after filtering)
  - **Decomposition helpers stored:** `d_frac` (doubles fraction of XBH) and `bb_frac` (BB fraction of BB+HBP) for downstream run-value decomposition
  - **hp_pa added as batting component:** not in step 2 aging curves; projects with K=1200 (strong regression, low skill signal)
  - **Volume projection:** PA and IP now have own CV-fit hyperparams in seasons units (not borrowed from rate components):
    - `pa_vol`: L=2, K=2.0 seasons, gamma=0.9 — short lookback, moderate regression (playing time roles change fast)
    - `ip_vol_SP`: L=3, K=5.0 seasons, gamma=0.7 — SP workload moderately predictable
    - `ip_vol_RP`: L=5, K=8.0 seasons, gamma=0.9 — heavy regression (RP roles are volatile)
  - **Hyperparameter grid improvements (from `research/step4_marcel_hyperparam_diag.py`):**
    - 11/14 components were hitting the K ceiling; extended grids found real improvements
    - `xbh_pa` K=3500 (was 1200), `single_pa` and `ubr_g` K=2000, pitching K=1200 for bb_hbp_bf/hra_bf/ha_n_bf
    - Gamma range extended to [0.1, 0.3, 0.5, 0.7, 0.9] — nearly all components want gamma=0.9; arm_rate wants 0.1 (almost no gap-year decay elsewhere matters)
    - Fielding K values still hit new ceiling (K=1500) — surface is relatively flat there (6-7% range); acceptable
    - Most sensitive component: k_pa (23% RMSE range across grid) — gets hyperparams right matters here
    - Outputs unchanged: 3,090 rows total; batting 558 projected (24 more due to longer lookback), pitching 503

### Step 5 — Contract regression

- [x] **5: Market price regression (#3)** (`src/pipeline/step5_market_regression.py`) — **rebuilt 2026-06-17 on transaction-log data**
  - Outputs: `intermediate/market_model_coefficients.csv`, `intermediate/market_training_data.csv`
  - **Training set now sourced from `intermediate/fa_signings_log.csv`**, not `contracts.csv` (see "Transaction-log rework" note above). **286 rows after floor (126 batters + 160 pitchers) as of the step 10 service-time fix (2026-06-17)** — up from 216 rows (84+132) under the old snapshot-based `service_at_signing` formula (see step 10: that formula silently misclassified some human-era rows too, not just AI-era ones). Same calibration threshold (`service_at_signing >= 6`), `human_era` (signing_year ≥ 2031) filter, extensions excluded.
  - **Salary floor**: $1,125,000 (= 1.5× replacement $750K).
  - **Retrospective projection**: unchanged — 3-year gamma=0.9 volume-and-recency weighted batting/pitching/fielding stats → RAR via step 3's linear weights (linear weights are stat-derived, not contract-derived, so they didn't change in the rework). Uses park-neutral panels.
  - **Feature sets unchanged** (decided before the rework, still valid): batters use the 7-feature decomposed set (`bat_raa`, `proj_ubr`, `proj_def`, `age`, `proj_pa`, `is_premium_def`, `proj_rar_sq`); pitchers use the 9-feature ablation-selected set (`proj_rar`, `age`, `age_sq`, `log_proj_ip`, `sp_flag`, `proj_rar_sq`, `k_rate`, `age_x_krate`, `rar_x_hra`). See `frostfire_pit_ablation*.py` for how the pitcher set was derived — that derivation is unaffected by the data-source fix.
  - **Alpha selection (re-run with step 10 fix)**: joint LOO-CV, 50-pt log-spaced grid [0.01–1000]; batter alpha=0.268, pitcher alpha=0.429.
  - **Results (re-run with step 10 fix)**: batter log(AAV) R²=0.661 / years R²=0.267; pitcher log(AAV) R²=0.502 / years R²=0.211.
  - **In-sample within ±15% of actual AAV: 19.2%** (within ±25%: 33.2%, median abs % error 39.5%) on the full training set — see step 8 for the proper held-out version of this metric.
  - **Primary limitation unchanged**: OOTP player ratings invisible. The expanded sample makes this more visible, not less — several of the biggest misses (e.g. player 41743, actual $11.9M vs predicted $2.5M; player 45402, actual $14.0M vs predicted $4.6M) are large enough that they're very likely rating-driven outliers the model has no signal for.
  - **Application recipe (step 7) — unchanged**:
    - Batters: `X_raw = [bat_raa, proj_ubr, proj_def, age, proj_pa, is_premium_def, proj_rar_sq]` where `proj_rar_sq = proj_rar²`
    - Pitchers: `X_raw = [proj_rar, age, age_sq, log_proj_ip, sp_flag, proj_rar_sq, k_rate, age_x_krate, rar_x_hra]` where `log_proj_ip = log(max(proj_ip, 10))`, `age_sq = age²`, `proj_rar_sq = proj_rar²`, `age_x_krate = age × k_rate`, `rar_x_hra = proj_rar × hra_rate`
    - Standardize with saved scaler params, apply ridge coefficients → predict log_aav + years. Floor predicted AAV at $750K.
  - *Old (`contracts.csv`-trained) feature-derivation history (ablation studies, ratings-limited examples like player 42636/39073) is unaffected by the rework and remains valid — see git history of this file for the full prior writeup if needed.*

### Step 6 — Monte Carlo

- [x] **6: Career simulations** (`src/pipeline/step6_monte_carlo.py`)
  - Outputs: `intermediate/mc_surplus_distributions.csv` (1,056 rows = 212 players × up to 5 lengths), `intermediate/mc_variance_model.csv`, `intermediate/mc_correlation_batting/fielding/pitching.csv`, `intermediate/mc_playtime_bat/pit.csv`
  - **N_SIMS = 40,000** careers per player; all valid candidate lengths computed simultaneously via vectorized numpy
  - **Variance decomposition**: total sigma = step-4 CV RMSE; 50% persistent (correlated across years), 50% transient (independent per year). Sigma_p = persist_frac × (corr × outer(rmse, rmse)); Sigma_t = (1-persist_frac) × diag(rmse²)
  - **Correlation matrices**: fit from 21-year pooled league-mean-demeaned raw rates (not Marcel residuals). PSD correction via eigenvalue clipping. Key empirical correlations: batting hr_pa vs single_pa = −0.341 (power vs contact, expected); pitching k_bf vs ha_n_bf = −0.516 (high-K pitchers allow fewer hits)
  - **Playing-time model**: year-over-year PA/IP log-normal ratio fit per age band (20–24, 25–27, 28–30, 31–33, 34–36, 37+). Mu declines from +0.21 (youngest) to −0.19 (37+) for batters; sigma ≈ 0.50–0.54. `p_catastrophic = 0.0` from data (qualification filter makes total-loss seasons invisible), replaced by age-specific cliff probs below.
  - **Park adjustment (Track A)**: Nationals Stadium effective factors applied by batter handedness (RHB: eff_avg=0.985, eff_hr=1.010; LHB: eff_avg=1.010, eff_hr=1.000; triples: 0.960 for all). Pitching uses blended (eff_avg=0.9938, eff_hr=1.0065).
  - **Preference weights**: infield defense (2B/3B/SS) zr+arm value × 1.20; pitcher HR suppression component × 1.15. Applied in Track A value only.
  - **Aging application**: Marcel already projects year+1 (age_proj applied in step 4). In MC, year-0 uses projection directly; year k adds cumulative delta from age_proj through age_proj+k−1 via `loess50_delta` (fallback `poly3_delta`). **Survivorship bias correction**: positive aging deltas at age ≥ 34 are clamped to 0 — apparent improvement in survivors is selection artifact, not genuine skill development.
  - **Track B (market price)**: step-5 ridge model applied to **retro-consistent** features for each player; produces fixed predicted AAV used as cost. **Luxury tax penalty**: AAV $35–50M incurs a 30% surcharge on excess; AAV >$50M incurs an additional 60% surcharge. The surplus calculation uses the tax-adjusted effective cost, not nominal AAV.
  - **Track B feature consistency fix (applied post-step-6-rerun)**: Two training/inference mismatches were corrected in `src/pipeline/step6_monte_carlo.py`:
    1. **Batter `proj_def`**: Step 5 training uses `def_runs = zr + arm + framing + pos_adj` as the single `proj_def` feature. Step 6 was computing `proj_def = zr + arm + frm` (missing pos_adj) then adding pos_adj separately to proj_rar. This caused the model to see catchers as having ~8 fewer `proj_def` runs at inference than in training → market AAV underpredicted by $4-6M for catchers. Fix: fold `pos_adj` into `proj_def` in `_bat_market_features` to match training definition.
    2. **Pitcher retro IP**: Step 5 training used 3-year gamma-weighted actual IP averages (~65-70 IP for RPs). Step 6 was passing Marcel's aggressively-regressed IP (~55-58 IP for RPs) into the market features. Lower IP → lower proj_rar → market AAV underpredicted by ~$2-3M for RPs. Fix: `_retro_pit_mkt_features()` recomputes proj_ip, proj_rar, k_rate, hra_rate from the historical panel (`pit_py`) using the identical 3-year gamma-weighted method as step 5. Marcel IP is still used for Track A simulation.
  - **After fix results**: catchers mean AAV $3.0M → $6.3M (all 8 now "do not sign"); RPs mean AAV $3.75M → $5.16M (33/44 do-not-sign). Remaining RP cases with very low predicted AAV ($1-2M) despite decent RAR are likely **rating-limited** — the same phenomenon as player 42636 (high stats, poor OOTP ratings) — and cannot be distinguished without the ratings token.
  - **`proj_rar` column in `mc_surplus_distributions.csv`**: for pitchers, this now stores the retro-computed RAR (used for Track B), not Marcel's RAR. Marcel RAR can be retrieved from `intermediate/marcel_projections.csv` if needed.
  - **Forecast horizon uncertainty** (`HORIZON_SIGMA_EXPANSION = 0.20`): transient shocks in year y are scaled by (1 + 0.20 × y), reflecting that Marcel accuracy degrades in years 2–5. This makes stdev grow faster than linearly with contract length and breaks the otherwise bimodal 1-or-5-year pattern.
  - **Age cap** (`MAX_CONTRACT_END_AGE = 44`): candidate lengths are filtered so the contract end age ≤ 44. A 42-year-old can receive at most a 2-year deal (plays at ages 42–43).
  - **Performance cliff** (`CLIFF_PROBS = [(35, 38, 0.04), (38, 41, 0.08), (41+, 0.14)]`): per-year probability that a sim career ends (catastrophic injury / retirement). Once triggered, that simulation contributes $0 value for the remainder of the contract. Not in historical panel due to survivorship; set as informed priors.
  - **FA threshold split (resolved)**: FA eligibility in Frostfire is 5 years (`>= 5`). However, the $/RAR curve (step 3) and market regression (step 5) are calibrated on `>= 6` contracts (established market pricing). Step 6 uses `>= 5` for evaluation. This split prevents 5-year FA contracts (younger, cheaper) from flattening the $/RAR curve and collapsing all surplus to zero. Using `>= 5` in steps 3/5 reduced b from 0.401 → 0.291 M/run and eliminated all positive recommendations.
  - **Pitcher Track A discount**: `PITCHER_TRACK_A_DISCOUNT_M = -2.0` in `rar_to_value_M(rar, is_pitcher=True)`. Carried over from the old curve-3 fit's residual pattern (~$4M SP overprediction). **Not re-validated against the rebuilt curve 3** — worth re-checking if revisiting step 6.
  - **Results summary (re-run 2026-06-17 with the step 10 service-time fix)**: 212 FA-eligible players evaluated (91 batters, 121 pitchers); **15/212 recommend "sign"** (down from 22/212 before the fix — the corrected, larger training set in steps 3/5 produced somewhat higher market-price predictions for several mid-tier players, closing some of the surplus that the old buggy eligibility filter had been showing). Top signings: Julio Torres (batter, 31yo, RAR=6.7, 5yr @ $2.96M, obj=+4.52), Bobby Hutchinson (batter, 32yo, RAR=5.7, 5yr @ $2.46M, obj=+4.30), Pamphile Gorman (batter, 30yo, RAR=9.7, 5yr @ $3.07M, obj=+4.25). Full list in `intermediate/recommendations.csv`.
  - **Why the recommendation count jumped, then came back down**: the original rebuild (contracts.csv → transaction log) lowered predicted market prices broadly, surfacing 22 positive-surplus signings from a baseline of 1. The step 10 service-time fix added ~70 more legitimately-eligible signings to steps 3/5's training set (mostly human-era rows the old buggy formula had been excluding) and shifted the market-price curve again, this time slightly upward for some mid-tier profiles — netting 15 recommended signings, still far above the original snapshot-biased baseline.
  - **Output**: `mc_surplus_distributions.csv` now includes `player_name` column for readability.

### Step 7 — Dollar conversion + length optimizer

- [x] **7: Length optimization** (`src/pipeline/step7_optimizer.py`) — **re-run 2026-06-17 on step 6 output (post step-10 service-time fix)**
  - Output: `intermediate/recommendations.csv` (212 rows, one per FA-eligible player)
  - Per-player argmax of `objective = mean_surplus − 1.15 × std_surplus` over candidate lengths 1–5; "do not sign" if best objective < 0
  - All preference weights (infield defense ×1.20, pitcher HR suppression ×1.15) were applied in step 6's Track A value calculation; step 7 reads the already-computed objective directly
  - **Results (post step-10 fix)**: 212 FA-eligible players; **15 recommended signings, 197 do not sign**. Mostly short-RAR-but-cheap profiles getting 3–5 year deals around $0.75M–$4.6M AAV. See `intermediate/recommendations.csv` for the full ranked list.
  - **Personal preference weights (value side only — do NOT apply to market price / Step 5), unchanged:**
    - Infield defense premium: `def_zr` and `def_arm` for positions 2B (4), 3B (5), SS (6) × 1.20
    - Pitcher HR suppression premium: `hra_suppression` component × 1.15
    - Everything else: weight 1.0
    - These apply when converting component runs to dollar value (Track A). Track B (market price from Step 5) stays neutral — it predicts what other GMs pay, not what you value the player at.

### Step 8 — Validation

- [x] **8: Nested k-fold validation** (`src/pipeline/step8_validation.py`) — **built 2026-06-17, re-run 2026-06-17 after the step 10 service-time fix**
  - Outputs: `intermediate/step8_validation_results.csv` (280 held-out predictions as of the step 10 fix), `intermediate/step8_fold_summary.csv`
  - **Refit-scope decision (confirmed with owner)**: aging curves (step 2) and Marcel hyperparameters (step 4) are fit on 21 years of performance data only and never touch contract/signing data, so there's no leakage path through them — NOT refit per fold. The $/RAR curve (step 3) and market ridge regression (step 5) directly fit on signing data — these ARE refit per fold. The retrospective Marcel-style features themselves (bat_raa, proj_def, k_rate, etc.) depend only on each player's own performance history, not on other players' contracts, so they're computed once and reused across folds — only the ridge coefficients are refit.
  - **Outer loop**: 5 folds, one per signing year 2031–2035. **Inner loop**: refit curve-3 quadratic + regression-5 ridge models (batter/pitcher) on every year except the held-out one, predict AAV for the held-out year's signings, compare to actual.
  - **Per-fold results (post step-10 fix)**: 2031 (n=37, within±15%=18.9%, R²=0.507), 2032 (n=48, 12.5%, R²=0.554), 2033 (n=63, 17.5%, R²=0.409), 2034 (n=65, 16.9%, R²=0.581), 2035 (n=67, 25.4%, R²=0.698).
  - **OVERALL (post step-10 fix): 18.6% within ±15% of actual AAV (280 held-out signings), 29.6% within ±25%, median abs % error 42.4%, R²=0.568.** Up from the pre-fix baseline of 16.7%/210 rows/R²=0.503 — see step 10 for the full diagnosis of the underlying service-time-at-signing bug that caused this. (Within±25% and median error moved slightly the other way, 31.0%→29.6% and 40.1%→42.4% — a mixed result on secondary metrics, but the primary success metric and R² both improved and the fix itself is independently correct.)
  - **DOES NOT MEET the 85% within ±15% success criteria.** Per the spec's own guidance, the response to missing this is to examine residuals, not relax the bar. The largest misses are heavily under-predicted high-actual-AAV signings (e.g. actual $11.9M vs predicted $2.5M; actual $14.0M vs predicted $4.1M) — consistent with the model's known, unfixable blind spot: **OOTP player ratings are invisible to this model**, and the real market clearly pays for them. This is the dominant source of the validation gap, not a methodology bug in the rework itself.
  - **Next step (not yet started): step 9 residual analysis** — break down misses by position/age/signing-year to see if any *correctable* (non-ratings) pattern remains, separate from the ratings-driven noise floor. (Note: step 9 and its follow-up round below were run against the pre-step-10-fix baseline of 210 rows/16.7%; re-running step 9 against the new 280-row/18.6% baseline has not been done and isn't required immediately, but bear the mismatch in mind if revisiting those findings.)

### Step 9 — Residual analysis

- [x] **9: Residual analysis** (`src/pipeline/step9_residual_analysis.py`) — **run 2026-06-17**
  - Outputs: `intermediate/step9_residuals.csv` (210 rows), `intermediate/step9_segment_summary.csv`
  - **Headline finding: a real, monotonic age-related under-prediction bias.** Mean signed % error (positive = under-predicted) by age bucket, pooled batters+pitchers: 28-30 = +14.5%, 31-33 = +12.0%, 34-36 = +30.8%, 37+ = **+57.8%**. Holds within both player types and within every proj_rar quartile (not just an artifact of old players happening to be elite outliers). Correlation of signed error with age: batters +0.249, pitchers +0.077.
  - **Position and RAR-tier breakdowns showed no other clean pattern** — position splits are noisy (n=2-17 per position) and proj_rar-tier error is roughly flat, so the age effect is the dominant *segment-level* signal in the residuals.
  - **Overall bias**: mean signed % error +20.9% across all 210 held-out signings — the model under-predicts more/worse than it over-predicts, consistent with the known ratings-blind-spot asymmetry (you can overpay for invisible positive traits; contracts are floored, so you can't symmetrically underpay for invisible negative ones).
  - **Tested and reverted: age² for batters** (`research/step9_bat_feature_ablation.py`). Re-ran the batter feature ablation on the rebuilt (post-rework) training set since the old "no benefit from age_sq" finding predates the transaction-log rework. Full-sample LOO-CV preferred adding it (combined loss −0.061, better than an age×proj_rar interaction at −0.020), so it was added to `BAT_FEATURES`, steps 5/6/8 were refit/rerun — but it produced **no improvement in the nested by-signing-year validation** (16.7% → 16.2% within ±15%, R² 0.503 → 0.498, age-37+ bucket bias 57.8% → 55.5%, within fold-to-fold noise). **Conclusion: the age bias is a small-sample extrapolation problem, not a missing-curvature problem** — only 19 age-37+ signings exist across all 5 cohorts combined, so nested-by-year validation is almost always asking the model to predict an age range it saw few or no examples of in training. No feature engineering on a sample this size fixes that; it would need more contract-year history or the (unavailable) ratings token. Reverted to the original 7-feature `BAT_FEATURES` — confirmed via re-run that step 5/8/6/7 outputs match the pre-experiment baseline exactly (23.1% in-sample, 16.7% held-out within ±15%, R²=0.503, 22/212 sign recommendations).
  - **Bottom line for the 85% success criteria**: the gap is real and not closeable with the current data. The dominant drivers are (a) invisible OOTP ratings (the documented, accepted limitation) and (b) thin contract history at the tails (very old/very young FAs, extreme RAR) that no amount of feature engineering on ~210-source rows can fix. Per the spec's own guidance ("examine residuals, not relax the bar"), this has now been done — the next lever, if pursued, is more pulled seasons of contract data, not new features.
  - **Follow-up diagnostic (`research/step9_age_holdout.py`, run 2026-06-17):** leave-one-age-bucket-out (pooling all 5 signing years for training, unlike step 8's by-year fold) found the age bias is a *general data-density* problem across the whole age axis, not an old-players-specific wall — holding out 28-30 is also bad (+42.1% bias) when its own data is excluded from training, and the 37+ bias roughly halves vs. the year-fold number once full temporal coverage is available (57.8%→29.7%). This is more encouraging for borrow-strength methods (partial pooling, track-record features) than the headline step 8 number suggested.
  - **Follow-up round run 2026-06-17 — #5, #1, #8 negative (reverted), #6 positive (kept as a new capability, not yet wired into steps 6/7). #3 not yet run.** All four validated against the real held-out nested-by-signing-year harness, not just in-sample LOO-CV (the `age_sq` lesson applied consistently):
    - **#5 (`research/step9_followup1_age_pooling.py`)** — empirical-Bayes shrinkage of age-bucket ridge residuals (`n/(n+K)`), swept K. Every finite K made within±15% worse (13.8-15.7% vs 16.7% baseline). Reverted.
    - **#1 (`research/step9_followup2_track_record.py`)** — added `n_career_seasons` / full-career unweighted `career_rar_cum` as features distinct from age. None beat 16.7% within±15% (best tied at 16.7%, most worse); R² improved for some variants but that's the same in-sample-only trap `age_sq` fell into. Reverted.
    - **#8 (`research/step9_followup3_gbm.py`)** — `HistGradientBoostingRegressor` with monotonic constraints vs ridge. Confirmed the documented overfitting risk: GBM lost on every metric (15.7% within±15%, R²=0.421 vs ridge's 16.7%/0.503) despite shallow trees + early stopping. n≈150-180/fold/type is too small for trees to beat a 7-9 feature linear model. Reverted.
    - **#6 (`research/step9_followup4_conformal.py`)** — CV+ (K-fold) conformal prediction intervals around the existing ridge point estimate, using out-of-fold residuals as nonconformity scores (no leakage: fold f's calibration set excludes fold f's own residuals). **Empirical coverage matches target tightly**: 80% target → 81.4% actual, 90% target → 91.0% actual. This is the one positive result of the round — it doesn't fix point accuracy, but gives a validated, honest uncertainty band (median width ~$14.5M at 80%, ~$20M at 90% — wide, but that's the model correctly admitting how little a ~210-row contract sample pins down AAV). Not yet integrated into step 6's Track B (which still uses the point estimate as a fixed cost) or step 7's optimizer.
    - **#3 (`research/step9_followup5_quantile.py`)** — HuberRegressor and QuantileRegressor(median) vs ridge, hyperparameters LOO-CV selected like ridge's alpha. Huber lost on every metric. Quantile nudged within±15% to 17.1% (from 16.7%) but gave up within±25% (27.1% vs 31.0%), median error (44.0% vs 40.1%), and R² (0.425 vs 0.503) — a single-signing noise effect on one metric, not a real win (same shape as the `age_sq` false positive). Reverted.
    - **Round closed.** All 5 planned items run. 4/5 negative, 1/5 (conformal intervals) positive and kept as a standalone capability. **The ridge point estimate (16.7% within±15%, R²=0.503) is very likely at or near the ceiling supportable by ~210 contract rows** — partial pooling, a new feature class, a different model family, and two robust-loss objectives all independently converged on not beating it. The binding constraint is contract-data quantity (5 cohorts), not modeling approach; the only remaining levers are external (more pulled seasons as the league continues, or an OOTP ratings token).
    - Full background, exact numbers, and integration notes: **[`docs/step9_followup_plan.md`](docs/step9_followup_plan.md)**.
    - Anchoring-on-prior-AAV was dropped (affected cohorts were mostly AI-managed, not human market behavior); Tobit/censoring for the salary floor was dropped (no signings actually fall below the floor); pooling batters+pitchers into one model was rejected.

### Step 10 — AI-era data expansion (closed 2026-06-17)

- [x] **10: Investigate incorporating pre-2031 "AI days" signings into training.** Two separate findings came out of this — one negative (the original ask), one positive (a bug fix discovered along the way). Both are final.
  - **Discovery:** the step 9 follow-up round concluded the ridge point estimate is ceilinged by contract-data quantity (~210 rows). Checked whether more transaction-log history exists in the OOTP saved-game folder than `src/data/transactions_parser.py` had previously consumed — it does. The news folder has a transaction-log HTML file for every month from **March 2015 through October 2035 with zero gaps**. Re-parsed: 251 files → 4,130 signing rows (1,521 FA signings + 2,609 extensions), of which 787 are `human_era=False` (pre-2031, the "AI days").
  - **First attempt — naive weighting (`research/step10_ai_era_weight_sweep.py`):** swept a fixed downweight `W` on AI-era rows in `sample_weight`, validating only on human-era folds (2031-2035) as proposed. Found only 37 of 787 AI-era rows survived the existing `service_at_signing >= 6` eligibility filter — and every `W > 0` underperformed `W=0` (excluding AI-era entirely).
  - **Root-cause dig — why only 37/787 survived:** the eligibility filter computed `service_at_signing` as `mlb_service_years` (current 2035 snapshot) minus years elapsed since signing. That assumes continuous accrual through 2035, which silently breaks for any player who has since retired (their `mlb_service_years` counter freezes, so subtracting elapsed time overcounts and pushes `service_at_signing` deeply negative). This wasn't just suppressing AI-era rows — it was misclassifying some human-era rows too.
  - **Fix (adopted into production, see below):** derive `service_at_signing` from each player's actual MLB appearance history (distinct years with a qualifying row in `batting_raw.csv`/`pitching_raw.csv`/`fielding_raw.csv`, `team_id` in `VALID_TIDS`) instead of the snapshot. Robust to retirement since it never depends on present-day state. Recovered AI-era rows from 37 → 452, and human-era rows from 210 → 280 (the bug was hurting both populations, not just AI-era).
  - **Re-swept weighting with the fix (`research/step10_ai_era_weight_sweep.py`, corrected):** even with 452 real AI-era rows available, every `W > 0` still underperformed `W=0` on both within±15% and R² — confirms the negative result isn't a small-sample artifact.
  - **Root-cause diagnosis of WHY blending still hurts (`research/step10_ai_era_diagnose.py`):** tested two suspected mechanisms in a 2×3×7 grid (LOO-CV scope `{all, human_only}` × features `{base, +is_ai_era dummy, +dummy+proj_rar interaction}` × weight `W`):
    1. **Alpha-tuning scope bug (confirmed real):** the original LOO-CV used to pick ridge's alpha scored against the blended human+AI pool, tuning regularization to fit AI noise too. Restricting LOO evaluation to human-era held-out points only (AI rows always stay in training) recovered most of the damage — e.g. at W=0.1, R² jumped from 0.430 (buggy scope) to 0.562 (fixed scope, vs. 0.567 baseline).
    2. **Domain-level price mismatch (confirmed real, smaller effect):** adding an `is_ai_era` dummy (+ a `proj_rar × is_ai_era` slope term) let AI-era rows have their own price level/slope instead of dragging the shared one. Recovered a further small amount in the buggy LOO scope (R² 0.500→0.509 at W=1.0).
    3. **Net result: both fixes combined still never beat `W=0`.** Best blended cell anywhere in the 42-combo grid (`loo=human_only`, `+dummy+slope`, W=0.1) **ties** baseline on within±15% (19.3% vs 19.3%) but is marginally behind on R² (0.563 vs 0.567). Conclusion: AI-era pricing differs from human pricing in more ways than one dummy + one interaction term can correct for, and the human-era sample is too small to spend more degrees of freedom finding out without overfitting. **This is a genuine, diagnosed negative result, not an under-explored one.**
  - **Decision: do NOT blend AI-era data into training, at any weight or feature-adaptation scheme tried.** `research/step10_ai_era_weight_sweep.py` and `research/step10_ai_era_diagnose.py` are kept as documentation of the investigation; not part of the production pipeline.
  - **Decision: DO adopt the service-time-at-signing bug fix into production.** It's correct on its own merits (eligibility should never depend on a frozen, present-day snapshot) independent of the AI-era question. Propagated into `src/pipeline/step3_dollar_curve.py`, `src/pipeline/step5_market_regression.py`, `src/pipeline/step8_validation.py`; steps 3/5/6/7/8 re-run 2026-06-17. **New production baseline: 18.6% within±15% (was 16.7%), R²=0.568 (was 0.503), n=280 held-out signings (was 210).** Some other metrics moved the other way (within±25% 29.6% vs 31.0%, median abs % error 42.4% vs 40.1%) — a mixed bag like several other results this round, but the primary success metric and R² both improved and the underlying fix is unambiguously correct, so it's kept. See updated numbers in steps 3/5/6/7/8 above.

---

### Step 11 — Hyperparameter sweep against step 8/9 accuracy (closed 2026-06-17)

- [x] **11: Sweep untested hand-set constants in steps 5/8 against the real within±15% metric.** Prompted by a request to do general hyperparameter tuning. Before sweeping, confirmed which constants can actually move step 8/9's numbers: steps 2 (aging) and 4 (Marcel) never touch contract data and were already CV-tuned on their own objective (RMSE), so they're untestable against AAV accuracy and weren't touched. Track A/Monte Carlo constants (risk-aversion 1.15, luxury tax tiers, `persist_frac`, `HORIZON_SIGMA_EXPANSION`, cliff probs) only affect the sign/no-sign decision, which has no ground-truth validation target — also not testable against step 8/9. That left three genuinely untested levers, all inside step 5's retrospective-projection feature builder (`src/pipeline/step5_market_regression.py`/`src/pipeline/step8_validation.py`):
  1. **`GAMMA` (gap-year decay) × `L_RETRO` (lookback window)** — currently 0.9/3, borrowed wholesale from step 4's Marcel CV-fit (which optimizes performance-projection RMSE, a different objective) and never independently tuned against AAV accuracy. Swept `research/step5_hyperparam_sweep.py`: GAMMA ∈ {0.5...1.0}, L_RETRO ∈ {2,3,4,5}, full nested-by-year refit each combo. Best within±15% (gamma=0.85, L_RETRO=2) hit 19.9% vs baseline 18.6% — but R² cratered (0.568→0.446) and per-fold breakdown showed it **loses** to baseline in the largest, best-determined fold (2035, n=67: 20.9% vs 25.4% within±15%, R² 0.166 vs 0.698) and only wins by accumulating small gains in smaller, noisier folds. Classic overfitting-to-fold-noise signature, same shape as the reverted age_sq/quantile-regression experiments in step 9's follow-up round. **Reverted — GAMMA=0.9, L_RETRO=3 unchanged.**
  2. **Salary-floor multiplier** (currently 1.5× replacement = $1.125M) — raising it to 2.5× improved within±15% to 21.8%, but this just narrows the evaluated population (n drops 280→248) by excluding more low-salary signings from both training AND eval. That's gaming the metric's denominator, not improving the model's predictions for the population the spec actually cares about. **Rejected on principle, not just on the numbers — left at 1.5×.**
  3. **MIN_PA/MIN_BF qualification threshold** (currently 100) — swept 50-200; thresh=150 nudged within±15% to 19.8% but R² dropped to 0.536 (vs 0.568) and within±25% also dropped slightly. Same noise pattern as #1. **Reverted — left at 100.**
  - **Conclusion: no combination of untested hyperparameters produced a fold-robust improvement.** This is now the third independent round of hyperparameter/model-family search (step 9's original follow-up round + step 10's AI-era blending attempts + this one) to reach the same wall. The ~280-row contract sample is very likely at its accuracy ceiling for this feature/model class; the remaining levers are external (more pulled contract-year history, or an OOTP ratings token), not further tuning. `research/step5_hyperparam_sweep.py` kept for documentation/reproducibility; not part of the production pipeline. Raw sweep grid in `intermediate/step5_gamma_lretro_sweep.csv`.
  - **Reliably-achievable accuracy, synthesized across all three search rounds: within±15% ≈ 18-20%, within±25% ≈ 28-32%, median abs % error ≈ 40-43%, R² ≈ 0.50-0.57.** Full writeup with practical interpretation (typical dollar error at various AAV levels) and why this is a real ceiling, not a tuning gap: **[`docs/step11_accuracy_ceiling.md`](docs/step11_accuracy_ceiling.md)**.

---

## Frostfire league rules (confirmed by owner)

- **Free agency:** Players become free agents after **5 years** of MLB service time (not 6). Step 6 evaluation uses `>= 5`. Calibration steps 3 and 5 intentionally remain at `>= 6` to avoid the younger-player bias that flattens the $/RAR curve — see FA threshold split rule in "Easy-to-get-wrong."
- **Arbitration:** Players are eligible after **3 years** of MLB service time.
- **Rule 5 Draft:** Maximum of **5 rounds** of selections. Attempting to claim beyond 5 rounds results in players being returned and a warning; repeat offenses incur a cash penalty.
- **No qualifying offers:** No QO contracts are used, and there are no compensation draft picks to teams that don't re-sign their own free agents in the offseason.

---

## Easy-to-get-wrong rules (do not violate)

- Reference columns by **header name**, never by positional index (schema versioning).
- Majors filter = **allowlist** of the 22 real team_ids, not a blocklist of All-Star teams.
- **FA eligibility threshold split**: Frostfire FA eligibility = 5 years (`mlb_service_years >= 5`). BUT calibration steps 3 and 5 use `>= 6` (established-market pricing). Step 6 evaluation uses `>= 5`. Using `>= 5` in steps 3/5 collapses the curve b coefficient and eliminates all positive recommendations — the 5-year FAs are younger/cheaper and flatten the $/RAR relationship. Do not change steps 3 or 5 to `>= 5` without expecting this consequence.
- Keep batting/pitching at the **stint level** through park adjustment; aggregate only after.
  - **Batting grain:** `(player_id, year, stint, split_id)` — batting stint DOES increment on mid-season trades.
  - **Pitching grain:** `(player_id, year, team_id, split_id)` — the `stint` column in pitching is always 0 and does NOT increment. Use `team_id` to differentiate stints.
  - **Fielding grain:** `(player_id, year, position, team_id)`.
- `contracts.csv`: each contract is **one row**. Signing year is `season_year`; `salary0` is the first-year AAV. Filter signings with `is_major=1 AND season_year>0`. Do NOT filter on `current_year=1` (documented mistake). **`contracts.csv` is a live snapshot only** (the `contract` API endpoint ignores all date params — confirmed empirically) and is **retired as a training-set source** for steps 3/5 because of severe survivorship bias (a contract is only visible if it hasn't expired as of the 2035 pull date). It's still fine to use for snapshot-appropriate things like the replacement-salary calibration (today's pay floor).
- **Market-regression training (steps 3, 5) now comes from `intermediate/fa_signings_log.csv`**, parsed from OOTP's in-game transaction-log HTML by `src/data/transactions_parser.py` (source: `news/html/leagues/league_203_all_transactions_{M}_{YYYY}.html` under the OOTP save folder). Columns: `player_id`, `player_name`, `team_id`, `team_name`, `position`, `file_year`, `file_month`, `signing_year` (season the contract is for — see mapping below), `years`, `total_value`, `aav` (`total_value/years`, used as the `salary0` proxy since the log has no year-by-year salary breakdown), `is_extension`, `human_era` (signing_year ≥ 2031). The parser keeps every row it finds — it does NOT drop pre-2031 rows itself; that filtering happens downstream in steps 3/5 via the `human_era` flag. Full background: `step8_transaction_log_rework.md`.
- **`signing_year` mapping**: a contract signed in month ≤ 6 of calendar year Y is for the Y season; signed in month > 6 of Y is for the Y+1 season (post-season offseason signing). This matches `contracts.csv`'s `season_year` semantics and was confirmed empirically from the signing-volume spike pattern (heavy Dec–Feb signing activity, near-zero in-season).
- Exclude `contract_extensions.csv` (and `is_extension=True` rows in `fa_signings_log.csv`) from the market regression (#3) — pre-FA discount bias.
- Derive run values **from Frostfire data**, not borrowed MLB constants.
- The league has **no inflation** — one pooled $/component fit, no year fixed effects. This is an indisputable fact confirmed by the owner. Do not investigate, test, or flag year-trend patterns in residuals as possible inflation. Any year-to-year residual pattern is selection bias from the thin contract sample (early cohorts are almost entirely multi-year deals for above-average players).
- The `_vsLHP`/`_vsRHP` and `_vsLHB`/`_vsRHB` split files are **redundant** — all splits (1/2/3) are already embedded in the main `player_batting_YYYY.csv` and `player_pitching_YYYY.csv` files. Load only the main file.
- `split_id=21` rows appear in batting and pitching across all years from real ML teams. **Drop them** — nature unknown, not used by the model.
- The `position` column in `player_batting_YYYY.csv` is **always 0** — OOTP does not store a fielding position in batting files. For aging curves and position-group assignment, derive the player's primary position from `fielding_raw.csv` (position with the most innings in that season), falling back to `players.csv` `Pos` column.
- **Step 5 market regression**: apply salary floor (salary0 > $1,125,000) before fitting. Batter features are decomposed components `bat_raa`, `proj_ubr`, `proj_def`, not combined `proj_rar`. Pitcher uses `log(proj_ip)` not raw IP. Both models include `proj_rar²` for quadratic saturation. `log_proj_ip` clipped at 10 IP minimum. Step 7 must floor predicted AAV at $750K.
- **Step 6 Track B feature definitions must match step 5 training exactly**: (a) Batter `proj_def = zr + arm + frm + pos_adj` — positional adjustment is INSIDE proj_def, not separate. (b) Pitcher market features use `_retro_pit_mkt_features()` (3-year gamma-weighted actual IP/BF from `pit_py`), NOT Marcel's regressed IP projection. Violating either causes systematic mis-pricing of catchers (a) and relievers (b).
- **Step 6 Track A `rar_to_value_M()` must pass `is_pitcher` flag**: `rar_to_value_M(rar, is_pitcher=False)` for batters; `rar_to_value_M(rar, is_pitcher=True)` for pitchers. The pitcher flag applies `PITCHER_TRACK_A_DISCOUNT_M = -2.0` (hardcoded constant in step 6, NOT loaded from `curve_coefficients.csv`). Omitting the flag causes the pooled curve to overvalue pitcher runs. There is NO `d_M_pitcher` in `curve_coefficients.csv` — the pitcher dummy approach was tried and reverted.

---

## How I want to work

- Fact-check before writing code. If instructions are unclear, ask questions.
- Favor accuracy over speed. Never give inaccurate or outdated information.
- Explain concepts as we go so I can discuss them with others.
- Small steps — finish and verify one sub-step before starting the next.
- **Update CLAUDE.md and `docs/data_summary.md` at the end of every working session** with completed steps, new findings, and any corrections to prior assumptions.
