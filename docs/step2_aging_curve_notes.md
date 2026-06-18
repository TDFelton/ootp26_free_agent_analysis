# Step 2 — Aging Curves: Implementation Notes

Script: `src/pipeline/step2_aging_curves.py`  
Status: Complete (exit correction applied, DH pooled with 1B)

---

## What step 2 does

Builds per-(position, component) aging curves using the **delta method**: for each player
who appears in consecutive seasons A and A+1, compute `delta = component(A+1) − component(A)`
and assign the delta to age A. Accumulate these deltas per (position, component, age) into
cells, then smooth with six curve shapes per cell sequence.

The output tells you: *for a typical player at position P and age A, how much does component C
change in the next year?* The cumulative version tells you how much total change you'd expect
from age 20 to age A.

---

## Components tracked

**Batting** (per plate appearance)
- `hr_pa` — home runs / PA (park-neutral hr_n)
- `xbh_pa` — extra-base hits / PA (park-neutral d_n + t_n + hr_n)
- `single_pa` — singles / PA (park-neutral singles_n)
- `bb_pa` — walks / PA
- `k_pa` — strikeouts / PA
- `ubr_g` — ultimate baserunning runs / game

**Pitching** (per batter faced)
- `k_bf` — strikeout rate
- `bb_hbp_bf` — walk + HBP rate
- `hra_bf` — HR allowed rate (park-neutral hra_n)

**Fielding** (per 1,000 innings)
- `zr_rate` — zone rating runs
- `arm_rate` — arm runs
- `framing_rate` — catcher framing runs (zero for non-catchers)

---

## Grouping structure

| Domain | Groups |
|---|---|
| Batting | 8 positions (2=C, 3=1B [includes DH], 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF) |
| Pitching | SP, RP |
| Fielding | 8 positions (same as batting, excludes pitchers) |

**SP/RP classification:** `gs/g >= 0.5` = SP, else RP. At bf≥100 this yields SP=2,426 and RP=4,591 qualifying seasons across 21 years. The gs/g threshold is a hyperparameter at the top of the script (`SP_THRESHOLD`).

---

## Data discoveries and surprises

### Batting position column is always 0

The `position` column in `player_batting_YYYY.csv` is **always 0**. OOTP does not store a
fielding position in batting rows. This was discovered via diagnostic: all 9,395 overall-split
batting rows had `position=0`.

**Fix:** `build_position_lookup()` derives each player's primary position from `fielding_raw.csv`
(the position with the most innings in that player-year). Falls back to `players.csv` `Pos`
column when a player-year has no fielding record (e.g., DH-only seasons or the 2035 partial
year). After the fix, positions 2–9 are distributed correctly across 6,137 batting player-years.

### DH (position 10) pooled with 1B

DH has only **4 qualifying batting seasons** (pa≥100) across 21 years. Completely inadequate
for a curve fit. Solution: remap DH → 1B in the position lookup:

```python
agg["position"] = agg["position"].replace({10: 3})
```

This is applied in `batting_seasons()` after merging in the position lookup. The effect is that
DH seasons contribute to the 1B aging curves, which is defensible since most DHs are washed-up
1Bs or corner outfielders anyway.

---

## Survivorship bias in defense — the Woolner exit correction

### The problem

Without correction, aging curves for positions that require athleticism (CF, SS) show zone
rating *rising* with age. This is backwards: it happens because below-average fielders at these
positions get moved to easier spots (LF, 1B, DH) as they age, so the remaining CF/SS pool
at age 35 is an elite subset. The delta method sees elite 34-year-olds turning into elite
35-year-olds and concludes fielding improves — wrong.

### The correction (Woolner exit method)

For each player who qualifies at age A but does **not** appear in the next season's qualifying
set (they "exited"):

1. Compute the **age-A+1 group mean** for every component (based on players who *did* return).
2. Create a **synthetic delta** = `age_A+1_group_mean − player_value_at_A` for that player.
3. Add it to the delta pool at weight `EXIT_CORRECTION_WEIGHT` (default 0.5, i.e., half a
   real observation).

The intuition: if a below-average CF (zr_rate = −30) retires at age 34, and the typical
returning 35-year-old CF has zr_rate = +20, the synthetic delta is +50 — a large positive
that counteracts the artificial upward pull. Adding it (at half weight) pulls the curve down
toward the true population trajectory.

### Results from final run

```
Batting exit-correction:  2,830 exiting player-years  ->  16,590 synthetic deltas (weight=0.5)
Pitching exit-correction: 2,641 exiting player-years  ->   7,905 synthetic deltas (weight=0.5)
Fielding exit-correction: 4,368 exiting player-years  ->   8,986 synthetic deltas (weight=0.5)
Total delta pairs: 99,923 (natural: 66,442, synthetic: 33,481)
```

The cell stats file now includes `n_natural` and `n_exit_correction` columns so you can see the
correction ratio per (group, component, age) cell.

### Residual survivorship in defense

Even with `EXIT_CORRECTION_WEIGHT=0.5`, CF and SS zr_rate still rise substantially with age:

| Position | zr_rate cumulative by age 40 |
|---|---|
| SS | +52.6 (per 1,000 IP) |
| CF | +45.7 |
| 2B | +30.4 |
| 3B | −6.6 |
| RF | −16.5 |
| LF | −19.8 |
| C | −7.8 |
| 1B | −57.4 |

SS and CF rising throughout a career is not realistic. Two options:
1. Increase `EXIT_CORRECTION_WEIGHT` toward 1.0 to apply a stronger correction.
2. Accept the curves and instead project *relative to same-age league average* rather than as
   absolute change from 20. The Marcel layer (step 4) regresses toward the position-age mean,
   which will implicitly handle this.

Option 2 is the spec's intent (spec §Marcel projection: "regressed toward position-and-age
league baseline"). So the residual survivorship in the raw delta curves is acceptable — the
Marcel step corrects for it by not projecting in absolute terms.

---

## Smoothing and fit metrics

Six curve shapes are fit for each (group, component) sequence:

| Model | Description |
|---|---|
| `poly2` | Degree-2 polynomial, weighted by n_pairs |
| `poly3` | Degree-3 polynomial |
| `poly4` | Degree-4 polynomial |
| `loess30` | LOESS at bandwidth 0.3 (tight, local) |
| `loess50` | LOESS at bandwidth 0.5 (medium) |
| `loess70` | LOESS at bandwidth 0.7 (smooth) |

**Fit statistics** (in `aging_fit_stats.csv`):
- `r2_weighted` — weighted R² against cell means (diagnostic only; cells are averages not raw obs)
- `rmse_looa` — leave-one-out age cross-validation RMSE (the primary selection criterion)
- `aic`, `bic` — information criteria for polynomial models

**Key findings:**
- `single_pa` fits poorly everywhere (poly3 R² < 0.05 for most positions). Singles rate is
  inherently noisy; the stat has weak aging signal. Use a wide LOESS bandwidth or wide
  uncertainty bounds.
- `poly4` often diverges at extreme ages (38–40) when cells are thin. Default to `loess50`
  when `flag_thin > 0` at edge ages.
- `arm_rate` occasionally favors poly4 over poly3 by AIC — inspect the edge-age behavior
  before accepting.

---

## Hyperparameters (all at top of `src/pipeline/step2_aging_curves.py`)

| Parameter | Default | What it controls |
|---|---|---|
| `MIN_PA` | 100 | Minimum plate appearances for a batting season to qualify |
| `MIN_BF` | 100 | Minimum batters faced for a pitching season to qualify |
| `MIN_IP_FLD` | 50 | Minimum innings for a fielding position to qualify |
| `SP_THRESHOLD` | 0.5 | gs/g cutoff to classify SP vs RP |
| `POLY_DEGREES` | [2, 3, 4] | Polynomial degrees to try |
| `LOESS_FRACS` | [0.3, 0.5, 0.7] | LOESS bandwidths to try |
| `APPLY_EXIT_CORRECTION` | True | Whether to apply the Woolner exit correction |
| `EXIT_CORRECTION_WEIGHT` | 0.5 | Weight of synthetic exit-correction deltas |
| `SENS_PA_THRESHOLDS` | [50, 100, 150, 200] | Thresholds for sensitivity table |
| `SENS_BF_THRESHOLDS` | [50, 100, 150, 200] | Same for pitching |
| `SENS_IP_THRESHOLDS` | [25, 50, 100, 150] | Same for fielding |

**Tuning guidance from the threshold sensitivity table:**
- Batting PA 50→200: peak_age shifts <2 years for most components. MIN_PA=100 is stable.
- Pitching BF 50→200: RP curves are more sensitive (thin at high thresholds). MIN_BF=100 is
  a reasonable compromise; 150 tightens RP sample too much.
- Fielding IP 50→150: zr_rate and arm_rate peak_age are stable. MIN_IP_FLD=50 is fine.

---

## Output files

| File | Rows | Description |
|---|---|---|
| `intermediate/aging_deltas.csv` | 99,923 | All delta pairs (natural + synthetic exit-correction). Columns: player_id, group_label, component, age_start, delta, weight, birth_year, year_start, is_exit_correction |
| `intermediate/aging_cell_stats.csv` | 1,434 | Per (group, component, age) stats: mean, std, n_pairs, n_natural, n_exit_correction, flag_thin |
| `intermediate/aging_curves_smooth.csv` | 1,491 | Smoothed per-age predictions for all 6 models + poly3_cumulative anchored at age 20 = 0 |
| `intermediate/aging_fit_stats.csv` | 426 | Per (group, component, model): r2_weighted, rmse_looa, aic, bic, peak_age, peak_value |
| `intermediate/aging_threshold_sens.csv` | 284 | Stability of peak_age / r2 / n_pairs as PA/BF/IP threshold varies |

The cumulative column (`poly3_cumulative`) is anchored at 0 at age 20. A value of −57 at age
40 for 1B zr_rate means a first-baseman is expected to be 57 runs per 1,000 innings worse
defensively at 40 than at 20.

---

## Notable curve shapes (poly3, per 1,000 IP for defense; per PA for offense)

**HR rate (hr_pa):** Peaks around age 28–32 for most positions, then modest decline. CF and
LF show unrealistically late peaks (age 38+) due to thin samples — use loess50 for those.

**Strikeout rate (k_pa):** Generally declines (improves) from age 20 into the early-to-mid
30s as hitters become more disciplined. SS shows continuous improvement, reflecting that only
elite contact hitters survive at that position.

**Zone rating (zr_rate):** Varies substantially by position. 1B declines steeply (−57 by 40),
C declines modestly (−8), while SS/CF still show artificial rises (survivorship residual, see
above).

**Catcher framing:** Declines from age 20 onward in the corrected curves (−2 by age 40). The
framing curves are noisy due to small catcher sample sizes; the correction may be slightly
overcorrecting toward decline.

---

## What step 3 needs from here

Step 3 (league $/component curve) does not consume the aging curves directly — it regresses
WAR/salary against the foundation data to calibrate the dollar value of each run component.
The aging curves feed into step 4 (Marcel projection), which uses the per-age delta in
`aging_curves_smooth.csv` to age the weighted-average historical rates forward.
