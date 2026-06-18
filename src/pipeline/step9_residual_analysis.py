"""
Step 9: Residual analysis on step 8's held-out nested k-fold predictions.

Goal: separate *correctable* miss patterns (something systematic the model
could fix -- a missing feature, a biased coefficient, a segment that's
mis-specified) from the *uncorrectable* ratings-blind-spot noise floor that
the spec already flags as a known limitation.

Breaks residuals down by:
  - player_type (batter/pitcher), and sp_flag within pitchers
  - position (primary_pos) within batters
  - age bucket
  - signing year (fold)
  - proj_rar tier (is the model worse for stars specifically?)
  - direction of miss (under- vs over-prediction) -- a systematic skew in
    one direction across a segment is a correctable bias; symmetric scatter
    around 0 is consistent with unmodeled (ratings) noise.

Inputs:
  intermediate/step8_validation_results.csv  (held-out preds, 210 rows)
  intermediate/market_training_data.csv      (full-sample features incl.
                                               primary_pos, proj_rar, sp_flag,
                                               n_seasons -- joined in for
                                               segmentation; these are
                                               player-own-history features,
                                               not leakage, since they only
                                               depend on each player's stats
                                               before their own signing)

Outputs:
  intermediate/step9_residuals.csv      -- per-signing residuals + features
  intermediate/step9_segment_summary.csv -- aggregated bias/error by segment
  Console: ranked list of worst segments, signed-bias check
"""

import os
import numpy as np
import pandas as pd

INT = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"

POS_NAMES = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF", 10: "DH", 0: "unk"}

print("=" * 60)
print("STEP 9: Residual analysis on step 8 held-out predictions")
print("=" * 60)

s8 = pd.read_csv(os.path.join(INT, "step8_validation_results.csv"))
mt = pd.read_csv(os.path.join(INT, "market_training_data.csv"))

# join own-history features (position, proj_rar, sp_flag, n_seasons) — keyed
# by (player_id, season_year, player_type) to disambiguate the rare
# same-year re-signing cases (8 rows in mt share a key with a different deal)
feat_cols = ["player_id", "season_year", "player_type", "primary_pos", "proj_rar",
             "n_seasons", "sp_flag", "age"]
mt_feat = mt[feat_cols].drop_duplicates(subset=["player_id", "season_year", "player_type"])
df = s8.merge(mt_feat, on=["player_id", "season_year", "player_type"], how="left")

missing = df["age"].isna().sum()
if missing:
    print(f"  WARNING: {missing} rows failed to join own-history features (dropped from segment analysis)")
df = df.dropna(subset=["age"]).copy()

# signed residual: positive = model UNDER-predicted (actual > pred)
df["resid_M"] = df["actual_aav_M"] - df["pred_aav_M"]
df["signed_pct_err"] = df["resid_M"] / df["pred_aav_M"]
df["pos_name"] = df["primary_pos"].fillna(0).astype(int).map(POS_NAMES).fillna("unk")
df["age_bucket"] = pd.cut(df["age"], bins=[19, 27, 30, 33, 36, 50],
                           labels=["<=27", "28-30", "31-33", "34-36", "37+"])
df["rar_tier"] = pd.qcut(df["proj_rar"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")
df["role"] = np.where(df["player_type"] == "pitcher",
                       np.where(df["sp_flag"] == 1, "SP", "RP"), "batter")

df.to_csv(os.path.join(INT, "step9_residuals.csv"), index=False)

def summarize(group_col, label):
    """Aggregate residual/error metrics (n, signed/abs % error, within±15%, mean AAVs) by group_col, tagging the result rows with segment_type=label."""
    g = df.groupby(group_col, observed=True).agg(
        n=("resid_M", "size"),
        mean_signed_pct_err=("signed_pct_err", "mean"),
        median_abs_pct_err=("pct_err", "median"),
        within_15=("within_15", "mean"),
        mean_actual_aav=("actual_aav_M", "mean"),
        mean_pred_aav=("pred_aav_M", "mean"),
    ).reset_index()
    g.insert(0, "segment_type", label)
    return g

segments = pd.concat([
    summarize("player_type", "player_type"),
    summarize("role", "role"),
    summarize("pos_name", "position (batters)"),
    summarize("age_bucket", "age_bucket"),
    summarize("rar_tier", "proj_rar_tier"),
    summarize("test_year", "signing_year"),
], ignore_index=True)
segments.to_csv(os.path.join(INT, "step9_segment_summary.csv"), index=False)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

print("\n[1] Overall signed bias (positive = under-predict, negative = over-predict)")
print(f"  mean signed %% error: {df['signed_pct_err'].mean():+.1%}   "
      f"median signed %% error: {df['signed_pct_err'].median():+.1%}")
print(f"  share under-predicted (actual > pred): {(df['resid_M'] > 0).mean():.1%}")

print("\n[2] By player_type / role")
print(segments[segments["segment_type"].isin(["player_type", "role"])]
      .to_string(index=False))

print("\n[3] By batter position")
print(segments[segments["segment_type"] == "position (batters)"].to_string(index=False))

print("\n[4] By age bucket")
print(segments[segments["segment_type"] == "age_bucket"].to_string(index=False))

print("\n[5] By proj_rar tier (is the model worse for stars?)")
print(segments[segments["segment_type"] == "proj_rar_tier"].to_string(index=False))

print("\n[6] By signing year (fold)")
print(segments[segments["segment_type"] == "signing_year"].to_string(index=False))

print("\n[7] Worst individual misses (top 15 by abs % error)")
worst = df.sort_values("pct_err", ascending=False).head(15)[
    ["player_id", "test_year", "player_type", "role", "pos_name", "age", "proj_rar",
     "actual_aav_M", "pred_aav_M", "signed_pct_err"]
]
print(worst.to_string(index=False))

print("\n[8] Correlation of signed %% error with proj_rar and age (within player_type)")
for ptype in ["batter", "pitcher"]:
    sub = df[df["player_type"] == ptype]
    corr_rar = sub["signed_pct_err"].corr(sub["proj_rar"])
    corr_age = sub["signed_pct_err"].corr(sub["age"])
    print(f"  {ptype}: corr(signed_err, proj_rar) = {corr_rar:+.3f}   "
          f"corr(signed_err, age) = {corr_age:+.3f}")

print("\n" + "=" * 60)
print("STEP 9 COMPLETE -- see intermediate/step9_residuals.csv and step9_segment_summary.csv")
print("=" * 60)
