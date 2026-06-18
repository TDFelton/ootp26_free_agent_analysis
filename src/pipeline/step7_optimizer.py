"""
step7_optimizer.py — Step 7: Length optimizer + contract recommendations

Reads the surplus distribution table from step 6 (one row per player × candidate
length) and selects the optimal contract length per player by maximizing the
risk-adjusted objective:

    objective = mean_surplus − 1.15 × std_surplus

If the best objective is negative the model recommends "do not sign".

Inputs:
  intermediate/mc_surplus_distributions.csv

Outputs:
  intermediate/recommendations.csv  — one row per FA-eligible player
"""

import pandas as pd
import numpy as np
import os

RISK_LAMBDA = 1.15  # risk-aversion coefficient (spec-locked)
SURPLUS_CSV = "intermediate/mc_surplus_distributions.csv"
OUTPUT_CSV  = "intermediate/recommendations.csv"


def main():
    """Load the step 6 surplus distributions, pick the objective-maximizing contract length per player, and write the sign/do-not-sign recommendations CSV."""
    os.makedirs("intermediate", exist_ok=True)

    df = pd.read_csv(SURPLUS_CSV)
    print(f"Loaded {len(df):,} rows, {df['player_id'].nunique()} players")

    # --- sanity-check: objective already stored matches our formula ---
    check = (df["mean_surplus_M"] - RISK_LAMBDA * df["std_surplus_M"]).round(6)
    stored = df["objective"].round(6)
    mismatches = (check - stored).abs().gt(1e-4).sum()
    if mismatches:
        print(f"WARNING: {mismatches} objective values don't match formula — recomputing")
        df["objective"] = df["mean_surplus_M"] - RISK_LAMBDA * df["std_surplus_M"]
    else:
        print("Objective check passed (stored values match formula)")

    # --- per-player best length ---
    idx_best = df.groupby("player_id")["objective"].idxmax()
    best = df.loc[idx_best].copy()
    best = best.rename(columns={"candidate_length": "recommended_years"})

    # --- sign / do-not-sign decision ---
    best["sign"] = best["objective"].gt(0)
    best["recommendation"] = best.apply(
        lambda r: f"Sign {int(r['recommended_years'])}yr @ ${r['pred_aav_M']:.2f}M/yr"
        if r["sign"] else "Do not sign",
        axis=1,
    )

    # --- clean output column order ---
    out_cols = [
        "player_id", "player_name", "player_type", "age",
        "sign", "recommendation", "recommended_years", "pred_aav_M",
        "market_pred_years", "proj_rar",
        "mean_value_M", "mean_surplus_M", "std_surplus_M",
        "objective", "p_underperform",
        "pct5_surplus", "pct25_surplus", "pct75_surplus", "pct95_surplus",
    ]
    best = best[out_cols].sort_values(
        ["sign", "objective"], ascending=[False, False]
    ).reset_index(drop=True)

    best.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(best):,} recommendations -> {OUTPUT_CSV}")

    # --- summary ---
    sign_df   = best[best["sign"]]
    nosign_df = best[~best["sign"]]
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION SUMMARY  ({len(best)} FA-eligible players)")
    print(f"{'='*60}")
    print(f"  Sign:        {len(sign_df):>4}")
    print(f"  Do not sign: {len(nosign_df):>4}")

    if len(sign_df) > 0:
        print(f"\n--- Recommended signings (sorted by objective) ---")
        for _, r in sign_df.iterrows():
            ptype = r["player_type"]
            print(
                f"  {r['player_name']:<22}  {ptype:<7}  age {r['age']:.0f}  "
                f"RAR={r['proj_rar']:.1f}  "
                f"{int(r['recommended_years'])}yr @ ${r['pred_aav_M']:.2f}M  "
                f"EV=${r['mean_value_M']:.1f}M  ES=${r['mean_surplus_M']:.1f}M  "
                f"sd=${r['std_surplus_M']:.1f}M  obj={r['objective']:.3f}  "
                f"P(under)={r['p_underperform']:.1%}"
            )
    else:
        print("\n  No players recommended — entire FA class is priced above value.")

    # --- near-miss table: best objective by type ---
    print(f"\n--- Closest 'do not sign' by player type ---")
    for ptype in ["batter", "pitcher"]:
        top5 = (
            nosign_df[nosign_df["player_type"] == ptype]
            .nlargest(5, "objective")
        )
        if len(top5) == 0:
            continue
        print(f"\n  {ptype.capitalize()}s:")
        for _, r in top5.iterrows():
            print(
                f"    {r['player_name']:<22}  age {r['age']:.0f}  "
                f"RAR={r['proj_rar']:.1f}  "
                f"{int(r['recommended_years'])}yr @ ${r['pred_aav_M']:.2f}M  "
                f"obj={r['objective']:.3f}  P(under)={r['p_underperform']:.1%}"
            )


if __name__ == "__main__":
    main()
