#!/usr/bin/env python3
"""
Generates two artifacts that show ABSOLUTE fit quality for step 2 aging curves:

  intermediate/aging_fit_quality.csv
      One row per group x component. Columns:
        group, group_label, component, n_pairs, n_thin_ages,
        poly3_r2, poly3_cv_rmse, poly3_rmse,
        best_poly_model, best_poly_cv_rmse,
        quality_tier  (good / marginal / noise)
        note          (e.g. "LOESS preferred — thin tail ages")

  intermediate/viz/aging_r2_heatmap.png
      Heatmap of poly3 R² in absolute terms. Colors mapped to [0, 1]
      so you can see which fits are actually meaningful.

  intermediate/viz/aging_delta_fit_<component>.png   (12 files)
      For each component: one subplot per applicable group.
      Shows the RAW per-age mean deltas (dots + SE bars) against the
      poly3 fitted delta (blue line) and loess50 fitted delta (red line).
      This is the actual space where fitting happens — not the cumulative.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INTER = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"
VIZ   = os.path.join(INTER, "viz")
os.makedirs(VIZ, exist_ok=True)

fs   = pd.read_csv(os.path.join(INTER, "aging_fit_stats.csv"))
cs   = pd.read_csv(os.path.join(INTER, "aging_curves_smooth.csv"))
cell = pd.read_csv(os.path.join(INTER, "aging_cell_stats.csv"))

COMPONENT_LABELS = {
    "hr_pa":      "HR / PA",
    "xbh_pa":     "XBH / PA",
    "single_pa":  "Singles / PA",
    "bb_pa":      "BB / PA",
    "k_pa":       "K / PA",
    "ubr_g":      "UBR / game",
    "k_bf":       "K / BF",
    "bb_hbp_bf":  "(BB+HBP) / BF",
    "hra_bf":     "HR allowed / BF",
    "zr_rate":    "ZR (per 1000 IP)",
    "arm_rate":   "Arm (per 1000 IP)",
    "framing_rate": "Framing (per 1000 IP)",
}

GROUP_ORDER  = ["2","3","4","5","6","7","8","9","SP","RP"]
GROUP_LABELS = {
    "2":"C","3":"1B","4":"2B","5":"3B","6":"SS",
    "7":"LF","8":"CF","9":"RF","SP":"SP","RP":"RP"
}

# ── 1. Build fit_quality.csv ───────────────────────────────────────────────────
print("Building aging_fit_quality.csv…")

poly = fs[fs["model"].isin(["poly2","poly3","poly4"])].copy()

rows = []
for (grp, comp), g in poly.groupby(["group","component"]):
    p3 = g[g["model"] == "poly3"].iloc[0] if (g["model"] == "poly3").any() else None

    # best polynomial by CV RMSE
    valid = g.dropna(subset=["cv_rmse"])
    if len(valid):
        best_row  = valid.loc[valid["cv_rmse"].idxmin()]
        best_model = best_row["model"]
        best_cv   = best_row["cv_rmse"]
    else:
        best_model = best_cv = np.nan

    r2  = p3["r2"]  if p3 is not None else np.nan
    cvr = p3["cv_rmse"] if p3 is not None else np.nan
    rmse = p3["rmse"]  if p3 is not None else np.nan
    n_pairs     = int(p3["n_pairs"])     if p3 is not None else np.nan
    n_thin      = int(p3["n_thin_ages"]) if p3 is not None else np.nan

    # quality tier
    if np.isnan(r2):
        tier = "unknown"
    elif r2 >= 0.30:
        tier = "good"
    elif r2 >= 0.10:
        tier = "marginal"
    else:
        tier = "noise"

    # note
    note = ""
    if not np.isnan(n_thin) and n_thin >= 4:
        note = "many thin age-cells — prefer LOESS"
    if tier == "noise":
        note = ("weak age signal — " + note).strip(" —")
    if comp == "single_pa":
        note = "known weak signal per spec"

    rows.append({
        "group": grp,
        "group_label": GROUP_LABELS.get(str(grp), str(grp)),
        "component": comp,
        "n_pairs": n_pairs,
        "n_thin_ages": n_thin,
        "poly3_r2": round(r2, 4) if not np.isnan(r2) else np.nan,
        "poly3_cv_rmse": round(cvr, 6) if not np.isnan(cvr) else np.nan,
        "poly3_rmse": round(rmse, 6) if not np.isnan(rmse) else np.nan,
        "best_poly_model": best_model,
        "best_poly_cv_rmse": round(best_cv, 6) if not np.isnan(best_cv) else np.nan,
        "quality_tier": tier,
        "note": note,
    })

fq = pd.DataFrame(rows).sort_values(["component","group_label"])
out_csv = os.path.join(INTER, "aging_fit_quality.csv")
fq.to_csv(out_csv, index=False)
print(f"  {out_csv}")
print(f"  Tier counts:\n{fq['quality_tier'].value_counts().to_string()}")


# ── 2. Absolute R² heatmap ────────────────────────────────────────────────────
print("\nPlotting absolute R² heatmap…")

# Pivot: rows = group_label + component, cols = poly model
fq["row_label"] = fq["group_label"] + " | " + fq["component"]

pivot_r2 = fq.pivot_table(index="row_label", columns="component",
                           values="poly3_r2", aggfunc="first")

# Actually let's make a cleaner pivot: rows = group_label, cols = component
pivot2 = fq.pivot_table(index="group_label", columns="component",
                         values="poly3_r2", aggfunc="first")

# Order rows and cols
row_order = ["C","1B","2B","3B","SS","LF","CF","RF","SP","RP"]
col_order  = list(COMPONENT_LABELS.keys())
pivot2 = pivot2.reindex(index=[r for r in row_order if r in pivot2.index],
                        columns=[c for c in col_order if c in pivot2.columns])

n_rows, n_cols = pivot2.shape
fig, ax = plt.subplots(figsize=(n_cols * 1.3 + 2, n_rows * 0.8 + 1.5))

vals = pivot2.values.astype(float)
im = ax.imshow(np.where(np.isnan(vals), -1, vals),
               aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.6)

ax.set_xticks(range(n_cols))
ax.set_xticklabels([COMPONENT_LABELS.get(c, c) for c in pivot2.columns],
                    rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(n_rows))
ax.set_yticklabels(pivot2.index, fontsize=9)
ax.set_title("Poly3 R² — absolute fit quality\n"
             "(green ≥ 0.30 good · yellow ~0.15 marginal · red < 0.10 noise · grey = N/A)",
             fontsize=11)

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
cbar.set_label("R²", fontsize=9)
cbar.set_ticks([0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])

for i in range(n_rows):
    for j in range(n_cols):
        v = vals[i, j]
        if np.isnan(v):
            ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#aaaaaa")
        else:
            text_color = "black" if v < 0.45 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color,
                    fontweight="bold" if v >= 0.30 else "normal")

plt.tight_layout()
out = os.path.join(VIZ, "aging_r2_heatmap.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  {os.path.basename(out)}")


# ── 3. Delta-level residual plots ─────────────────────────────────────────────
print("\nPlotting delta-level fit plots…")

for comp, comp_label in COMPONENT_LABELS.items():
    sub_cs   = cs[cs["component"] == comp]
    sub_cell = cell[cell["component"] == comp]
    groups   = [g for g in GROUP_ORDER if g in sub_cs["group"].values]
    if not groups:
        continue

    ncols = min(4, len(groups))
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(f"Delta-level fit — {comp_label}\n"
                 f"(dots = per-age mean delta ± SE  |  blue = poly3  |  red = loess50)",
                 fontsize=12, y=1.01)

    for i, grp in enumerate(groups):
        r, c = divmod(i, ncols)
        ax   = axes[r][c]

        # cell stats (per-age summary of observed deltas)
        sc = sub_cell[sub_cell["group"] == grp].sort_values("age")
        # smooth curves
        sm = sub_cs[sub_cs["group"] == grp].sort_values("age")

        if sc.empty or sm.empty:
            ax.set_visible(False)
            continue

        ages       = sc["age"].values
        mean_delta = sc["weighted_mean"].fillna(0).values
        se_delta   = sc["se_delta"].fillna(0).values
        flag_thin  = sc["flag_thin"].fillna(0).astype(int).values

        # shade thin ages
        for age, thin in zip(ages, flag_thin):
            if thin:
                ax.axvspan(age - 0.5, age + 0.5, color="#eeeeee", zorder=0)

        ax.axhline(0, color="black", lw=0.5, zorder=1)

        # raw observed mean delta + SE bars
        ax.errorbar(ages, mean_delta, yerr=se_delta,
                    fmt="o", ms=4, color="black", alpha=0.55,
                    elinewidth=0.8, capsize=2, zorder=3, label="obs wt mean ± SE")

        # poly3 fitted delta
        if "poly3_delta" in sm.columns:
            ax.plot(sm["age"], sm["poly3_delta"].astype(float),
                    color="#1f77b4", lw=2.0, label="poly3 fit", zorder=4)

        # loess50 fitted delta
        if "loess50_delta" in sm.columns:
            ax.plot(sm["age"], sm["loess50_delta"].astype(float),
                    color="#d62728", lw=1.5, ls="--", label="loess50 fit", zorder=4)

        # R² annotation
        r2_row = fq[(fq["group"] == grp) & (fq["component"] == comp)]
        r2_val = r2_row["poly3_r2"].values[0] if len(r2_row) else np.nan
        tier   = r2_row["quality_tier"].values[0] if len(r2_row) else ""
        tier_color = {"good": "#2ca02c", "marginal": "#ff7f0e",
                      "noise": "#d62728", "unknown": "gray"}.get(tier, "gray")
        ax.set_title(f"{GROUP_LABELS.get(grp, grp)}  —  R²={r2_val:.3f}  [{tier}]",
                     fontsize=9, color=tier_color, pad=3)
        ax.set_xlim(19.5, 40.5)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Age", fontsize=7)
        ax.set_ylabel("Δ per year", fontsize=7)

    for j in range(len(groups), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    # shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="black", ls="", marker="o", ms=4, alpha=0.55, label="obs wt mean ± SE"),
        Line2D([], [], color="#1f77b4", lw=2.0, label="poly3 fit"),
        Line2D([], [], color="#d62728", lw=1.5, ls="--", label="loess50 fit"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), fontsize=8)

    plt.tight_layout()
    out = os.path.join(VIZ, f"aging_delta_fit_{comp}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(out)}")

print(f"\nDone. New files in {INTER} and {VIZ}")
