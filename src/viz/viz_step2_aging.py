#!/usr/bin/env python3
"""
Visualizations for Step 2 aging curves.
Produces PNGs in intermediate/viz/.

Figures:
  1–12  aging_<component>.png       — cumulative curves, all groups, per component
  13    aging_model_fit_heatmap.png — CV RMSE color comparison across models
  14-16 aging_threshold_sensitivity_<label>.png — n_pairs / R² vs threshold
  17-19 aging_sample_coverage_<label>.png       — n_pairs by age bar charts
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Paths ─────────────────────────────────────────────────────────────────────
INTER = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"
VIZ   = os.path.join(INTER, "viz")
os.makedirs(VIZ, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
cs    = pd.read_csv(os.path.join(INTER, "aging_curves_smooth.csv"))
fs    = pd.read_csv(os.path.join(INTER, "aging_fit_stats.csv"))
cell  = pd.read_csv(os.path.join(INTER, "aging_cell_stats.csv"))
thresh = pd.read_csv(os.path.join(INTER, "aging_threshold_sens.csv"))

# ── Labels / ordering ─────────────────────────────────────────────────────────
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

# model appearance: color, linestyle, alpha, linewidth
MODEL_STYLE = {
    "poly2":   ("#aaaaaa", "--", 0.65, 1.2),
    "poly3":   ("#1f77b4", "-",  1.0,  2.5),
    "poly4":   ("#aec7e8", ":",  0.75, 1.2),
    "loess30": ("#ffa040", "--", 0.65, 1.2),
    "loess50": ("#d62728", "-",  1.0,  2.0),
    "loess70": ("#9467bd", ":",  0.85, 1.5),
}

CUM_COLS = {m: f"{m}_cumulative" for m in MODEL_STYLE}


# ── Helpers ───────────────────────────────────────────────────────────────────

def shade_thin(ax, ages, flag_thin):
    """Shade a vertical band on ax for each age flagged as thin-sample."""
    for age, thin in zip(ages, flag_thin):
        if thin:
            ax.axvspan(age - 0.5, age + 0.5, color="#eeeeee", zorder=0)


def plot_panel(ax, sub, title):
    """Render a single group×component aging-curve panel."""
    sub = sub.sort_values("age")
    ages     = sub["age"].values
    flag_thin = sub["flag_thin"].fillna(0).astype(int).values

    shade_thin(ax, ages, flag_thin)
    ax.axhline(0, color="black", lw=0.6, zorder=1)

    # Raw weighted mean cumulative (anchored at 0 at age 20)
    raw_cum = sub["weighted_mean"].fillna(0).cumsum().values
    raw_cum = raw_cum - raw_cum[0]
    ax.scatter(ages, raw_cum, s=9, color="black", alpha=0.45, zorder=4, label="raw")

    # Smoothers
    for model, (color, ls, alpha, lw) in MODEL_STYLE.items():
        col = CUM_COLS[model]
        if col not in sub.columns:
            continue
        vals = sub[col].values.astype(float)
        if np.all(np.isnan(vals)):
            continue
        ax.plot(ages, vals, color=color, ls=ls, alpha=alpha, lw=lw,
                label=model, zorder=3)

    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlim(19.5, 40.5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Age", fontsize=7)
    ax.set_ylabel("Cumulative Δ", fontsize=7)

    total_n = int(sub["n_pairs"].sum())
    ax.text(0.97, 0.03, f"Σn={total_n:,}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color="gray")


def make_legend_handles():
    """Build the shared legend handles (one per smoother model plus the raw-mean marker)."""
    handles = [
        mlines.Line2D([], [], color=col, ls=ls, lw=lw, alpha=alpha, label=m)
        for m, (col, ls, alpha, lw) in MODEL_STYLE.items()
    ]
    handles.append(
        mlines.Line2D([], [], color="black", ls="", marker="o",
                      markersize=4, alpha=0.5, label="raw wt mean")
    )
    return handles


# ── Figures 1–12: aging curves per component ──────────────────────────────────
print("Plotting aging curves per component…")
for comp, comp_label in COMPONENT_LABELS.items():
    sub_all = cs[cs["component"] == comp]
    groups  = [g for g in GROUP_ORDER if g in sub_all["group"].values]
    if not groups:
        continue

    ncols = min(4, len(groups))
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.6 * nrows),
                             squeeze=False)
    fig.suptitle(f"Aging curves — {comp_label}", fontsize=13, y=1.01)

    for i, grp in enumerate(groups):
        r, c = divmod(i, ncols)
        ax   = axes[r][c]
        sub  = sub_all[sub_all["group"] == grp]
        plot_panel(ax, sub, GROUP_LABELS.get(grp, grp))

    for j in range(len(groups), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.legend(handles=make_legend_handles(),
               loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.04), fontsize=8)

    plt.tight_layout()
    out = os.path.join(VIZ, f"aging_{comp}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(out)}")


# ── Figure 13: Model fit heatmap (CV RMSE) ────────────────────────────────────
print("Plotting model fit heatmap…")

# Build group|component label for axis
fs["row_label"] = fs["group_label"] + " | " + fs["component"]

# Pivot: rows = (group, component), cols = model
pivot = fs.pivot_table(
    index="row_label", columns="model", values="cv_rmse", aggfunc="first"
)

# R² pivot (for polynomial models only; LOESS lacks R²)
r2_pivot = fs.pivot_table(
    index="row_label", columns="model", values="r2", aggfunc="first"
)

# Normalize each row: 0 = best CV RMSE, 1 = worst
row_min  = pivot.min(axis=1)
row_max  = pivot.max(axis=1)
norm     = pivot.sub(row_min, axis=0).div((row_max - row_min).replace(0, np.nan), axis=0)

n_rows, n_cols = norm.shape
fig, ax = plt.subplots(figsize=(n_cols * 1.6 + 1, n_rows * 0.45 + 2))

im = ax.imshow(norm.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
ax.set_xticks(range(n_cols))
ax.set_xticklabels(norm.columns, fontsize=9)
ax.set_yticks(range(n_rows))
ax.set_yticklabels(norm.index, fontsize=7)
ax.set_title("Model fit — CV RMSE\n(green = best per row, red = worst)", fontsize=11)
plt.colorbar(im, ax=ax, label="Relative excess CV RMSE (0=best)", fraction=0.03)

for i in range(n_rows):
    for j in range(n_cols):
        raw = pivot.values[i, j]
        if not np.isnan(raw):
            ax.text(j, i, f"{raw:.4f}", ha="center", va="center", fontsize=5.5,
                    color="black" if norm.values[i, j] < 0.8 else "white")

plt.tight_layout()
out = os.path.join(VIZ, "aging_model_fit_heatmap.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  {os.path.basename(out)}")


# ── Figures 14–16: Threshold sensitivity ──────────────────────────────────────
print("Plotting threshold sensitivity…")

THRESH_COL = {"batting": "min_pa", "pitching": "min_bf", "fielding": "min_ip"}

for label in thresh["label"].unique():
    sub     = thresh[thresh["label"] == label]
    comps   = [c for c in COMPONENT_LABELS if c in sub["component"].values]
    groups  = sorted(sub["group_label"].unique())
    tcol    = THRESH_COL.get(label, "min_pa")

    n_comps = len(comps)
    if n_comps == 0:
        continue

    fig, axes = plt.subplots(2, n_comps,
                             figsize=(3.8 * n_comps, 6.5),
                             squeeze=False)
    fig.suptitle(f"Threshold sensitivity — {label}", fontsize=12)

    for ci, comp in enumerate(comps):
        ax_n  = axes[0][ci]
        ax_r2 = axes[1][ci]
        for grp in groups:
            s = sub[(sub["component"] == comp) & (sub["group_label"] == grp)].sort_values(tcol)
            if s.empty:
                continue
            ax_n.plot(s[tcol], s["n_pairs"], marker="o", ms=4, label=grp)
            ax_r2.plot(s[tcol], s["poly3_r2"], marker="o", ms=4, label=grp)

        ax_n.set_title(COMPONENT_LABELS.get(comp, comp), fontsize=9)
        ax_n.set_ylabel("n_pairs", fontsize=8)
        ax_n.tick_params(labelsize=7)
        ax_r2.set_ylabel("poly3 R²", fontsize=8)
        ax_r2.set_xlabel(tcol, fontsize=8)
        ax_r2.tick_params(labelsize=7)
        ax_r2.set_ylim(-0.05, 1.05)

    axes[0][0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out = os.path.join(VIZ, f"aging_threshold_sensitivity_{label}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(out)}")


# ── Figures 17–19: Sample coverage (n_pairs by age) ───────────────────────────
print("Plotting sample coverage…")

COVERAGE_GROUPS = {
    "batting":  (["2","3","4","5","6","7","8","9"],
                 ["hr_pa","xbh_pa","single_pa","bb_pa","k_pa","ubr_g"]),
    "pitching": (["SP","RP"],
                 ["k_bf","bb_hbp_bf","hra_bf"]),
    "fielding": (["2","3","4","5","6","7","8","9"],
                 ["zr_rate","arm_rate","framing_rate"]),
}

for type_label, (groups, comps) in COVERAGE_GROUPS.items():
    n_grp  = len(groups)
    n_comp = len(comps)
    fig, axes = plt.subplots(n_grp, n_comp,
                             figsize=(3.4 * n_comp, 2.4 * n_grp),
                             squeeze=False)
    fig.suptitle(f"Sample coverage (n_pairs by age) — {type_label}", fontsize=12)

    for ri, grp in enumerate(groups):
        for ci, comp in enumerate(comps):
            ax  = axes[ri][ci]
            sub = cell[(cell["group"] == grp) & (cell["component"] == comp)].sort_values("age")
            if sub.empty:
                ax.set_visible(False)
                continue

            bar_colors = ["#e06060" if t else "#6699cc"
                          for t in sub["flag_thin"].fillna(0).astype(int)]
            ax.bar(sub["age"], sub["n_pairs"], color=bar_colors, width=0.8, edgecolor="none")
            ax.set_title(
                f"{GROUP_LABELS.get(grp, grp)} | {COMPONENT_LABELS.get(comp, comp)}",
                fontsize=7
            )
            ax.tick_params(labelsize=6)
            ax.set_xlabel("Age", fontsize=6)
            ax.set_ylabel("n_pairs", fontsize=6)

    # Shared legend for thin flag
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color="#6699cc", label="normal"),
                 Patch(color="#e06060", label="thin (flag)")],
        loc="lower center", ncol=2, fontsize=8,
        bbox_to_anchor=(0.5, -0.02)
    )
    plt.tight_layout()
    out = os.path.join(VIZ, f"aging_sample_coverage_{type_label}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(out)}")


print(f"\nDone — all plots in {VIZ}")
