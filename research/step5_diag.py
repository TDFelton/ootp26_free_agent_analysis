"""
Step 5 diagnostics: extended hyperparameter tuning + 10 interpretability plots.

Inputs:  intermediate/market_training_data.csv
         intermediate/market_model_coefficients.csv
Outputs: intermediate/viz/step5_*.png  (10 files)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
INT = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"
VIZ = os.path.join(INT, "viz")
os.makedirs(VIZ, exist_ok=True)

def save(name):
    """Save the current matplotlib figure to intermediate/viz/<name> and close it."""
    path = os.path.join(VIZ, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    kb = os.path.getsize(path) // 1024
    print(f"  Saved {name}  ({kb} KB)")

# ── Style ──────────────────────────────────────────────────────────────────────
BAT_C  = "#1565C0"
PIT_C  = "#BF360C"
POS_C  = "#2E7D32"
NEG_C  = "#C62828"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "DejaVu Sans",
})

BAT_FEATURES = ["proj_rar", "age", "proj_pa", "is_premium_def"]
PIT_FEATURES = ["proj_rar", "age", "proj_ip", "sp_flag"]
BAT_LABELS   = ["Proj RAR", "Age", "Proj PA", "Premium Def\n(C/SS)"]
PIT_LABELS   = ["Proj RAR", "Age", "Proj IP", "SP Flag"]
ORIG_ALPHA   = {"batter": 5.0, "pitcher": 1.0}

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5 DIAGNOSTICS")
print("=" * 60)
print("\n[1] Loading data...")

train_df = pd.read_csv(os.path.join(INT, "market_training_data.csv"))

bat_df = train_df[train_df["player_type"] == "batter"].copy().reset_index(drop=True)
pit_df = train_df[train_df["player_type"] == "pitcher"].copy().reset_index(drop=True)

bat_df["is_premium_def"] = bat_df["is_premium_def"].fillna(0).astype(float)
pit_df["sp_flag"]        = pit_df["sp_flag"].fillna(0).astype(float)

X_bat     = bat_df[BAT_FEATURES].values.astype(float)
y_bat_aav = bat_df["log_aav"].values.astype(float)
y_bat_yrs = bat_df["years"].values.astype(float)

X_pit     = pit_df[PIT_FEATURES].values.astype(float)
y_pit_aav = pit_df["log_aav"].values.astype(float)
y_pit_yrs = pit_df["years"].values.astype(float)

n_bat, n_pit = len(bat_df), len(pit_df)
print(f"  Batters: {n_bat}  Pitchers: {n_pit}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. EXTENDED LOO-CV
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Extended LOO-CV alpha search (150 points, log space 1e-3 to 1e4)...")

def run_loo_cv(X_raw, y_aav, y_yrs, alpha_grid):
    """Leave-one-out CV over a ridge alpha grid; returns per-alpha normalized MSE for log(AAV) and years, plus their sum."""
    sc  = StandardScaler()
    X   = sc.fit_transform(X_raw)
    var_a = np.var(y_aav, ddof=1)
    var_y = np.var(y_yrs, ddof=1)
    loo = LeaveOneOut()
    rows = []
    for alpha in alpha_grid:
        m    = Ridge(alpha=alpha)
        ea, ey = [], []
        for tr, te in loo.split(X):
            m.fit(X[tr], y_aav[tr]); ea.append((y_aav[te[0]] - m.predict(X[te])[0])**2)
            m.fit(X[tr], y_yrs[tr]); ey.append((y_yrs[te[0]] - m.predict(X[te])[0])**2)
        mse_a = np.mean(ea) / max(var_a, 1e-12)
        mse_y = np.mean(ey) / max(var_y, 1e-12)
        rows.append({"alpha": alpha, "mse_aav": mse_a, "mse_yrs": mse_y,
                     "combined": mse_a + mse_y})
    return pd.DataFrame(rows)

ALPHA_FINE = np.logspace(-3, 4, 150)
bat_cv = run_loo_cv(X_bat, y_bat_aav, y_bat_yrs, ALPHA_FINE)
pit_cv = run_loo_cv(X_pit, y_pit_aav, y_pit_yrs, ALPHA_FINE)

bat_best = float(bat_cv.loc[bat_cv["combined"].idxmin(), "alpha"])
pit_best = float(pit_cv.loc[pit_cv["combined"].idxmin(), "alpha"])
print(f"  Batter best alpha: {bat_best:.3f}  (orig {ORIG_ALPHA['batter']})")
print(f"  Pitcher best alpha: {pit_best:.3f}  (orig {ORIG_ALPHA['pitcher']})")

# Save tuning table
tuning_rows = []
for _, row in bat_cv.iterrows():
    tuning_rows.append({"player_type": "batter", **row})
for _, row in pit_cv.iterrows():
    tuning_rows.append({"player_type": "pitcher", **row})
pd.DataFrame(tuning_rows).to_csv(os.path.join(INT, "step5_hp_tuning.csv"), index=False)
print("  Saved step5_hp_tuning.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT MODELS WITH BEST ALPHA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Fitting models with extended best alpha...")

def fit_pair(X_raw, y_aav, y_yrs, alpha):
    """Standardize features and fit a ridge model each for log(AAV) and years at a given alpha."""
    sc    = StandardScaler()
    Xs    = sc.fit_transform(X_raw)
    m_aav = Ridge(alpha=alpha).fit(Xs, y_aav)
    m_yrs = Ridge(alpha=alpha).fit(Xs, y_yrs)
    return sc, m_aav, m_yrs, Xs

bat_sc, bat_aav_m, bat_yrs_m, X_bat_sc = fit_pair(X_bat, y_bat_aav, y_bat_yrs, bat_best)
pit_sc, pit_aav_m, pit_yrs_m, X_pit_sc = fit_pair(X_pit, y_pit_aav, y_pit_yrs, pit_best)

bat_df["pred_log_aav"] = bat_aav_m.predict(X_bat_sc)
bat_df["pred_years"]   = bat_yrs_m.predict(X_bat_sc)
bat_df["resid_aav"]    = bat_df["log_aav"] - bat_df["pred_log_aav"]
bat_df["resid_yrs"]    = bat_df["years"]   - bat_df["pred_years"]
bat_df["pred_aav_M"]   = np.exp(bat_df["pred_log_aav"]) / 1e6
bat_df["actual_aav_M"] = bat_df["salary0"] / 1e6

pit_df["pred_log_aav"] = pit_aav_m.predict(X_pit_sc)
pit_df["pred_years"]   = pit_yrs_m.predict(X_pit_sc)
pit_df["resid_aav"]    = pit_df["log_aav"] - pit_df["pred_log_aav"]
pit_df["resid_yrs"]    = pit_df["years"]   - pit_df["pred_years"]
pit_df["pred_aav_M"]   = np.exp(pit_df["pred_log_aav"]) / 1e6
pit_df["actual_aav_M"] = pit_df["salary0"] / 1e6

r2 = lambda y, yhat: 1 - np.var(y - yhat) / np.var(y)
print(f"  Batter:  log(AAV) R²={r2(y_bat_aav, bat_df['pred_log_aav'].values):.3f}  "
      f"years R²={r2(y_bat_yrs, bat_df['pred_years'].values):.3f}")
print(f"  Pitcher: log(AAV) R²={r2(y_pit_aav, pit_df['pred_log_aav'].values):.3f}  "
      f"years R²={r2(y_pit_yrs, pit_df['pred_years'].values):.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. BOOTSTRAP COEFS (500 iterations)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Bootstrapping (n=500) for feature importance CI...")

def bootstrap_coefs(X_raw, y, alpha, n_boot=500, seed=42):
    """Bootstrap-resample the training rows n_boot times and refit a ridge model each time, returning the stacked coefficient array for CI estimation."""
    rng   = np.random.default_rng(seed)
    n     = len(y)
    coefs = np.zeros((n_boot, X_raw.shape[1]))
    for i in range(n_boot):
        idx       = rng.integers(0, n, size=n)
        sc        = StandardScaler()
        Xs        = sc.fit_transform(X_raw[idx])
        coefs[i]  = Ridge(alpha=alpha).fit(Xs, y[idx]).coef_
    return coefs

bat_boot_aav = bootstrap_coefs(X_bat, y_bat_aav, bat_best)
bat_boot_yrs = bootstrap_coefs(X_bat, y_bat_yrs, bat_best)
pit_boot_aav = bootstrap_coefs(X_pit, y_pit_aav, pit_best)
pit_boot_yrs = bootstrap_coefs(X_pit, y_pit_yrs, pit_best)
print("  Done.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. RIDGE LEVERAGES
# ══════════════════════════════════════════════════════════════════════════════
def ridge_leverage(Xs, alpha):
    """Compute the diagonal of the ridge hat matrix (per-row leverage) for the standardized design matrix Xs."""
    p = Xs.shape[1]
    H = Xs @ np.linalg.inv(Xs.T @ Xs + alpha * np.eye(p)) @ Xs.T
    return np.diag(H)

bat_lev = ridge_leverage(X_bat_sc, bat_best)
pit_lev = ridge_leverage(X_pit_sc, pit_best)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
print("\n[P1] Feature importance...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Step 5 — Feature Importance (Standardized Coefficients)\n"
             "Bootstrap 95% CI  |  Blue = positive, Red = negative",
             fontsize=12, fontweight="bold")

configs = [
    (axes[0, 0], bat_aav_m.coef_, bat_boot_aav, BAT_LABELS, "Batters — log(AAV)", BAT_C),
    (axes[0, 1], bat_yrs_m.coef_, bat_boot_yrs, BAT_LABELS, "Batters — Years",    BAT_C),
    (axes[1, 0], pit_aav_m.coef_, pit_boot_aav, PIT_LABELS, "Pitchers — log(AAV)", PIT_C),
    (axes[1, 1], pit_yrs_m.coef_, pit_boot_yrs, PIT_LABELS, "Pitchers — Years",   PIT_C),
]
for ax, coef, boot, labels, title, color in configs:
    ci_lo = np.percentile(boot, 2.5,  axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)
    yerr  = np.array([coef - ci_lo, ci_hi - coef])
    colors = [POS_C if c > 0 else NEG_C for c in coef]
    ax.barh(range(len(labels)), coef, xerr=yerr, color=colors,
            alpha=0.8, capsize=4, error_kw={"linewidth": 1.5})
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Coefficient (std. units)")
    for i, (c, lo, hi) in enumerate(zip(coef, ci_lo, ci_hi)):
        x_pos = hi + 0.005 if c >= 0 else lo - 0.005
        ha    = "left" if c >= 0 else "right"
        ax.text(x_pos, i, f"{c:+.3f}", va="center", ha=ha, fontsize=7)

plt.tight_layout()
save("step5_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Loss Curves (Extended Alpha Grid)
# ══════════════════════════════════════════════════════════════════════════════
print("[P2] Loss curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Step 5 — LOO-CV Loss vs Alpha (150-point log grid)\n"
             "Green dashed = extended best  |  Red dotted = original best",
             fontsize=12, fontweight="bold")

for ax, cv_df, best_a, orig_a, label, color, n in [
    (axes[0], bat_cv, bat_best, ORIG_ALPHA["batter"],  "Batters",  BAT_C, n_bat),
    (axes[1], pit_cv, pit_best, ORIG_ALPHA["pitcher"], "Pitchers", PIT_C, n_pit),
]:
    ax.semilogx(cv_df["alpha"], cv_df["mse_aav"],   color=color,    lw=2,   label="MSE log(AAV) [norm]",  alpha=0.9)
    ax.semilogx(cv_df["alpha"], cv_df["mse_yrs"],   color="orange", lw=2,   label="MSE years [norm]",     alpha=0.9)
    ax.semilogx(cv_df["alpha"], cv_df["combined"],  color="black",  lw=2.5, label="Combined (sum)",       alpha=1.0)
    ax.axvline(best_a, color="green", lw=2,   ls="--", label=f"Extended best α={best_a:.2f}")
    ax.axvline(orig_a, color="red",   lw=1.5, ls=":",  label=f"Original best α={orig_a:.0f}")

    # Shade region within 2% of optimal
    thr = cv_df["combined"].min() * 1.02
    within = cv_df[cv_df["combined"] <= thr]
    if len(within) > 1:
        ax.axvspan(within["alpha"].min(), within["alpha"].max(),
                   alpha=0.08, color="green", label="Within 2% of optimal")

    ax.set_title(f"{label} (n={n})", fontweight="bold")
    ax.set_xlabel("Alpha (regularization strength)")
    ax.set_ylabel("Normalized LOO-CV MSE")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(bottom=0)

plt.tight_layout()
save("step5_loss_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("[P3] Correlation heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Step 5 — Feature + Target Correlation Matrix\n"
             "Bold blue = target variables  |  lower triangle only",
             fontsize=12, fontweight="bold")

bat_corr_df = bat_df[BAT_FEATURES + ["log_aav", "years"]].copy()
bat_corr_df.columns = ["Proj RAR", "Age", "Proj PA", "Premium Def", "log(AAV)", "Years"]

pit_corr_df = pit_df[PIT_FEATURES + ["log_aav", "years"]].copy()
pit_corr_df.columns = ["Proj RAR", "Age", "Proj IP", "SP Flag", "log(AAV)", "Years"]

for ax, cdf, title in [
    (axes[0], bat_corr_df, f"Batters (n={n_bat})"),
    (axes[1], pit_corr_df, f"Pitchers (n={n_pit})"),
]:
    corr = cdf.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
                annot_kws={"size": 9}, cmap="RdBu_r", vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})
    ax.set_title(title, fontweight="bold")
    # Bold the target row/col tick labels
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        if tl.get_text() in ("log(AAV)", "Years"):
            tl.set_fontweight("bold")
            tl.set_color("darkblue")

plt.tight_layout()
save("step5_corr_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Residual Heatmap (Age × RAR bins)
# ══════════════════════════════════════════════════════════════════════════════
print("[P4] Residual heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Step 5 — Mean log(AAV) Residual Heatmap\n"
             "Red = model over-predicts  |  Blue = model under-predicts",
             fontsize=12, fontweight="bold")

age_bins = [20, 27, 30, 33, 36, 99]
age_lbls = ["≤26", "27–29", "30–32", "33–35", "36+"]

for ax, df, title in [
    (axes[0], bat_df, f"Batters (n={n_bat})"),
    (axes[1], pit_df, f"Pitchers (n={n_pit})"),
]:
    df2 = df.copy()
    rar_vals = df2["proj_rar"].values
    rar_bins = np.percentile(rar_vals, [0, 25, 50, 75, 100])
    rar_bins[0] -= 0.01  # ensure inclusion of minimum
    rar_lbls = ["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"]

    df2["age_bin"] = pd.cut(df2["age"], bins=age_bins, labels=age_lbls, right=False)
    df2["rar_bin"] = pd.cut(df2["proj_rar"], bins=rar_bins, labels=rar_lbls, include_lowest=True)

    pivot = df2.groupby(["rar_bin", "age_bin"], observed=False)["resid_aav"].mean().unstack()
    count = df2.groupby(["rar_bin", "age_bin"], observed=False)["resid_aav"].count().unstack()

    # Build annotation strings
    annot = np.full(pivot.shape, "", dtype=object)
    for ri, row in enumerate(pivot.index):
        for ci, col in enumerate(pivot.columns):
            v = pivot.iloc[ri, ci]
            n = count.iloc[ri, ci] if ri < len(count) and ci < len(count.columns) else 0
            if pd.notna(v) and n > 0:
                annot[ri, ci] = f"{v:+.2f}\n(n={int(n)})"
            else:
                annot[ri, ci] = "–"

    absmax = max(np.nanmax(np.abs(pivot.values)), 0.01)
    sns.heatmap(pivot, ax=ax, annot=annot, fmt="",
                annot_kws={"size": 8}, cmap="RdBu_r", center=0,
                vmin=-absmax, vmax=absmax, linewidths=0.5,
                cbar_kws={"label": "Mean residual (log AAV)"})
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Age at Signing")
    ax.set_ylabel("Proj RAR Quartile")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
save("step5_resid_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Predicted vs Actual (AAV + Years, 2×2)
# ══════════════════════════════════════════════════════════════════════════════
print("[P5] Predicted vs actual...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Step 5 — Predicted vs Actual (In-Sample)\nDashed = ±15% band for AAV panels",
             fontsize=12, fontweight="bold")

def pred_vs_actual_panel(ax, y_pred, y_actual, title, color, is_aav=True, label_unit=""):
    """Draw a single predicted-vs-actual scatter panel on ax, with a 1:1 reference line and (for AAV panels) a dashed +-15% band."""
    ax.scatter(y_pred, y_actual, color=color, alpha=0.55, s=40,
               edgecolors="white", linewidths=0.4)
    lo = min(y_pred.min(), y_actual.min())
    hi = max(y_pred.max(), y_actual.max())
    pad = (hi - lo) * 0.05
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "k-", lw=1.5, label="Perfect")
    if is_aav:
        ax.plot(lim, [v * 0.85 for v in lim], "--", color="gray", lw=1, alpha=0.6)
        ax.plot(lim, [v * 1.15 for v in lim], "--", color="gray", lw=1, alpha=0.6, label="±15%")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Predicted{label_unit}")
    ax.set_ylabel(f"Actual{label_unit}")
    ax.set_title(title, fontweight="bold")
    ss_res = np.sum((y_actual - y_pred)**2)
    ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
    r2_val = 1 - ss_res / max(ss_tot, 1e-12)
    txt = f"R²={r2_val:.3f}"
    if is_aav:
        within15 = np.mean(np.abs(y_actual - y_pred) / np.abs(y_pred + 1e-9) <= 0.15)
        txt += f"\n±15%: {within15:.0%}"
        rmse = np.sqrt(np.mean((np.log(y_actual + 1e-9) - np.log(y_pred + 1e-9))**2))
        txt += f"\nRMSE log: {rmse:.3f}"
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.legend(fontsize=7, loc="lower right")

pred_vs_actual_panel(axes[0, 0], bat_df["pred_aav_M"].values, bat_df["actual_aav_M"].values,
                     f"Batters — AAV ($M)", BAT_C, is_aav=True, label_unit=" AAV ($M)")
pred_vs_actual_panel(axes[0, 1], pit_df["pred_aav_M"].values, pit_df["actual_aav_M"].values,
                     f"Pitchers — AAV ($M)", PIT_C, is_aav=True, label_unit=" AAV ($M)")
pred_vs_actual_panel(axes[1, 0], bat_df["pred_years"].values, bat_df["years"].values.astype(float),
                     "Batters — Years", BAT_C, is_aav=False, label_unit=" Years")
pred_vs_actual_panel(axes[1, 1], pit_df["pred_years"].values, pit_df["years"].values.astype(float),
                     "Pitchers — Years", PIT_C, is_aav=False, label_unit=" Years")

plt.tight_layout()
save("step5_pred_vs_actual.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Residual Diagnostics (4-panel × 2 player types)
# ══════════════════════════════════════════════════════════════════════════════
print("[P6] Residual diagnostics...")
fig, axes = plt.subplots(2, 4, figsize=(19, 8))
fig.suptitle("Step 5 — Residual Diagnostics (log AAV model)\nTop: Batters  |  Bottom: Pitchers",
             fontsize=12, fontweight="bold")

for ri, (df, lev, label, color, feats) in enumerate([
    (bat_df, bat_lev, "Batters",  BAT_C, BAT_FEATURES),
    (pit_df, pit_lev, "Pitchers", PIT_C, PIT_FEATURES),
]):
    resid  = df["resid_aav"].values
    fitted = df["pred_log_aav"].values
    p      = len(feats)
    n      = len(resid)
    mse    = np.mean(resid ** 2)
    std_r  = (resid - resid.mean()) / max(resid.std(), 1e-9)
    x_line = np.linspace(fitted.min(), fitted.max(), 100)

    # Panel 0: Residuals vs Fitted
    ax = axes[ri, 0]
    ax.scatter(fitted, resid, color=color, alpha=0.55, s=35)
    ax.axhline(0, color="black", lw=1)
    z = np.polyfit(fitted, resid, 1)
    ax.plot(x_line, np.polyval(z, x_line), "r--", lw=1.5, label="Trend")
    ax.set_xlabel("Fitted log(AAV)"); ax.set_ylabel("Residual")
    ax.set_title(f"{label}: Residuals vs Fitted"); ax.legend(fontsize=7)

    # Panel 1: Q-Q
    ax = axes[ri, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, color=color, alpha=0.55, s=35)
    ax.plot([osm.min(), osm.max()],
            [slope * osm.min() + intercept, slope * osm.max() + intercept],
            "r-", lw=1.5)
    w_stat, w_p = stats.shapiro(resid)
    ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"{label}: Normal Q-Q")
    ax.text(0.05, 0.95, f"Shapiro W={w_stat:.3f}\np={w_p:.3f}",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Panel 2: Scale-Location
    ax = axes[ri, 2]
    sqrt_abs = np.sqrt(np.abs(std_r))
    ax.scatter(fitted, sqrt_abs, color=color, alpha=0.55, s=35)
    z2 = np.polyfit(fitted, sqrt_abs, 1)
    ax.plot(x_line, np.polyval(z2, x_line), "r--", lw=1.5)
    ax.set_xlabel("Fitted log(AAV)"); ax.set_ylabel("√|Standardized Residual|")
    ax.set_title(f"{label}: Scale-Location")

    # Panel 3: Leverage vs Residual (influence map)
    ax = axes[ri, 3]
    sc_plot = ax.scatter(lev, resid, c=np.abs(resid), cmap="YlOrRd", s=40, zorder=5)
    plt.colorbar(sc_plot, ax=ax, label="|Residual|", shrink=0.8)
    ax.axhline(0, color="black", lw=1)
    hline = 2 * p / n
    ax.axvline(hline, color="gray", lw=1.5, ls="--", label=f"High lev (2p/n={hline:.3f})")
    # Cook's D contours
    for cd_thr in [0.5, 1.0]:
        hv = np.linspace(1e-4, min(lev.max() * 1.05, 0.999), 200)
        rv = np.sqrt(cd_thr * p * mse * (1 - hv)**2 / np.maximum(hv, 1e-9))
        ax.plot(hv, rv,  "r--", lw=0.8, alpha=0.6)
        ax.plot(hv, -rv, "r--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Leverage (ridge hat matrix)"); ax.set_ylabel("Residual log(AAV)")
    ax.set_title(f"{label}: Leverage–Residual (Cook's D contours at 0.5, 1.0)")
    ax.legend(fontsize=7)

plt.tight_layout()
save("step5_resid_diagnostics.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7: Partial Dependence Plots
# ══════════════════════════════════════════════════════════════════════════════
print("[P7] Partial dependence...")

def partial_dep(X_raw, model, sc, feat_idx, n_pts=100):
    """Sweep one feature across its observed range (holding others at their median) and return the fitted model's predicted curve."""
    X_med = np.median(X_raw, axis=0)
    feat_range = np.linspace(X_raw[:, feat_idx].min(), X_raw[:, feat_idx].max(), n_pts)
    preds = []
    for v in feat_range:
        Xtmp = X_med.copy(); Xtmp[feat_idx] = v
        preds.append(model.predict(sc.transform(Xtmp.reshape(1, -1)))[0])
    return feat_range, np.array(preds)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Step 5 — Partial Dependence (log AAV)\n"
             "All other features at training median  |  Rug = actual data distribution",
             fontsize=12, fontweight="bold")

for ri, (feats, sc, model, labels, df, color, row_lbl) in enumerate([
    (BAT_FEATURES, bat_sc, bat_aav_m, BAT_LABELS, bat_df, BAT_C, "Batters"),
    (PIT_FEATURES, pit_sc, pit_aav_m, PIT_LABELS, pit_df, PIT_C, "Pitchers"),
]):
    X_raw = df[feats].values.astype(float)
    for fi in range(len(feats)):
        ax = axes[ri, fi]
        fv, preds = partial_dep(X_raw, model, sc, fi)
        ax.plot(fv, preds, color=color, lw=2.5)
        # Rug plot
        rug_y = preds.min() - (preds.max() - preds.min()) * 0.06
        ax.plot(X_raw[:, fi], np.full(len(X_raw), rug_y), "|",
                color="black", alpha=0.35, markersize=7)
        ax.set_xlabel(labels[fi].replace("\n", " "), fontsize=8)
        ax.set_title(f"{row_lbl}: {labels[fi]}", fontsize=9, fontweight="bold")
        if fi == 0:
            ax.set_ylabel("Predicted log(AAV)", fontsize=8)
        # Secondary y-axis showing $M
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        yticks = ax.get_yticks()
        valid  = yticks[(yticks >= 12) & (yticks <= 22)]
        ax2.set_yticks(valid)
        ax2.set_yticklabels([f"${np.exp(y)/1e6:.0f}M" for y in valid], fontsize=7)
        ax2.tick_params(axis="y", length=0)

plt.tight_layout()
save("step5_partial_dependence.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8: Calibration Curve
# ══════════════════════════════════════════════════════════════════════════════
print("[P8] Calibration curve...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 5 — AAV Calibration: Mean Predicted vs Mean Actual by Predicted Quintile\n"
             "Bubble size = n in bin  |  Error bars = ±1 SE",
             fontsize=12, fontweight="bold")

for ax, df, title, color in [
    (axes[0], bat_df, f"Batters (n={n_bat})",  BAT_C),
    (axes[1], pit_df, f"Pitchers (n={n_pit})", PIT_C),
]:
    df2 = df.copy()
    try:
        df2["bin"] = pd.qcut(df2["pred_aav_M"], q=5, labels=False, duplicates="drop")
    except Exception:
        df2["bin"] = pd.qcut(df2["pred_aav_M"], q=4, labels=False, duplicates="drop")

    grp = df2.groupby("bin").agg(
        mean_pred  = ("pred_aav_M",   "mean"),
        mean_actual= ("actual_aav_M", "mean"),
        std_actual = ("actual_aav_M", "std"),
        count      = ("actual_aav_M", "count"),
    ).dropna()
    grp["se"] = grp["std_actual"] / np.sqrt(grp["count"])

    lim = [0, max(grp["mean_pred"].max(), grp["mean_actual"].max()) * 1.15]
    ax.plot(lim, lim, "k-", lw=1.5, alpha=0.6, label="Perfect calibration")
    ax.scatter(grp["mean_pred"], grp["mean_actual"],
               s=grp["count"] * 12, color=color, alpha=0.8, zorder=5)
    ax.errorbar(grp["mean_pred"], grp["mean_actual"], yerr=grp["se"],
                fmt="none", color=color, capsize=5, alpha=0.7)
    for _, row in grp.iterrows():
        ax.annotate(f"n={int(row['count'])}\n${row['mean_pred']:.1f}M",
                    (row["mean_pred"], row["mean_actual"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Mean Predicted AAV ($M)")
    ax.set_ylabel("Mean Actual AAV ($M)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)

plt.tight_layout()
save("step5_calibration.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9: Cook's Distance (influence analysis)
# ══════════════════════════════════════════════════════════════════════════════
print("[P9] Cook's distance...")

def cooks_d(resid, lev, p):
    """Compute Cook's distance per observation from residuals, leverage, and the number of parameters p."""
    mse = np.mean(resid ** 2)
    lev_clip = np.clip(lev, 1e-6, 1 - 1e-6)
    return (resid ** 2 / (p * mse)) * (lev_clip / (1 - lev_clip) ** 2)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Step 5 — Influence Analysis (Ridge Cook's Distance Approximation)\n"
             "Red bars = above 4/n threshold",
             fontsize=12, fontweight="bold")

for col, (df, lev, label, color, feats) in enumerate([
    (bat_df, bat_lev, "Batters",  BAT_C, BAT_FEATURES),
    (pit_df, pit_lev, "Pitchers", PIT_C, PIT_FEATURES),
]):
    resid = df["resid_aav"].values
    p     = len(feats) + 1
    n     = len(resid)
    cd    = cooks_d(resid, lev, p)
    thr   = 4.0 / n

    # Bar chart
    ax = axes[0, col]
    bar_colors = [NEG_C if d > thr else color for d in cd]
    ax.bar(np.arange(n), cd, color=bar_colors, alpha=0.75)
    ax.axhline(thr, color="red", lw=1.5, ls="--", label=f"4/n = {thr:.3f}")
    high_idx = np.where(cd > thr)[0]
    for idx in high_idx:
        pid = int(df.iloc[idx]["player_id"])
        ax.text(idx, cd[idx] + cd.max() * 0.01, str(pid),
                ha="center", fontsize=6, rotation=80, color=NEG_C)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's Distance")
    ax.set_title(f"{label}: Cook's D  (n_influential={len(high_idx)})", fontweight="bold")
    ax.legend(fontsize=8)

    # Leverage vs residual scatter
    ax = axes[1, col]
    mse = np.mean(resid ** 2)
    sc2 = ax.scatter(lev, resid, c=cd, cmap="YlOrRd", s=55,
                     vmin=0, vmax=max(cd.max() * 0.8, thr * 2), zorder=5)
    plt.colorbar(sc2, ax=ax, label="Cook's D", shrink=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(2 * p / n, color="gray", lw=1.5, ls="--",
               label=f"High leverage (2p/n={2*p/n:.3f})")
    rmse = np.sqrt(mse)
    ax.axhline( 2 * rmse, color="steelblue", lw=1, ls=":", alpha=0.7, label="±2·RMSE")
    ax.axhline(-2 * rmse, color="steelblue", lw=1, ls=":", alpha=0.7)
    # Cook's D contours
    hv = np.linspace(1e-4, min(lev.max() * 1.05, 0.9), 200)
    for cd_c in [0.5, 1.0]:
        rv = np.sqrt(cd_c * p * mse * (1 - hv)**2 / np.maximum(hv, 1e-9))
        ax.plot(hv, rv,  "r--", lw=0.8, alpha=0.5)
        ax.plot(hv, -rv, "r--", lw=0.8, alpha=0.5)
    ax.text(lev.max() * 0.7, resid.max() * 0.85, "D=0.5\nD=1.0",
            fontsize=7, color="red", alpha=0.7)
    # Label top 5 by Cook's D
    top5 = np.argsort(cd)[-5:]
    for idx in top5:
        pid = int(df.iloc[idx]["player_id"])
        ax.annotate(str(pid), (lev[idx], resid[idx]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    ax.set_xlabel("Leverage (ridge hat matrix)")
    ax.set_ylabel("Residual log(AAV)")
    ax.set_title(f"{label}: Leverage–Residual Map", fontweight="bold")
    ax.legend(fontsize=7)

plt.tight_layout()
save("step5_cooks_distance.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 10: Coefficient Shrinkage Path (regularization path)
# ══════════════════════════════════════════════════════════════════════════════
print("[P10] Coefficient shrinkage paths...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Step 5 — Ridge Coefficient Shrinkage Paths\n"
             "How each feature's coefficient changes with regularization strength",
             fontsize=12, fontweight="bold")

ALPHA_PATH = np.logspace(-3, 4, 80)
cmap10     = plt.get_cmap("tab10")

configs_path = [
    (axes[0, 0], X_bat, y_bat_aav, bat_sc, bat_best, ORIG_ALPHA["batter"],
     BAT_LABELS, BAT_C, "Batters — log(AAV)"),
    (axes[0, 1], X_bat, y_bat_yrs, bat_sc, bat_best, ORIG_ALPHA["batter"],
     BAT_LABELS, BAT_C, "Batters — Years"),
    (axes[1, 0], X_pit, y_pit_aav, pit_sc, pit_best, ORIG_ALPHA["pitcher"],
     PIT_LABELS, PIT_C, "Pitchers — log(AAV)"),
    (axes[1, 1], X_pit, y_pit_yrs, pit_sc, pit_best, ORIG_ALPHA["pitcher"],
     PIT_LABELS, PIT_C, "Pitchers — Years"),
]

for ax, X_raw, y, sc_ref, best_a, orig_a, labels, color, title in configs_path:
    sc_path = StandardScaler()
    Xs_path = sc_path.fit_transform(X_raw)
    coef_paths = np.zeros((len(ALPHA_PATH), X_raw.shape[1]))
    for i, a in enumerate(ALPHA_PATH):
        coef_paths[i] = Ridge(alpha=a).fit(Xs_path, y).coef_

    for fi in range(X_raw.shape[1]):
        ax.semilogx(ALPHA_PATH, coef_paths[:, fi],
                    color=cmap10(fi), lw=2,
                    label=labels[fi].replace("\n", " "))

    ax.axvline(best_a, color="green", lw=2,   ls="--", alpha=0.8,
               label=f"Extended best α={best_a:.2f}")
    ax.axvline(orig_a, color="red",   lw=1.5, ls=":",  alpha=0.8,
               label=f"Original α={orig_a:.0f}")
    ax.axhline(0, color="black", lw=0.7, alpha=0.4)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Standardized coefficient")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, loc="center left")

plt.tight_layout()
save("step5_shrinkage_paths.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 DIAGNOSTICS COMPLETE")
print("=" * 60)
print(f"\nExtended best alpha: batter={bat_best:.3f}, pitcher={pit_best:.3f}")
print(f"Original best alpha: batter={ORIG_ALPHA['batter']}, pitcher={ORIG_ALPHA['pitcher']}")
print(f"\nGenerated in {VIZ}:")
for f in sorted(os.listdir(VIZ)):
    if f.startswith("step5_") and f.endswith(".png"):
        kb = os.path.getsize(os.path.join(VIZ, f)) // 1024
        print(f"  {f:50s}  {kb:4d} KB")
