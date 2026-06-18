"""
Extended pitcher feature ablation: FIP/SIERA-inspired additions to BASE7.

Tests individual features and interaction terms on top of the current best
7-feature model (BASE7 = proj_rar, age, age_sq, log_proj_ip, sp_flag,
proj_rar_sq, k_rate).

Features tested:
  - bb_hbp_rate  : command signal (BB+HBP per BF)
  - hra_rate     : HR suppression (HA per BF)
  - gb_rate      : groundball tendency (GB/(GB+FB))
  - k_minus_bb   : K-BB rate (combined efficiency signal)
  - fip_comp     : FIP-like composite (13*hra + 3*bb_hbp - 2*k, rate form)
  Interactions:
  - sp_x_krate   : does K-rate premium differ by role?
  - sp_x_hra     : HR matters differently for SPs (more BF exposed)?
  - k_x_gb       : SIERA core — high-K GB pitchers outperform
  - hra_x_gb     : GB pitchers suppress HR beyond raw hra_rate
  - kmbb_sq      : (K-BB)^2 non-linear term (from SIERA)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

INT = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"
train_df = pd.read_csv(INT + r"\market_training_data.csv")

pit = train_df[train_df["player_type"] == "pitcher"].copy().reset_index(drop=True)

# Ensure derived columns exist
pit["sp_flag"]     = pit["sp_flag"].fillna(0)
pit["log_proj_ip"] = np.log(pit["proj_ip"].clip(lower=10).fillna(pit["proj_ip"].median()))
pit["proj_rar_sq"] = pit["proj_rar"] ** 2
pit["age_sq"]      = pit["age"] ** 2

# New derived features
pit["k_minus_bb"]  = pit["k_rate"] - pit["bb_hbp_rate"]
pit["fip_comp"]    = 13 * pit["hra_rate"] + 3 * pit["bb_hbp_rate"] - 2 * pit["k_rate"]
pit["sp_x_krate"]  = pit["sp_flag"] * pit["k_rate"]
pit["sp_x_hra"]    = pit["sp_flag"] * pit["hra_rate"]
pit["k_x_gb"]      = pit["k_rate"]  * pit["gb_rate"]
pit["hra_x_gb"]    = pit["hra_rate"] * pit["gb_rate"]
pit["kmbb_sq"]     = pit["k_minus_bb"] ** 2

y_aav = pit["log_aav"].values
y_yrs = pit["years"].values.astype(float)

ALPHAS = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
loo = LeaveOneOut()

def best_combined(X_raw, y_aav, y_yrs, alphas):
    """LOO-CV over alphas; return the best combined (log(AAV)+years) normalized MSE, its alpha, and the two component MSEs."""
    sc = StandardScaler(); X = sc.fit_transform(X_raw)
    va = np.var(y_aav, ddof=1); vy = np.var(y_yrs, ddof=1)
    best = 9999; best_a = None; best_mse_a = None; best_mse_y = None
    for alpha in alphas:
        m = Ridge(alpha=alpha); ea = []; ey = []
        for tr, te in loo.split(X):
            m.fit(X[tr], y_aav[tr])
            ea.append((y_aav[te[0]] - m.predict(X[te])[0]) ** 2)
            m.fit(X[tr], y_yrs[tr])
            ey.append((y_yrs[te[0]] - m.predict(X[te])[0]) ** 2)
        comb = np.mean(ea) / va + np.mean(ey) / vy
        if comb < best:
            best = comb; best_a = alpha
            best_mse_a = np.mean(ea) / va
            best_mse_y = np.mean(ey) / vy
    return round(best, 4), best_a, round(best_mse_a, 4), round(best_mse_y, 4)

BASE7 = ["proj_rar", "age", "age_sq", "log_proj_ip", "sp_flag", "proj_rar_sq", "k_rate"]

print("Extended pitcher ablation -- additions to BASE7 (current best, LOO-CV ~1.200)")
print(f"n = {len(pit)} pitchers\n")

tests = {
    # ── Baseline ─────────────────────────────────────────────────────────────
    "BASE7 (current best)":             BASE7,
    # ── Individual FIP/SIERA components ──────────────────────────────────────
    "+bb_hbp_rate":                     BASE7 + ["bb_hbp_rate"],
    "+hra_rate":                         BASE7 + ["hra_rate"],
    "+gb_rate":                          BASE7 + ["gb_rate"],
    "+k_minus_bb":                       BASE7 + ["k_minus_bb"],
    "+fip_comp":                         BASE7 + ["fip_comp"],
    # ── Interaction terms (SIERA-inspired) ───────────────────────────────────
    "+sp_x_krate":                       BASE7 + ["sp_x_krate"],
    "+sp_x_hra":                         BASE7 + ["sp_x_hra"],
    "+k_x_gb":                           BASE7 + ["k_x_gb"],
    "+hra_x_gb":                         BASE7 + ["hra_x_gb"],
    "+kmbb_sq":                          BASE7 + ["kmbb_sq"],
    # ── Two-feature additions ─────────────────────────────────────────────────
    "+bb_hbp_rate+hra_rate":             BASE7 + ["bb_hbp_rate", "hra_rate"],
    "+bb_hbp_rate+gb_rate":              BASE7 + ["bb_hbp_rate", "gb_rate"],
    "+hra_rate+gb_rate":                 BASE7 + ["hra_rate", "gb_rate"],
    "+k_minus_bb+gb_rate":               BASE7 + ["k_minus_bb", "gb_rate"],
    "+k_x_gb+hra_rate":                  BASE7 + ["k_x_gb", "hra_rate"],
    "+k_x_gb+hra_x_gb":                  BASE7 + ["k_x_gb", "hra_x_gb"],
    "+sp_x_krate+sp_x_hra":             BASE7 + ["sp_x_krate", "sp_x_hra"],
    # ── Three-feature / SIERA-like combinations ───────────────────────────────
    "+bb_hbp_rate+hra_rate+gb_rate":     BASE7 + ["bb_hbp_rate", "hra_rate", "gb_rate"],
    "+k_minus_bb+gb_rate+k_x_gb":        BASE7 + ["k_minus_bb", "gb_rate", "k_x_gb"],
    "+bb_hbp_rate+gb_rate+k_x_gb":       BASE7 + ["bb_hbp_rate", "gb_rate", "k_x_gb"],
    "+hra_rate+gb_rate+hra_x_gb":        BASE7 + ["hra_rate", "gb_rate", "hra_x_gb"],
}

hdr = f"{'Model':<46}  {'nfeat':>5}  {'combined':>9}  {'mse_aav':>9}  {'mse_yrs':>9}  {'best_a':>7}"
print(hdr)
print("-" * 95)
base_comb = None
for label, feats in tests.items():
    comb, ba, ma, my = best_combined(pit[feats].values.astype(float), y_aav, y_yrs, ALPHAS)
    if base_comb is None:
        base_comb = comb
    delta = comb - base_comb
    flag = "  <<" if delta < -0.005 else ("  >>" if delta > 0.01 else "")
    print(f"{label:<46}  {len(feats):>5}  {comb:>9.4f} ({delta:+.4f})  {ma:>9.4f}  {my:>9.4f}  {ba:>7}{flag}")

# ── Print feature stats for context ──────────────────────────────────────────
print("\nFeature summary (pitchers):")
cols = ["proj_rar","k_rate","bb_hbp_rate","hra_rate","gb_rate","k_minus_bb"]
print(pit[cols].describe().loc[["mean","std","min","max"]].round(4).to_string())
