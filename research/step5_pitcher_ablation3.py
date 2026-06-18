"""
Comprehensive pitcher feature search.

Phase 1: test every candidate feature individually against BASE7.
Phase 2: greedy forward selection — keep adding the best next feature
         as long as combined LOO-CV keeps falling.
Alpha grid: 40 log-spaced values [0.01 .. 1000].
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

INT   = r"C:\Users\Felto\Downloads\ootp_analysis\intermediate"
df    = pd.read_csv(INT + r"\market_training_data.csv")
pit   = df[df["player_type"] == "pitcher"].copy().reset_index(drop=True)

# ── Ensure all existing columns ───────────────────────────────────────────────
pit["sp_flag"]     = pit["sp_flag"].fillna(0)
pit["log_proj_ip"] = np.log(pit["proj_ip"].clip(lower=10).fillna(pit["proj_ip"].median()))
pit["proj_rar_sq"] = pit["proj_rar"] ** 2
pit["age_sq"]      = pit["age"] ** 2

# ── Build full candidate feature pool ─────────────────────────────────────────
# Rate components (in retro_pit but not currently in model)
pit["bb_hbp_rate"] = pit["bb_hbp_rate"].fillna(pit["bb_hbp_rate"].median())
pit["hra_rate"]    = pit["hra_rate"].fillna(pit["hra_rate"].median())
pit["gb_rate"]     = pit["gb_rate"].fillna(pit["gb_rate"].median())
pit["k_minus_bb"]  = pit["k_rate"] - pit["bb_hbp_rate"]

# Polynomial terms
pit["krate_sq"]    = pit["k_rate"]      ** 2
pit["bb_sq"]       = pit["bb_hbp_rate"] ** 2
pit["hra_sq"]      = pit["hra_rate"]    ** 2
pit["gb_sq"]       = pit["gb_rate"]     ** 2
pit["kmbb_sq"]     = pit["k_minus_bb"]  ** 2
pit["proj_rar_cb"] = pit["proj_rar"]    ** 3   # cubic RAR

# k_rate interactions
pit["k_x_gb"]      = pit["k_rate"]    * pit["gb_rate"]
pit["k_x_bb"]      = pit["k_rate"]    * pit["bb_hbp_rate"]
pit["k_x_hra"]     = pit["k_rate"]    * pit["hra_rate"]
pit["rar_x_krate"] = pit["proj_rar"]  * pit["k_rate"]
pit["age_x_krate"] = pit["age"]       * pit["k_rate"]
pit["ip_x_krate"]  = pit["log_proj_ip"] * pit["k_rate"]

# sp_flag interactions
pit["sp_x_krate"]  = pit["sp_flag"]   * pit["k_rate"]
pit["sp_x_hra"]    = pit["sp_flag"]   * pit["hra_rate"]
pit["sp_x_bb"]     = pit["sp_flag"]   * pit["bb_hbp_rate"]
pit["sp_x_gb"]     = pit["sp_flag"]   * pit["gb_rate"]
pit["age_x_sp"]    = pit["age"]       * pit["sp_flag"]

# RAR cross-interactions
pit["rar_x_age"]   = pit["proj_rar"]  * pit["age"]
pit["rar_x_gb"]    = pit["proj_rar"]  * pit["gb_rate"]
pit["rar_x_ip"]    = pit["proj_rar"]  * pit["log_proj_ip"]
pit["rar_x_bb"]    = pit["proj_rar"]  * pit["bb_hbp_rate"]
pit["rar_x_hra"]   = pit["proj_rar"]  * pit["hra_rate"]

# age cross-interactions
pit["age_x_bb"]    = pit["age"]       * pit["bb_hbp_rate"]
pit["age_x_hra"]   = pit["age"]       * pit["hra_rate"]
pit["age_x_gb"]    = pit["age"]       * pit["gb_rate"]

# ip cross-interactions
pit["ip_x_gb"]     = pit["log_proj_ip"] * pit["gb_rate"]
pit["ip_x_hra"]    = pit["log_proj_ip"] * pit["hra_rate"]
pit["ip_x_bb"]     = pit["log_proj_ip"] * pit["bb_hbp_rate"]

y_aav = pit["log_aav"].values
y_yrs = pit["years"].values.astype(float)

ALPHAS = np.logspace(-2, 3, 40).tolist()   # 40 points 0.01 -> 1000
loo    = LeaveOneOut()

def best_combined(feat_cols):
    """LOO-CV over ALPHAS for the given pitcher feature set; return the best combined (log(AAV)+years) normalized MSE and its alpha."""
    X_raw = pit[feat_cols].values.astype(float)
    sc = StandardScaler(); X = sc.fit_transform(X_raw)
    va = np.var(y_aav, ddof=1); vy = np.var(y_yrs, ddof=1)
    best = 9999; best_a = None; best_ma = None; best_my = None
    for alpha in ALPHAS:
        m = Ridge(alpha=alpha); ea = []; ey = []
        for tr, te in loo.split(X):
            m.fit(X[tr], y_aav[tr])
            ea.append((y_aav[te[0]] - m.predict(X[te])[0]) ** 2)
            m.fit(X[tr], y_yrs[tr])
            ey.append((y_yrs[te[0]] - m.predict(X[te])[0]) ** 2)
        comb = np.mean(ea) / va + np.mean(ey) / vy
        if comb < best:
            best = comb; best_a = round(alpha, 4)
            best_ma = np.mean(ea) / va; best_my = np.mean(ey) / vy
    return round(best, 4), best_a, round(best_ma, 4), round(best_my, 4)

# ── Phase 1: individual additions to BASE7 ───────────────────────────────────
BASE7 = ["proj_rar", "age", "age_sq", "log_proj_ip", "sp_flag", "proj_rar_sq", "k_rate"]

CANDIDATES = [
    # raw rates
    "bb_hbp_rate", "hra_rate", "gb_rate", "k_minus_bb",
    # polynomials
    "krate_sq", "bb_sq", "hra_sq", "gb_sq", "kmbb_sq", "proj_rar_cb",
    # k_rate interactions
    "k_x_gb", "k_x_bb", "k_x_hra", "rar_x_krate", "age_x_krate", "ip_x_krate",
    # sp_flag interactions
    "sp_x_krate", "sp_x_hra", "sp_x_bb", "sp_x_gb", "age_x_sp",
    # RAR cross-interactions
    "rar_x_age", "rar_x_gb", "rar_x_ip", "rar_x_bb", "rar_x_hra",
    # age cross-interactions
    "age_x_bb", "age_x_hra", "age_x_gb",
    # ip cross-interactions
    "ip_x_gb", "ip_x_hra", "ip_x_bb",
]

print(f"Phase 1: individual additions to BASE7  (n={len(pit)}, alpha grid: {len(ALPHAS)} pts)\n")
hdr = f"{'Feature':<18}  {'nfeat':>5}  {'combined':>9}  {'delta':>8}  {'mse_aav':>9}  {'mse_yrs':>9}  {'alpha':>8}"
print(hdr); print("-" * 80)

base_comb, base_a, base_ma, base_my = best_combined(BASE7)
print(f"{'BASE7':18}  {len(BASE7):>5}  {base_comb:>9.4f}  {'---':>8}  {base_ma:>9.4f}  {base_my:>9.4f}  {base_a:>8}")
print()

results = {}
for c in CANDIDATES:
    comb, ba, ma, my = best_combined(BASE7 + [c])
    delta = comb - base_comb
    flag  = "  <<" if delta < -0.005 else ("  >>" if delta > 0.01 else "")
    print(f"{c:<18}  {len(BASE7)+1:>5}  {comb:>9.4f}  {delta:>+8.4f}  {ma:>9.4f}  {my:>9.4f}  {ba:>8}{flag}")
    results[c] = comb

# ── Phase 2: greedy forward selection ────────────────────────────────────────
print("\n" + "=" * 80)
print("Phase 2: greedy forward selection (keep adding best feature while LOO-CV falls)\n")

current_feats = BASE7[:]
current_comb  = base_comb
remaining     = [c for c in CANDIDATES]

for step in range(6):
    best_feat = None; best_comb = current_comb
    for c in remaining:
        comb, _, _, _ = best_combined(current_feats + [c])
        if comb < best_comb:
            best_comb = comb; best_feat = c
    if best_feat is None:
        print(f"  Step {step+1}: no improvement found -- stopping.")
        break
    comb_final, ba, ma, my = best_combined(current_feats + [best_feat])
    delta = comb_final - current_comb
    current_feats.append(best_feat)
    remaining.remove(best_feat)
    current_comb = comb_final
    print(f"  Step {step+1}: +{best_feat:<18}  feats={len(current_feats)}  "
          f"combined={comb_final:.4f} ({delta:+.4f})  alpha={ba}")

print(f"\nFinal feature set ({len(current_feats)} features):")
for f in current_feats:
    print(f"  {f}")
comb_f, ba_f, ma_f, my_f = best_combined(current_feats)
print(f"Final LOO-CV: {comb_f:.4f}  (BASE7 was {base_comb:.4f}, improvement {comb_f-base_comb:+.4f})")
print(f"Best alpha: {ba_f}  mse_aav: {ma_f:.4f}  mse_yrs: {my_f:.4f}")
