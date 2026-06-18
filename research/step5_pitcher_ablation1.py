"""
Pitcher feature ablation: LOO-CV comparison of adding individual features
to the base 5-feature pitcher model.
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
pit["sp_flag"]     = pit["sp_flag"].fillna(0)
pit["log_proj_ip"] = np.log(pit["proj_ip"].clip(lower=10).fillna(pit["proj_ip"].median()))
pit["proj_rar_sq"] = pit["proj_rar"] ** 2
pit["age_sq"]      = pit["age"] ** 2
pit["sp_x_rar"]    = pit["sp_flag"] * pit["proj_rar"]
pit["sp_x_logip"]  = pit["sp_flag"] * pit["log_proj_ip"]

y_aav = pit["log_aav"].values
y_yrs = pit["years"].values.astype(float)

ALPHAS = [0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
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

BASE5 = ["proj_rar", "age", "log_proj_ip", "sp_flag", "proj_rar_sq"]

tests = {
    "BASE (5)":                            BASE5,
    "+k_rate":                             BASE5 + ["k_rate"],
    "+n_seasons":                          BASE5 + ["n_seasons"],
    "+age_sq":                             BASE5 + ["age_sq"],
    "+sp_x_rar":                           BASE5 + ["sp_x_rar"],
    "+sp_x_logip":                         BASE5 + ["sp_x_logip"],
    "+hra_rate":                           BASE5 + ["hra_rate"],
    "+bb_hbp_rate":                        BASE5 + ["bb_hbp_rate"],
    "+k_rate+n_seasons":                   BASE5 + ["k_rate", "n_seasons"],
    "+k_rate+age_sq":                      BASE5 + ["k_rate", "age_sq"],
    "+k_rate+hra_rate":                    BASE5 + ["k_rate", "hra_rate"],
    "+k_rate+n_seasons+age_sq":            BASE5 + ["k_rate", "n_seasons", "age_sq"],
    "+k_rate+n_seasons+hra_rate":          BASE5 + ["k_rate", "n_seasons", "hra_rate"],
    "+k_rate+n_seasons+age_sq+sp_x_rar":   BASE5 + ["k_rate", "n_seasons", "age_sq", "sp_x_rar"],
    "+k_rate+n_seasons+age_sq+hra_rate":   BASE5 + ["k_rate", "n_seasons", "age_sq", "hra_rate"],
}

hdr = f"{'Model':<44}  {'nfeat':>5}  {'combined':>9}  {'mse_aav':>9}  {'mse_yrs':>9}  {'best_a':>7}"
print(hdr)
print("-" * 90)
base_comb = None
for label, feats in tests.items():
    comb, ba, ma, my = best_combined(pit[feats].values.astype(float), y_aav, y_yrs, ALPHAS)
    if base_comb is None:
        base_comb = comb
    delta = comb - base_comb
    flag = "  <<" if delta < -0.005 else ""
    print(f"{label:<44}  {len(feats):>5}  {comb:>9.4f} ({delta:+.4f})  {ma:>9.4f}  {my:>9.4f}  {ba:>7}{flag}")
