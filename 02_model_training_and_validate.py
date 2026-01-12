#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hardness regression — GBM + NN + manual SOTA ensemble (A/B/C/D) + CSV

Modeller:
- MLPRegressor
- XGBRegressor (opsiyonel)
- LGBMRegressor (opsiyonel)
- CatBoostRegressor (opsiyonel)
- Ensemble_SOTA (elle ortalama: XGB + LGBM + CatBoost; eşit ağırlık)

Ayarlar:
- Train: 'e', Test: ['a','b','c','d']
- FS: SelectKBest(f_regression), K=65
- Preproc: mean-impute -> StandardScaler -> MinMaxScaler
- Outliers (>|EXTREME_THRESH|): NaN -> linear interpolate
- Smoothing: KATMAN içinde komşu ortalaması (window=N_NEIGHBORS)
- Metrikler: R2, "RMSE"(=MSE), MAE (Raw & Smooth) + Layer-Mean (R2/MAE)
- CSV: models/bench_hardness_gbm_nn_with_sota.csv
"""

import os, warnings
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor

# Opsiyonel GBM kütüphaneleri (regresyon)
HAS_XGB = HAS_LGBM = HAS_CATB = False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    pass
try:
    from catboost import CatBoostRegressor
    HAS_CATB = True
except Exception:
    pass

warnings.filterwarnings("ignore")

# ---- PATHS / CONSTS ----
FEAT_DIR = "./features_out"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS     = ["Young's Modulus, apparent", "Young's Modulus, film", "Hardness"]
TARGET_HARD = "Hardness"
TRAIN_KEY   = "e"
TEST_KEYS   = ["a","b","c","d"]

DROP_COLS = {
    "Load On Sample (mN)",
    "Time On Sample (s)",
}

EXTREME_THRESH = 1e10
K = 60                    # SelectKBest k
N_NEIGHBORS = 50           # smoothing penceresi (toplam pencere boyu)
RANDOM_STATE = 42

# ---- IO ----
def read_feat(key: str) -> pd.DataFrame:
    p = os.path.join(FEAT_DIR, f"{key}_features.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} bulunamadı.")
    return pd.read_csv(p)

# ---- Cleaning ----
def clean_extreme_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    if target in out.columns:
        s = pd.to_numeric(out[target], errors="coerce")
        s = s.where(~(np.isfinite(s) & (np.abs(s) > EXTREME_THRESH)), np.nan)
        s = s.interpolate(method="linear", limit_direction="both")
        out[target] = s
    return out

# ---- Features & Preproc ----
def all_numeric_features_strict(df: pd.DataFrame) -> List[str]:
    exclude = set(TARGETS + ["KATMAN"]) | DROP_COLS
    return [c for c in df.columns if (c not in exclude) and pd.api.types.is_numeric_dtype(df[c])]

def fit_preproc(df_train: pd.DataFrame, feat_cols: List[str]):
    imputer = SimpleImputer(strategy="mean")
    std = StandardScaler(with_mean=True, with_std=True)
    mm  = MinMaxScaler(feature_range=(0.0, 1.0))
    X = df_train[feat_cols].values
    X = imputer.fit_transform(X)
    X = std.fit_transform(X)
    X = mm.fit_transform(X)
    return imputer, std, mm, X

def transform_with(df: pd.DataFrame, feat_cols: List[str], imputer, std, mm):
    # Eksik feature varsa 0.0 ile oluştur
    missing = [c for c in feat_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    X = df[feat_cols].values
    X = imputer.transform(X)
    X = std.transform(X)
    X = mm.transform(X)
    return X

# ---- Post: smoothing ----
def smooth_pred_by_neighbors(y_pred: np.ndarray,
                             layer_series: Optional[pd.Series],
                             n_neighbors: int = 3) -> np.ndarray:
    """
    KATMAN içinde komşu ortalaması (sınır aşmadan).
    n_neighbors toplam pencere uzunluğu (örn: 7 -> radius=3).
    """
    y = np.asarray(y_pred, dtype=float)
    n = len(y)
    if n <= 1 or n_neighbors <= 1:
        return y.copy()
    r = max(1, n_neighbors // 2)
    out = np.copy(y)

    if layer_series is None or layer_series.isna().all():
        for i in range(n):
            lo = max(0, i - r); hi = min(n - 1, i + r)
            out[i] = np.nanmean(y[lo:hi+1])
        return out

    layers = layer_series.fillna("__NA__").astype(str).values
    start = 0
    while start < n:
        layer = layers[start]
        end = start
        while end + 1 < n and layers[end + 1] == layer:
            end += 1
        for i in range(start, end + 1):
            lo = max(start, i - r); hi = min(end, i + r)
            out[i] = np.nanmean(y[lo:hi+1])
        start = end + 1
    return out

# ---- Layer mean metrics ----
def layer_mean_metrics(y_true: np.ndarray, y_pred: np.ndarray, layer_series: Optional[pd.Series]) -> Tuple[float, float]:
    if layer_series is None or layer_series.isna().all():
        return np.nan, np.nan

    dfm = pd.DataFrame({
        "layer": layer_series.astype(str),
        "yt": pd.to_numeric(y_true, errors="coerce"),
        "yp": pd.to_numeric(y_pred, errors="coerce"),
    })
    dfm = dfm[np.isfinite(dfm["yt"]) & np.isfinite(dfm["yp"])]
    if dfm.empty:
        return np.nan, np.nan

    g = dfm.groupby("layer", dropna=False).agg(yt_mean=("yt","mean"),
                                               yp_mean=("yp","mean"))
    if len(g) < 2:
        return np.nan, float(np.abs(g["yt_mean"] - g["yp_mean"]).mean())

    return float(r2_score(g["yt_mean"].values, g["yp_mean"].values)), \
           float(mean_absolute_error(g["yt_mean"].values, g["yp_mean"].values))

# ---- Metrics helper ----
def metrics_pack(y_true, y_hat) -> Tuple[float, float, float]:
    ok = np.isfinite(y_true) & np.isfinite(y_hat)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan
    # Not: önceki koddakiyle uyum için squared=False kullanmıyoruz (etiket "RMSE" olsa da bu MSE).
    return (float(r2_score(y_true[ok], y_hat[ok])),
            float(mean_squared_error(y_true[ok], y_hat[ok])),
            float(mean_absolute_error(y_true[ok], y_hat[ok])))

# ---- Build ONLY the 4 regression models ----
def build_models():
    models = []
    if HAS_XGB:
        models.append(("XGBRegressor",
                       XGBRegressor(
                           n_estimators=1000, max_depth=2, learning_rate=0.175,
                           subsample=0.8, colsample_bytree=0.8,
                           objective="reg:squarederror",
                           eval_metric="rmse",
                           tree_method="hist",
                           random_state=RANDOM_STATE,
                           n_jobs=-1
                       )))
    if HAS_LGBM:
        models.append(("LGBMRegressor",
                       LGBMRegressor(
                           n_estimators=500, num_leaves=10, learning_rate=0.25,
                           objective="regression", max_depth=2,
                           random_state=RANDOM_STATE,
                           n_jobs=-1
                       )))

    return models

# ---- MAIN ----
def main():
    # Train set
    df_tr = read_feat(TRAIN_KEY)
    df_tr = clean_extreme_target(df_tr, TARGET_HARD)

    feat_cols = all_numeric_features_strict(df_tr)
    print(f"[INFO] #features (after strict exclude): {len(feat_cols)}")

    imputer, std, mm, Xtr_full = fit_preproc(df_tr, feat_cols)
    ytr_full = pd.to_numeric(df_tr[TARGET_HARD], errors="coerce").values
    mtr = np.isfinite(ytr_full)
    Xtr_full, ytr_full = Xtr_full[mtr], ytr_full[mtr]

    # Feature selection
    n_feat = Xtr_full.shape[1]
    if K >= n_feat:
        print(f"[INFO] K({K}) >= #feat({n_feat}) -> FS PASSTHROUGH")
        sel_cols = feat_cols[:]
        Xtr = Xtr_full
        selector = None
    else:
        selector = SelectKBest(score_func=f_regression, k=K)
        selector.fit(Xtr_full, ytr_full)
        mask = selector.get_support()
        sel_cols = [c for c, m in zip(feat_cols, mask) if m]
        Xtr = selector.transform(Xtr_full)
        with open(os.path.join(MODEL_DIR, f"selected_features_k{K}_anova.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sel_cols))
        print(f"[INFO] Selected k={K} features -> models/selected_features_k{K}_anova.txt")

    models = build_models()
    if not models:
        raise SystemExit("Hiç model yok (en az MLPRegressor olmalı).")

    # ---- Train all models once ----
    fitted: Dict[str, object] = {}
    for mname, model in models:
        print("\n" + "="*68)
        print(f"[MODEL] {mname}")
        model.fit(Xtr, ytr_full)
        print("[INFO] trained.")
        fitted[mname] = model

    # ---- Evaluate per dataset for each model ----
    all_rows = []

    # İleride Ensemble_SOTA'da kullanmak üzere GBM model isimleri
    gbm_names = [name for name in ["XGBRegressor", "LGBMRegressor", "CatBoostRegressor"] if name in fitted]

    for mname, model in fitted.items():
        rows = []
        for key in TEST_KEYS:
            df_te = read_feat(key)
            df_te = clean_extreme_target(df_te, TARGET_HARD)

            Xte_full = transform_with(df_te.copy(), feat_cols, imputer, std, mm)
            Xte = selector.transform(Xte_full) if selector is not None else Xte_full
            yte = pd.to_numeric(df_te[TARGET_HARD], errors="coerce").values
            layer_series = df_te["KATMAN"] if "KATMAN" in df_te.columns else None

            # Tahminler
            y_pred = model.predict(Xte)
            y_pred_s = smooth_pred_by_neighbors(y_pred, layer_series, n_neighbors=N_NEIGHBORS)

            # Metrikler
            r2_raw, rmse_raw, mae_raw = metrics_pack(yte, y_pred)
            r2_sm , rmse_sm , mae_sm  = metrics_pack(yte, y_pred_s)
            r2_layer, mae_layer = layer_mean_metrics(yte, y_pred_s, layer_series)

            print(f"\n[{key.upper()}]  (Hardness)  -- {mname}")
            print(f"  Raw     -> R2={r2_raw:.4f}  RMSE={rmse_raw:.4f}  MAE={mae_raw:.4f}")
            print(f"  Smooth  -> R2={r2_sm:.4f}  RMSE={rmse_sm:.4f}  MAE={mae_sm:.4f}")
            if np.isfinite(r2_layer):
                print(f"  Layer-Mean -> R2(mean-by-layer)={r2_layer:.4f}  MAE(mean-by-layer)={mae_layer:.4f}")
            else:
                print(f"  Layer-Mean -> insufficient distinct layers to compute R2")

            rows.append({
                "model": mname, "dataset": key.upper(), "k": min(K, n_feat),
                "R2_raw": r2_raw, "RMSE_raw": rmse_raw, "MAE_raw": mae_raw,
                "R2_smooth": r2_sm, "RMSE_smooth": rmse_sm, "MAE_smooth": mae_sm,
                "R2_layer_mean": r2_layer, "MAE_layer_mean": mae_layer,
                "n_features": len(sel_cols) if selector is not None else n_feat
            })

        # Ortalama satırı (A/B/C/D)
        dfm = pd.DataFrame(rows)
        mean_vals = {
            "model": mname, "dataset": "MEAN", "k": min(K, n_feat),
            "R2_raw": float(np.nanmean(dfm["R2_raw"])),
            "RMSE_raw": float(np.nanmean(dfm["RMSE_raw"])),
            "MAE_raw": float(np.nanmean(dfm["MAE_raw"])),
            "R2_smooth": float(np.nanmean(dfm["R2_smooth"])),
            "RMSE_smooth": float(np.nanmean(dfm["RMSE_smooth"])),
            "MAE_smooth": float(np.nanmean(dfm["MAE_smooth"])),
            "R2_layer_mean": float(np.nanmean(dfm["R2_layer_mean"])),
            "MAE_layer_mean": float(np.nanmean(dfm["MAE_layer_mean"])),
            "n_features": dfm["n_features"].iloc[0],
        }
        print("\n===== OVERALL (avg of A,B,C,D) -> {} =====".format(mname))
        print("Raw     -> R2={R2_raw:.4f}  RMSE={RMSE_raw:.4f}  MAE={MAE_raw:.4f}".format(**mean_vals))
        print("Smooth  -> R2={R2_smooth:.4f}  RMSE={RMSE_smooth:.4f}  MAE={MAE_smooth:.4f}".format(**mean_vals))
        print("Layer-Mean -> R2={R2_layer_mean:.4f}  MAE={MAE_layer_mean:.4f}".format(**mean_vals))

        all_rows.extend(rows)
        all_rows.append(mean_vals)

    # ---- Manual Ensemble SOTA: (XGB + LGBM + CatBoost) eşit ağırlık ----
    if len(gbm_names) >= 2:
        print("\n" + "="*68)
        print("[MODEL] Ensemble_SOTA (manual avg of: {})".format(", ".join(gbm_names)))
        ens_rows = []

        for key in TEST_KEYS:
            df_te = read_feat(key)
            df_te = clean_extreme_target(df_te, TARGET_HARD)

            Xte_full = transform_with(df_te.copy(), feat_cols, imputer, std, mm)
            Xte = selector.transform(Xte_full) if selector is not None else Xte_full
            yte = pd.to_numeric(df_te[TARGET_HARD], errors="coerce").values
            layer_series = df_te["KATMAN"] if "KATMAN" in df_te.columns else None

            preds = []
            for name in gbm_names:
                preds.append(fitted[name].predict(Xte))

            # Eşit ağırlıklı ortalama (mevcut model sayısına göre)
            y_pred = np.mean(np.vstack(preds), axis=0)
            y_pred_s = smooth_pred_by_neighbors(y_pred, layer_series, n_neighbors=N_NEIGHBORS)

            r2_raw, rmse_raw, mae_raw = metrics_pack(yte, y_pred)
            r2_sm , rmse_sm , mae_sm  = metrics_pack(yte, y_pred_s)
            r2_layer, mae_layer = layer_mean_metrics(yte, y_pred_s, layer_series)

            print(f"\n[{key.upper()}]  (Hardness)  -- Ensemble_SOTA")
            print(f"  Raw     -> R2={r2_raw:.4f}  RMSE={rmse_raw:.4f}  MAE={mae_raw:.4f}")
            print(f"  Smooth  -> R2={r2_sm:.4f}  RMSE={rmse_sm:.4f}  MAE={mae_sm:.4f}")
            if np.isfinite(r2_layer):
                print(f"  Layer-Mean -> R2(mean-by-layer)={r2_layer:.4f}  MAE(mean-by-layer)={mae_layer:.4f}")
            else:
                print(f"  Layer-Mean -> insufficient distinct layers to compute R2")

            ens_rows.append({
                "model": "Ensemble_SOTA", "dataset": key.upper(), "k": min(K, n_feat),
                "R2_raw": r2_raw, "RMSE_raw": rmse_raw, "MAE_raw": mae_raw,
                "R2_smooth": r2_sm, "RMSE_smooth": rmse_sm, "MAE_smooth": mae_sm,
                "R2_layer_mean": r2_layer, "MAE_layer_mean": mae_layer,
                "n_features": len(sel_cols) if selector is not None else n_feat
            })

        df_ens = pd.DataFrame(ens_rows)
        ens_mean = {
            "model": "Ensemble_SOTA", "dataset": "MEAN", "k": min(K, n_feat),
            "R2_raw": float(np.nanmean(df_ens["R2_raw"])),
            "RMSE_raw": float(np.nanmean(df_ens["RMSE_raw"])),
            "MAE_raw": float(np.nanmean(df_ens["MAE_raw"])),
            "R2_smooth": float(np.nanmean(df_ens["R2_smooth"])),
            "RMSE_smooth": float(np.nanmean(df_ens["RMSE_smooth"])),
            "MAE_smooth": float(np.nanmean(df_ens["MAE_smooth"])),
            "R2_layer_mean": float(np.nanmean(df_ens["R2_layer_mean"])),
            "MAE_layer_mean": float(np.nanmean(df_ens["MAE_layer_mean"])),
            "n_features": df_ens["n_features"].iloc[0],
        }
        print("\n===== OVERALL (avg of A,B,C,D) -> Ensemble_SOTA =====")
        print("Raw     -> R2={R2_raw:.4f}  RMSE={RMSE_raw:.4f}  MAE={MAE_raw:.4f}".format(**ens_mean))
        print("Smooth  -> R2={R2_smooth:.4f}  RMSE={RMSE_smooth:.4f}  MAE={MAE_smooth:.4f}".format(**ens_mean))
        print("Layer-Mean -> R2={R2_layer_mean:.4f}  MAE={MAE_layer_mean:.4f}".format(**ens_mean))

        all_rows.extend(ens_rows)
        all_rows.append(ens_mean)
    else:
        print("\n[WARN] Ensemble_SOTA kurulamadı: GBM modellerinden en az 2 tanesi gerekli (bulunan: {}).".format(len(gbm_names)))

    out_csv = os.path.join(MODEL_DIR, "bench_hardness_gbm_nn_with_sota_100.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\n[OK] Saved: {out_csv}")

if __name__ == "__main__":
    main()
