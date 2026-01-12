#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_build_features_plus.py (cleaned)

- ./data altında a.xlsx..e.xlsx bekler.
- KATMAN metninden (YbDSi, Mullite, Si) oranlarını çıkarır.
- depth_norm = Derinlik(µm) / max(Derinlik)
- Sentetikler (sadece izinli cihaz kolonlarıyla):
    * Oranlar: kareler, ikili çarpımlar, saflık, Shannon entropisi
    * Cihaz: kare/log1p ve seçilmiş çift etkileşimleri
    * Oran×cihaz ve Derinlik×(oran/cihaz)
    * HİBRİT: tek-faz şablonlarına oran-ağırlıklı karışım
    * EK ÇARPIMLAR (PRODUCED_FEATURES ile):
        - Derinlik (µm) * PRODUCED_FEATURES
        - Displacement nm * PRODUCED_FEATURES
        - Harmonic Contact Stiffness * PRODUCED_FEATURES
        - Normalized Displacement Into Surface * PRODUCED_FEATURES
        - Derinlik (µm) * Displacement nm * PRODUCED_FEATURES
- NOT: "Load On Sample (mN)" ve "Time On Sample (s)" KESİNLİKLE kullanılmaz.
- Çıktı: features_out/{key}_features.csv
"""

import os, re, glob, warnings
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "./data"
OUT_DIR  = "./features_out"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Kolon adları ---
COL_KATMAN = "KATMAN"
COL_DEPTH  = "Derinlik (µm)"
COL_DISP   = "Displacement nm"
COL_HCS    = "Harmonic Contact Stiffness"
COL_NDIS   = "Normalized Displacement Into Surface"

# SADECE İZİNLİ cihaz kolonları (Load / Time YOK!)
DEVICE_COLS = [COL_DISP, COL_HCS, COL_NDIS]

TARGETS = ["Young's Modulus, apparent", "Young's Modulus, film", "Hardness"]

# --------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------
def infer_key(path: str) -> str:
    base = os.path.basename(path).lower()
    for k in ["a","b","c","d","e"]:
        if re.search(rf"(^|[_\-]){k}(\.|[_\-])", base) or base.startswith(f"{k}.") or base.endswith(f"_{k}.xlsx"):
            return k
    return base[:1] if base[:1] in ["a","b","c","d","e"] else ""

def _pct_from_token(tok: str) -> float:
    m = re.search(r"(\d{1,3})", tok)
    return float(m.group(1))/100.0 if m else np.nan

def parse_katman(label: str) -> Tuple[float,float,float]:
    s = str(label).lower().strip()
    if re.match(r"^si\b", s): return 0.0, 0.0, 1.0
    if re.match(r"^ybdsi\b|^ybsi\b|^yb d?si\b", s): return 1.0, 0.0, 0.0
    if re.match(r"^mü|^mullite", s): return 0.0, 1.0, 0.0
    if "sınır" in s and (("ybdsi" in s) or ("ybsi" in s)) and (("mü" in s) or ("mullite" in s)):
        return 0.5, 0.5, 0.0

    s_norm = s.replace("müllite","mü").replace("mullite","mü").replace("  "," ")
    parts = re.split(r"\+|,|/", s_norm)
    yb = mu = si = 0.0
    for tok in parts:
        tok = tok.strip()
        p = _pct_from_token(tok)
        if not np.isfinite(p):
            if re.search(r"\bybd?si\b", tok): yb += 1.0
            elif re.search(r"\bmü\b", tok):   mu += 1.0
            elif re.search(r"\bsi\b", tok):   si += 1.0
            continue
        if re.search(r"\bybd?si\b", tok): yb += p
        elif re.search(r"\bmü\b", tok):   mu += p
        elif re.search(r"\bsi\b", tok):   si += p

    if (yb+mu+si) == 0.0: return 0.0, 0.0, 0.0
    ssum = yb+mu+si
    return yb/ssum, mu/ssum, si/ssum

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in DEVICE_COLS + TARGETS + [COL_DEPTH]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def safe_log1p(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return np.sign(x) * np.log1p(np.abs(x))

# --------------------------------------------------------
# Hibrit şablonlar
# --------------------------------------------------------
def build_phase_templates(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    tmpl = { "YbDSi": {}, "Mullite": {}, "Si": {} }
    masks = {
        "YbDSi": (df["ybds_i_pct"] >= 0.98),
        "Mullite": (df["mullite_pct"] >= 0.98),
        "Si": (df["si_pct"] >= 0.98),
    }
    for phase, m in masks.items():
        for c in DEVICE_COLS:
            if c in df.columns and m.any():
                tmpl[phase][c] = float(np.nanmedian(df.loc[m, c]))
            else:
                tmpl[phase][c] = 0.0
    return tmpl

def add_hybrid_columns(df: pd.DataFrame, templates: Dict[str, Dict[str, float]], produced: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    yb = out["ybds_i_pct"].astype(float).fillna(0.0)
    mu = out["mullite_pct"].astype(float).fillna(0.0)
    si = out["si_pct"].astype(float).fillna(0.0)

    for c in DEVICE_COLS:
        if c not in out.columns:
            continue
        t_yb = templates["YbDSi"].get(c, 0.0)
        t_mu = templates["Mullite"].get(c, 0.0)
        t_si = templates["Si"].get(c, 0.0)

        col_hyb = f"HYB_{c}"
        out[col_hyb] = yb*t_yb + mu*t_mu + si*t_si
        produced.append(col_hyb)

        col_dy = f"DIFF_{c}_yb"
        col_dm = f"DIFF_{c}_mu"
        col_ds = f"DIFF_{c}_si"
        out[col_dy] = out[c] - t_yb
        out[col_dm] = out[c] - t_mu
        out[col_ds] = out[c] - t_si
        produced.extend([col_dy, col_dm, col_ds])

        col_y = f"yb_DIFF_{c}"
        col_m = f"mu_DIFF_{c}"
        col_s = f"si_DIFF_{c}"
        out[col_y] = yb * out[col_dy]
        out[col_m] = mu * out[col_dm]
        out[col_s] = si * out[col_ds]
        produced.extend([col_y, col_m, col_s])

    return out, produced

# --------------------------------------------------------
# Ana zenginleştirme
# --------------------------------------------------------
# Yalnız izinli cihazlar arası çiftler
PAIR_DEVICE_INTERACTIONS: List[Tuple[str,str]] = [
    (COL_DISP, COL_HCS),
    (COL_HCS, COL_NDIS),
    (COL_DISP, COL_NDIS),
]

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = clean_numeric(df)

    produced_cols: List[str] = []  # bu scriptin ÜRETTİĞİ kolonlar

    # oranlar
    yb_list, mu_list, si_list = [], [], []
    for s in df[COL_KATMAN].astype(str).tolist():
        yb, mu, si = parse_katman(s)
        yb_list.append(yb); mu_list.append(mu); si_list.append(si)
    df["ybds_i_pct"] = yb_list
    df["mullite_pct"] = mu_list
    df["si_pct"] = si_list
    produced_cols.extend(["ybds_i_pct","mullite_pct","si_pct"])

    # derinlik normalizasyonu
    if COL_DEPTH in df.columns and df[COL_DEPTH].notna().any():
        maxd = float(np.nanmax(df[COL_DEPTH].values))
        df["depth_norm"] = (df[COL_DEPTH].astype(float) / max(1e-12, maxd)).clip(0,1)
    else:
        n = len(df)
        df["depth_norm"] = (np.arange(n)+0.5)/max(1,n)
    produced_cols.append("depth_norm")

    # Oran kareleri & ikililer
    for newc, series in [
        ("yb_sq",  df["ybds_i_pct"]**2),
        ("mu_sq",  df["mullite_pct"]**2),
        ("si_sq",  df["si_pct"]**2),
        ("yb_mu",  df["ybds_i_pct"]*df["mullite_pct"]),
        ("yb_si",  df["ybds_i_pct"]*df["si_pct"]),
        ("mu_si",  df["mullite_pct"]*df["si_pct"]),
    ]:
        df[newc] = series
        produced_cols.append(newc)

    # Saflık ve entropi
    ratios = df[["ybds_i_pct","mullite_pct","si_pct"]].clip(1e-12, 1.0)
    df["purity_max"] = ratios.max(axis=1)
    df["mix_entropy"] = -(ratios*np.log(ratios)).sum(axis=1)
    produced_cols.extend(["purity_max","mix_entropy"])

    # Cihaz ^2 ve log1p (SADECE izinli cihazlar)
    for c in DEVICE_COLS:
        if c in df.columns:
            sq = f"{c}__sq"
            lg = f"{c}__log"
            df[sq]  = df[c].astype(float)**2
            df[lg]  = safe_log1p(df[c])
            produced_cols.extend([sq, lg])

    # Seçilmiş cihaz-çift etkileşimleri (SADECE izinli cihaz çiftleri)
    for a,b in PAIR_DEVICE_INTERACTIONS:
        if (a in df.columns) and (b in df.columns):
            col = f"{a}__x__{b}"
            df[col] = df[a]*df[b]
            produced_cols.append(col)

    # Oran×cihaz ve Derinlik×{oran,cihaz} (SADECE izinli cihazlar)
    for c in DEVICE_COLS:
        if c in df.columns:
            for newc, series in [
                (f"yb__{c}", df["ybds_i_pct"] * df[c]),
                (f"mu__{c}", df["mullite_pct"] * df[c]),
                (f"si__{c}", df["si_pct"] * df[c]),
                (f"depth__{c}", df["depth_norm"] * df[c]),
            ]:
                df[newc] = series
                produced_cols.append(newc)

    for newc, series in [
        ("depth__yb", df["depth_norm"] * df["ybds_i_pct"]),
        ("depth__mu", df["depth_norm"] * df["mullite_pct"]),
        ("depth__si", df["depth_norm"] * df["si_pct"]),
        ("depth__purity", df["depth_norm"] * df["purity_max"]),
    ]:
        df[newc] = series
        produced_cols.append(newc)

    # HİBRİT şablonlardan karışım-ağırlıklı özellikler (SADECE izinli cihazlar)
    templates = build_phase_templates(df)
    df, produced_cols = add_hybrid_columns(df, templates, produced_cols)

    # --------------------------------------------------------
    # EK: Üretilen kolonlarla derinlik/cihaz çaprazları (Load/Time YOK)
    # --------------------------------------------------------
    produced_numeric = [c for c in produced_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    # Baz kolonların kendileriyle çarpma yapmayalım:
    base_exclude = {COL_DEPTH, COL_DISP, COL_HCS, COL_NDIS}
    produced_numeric = [c for c in produced_numeric if c not in TARGETS and c not in base_exclude]

    # 1) Depth * PRODUCED
    if COL_DEPTH in df.columns:
        for feat in produced_numeric:
            df[f"{COL_DEPTH}__x__{feat}"] = df[COL_DEPTH] * df[feat]

    # 2) Displacement * PRODUCED
    if COL_DISP in df.columns:
        for feat in produced_numeric:
            df[f"{COL_DISP}__x__{feat}"] = df[COL_DISP] * df[feat]

    # 3) HCS * PRODUCED
    if COL_HCS in df.columns:
        for feat in produced_numeric:
            df[f"{COL_HCS}__x__{feat}"] = df[COL_HCS] * df[feat]

    # 4) NDIS * PRODUCED
    if COL_NDIS in df.columns:
        for feat in produced_numeric:
            df[f"{COL_NDIS}__x__{feat}"] = df[COL_NDIS] * df[feat]

    # 5) Depth * Displacement * PRODUCED (sadece bu üçlü)
    if (COL_DEPTH in df.columns) and (COL_DISP in df.columns):
        prod_depth_disp = df[COL_DEPTH] * df[COL_DISP]
        for feat in produced_numeric:
            df[f"{COL_DEPTH}__x__{COL_DISP}__x__{feat}"] = prod_depth_disp * df[feat]

    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))
    if not files:
        raise SystemExit("data/*.xlsx altında dosya bulunamadı.")
    for p in files:
        key = infer_key(p)
        if key not in ["a","b","c","d","e"]:
            continue
        df = pd.read_excel(p)
        if COL_KATMAN not in df.columns:
            raise SystemExit(f"{p} içinde '{COL_KATMAN}' kolonu yok.")
        out = enrich(df)
        out_path = os.path.join(OUT_DIR, f"{key}_features.csv")
        out.to_csv(out_path, index=False)
        print(f"[{key}] saved -> {out_path}  (rows={len(out)}, cols={len(out.columns)})")

if __name__ == "__main__":
    main()
