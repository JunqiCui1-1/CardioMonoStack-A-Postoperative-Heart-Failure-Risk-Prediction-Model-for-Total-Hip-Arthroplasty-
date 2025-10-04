#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP · Branch A (Monotone LGBM) · Top-8 Only
============================================

- Loads /content/final-analysis.csv
- Builds the same feature pipeline used in CardioMonoStack benchmarks
- Trains Branch A (monotone LightGBM) on a stratified 80/20 split
- Saves 400-dpi Top-8 SHAP bar & beeswarm plots into /content/pmrspp_out/benchmarks

Run:
  python cms_shap_branchA_top8.py
"""

import sys
import subprocess
import importlib
import os
import re
import json
import math
import warnings
import inspect
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ----- lazy installs (Colab-friendly) -----
def _ensure(pkg: str):
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_ensure("lightgbm")
_ensure("shap")
_ensure("scikit-learn")

import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator

# ---------- Paths ----------
INPUT = "/content/final-analysis.csv"
OUTDIR = Path("/content/pmrspp_out/benchmarks"); OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- Regex & helpers ----------
ID_PAT = re.compile(r"(?:^id$|subject|hadm|icustay|patientunit|stayid|admission|visit|encounter|zid)$", re.I)

PHYS_LIMITS = {  # coarse physiological limits; used to create quality flags
    "na": (110, 170), "sodium": (110, 170),
    "k": (1.5, 7.5),  "potassium": (1.5, 7.5),
    "sbp": (60, 260), "systolic": (60, 260),
    "dbp": (30, 160), "diastolic": (30, 160),
}

# Monotonicity hints: + increases risk, - decreases risk
MONO_PLUS  = [re.compile(p, re.I) for p in [
    r"^age$", r"creat|creatinine", r"bnp|nt.?pro.?bnp",
    r"troponin", r"lactate", r"nyha", r"prior.?hf|\bhf\b",
    r"atrial.?fib|afib"
]]
MONO_MINUS = [re.compile(p, re.I) for p in [r"egfr", r"lvef|\bef\b", r"albumin"]]

def build_monotone_map(num_cols: List[str]) -> List[int]:
    cons = []
    for c in num_cols:
        s = 0
        if any(rx.search(c) for rx in MONO_PLUS): s = 1
        elif any(rx.search(c) for rx in MONO_MINUS): s = -1
        cons.append(s)
    return cons

def _make_ohe() -> OneHotEncoder:
    params = inspect.signature(OneHotEncoder.__init__).parameters
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False) if "sparse_output" in params \
           else OneHotEncoder(handle_unknown="ignore", sparse=False)

# ---------- Feature makers ----------
class SafeFeatureMaker(BaseEstimator, TransformerMixin):
    """Create clinically meaningful composites if source columns exist."""
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        cols = list(X.columns)

        def _find(pats):
            for p in pats:
                r = re.compile(p, re.I)
                for c in cols:
                    if r.search(c):
                        return c
            return None

        # SBP/DBP -> PP & MAP
        sbp = _find([r"\b(sbp|sys|systolic)\b"])
        dbp = _find([r"\b(dbp|dia|diastolic)\b"])
        if sbp and dbp and sbp in X and dbp in X:
            s = pd.to_numeric(X[sbp], errors='coerce')
            d = pd.to_numeric(X[dbp], errors='coerce')
            X["pp"] = s - d
            X["map_calc"] = (2*d + s) / 3.0

        # BUN/Cr ratio
        bun = _find([r"\b(bun|urea)\b"])
        cr  = _find([r"\b(creat|creatinine)\b"])
        if bun and cr and bun in X and cr in X:
            with np.errstate(divide='ignore', invalid='ignore'):
                X["bun_cr_ratio"] = pd.to_numeric(X[bun], errors='coerce') / pd.to_numeric(X[cr], errors='coerce')

        # Na/K ratio
        na = _find([r"\b(na|sodium)\b"])
        k  = _find([r"\b(k|potassium)\b"])
        if na and k and na in X and k in X:
            with np.errstate(divide='ignore', invalid='ignore'):
                X["na_k_ratio"] = pd.to_numeric(X[na], errors='coerce') / pd.to_numeric(X[k], errors='coerce')

        # Quality flags based on PHYS_LIMITS
        for key, (lo, hi) in PHYS_LIMITS.items():
            for c in X.columns:
                if re.search(fr"\b{key}\b", c, re.I):
                    v = pd.to_numeric(X[c], errors='coerce')
                    X[c + "_qflag"] = ((v < lo) | (v > hi)).astype(float)

        return X

class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    """Winsorize numeric features by global quantiles (0.5% / 99.5%)."""
    def __init__(self, q_low=0.005, q_high=0.995):
        self.q_low = q_low; self.q_high = q_high
        self.bounds_ = {}

    def fit(self, X, y=None):
        Xdf = X if hasattr(X, "columns") else pd.DataFrame(X)
        for c in Xdf.select_dtypes(include=[np.number]).columns:
            s = pd.to_numeric(Xdf[c], errors="coerce").dropna()
            if len(s) >= 50:
                lo, hi = float(s.quantile(self.q_low)), float(s.quantile(self.q_high))
                if math.isfinite(lo) and math.isfinite(hi):
                    self.bounds_[c] = (lo, hi)
        return self

    def transform(self, X):
        Xdf = X if hasattr(X, "columns") else pd.DataFrame(X)
        for c, (lo, hi) in self.bounds_.items():
            if c in Xdf.columns:
                Xdf[c] = np.clip(pd.to_numeric(Xdf[c], errors="coerce"), lo, hi)
        return Xdf.values

class MissingPatternEncoder(BaseEstimator, TransformerMixin):
    """Add missing-count/fraction and (optional) low-rank pattern embeddings."""
    def __init__(self, n_svd=5):
        self.n_svd = n_svd
        self.svd_ = None

    def fit(self, X, y=None):
        mask = X.isna().astype(float)
        from sklearn.decomposition import TruncatedSVD
        if mask.shape[1] >= 5 and mask.shape[0] >= 100:
            self.svd_ = TruncatedSVD(n_components=min(self.n_svd, mask.shape[1]-1), random_state=42)
            self.svd_.fit(mask)
        return self

    def transform(self, X):
        X = X.copy()
        mask = X.isna().astype(float)
        X["missing_count"] = mask.sum(axis=1)
        X["missing_frac"]  = X["missing_count"] / max(1, mask.shape[1])
        if self.svd_ is not None:
            Z = self.svd_.transform(mask)
            for i in range(Z.shape[1]):
                X[f"miss_svd_{i+1}"] = Z[:, i]
        return X

# ---------- Preprocessor ----------
def build_preprocessor(Xschema: pd.DataFrame):
    num_cols = Xschema.select_dtypes(include=[float, int, "float64", "int64"]).columns.tolist()
    cat_cols = Xschema.select_dtypes(exclude=[float, int, "float64", "int64"]).columns.tolist()
    num_vals = Pipeline([("winsor", WinsorizeTransformer(0.005, 0.995)),
                         ("imp", SimpleImputer(strategy="median"))])
    num_ind  = MissingIndicator(features="all")
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", _make_ohe())])
    pre = ColumnTransformer([
        ("num_vals", num_vals, num_cols),
        ("num_ind",  num_ind,  num_cols),
        ("cat",      cat_pipe, cat_cols),
    ], remainder="drop", sparse_threshold=0.0)
    return pre, num_cols, cat_cols

def get_feature_names_from_pre(pre: ColumnTransformer) -> Tuple[List[str], List[str]]:
    """Return expanded feature names after ColumnTransformer and the ordered numeric-column list."""
    name_to_cols = {name: cols for name, _, cols in pre.transformers}
    num_cols = list(name_to_cols.get("num_vals", []))
    cat_cols = list(name_to_cols.get("cat", []))
    num_names  = list(num_cols)
    miss_names = [f"{c}_missing" for c in num_cols]
    cat_names = []
    try:
        oh = pre.named_transformers_["cat"].named_steps["oh"]
        if hasattr(oh, "get_feature_names_out"):
            cat_names = list(oh.get_feature_names_out(cat_cols))
        else:
            cats = getattr(oh, "categories_", None)
            if cats is not None:
                for c, levels in zip(cat_cols, cats):
                    cat_names += [f"{c}_{str(v)}" for v in levels]
    except Exception:
        cat_names = []
    return num_names + miss_names + cat_names, num_cols

def read_csv_smart(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False, encoding="latin1")

# ---------- Load & clean ----------
df = read_csv_smart(INPUT)

# locate target column (HF)
rx = re.compile(r"(?i)^hf$")
hits = [c for c in df.columns if rx.search(c)]
if not hits:
    for guess in [r"(?i)^heart.?failure$", r"(?i)hf_label$", r"(?i)label_hf$"]:
        rx2 = re.compile(guess)
        hits = [c for c in df.columns if rx2.search(c)]
        if hits: break
if not hits:
    raise ValueError("Target column not found (try ^hf$).")
y_col = hits[0]

# drop fully empty cols
empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
if empty_cols:
    df = df.drop(columns=empty_cols)

# ensure y in {0,1}
y_map = {"yes":1,"y":1,"true":1,"t":1,"pos":1,"positive":1,"1":1,
         "no":0,"n":0,"false":0,"f":0,"neg":0,"negative":0,"0":0}
df[y_col] = df[y_col].map(lambda v: y_map.get(str(v).strip().lower(), v))
df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)

# features
X = df.drop(columns=[y_col]).copy()
X = X.drop(columns=[c for c in X.columns if ID_PAT.search(c)], errors="ignore")
for c in X.columns:
    if X[c].dtype == object:
        X[c] = pd.to_numeric(X[c], errors="ignore")
y = df[y_col].astype(int).values

# ---------- Split (stratified 80/20) ----------
X_train_df, X_valid_df, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42, shuffle=True
)
print(f"Split: train={len(y_train)}, valid={len(y_valid)} | "
      f"pos_rate train={y_train.mean():.4f}, valid={y_valid.mean():.4f}")

# ---------- Schema & pre ----------
Xs = SafeFeatureMaker().fit(X_train_df).transform(X_train_df.copy())
mpe = MissingPatternEncoder().fit(Xs)
Xschema = mpe.transform(Xs)
pre, num_cols, cat_cols = build_preprocessor(Xschema)
pre.fit(Xschema)

Xt_train = pre.transform(mpe.transform(SafeFeatureMaker().transform(X_train_df.copy())))
Xt_train = np.asarray(Xt_train)

# ---------- Monotone constraints ----------
feat_names, num_cols_order = get_feature_names_from_pre(pre)
mono_base = build_monotone_map(num_cols_order)
# Columns expand as: [num_vals ...] + [num_missing_indicators ...] + [one-hot ...]
oh_len = len(feat_names) - len(num_cols_order) - len(num_cols_order)
mono_vec = mono_base + [0] * len(num_cols_order) + [0] * max(0, oh_len)

clfA = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.5, reg_lambda=1.0,
    random_state=42, n_jobs=-1, class_weight="balanced",
    monotone_constraints=mono_vec if len(mono_vec) == Xt_train.shape[1] else None
)
clfA.fit(Xt_train, y_train)
print("Branch A trained. Monotone applied:", len(mono_vec) == Xt_train.shape[1])

# ---------- SHAP (Top-8) ----------
TOPK = 8
n_samp = min(1000, Xt_train.shape[0])
rng = np.random.RandomState(42)
idx = rng.choice(Xt_train.shape[0], n_samp, replace=False)
Xt_sample = Xt_train[idx]

if len(feat_names) != Xt_train.shape[1]:
    feat_names = [f"f{i}" for i in range(Xt_train.shape[1])]

explainer = shap.TreeExplainer(clfA)
sv_raw = explainer.shap_values(Xt_sample)
SV = sv_raw[1] if isinstance(sv_raw, list) and len(sv_raw) > 1 else (sv_raw[0] if isinstance(sv_raw, list) else sv_raw)

# Select top-8 by mean |SHAP|
imp = np.mean(np.abs(SV), axis=0)
top_idx = np.argsort(-imp)[:TOPK]
SV_top = SV[:, top_idx]
Xt_top = Xt_sample[:, top_idx]
feat_top = [feat_names[i] for i in top_idx]

# ---------- Save plots (400 dpi) ----------
bar_path   = OUTDIR / "shap_bar_CMS_BranchA_top8.png"
swarm_path = OUTDIR / "shap_beeswarm_CMS_BranchA_top8.png"

plt.figure(figsize=(8, 6), dpi=400)
shap.summary_plot(SV_top, feature_names=feat_top, plot_type='bar', show=False, max_display=TOPK)
plt.tight_layout(); plt.savefig(bar_path, dpi=400, bbox_inches="tight"); plt.close()
print("✅ Saved:", bar_path)

plt.figure(figsize=(8, 6), dpi=400)
shap.summary_plot(SV_top, features=Xt_top, feature_names=feat_top, show=False, max_display=TOPK)
plt.tight_layout(); plt.savefig(swarm_path, dpi=400, bbox_inches="tight"); plt.close()
print("✅ Saved:", swarm_path)

# ---------- Auto-download (Colab only) ----------
try:
    from google.colab import files
    files.download(str(bar_path))
    files.download(str(swarm_path))
except Exception:
    pass

print("\nDone. Files in:", OUTDIR)
