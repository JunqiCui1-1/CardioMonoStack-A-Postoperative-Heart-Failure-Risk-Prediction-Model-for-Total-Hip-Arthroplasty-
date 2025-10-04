#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CardioMonoStack vs 6 Baselines (Stratified 80/20, Holdout Isotonic Calibration)
===============================================================================

- One-shot benchmark with 7 models total:
  * CardioMonoStack (monotone-aware LightGBM wrapper)
  * CatBoost, LightGBM (free), XGBoost, RandomForest, LogisticRegression, KNN
- Pipeline includes:
  * SafeFeatureMaker (PP/MAP, BUN/Cr, Na/K, QFlags)
  * MissingPatternEncoder (missing_count/frac + SVD comps)
  * ColumnTransformer (winsorization + imputation + OHE)
- Holdout isotonic calibration on a stratified 20% of the TRAIN split.
- Threshold tuning on the calibration fold (F1 by default).
- Evaluation on the untouched VALID split.
- Outputs: metrics table CSV, ROC overlay, reliability diagram, confusion matrices,
  per-model calibrated probabilities for VALID, and baseline params.

USAGE (Colab / local):
  %pip install -q numpy pandas scikit-learn lightgbm xgboost catboost matplotlib
  !python bench_cardio_monostack.py \
      --input "/content/final-analysis.csv" \
      --outdir "/content/pmrspp_out" \
      --target_regex "(?i)^hf$" \
      --seed 42 --calib_frac 0.2 --goal f1
"""

# --- Idempotent installs (only when needed) ---
import sys, subprocess, importlib
def _ensure(pkg):
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_ensure("lightgbm")
_ensure("xgboost")
_ensure("catboost")

# --- Imports ---
import os, re, json, math, warnings, inspect, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, log_loss
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ---------- Utilities ----------
ID_PAT = re.compile(r"(?:^id$|subject|hadm|icustay|patientunit|stayid|admission|visit|encounter|zid)$", re.I)

PHYS_LIMITS = {
    "na": (110, 170), "sodium": (110, 170),
    "k": (1.5, 7.5),  "potassium": (1.5, 7.5),
    "sbp": (60, 260), "systolic": (60, 260),
    "dbp": (30, 160), "diastolic": (30, 160),
}

MONO_PLUS = [re.compile(p, re.I) for p in [
    r"^age$", r"creat|creatinine", r"bnp|nt.?pro.?bnp", r"troponin", r"lactate",
    r"nyha", r"prior.?hf|\bhf\b", r"atrial.?fib|afib",
]]
MONO_MINUS = [re.compile(p, re.I) for p in [r"egfr", r"lvef|\bef\b", r"albumin"]]

def build_monotone_map(num_cols):
    cons=[]
    for c in num_cols:
        sign=0
        if any(rx.search(c) for rx in MONO_PLUS): sign=1
        elif any(rx.search(c) for rx in MONO_MINUS): sign=-1
        cons.append(sign)
    return cons

class SafeFeatureMaker(BaseEstimator, TransformerMixin):
    """Adds PP/MAP, BUN/Cr, Na/K and out-of-range flags if source columns exist."""
    def __init__(self): self.new_cols_=[]
    @staticmethod
    def _find(cols, pats):
        for p in pats:
            r=re.compile(p, re.I)
            for c in cols:
                if r.search(c): return c
        return None
    def fit(self, X, y=None): return self
    def transform(self, X):
        X=X.copy(); cols=list(X.columns)
        # SBP/DBP -> PP/MAP
        sbp=self._find(cols,[r"\b(sbp|sys|systolic)\b"])
        dbp=self._find(cols,[r"\b(dbp|dia|diastolic)\b"])
        if sbp and dbp and sbp in X and dbp in X:
            s=pd.to_numeric(X[sbp], errors='coerce'); d=pd.to_numeric(X[dbp], errors='coerce')
            X['pp']=s-d; X['map_calc']=(2*d+s)/3.0
        # BUN/Cr
        bun=self._find(cols,[r"\b(bun|urea)\b"]); cr=self._find(cols,[r"\b(creat|creatinine)\b"])
        if bun and cr and bun in X and cr in X:
            with np.errstate(divide='ignore', invalid='ignore'):
                X['bun_cr_ratio']=pd.to_numeric(X[bun], errors='coerce')/pd.to_numeric(X[cr], errors='coerce')
        # Na/K
        na=self._find(cols,[r"\b(na|sodium)\b"]); k=self._find(cols,[r"\b(k|potassium)\b"])
        if na and k and na in X and k in X:
            with np.errstate(divide='ignore', invalid='ignore'):
                X['na_k_ratio']=pd.to_numeric(X[na], errors='coerce')/pd.to_numeric(X[k], errors='coerce')
        # QFlags
        for key,(lo,hi) in PHYS_LIMITS.items():
            for c in X.columns:
                if re.search(fr"\b{key}\b", c, re.I):
                    v=pd.to_numeric(X[c], errors='coerce'); X[c+"_qflag"]=((v<lo)|(v>hi)).astype(float)
        return X

class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    """Per-column winsorization with learned bounds."""
    def __init__(self, q_low=0.005, q_high=0.995):
        self.q_low=q_low; self.q_high=q_high; self.bounds_={}; self.cols_=None
    def fit(self, X, y=None):
        Xdf=X if hasattr(X,"columns") else pd.DataFrame(X); self.cols_=list(Xdf.columns)
        for c in self.cols_:
            s=pd.to_numeric(Xdf[c], errors="coerce").dropna()
            if len(s)>=50:
                lo,hi=float(s.quantile(self.q_low)), float(s.quantile(self.q_high))
                if math.isfinite(lo) and math.isfinite(hi): self.bounds_[c]=(lo,hi)
        return self
    def transform(self, X):
        Xdf=X if hasattr(X,"columns") else pd.DataFrame(X, columns=self.cols_)
        for c,(lo,hi) in self.bounds_.items():
            Xdf[c]=np.clip(pd.to_numeric(Xdf[c], errors="coerce"), lo, hi)
        return Xdf.values

class MissingPatternEncoder(BaseEstimator, TransformerMixin):
    """Adds missing_count/frac and (optional) SVD components of missing mask."""
    def __init__(self, n_svd=5): 
        self.n_svd=n_svd; self.svd_=None
    def fit(self, X, y=None):
        from sklearn.decomposition import TruncatedSVD
        mask=X.isna().astype(float)
        if mask.shape[1]>=5 and mask.shape[0]>=100:
            self.svd_=TruncatedSVD(n_components=min(self.n_svd, mask.shape[1]-1), random_state=42)
            self.svd_.fit(mask)
        return self
    def transform(self, X):
        X=X.copy(); mask=X.isna().astype(float)
        X['missing_count']=mask.sum(axis=1); X['missing_frac']=X['missing_count']/max(1,mask.shape[1])
        if self.svd_ is not None:
            Z=self.svd_.transform(mask)
            for i in range(Z.shape[1]): X[f'miss_svd_{i+1}']=Z[:,i]
        return X

def _make_ohe():
    params = inspect.signature(OneHotEncoder.__init__).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[float, int, "float64", "int64"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[float, int, "float64", "int64"]).columns.tolist()
    num_vals = Pipeline([("winsor", WinsorizeTransformer(0.005,0.995)),
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

class MonotoneAwareLGBM(BaseEstimator, ClassifierMixin):
    """LightGBM classifier with monotone constraints aligned to numeric columns."""
    _estimator_type = "classifier"
    def __init__(self, num_cols=None, **kwargs):
        self.num_cols = num_cols
        self._kwargs  = dict(kwargs)
        self.model_   = None
    def fit(self, X, y, sample_weight=None):
        num_cols=list(self.num_cols) if self.num_cols is not None else []
        k=len(num_cols); n_total=X.shape[1]
        cons = build_monotone_map(num_cols) + [0]*k + [0]*max(0, n_total-2*k)
        params=dict(self._kwargs); params["monotone_constraints"]=cons
        self.model_ = lgb.LGBMClassifier(**params)
        fit_kw={"sample_weight":sample_weight} if sample_weight is not None else {}
        self.model_.fit(X,y,**fit_kw)
        # sklearn compatibility
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self
    def predict_proba(self, X): return self.model_.predict_proba(X)
    def predict(self, X): return self.model_.predict(X)
    def get_params(self, deep=True): return {"num_cols": self.num_cols, **self._kwargs}
    def set_params(self, **params):
        if "num_cols" in params: self.num_cols = params.pop("num_cols")
        self._kwargs.update(params); return self

# --- Metric helpers ---
def tune_threshold(y_true, y_prob, goal="f1", thresholds=None):
    if thresholds is None: thresholds = np.linspace(0.05, 0.95, 19)
    best = {"threshold":0.5, "precision":0, "recall":0, "f1":0}
    for thr in thresholds:
        yhat = (y_prob >= thr).astype(int)
        p = precision_score(y_true, yhat, zero_division=0)
        r = recall_score(y_true, yhat, zero_division=0)
        f = f1_score(y_true, yhat, zero_division=0)
        if (goal=="f1" and f>best["f1"]) or (goal=="recall" and r>best["recall"]):
            best = {"threshold":float(thr), "precision":float(p), "recall":float(r), "f1":float(f)}
    return best

def brier(y_true, p): return float(np.mean((p - y_true)**2))

def ece(y_true, p, n_bins=10):
    order = np.argsort(p)
    p_sorted, y_sorted = p[order], y_true[order]
    bins = np.array_split(np.arange(len(p_sorted)), n_bins)
    gap_sum, n = 0.0, len(p)
    for idx in bins:
        if len(idx)==0: continue
        conf = np.mean(p_sorted[idx]); acc  = np.mean(y_sorted[idx])
        gap_sum += (len(idx)/n) * abs(acc - conf)
    return float(gap_sum)

# ---------- Data I/O ----------
def read_csv_smart(path):
    try: return pd.read_csv(path, low_memory=False)
    except Exception: return pd.read_csv(path, low_memory=False, encoding="latin1")

def find_target(df: pd.DataFrame, target_regex: str) -> str:
    rx = re.compile(target_regex)
    hits = [c for c in df.columns if rx.search(c)]
    if not hits:
        for guess in [r"(?i)^hf$", r"(?i)^heart.?failure$", r"(?i)hf_label$", r"(?i)label_hf$"]:
            rx2 = re.compile(guess)
            hits = [c for c in df.columns if rx2.search(c)]
            if hits: break
    if not hits:
        raise ValueError("Target column not found (e.g., ^hf$). Use --target_regex to specify.")
    return hits[0]

# ---------- Core run ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/content/final-analysis.csv")
    ap.add_argument("--outdir", default="/content/pmrspp_out")
    ap.add_argument("--target_regex", default=r"(?i)^hf$")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calib_frac", type=float, default=0.20)
    ap.add_argument("--goal", choices=["f1","recall"], default="f1")
    args = ap.parse_args()

    INPUT = args.input
    OUTDIR = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)
    BENCH = OUTDIR / "benchmarks"; BENCH.mkdir(parents=True, exist_ok=True)

    # Load
    df = read_csv_smart(INPUT)
    y_col = find_target(df, args.target_regex)
    # normalize y
    y_map = {"yes":1,"y":1,"true":1,"t":1,"pos":1,"positive":1,"1":1,
             "no":0,"n":0,"false":0,"f":0,"neg":0,"negative":0,"0":0}
    df[y_col] = df[y_col].map(lambda v: y_map.get(str(v).strip().lower(), v))
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)

    # drop empty cols & ID-like cols from features
    empty_cols = [c for c in df.columns if df[c].notna().sum()==0]
    if empty_cols: df = df.drop(columns=empty_cols)
    X = df.drop(columns=[y_col])
    X = X.drop(columns=[c for c in X.columns if ID_PAT.search(c)], errors="ignore")
    for c in X.columns:
        if X[c].dtype==object:
            X[c] = pd.to_numeric(X[c], errors="ignore")
    y = df[y_col].astype(int).values

    # Schema fit (so ColumnTransformer sees post-MPE columns)
    Xs = SafeFeatureMaker().fit(X).transform(X.copy())
    mpe = MissingPatternEncoder().fit(Xs)
    X_schema = mpe.transform(Xs)
    pre, num_cols, cat_cols = build_preprocessor(X_schema)

    # Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.seed, shuffle=True
    )
    print(f"Split -> train:{len(y_train)} valid:{len(y_valid)} "
          f"| pos_rate train={y_train.mean():.4f}, valid={y_valid.mean():.4f}")

    # Build model pipelines
    def pipe_core(estimator, add_scaler=False):
        steps = [("feat", SafeFeatureMaker()),
                 ("miss", MissingPatternEncoder()),
                 ("pre",  pre)]
        if add_scaler:
            steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        steps.append(("clf", estimator))
        return Pipeline(steps)

    models = {}

    # 1) CardioMonoStack (monotone-aware LGBM)
    cms_est = MonotoneAwareLGBM(
        num_cols=num_cols,
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=args.seed, n_jobs=-1, class_weight="balanced"
    )
    models["CardioMonoStack"] = pipe_core(cms_est, add_scaler=False)

    # 2) CatBoost
    models["CatBoost"] = pipe_core(
        CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08,
                           loss_function='Logloss', eval_metric='AUC',
                           random_seed=args.seed, verbose=False, auto_class_weights='Balanced'),
        add_scaler=False
    )

    # 3) LightGBM (no monotone)
    models["LightGBM"] = pipe_core(
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=31,
                           subsample=0.9, colsample_bytree=0.9,
                           reg_alpha=0.0, reg_lambda=1.0, random_state=7,
                           n_jobs=-1, class_weight="balanced"),
        add_scaler=False
    )

    # 4) XGBoost
    models["XGBoost"] = pipe_core(
        XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.10,
                      subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
                      objective="binary:logistic", eval_metric="logloss",
                      random_state=2024, tree_method="hist", n_jobs=-1),
        add_scaler=False
    )

    # 5) RandomForest
    models["RF"] = pipe_core(
        RandomForestClassifier(n_estimators=300, max_depth=None,
                               class_weight="balanced_subsample", n_jobs=-1, random_state=1337),
        add_scaler=False
    )

    # 6) LogisticRegression
    models["LR"] = pipe_core(
        LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                           class_weight="balanced", max_iter=2000, random_state=args.seed),
        add_scaler=True
    )

    # 7) KNN
    models["KNN"] = pipe_core(
        KNeighborsClassifier(n_neighbors=25, weights="distance", p=2),
        add_scaler=True
    )

    # --- Save baseline params (excluding CardioMonoStack which is custom) ---
    baseline_params = {}
    for name in ["CatBoost","LightGBM","XGBoost","RF","LR","KNN"]:
        est = models[name].named_steps["clf"]
        try:
            baseline_params[name] = est.get_params()
        except Exception:
            baseline_params[name] = str(est)

    with open(BENCH / "baselines_params.json", "w", encoding="utf-8") as f:
        json.dump(baseline_params, f, ensure_ascii=False, indent=2)
    print("Saved six baseline params ->", BENCH / "baselines_params.json")

    # ---------- Calibration (holdout on TRAIN) ----------
    def calibrate_with_holdout(pipe, Xtr, ytr, calib_frac=0.2, seed=42):
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            Xtr, ytr, test_size=calib_frac, stratify=ytr, random_state=seed, shuffle=True
        )
        # fit on fit-split
        pipe.fit(X_fit, y_fit)
        # isotonic on calibration split
        p_cal_raw = pipe.predict_proba(X_cal)[:,1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal_raw, y_cal)
        # threshold tuning on calibrated calibration predictions
        thr_info = tune_threshold(y_cal, iso.predict(p_cal_raw), goal=args.goal)
        # refit on full TRAIN
        pipe.fit(Xtr, ytr)
        return pipe, iso, thr_info

    # ---------- Train + Calibrate + Evaluate on VALID ----------
    def eval_on_valid(pipe, iso, thr_info, Xva, yva, name):
        p_raw = pipe.predict_proba(Xva)[:,1]
        p_cal = iso.predict(p_raw)
        thr = float(thr_info["threshold"])
        yhat = (p_cal >= thr).astype(int)
        prec = precision_score(yva, yhat, zero_division=0)
        rec  = recall_score(yva, yhat, zero_division=0)
        f1   = f1_score(yva, yhat, zero_division=0)
        roc  = roc_auc_score(yva, p_cal)
        ll   = log_loss(yva, p_cal, eps=1e-15)
        br   = brier(yva, p_cal)
        ec   = ece(yva, p_cal, n_bins=10)
        # save probs for VALID
        pd.DataFrame({"y_true": yva, "p_calibrated": p_cal}).to_csv(BENCH / f"probs_valid_{name}.csv", index=False)
        return dict(
            name=name, thr=thr, precision=float(prec), recall=float(rec), f1=float(f1),
            roc_auc=float(roc), logloss=float(ll), brier=float(br), ece=float(ec),
            p_cal=p_cal, y=yva
        )

    results = {}
    for name, pipe in models.items():
        print(f"[{name}] training & calibrating ...")
        fitted, iso, thr_info = calibrate_with_holdout(pipe, X_train, y_train, calib_frac=args.calib_frac, seed=args.seed)
        res = eval_on_valid(fitted, iso, thr_info, X_valid, y_valid, name)
        results[name] = {"fitted": fitted, "iso": iso, "thr_info": thr_info, "eval": res}

    # ---------- Plots: ROC overlay (valid) ----------
    plt.figure(figsize=(7.2,5.2))
    for name in results:
        yv = results[name]["eval"]["y"]; p = results[name]["eval"]["p_cal"]
        fpr, tpr, _ = roc_curve(yv, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],'--',label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC (Valid, Calibrated)")
    plt.legend()
    plt.tight_layout(); plt.savefig(BENCH / "bench_roc_overlay.png", dpi=400); plt.close()
    print("Saved ROC overlay ->", BENCH / "bench_roc_overlay.png")

    # ---------- Plots: Calibration overlay (valid) ----------
    def calib_points(y, p, n_bins=10):
        order = np.argsort(p); p_sorted = p[order]; y_sorted = y[order]
        bins = np.array_split(np.arange(len(p_sorted)), n_bins)
        x = []; yout=[]
        for idx in bins:
            if len(idx)==0: continue
            x.append(np.mean(p_sorted[idx])); yout.append(np.mean(y_sorted[idx]))
        return np.array(x), np.array(yout)

    plt.figure(figsize=(7.2,5.2))
    grid = np.linspace(0,1,101)
    plt.plot(grid, grid, linestyle="--", label="Perfect")
    for name in results:
        yv = results[name]["eval"]["y"]; p = results[name]["eval"]["p_cal"]
        xpt, ypt = calib_points(yv, p, n_bins=10)
        plt.plot(xpt, ypt, marker="o", linestyle="-", label=name)
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical frequency")
    plt.title("Reliability Diagram (Valid, Calibrated)")
    plt.legend()
    plt.tight_layout(); plt.savefig(BENCH / "benc_
