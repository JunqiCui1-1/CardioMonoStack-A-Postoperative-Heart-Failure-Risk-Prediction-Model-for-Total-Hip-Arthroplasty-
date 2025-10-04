#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMRS++ (Colab-ready) — Research-grade HF=1 classifier with saved artifacts
==========================================================================
This version is cleaned and **Colab-friendly**:
- Works via CLI (`!python pmrs_pp_hf.py --input "..."`) or programmatic call.
- Trains **full models** for each enabled branch (A/B/C) + **meta stacker** and optional **residual head**.
- **Saves all artifacts** for future comparison with baselines:
  models/pmrs_predictor.joblib (one-file predictor), models/branch_*.joblib, models/meta.joblib,
  models/residual.joblib, models/baseline_*.joblib, plus metrics JSON/CSVs.
- Includes **baseline** models (Logistic, unconstrained LGBM) for side-by-side metrics.

USAGE (Colab):
  %pip install -q pandas numpy scikit-learn lightgbm shap matplotlib catboost interpret joblib
  !python pmrs_pp_hf.py \
    --input "/content/inspire final.csv" \
    --target_regex "(?i)^hf$" \
    --outdir "/content/pmrs_pp" \
    --enable_branch_b 1 --enable_branch_c 1 \
    --enable_residual_head 1 --enable_interactions 1 \
    --enable_conformal 1 --dro_group_regex "(?i)(subject|patientunit)" \
    --goal f1 --min_precision 0.70

Programmatic:
  from pmrs_pp_hf import train_main
  train_main(input_path="/content/inspire final.csv", outdir="/content/pmrs_pp")
"""

import argparse, os, re, json, math, warnings, itertools, datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score, precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# Optional packages
try:
    import lightgbm as lgb
except Exception:
    raise SystemExit("LightGBM is not installed. Please run: pip install lightgbm")

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    _HAS_EBM = True
except Exception:
    _HAS_EBM = False

# -----------------------
# Helper transformers & utils
# -----------------------
ID_PAT = re.compile(r"(subject|hadm|icustay|patientunit|stayid|admission|visit|encounter|zid)$", re.I)

MONO_PLUS = [  # expected higher -> higher HF risk
    re.compile(p, re.I) for p in [
        r"^age$", r"creat|creatinine", r"bnp|nt.?pro.?bnp", r"troponin", r"lactate",
        r"nyha", r"prior.?hf|\bhf\b", r"atrial.?fib|afib",
    ]
]
MONO_MINUS = [  # expected higher -> lower HF risk
    re.compile(p, re.I) for p in [r"egfr", r"lvef|\bef\b", r"albumin"]
]

PHYS_LIMITS = {
    # coarse physiological ranges for QFlags (edit as needed)
    "na": (110, 170),
    "sodium": (110, 170),
    "k": (1.5, 7.5),
    "potassium": (1.5, 7.5),
    "sbp": (60, 260),
    "systolic": (60, 260),
    "dbp": (30, 160),
    "diastolic": (30, 160),
}

class SafeFeatureMaker(BaseEstimator, TransformerMixin):
    """Create clinically meaningful features if source cols exist.
    New features (if inputs exist): pp, map_calc, bun_cr_ratio, na_k_ratio
    """
    def __init__(self):
        self.new_cols_ = []

    @staticmethod
    def _find(cols: List[str], pats: List[str]) -> Optional[str]:
        for p in pats:
            r = re.compile(p, re.I)
            for c in cols:
                if r.search(c):
                    return c
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(X.columns)

        # SBP/DBP -> PP/MAP
        sbp = self._find(cols, [r"\b(sbp|sys|systolic)\b"])
        dbp = self._find(cols, [r"\b(dbp|dia|diastolic)\b"])
        if sbp and dbp and sbp in X and dbp in X:
            s = pd.to_numeric(X[sbp], errors='coerce')
            d = pd.to_numeric(X[dbp], errors='coerce')
            X['pp'] = s - d
            X['map_calc'] = (2*d + s)/3.0
            self.new_cols_ += ['pp', 'map_calc']

        # BUN/Cr
        bun = self._find(cols, [r"\b(bun|urea)\b"])
        cr = self._find(cols, [r"\b(creat|creatinine)\b"])
        if bun and cr and bun in X and cr in X:
            with np.errstate(divide='ignore', invalid='ignore'):
                X['bun_cr_ratio'] = pd.to_numeric(X[bun], errors='coerce') / pd.to_numeric(X[cr], errors='coerce')
            self.new_cols_ += ['bun_cr_ratio']

        # Na/K
        na = self._find(cols, [r"\b(na|sodium)\b"])
        k = self._find(cols, [r"\b(k|potassium)\b"])
        if na and k and na in X and k in X:
            with np.errstate(divide='ignore', invalid='ignore'):
                X['na_k_ratio'] = pd.to_numeric(X[na], errors='coerce') / pd.to_numeric(X[k], errors='coerce')
            self.new_cols_ += ['na_k_ratio']

        return X

class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, q_low=0.005, q_high=0.995):
        self.q_low = q_low; self.q_high = q_high
        self.bounds_: Dict[str, Tuple[float,float]] = {}

    def fit(self, X, y=None):
        Xn = X.select_dtypes(include=[np.number])
        for c in Xn.columns:
            s = Xn[c].dropna()
            if len(s) >= 50:
                lo, hi = float(s.quantile(self.q_low)), float(s.quantile(self.q_high))
                if math.isfinite(lo) and math.isfinite(hi):
                    self.bounds_[c] = (lo, hi)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            if c in X.columns:
                X[c] = np.clip(pd.to_numeric(X[c], errors='coerce'), lo, hi)
        # quality flags
        for key, (lo, hi) in PHYS_LIMITS.items():
            for c in X.columns:
                if re.search(fr"\b{key}\b", c, re.I):
                    v = pd.to_numeric(X[c], errors='coerce')
                    X[c+"_qflag"] = ((v < lo) | (v > hi)).astype(float)
        return X

class MissingPatternEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_svd=5):
        self.n_svd = n_svd
        self.svd_: Optional[TruncatedSVD] = None
        self.cols_: List[str] = []

    def fit(self, X, y=None):
        self.cols_ = list(X.columns)
        mask = X.isna().astype(float)
        if mask.shape[1] >= 5 and mask.shape[0] >= 100:
            self.svd_ = TruncatedSVD(n_components=min(self.n_svd, mask.shape[1]-1), random_state=42)
            self.svd_.fit(mask)
        return self

    def transform(self, X):
        X = X.copy()
        mask = X.isna().astype(float)
        X['missing_count'] = mask.sum(axis=1)
        X['missing_frac'] = X['missing_count'] / max(1, mask.shape[1])
        if self.svd_ is not None:
            Z = self.svd_.transform(mask)
            for i in range(Z.shape[1]):
                X[f'miss_svd_{i+1}'] = Z[:, i]
        return X

# Monotone constraints map aligned to numeric columns
def build_monotone_map(num_cols: List[str]) -> List[int]:
    cons = []
    for c in num_cols:
        sign = 0
        for rx in MONO_PLUS:
            if rx.search(c):
                sign = 1; break
        if sign == 0:
            for rx in MONO_MINUS:
                if rx.search(c):
                    sign = -1; break
        cons.append(sign)
    return cons

# Utility: metrics & thresholding
def metrics_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        threshold=float(thr),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )

def tune_threshold(y_true, y_prob, goal="f1", min_precision=None, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best = {"score": -1, "threshold": 0.5, "precision": 0, "recall": 0, "f1": 0}
    for thr in thresholds:
        m = metrics_at_threshold(y_true, y_prob, thr)
        if min_precision is not None and m["precision"] < min_precision:
            continue
        score = m[goal]
        if score > best["score"]:
            best = {"score": score, **m}
    return best

# Decision Curve Analysis (Net Benefit)
def net_benefit(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    N = len(y_true)
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    pt = thr
    return (tp/N) - (fp/N) * (pt/(1-pt+1e-12))

def decision_curve(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    return [(float(t), float(net_benefit(y_true, y_prob, t))) for t in thresholds]

# Conformal selective prediction (split-conformal)
def conformal_thresholds(y_calib, p_calib, alpha=0.1):
    scores = np.where(y_calib==1, 1.0 - p_calib, p_calib)
    q = np.quantile(scores, 1-alpha, method='higher') if hasattr(np, 'quantile') else np.percentile(scores, 100*(1-alpha))
    return float(q)

def apply_conformal(p, q):
    pred_set = []
    for pi in p:
        s_pos = 1.0 - pi
        s_neg = pi
        take1 = s_pos <= q
        take0 = s_neg <= q
        if take1 and not take0:
            pred_set.append({1})
        elif take0 and not take1:
            pred_set.append({0})
        else:
            pred_set.append({0,1})
    return pred_set

# Group reweighting (approx DRO)
def group_sample_weights(groups: Optional[ArrayLike], worst_group: Optional[int]=None, upweight=2.0):
    if groups is None:
        return None
    groups = np.asarray(groups)
    vals, counts = np.unique(groups, return_counts=True)
    inv = {v: 1.0/c for v,c in zip(vals, counts)}
    w = np.vectorize(lambda g: inv.get(g, 1.0))(groups)
    if worst_group is not None:
        w[groups==worst_group] *= upweight
    w = w * (len(w)/np.sum(w))
    return w

# -----------------------
# Core builders
# -----------------------
def find_target(df: pd.DataFrame, target_regex: str) -> str:
    rx = re.compile(target_regex)
    matches = [c for c in df.columns if rx.search(c)]
    if not matches:
        for guess in [r"^hf$", r"^heart.?failure$", r"hf_label", r"label_hf"]:
            rx2 = re.compile(guess, re.I)
            matches = [c for c in df.columns if rx2.search(c)]
            if matches:
                break
    if not matches:
        raise ValueError("Target column (HF) not found. Please specify with --target_regex (e.g., --target_regex '(?i)^hf$').")
    return matches[0]

def detect_groups(df: pd.DataFrame, regex: Optional[str]=None) -> Optional[str]:
    if regex:
        for c in df.columns:
            if re.search(regex, c):
                return c
    for c in df.columns:
        if ID_PAT.search(c):
            return c
    return None

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("winsor", WinsorizeTransformer(0.005, 0.995)),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ]), cat_cols)
    ], remainder="drop")
    return pre, num_cols, cat_cols

# Branch factories
def make_branch_A(num_cols):
    mono_map = build_monotone_map(num_cols)
    clf = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
        random_state=42, n_jobs=-1, class_weight="balanced",
        monotone_constraints=mono_map if mono_map else None,
    )
    return clf

def make_branch_B():
    if _HAS_CATBOOST:
        return CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05, loss_function='Logloss', eval_metric='AUC',
            random_seed=42, verbose=False, auto_class_weights='Balanced'
        )
    else:
        return lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.05, num_leaves=63, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.5, reg_lambda=1.0, random_state=1337, n_jobs=-1, class_weight="balanced"
        )

def make_branch_C():
    if _HAS_EBM:
        return ExplainableBoostingClassifier(interactions=10, max_bins=256, outer_bags=8, inner_bags=8, random_state=7)
    else:
        return LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=200)

# Residual head (LGBMRegressor)
def make_residual_head():
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0, random_state=2025, n_jobs=-1
    )

# -----------------------
# CV fit util (OOF)
# -----------------------
def fit_branch_with_cv(X: pd.DataFrame, y: ArrayLike, pre: ColumnTransformer, base_clf, groups: Optional[ArrayLike],
                        n_splits=5, random_state=42, sample_weight: Optional[ArrayLike]=None, calibrate=True):
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splitter = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splitter = cv.split(X, y)

    oof = np.zeros(len(y), dtype=float)
    models = []
    for fold, (tr, te) in enumerate(splitter, 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]
        sw_tr = None if sample_weight is None else np.asarray(sample_weight)[tr]

        pipe = Pipeline([
            ("feat", SafeFeatureMaker()),
            ("miss", MissingPatternEncoder()),
            ("pre", pre),
            ("clf", base_clf),
        ])
        if calibrate:
            model = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
        else:
            model = pipe
        model.fit(X_tr, y_tr, **({"sample_weight": sw_tr} if sw_tr is not None else {}))
        prob_te = model.predict_proba(X_te)[:,1]
        oof[te] = prob_te
        models.append(model)
    return oof, models

# Meta stacking
def fit_stacker(oof_list: List[np.ndarray], y: ArrayLike):
    Z = np.column_stack(oof_list)
    meta = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', max_iter=500)
    meta.fit(Z, y)
    p = meta.predict_proba(Z)[:,1]
    return meta, p

# Residual head training (additive on logit scale, OOF)
def add_residual_head_oof(pred_prob: np.ndarray, X: pd.DataFrame, y: ArrayLike, pre: ColumnTransformer, groups: Optional[ArrayLike],
                          reg_model=None, n_splits=5, random_state=123):
    if reg_model is None:
        reg_model = make_residual_head()
    eps = 1e-6
    z = np.log(np.clip(pred_prob, eps, 1-eps) / np.clip(1-pred_prob, eps, 1-eps))
    r = y - 1/(1+np.exp(-z))

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splitter = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splitter = cv.split(X, y)

    oof_delta = np.zeros_like(pred_prob)
    models = []

    for tr, te in splitter:
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        r_tr = r[tr]
        pipe = Pipeline([
            ("feat", SafeFeatureMaker()),
            ("miss", MissingPatternEncoder()),
            ("pre", pre),
            ("reg", reg_model),
        ])
        pipe.fit(X_tr, r_tr)
        delta_te = pipe.predict(X_te)
        oof_delta[te] = delta_te
        models.append(pipe)

    p_adj = 1/(1+np.exp(-(z + oof_delta)))
    return oof_delta, p_adj, models

# -----------------------
# Full-fit models for deployment
# -----------------------
def fit_full_branch(X: pd.DataFrame, y: ArrayLike, pre: ColumnTransformer, base_clf, sample_weight=None, calibrate=True):
    pipe = Pipeline([
        ("feat", SafeFeatureMaker()),
        ("miss", MissingPatternEncoder()),
        ("pre", pre),
        ("clf", base_clf),
    ])
    model = CalibratedClassifierCV(pipe, method="isotonic", cv=5) if calibrate else pipe
    model.fit(X, y, **({"sample_weight": sample_weight} if sample_weight is not None else {}))
    return model

class PMRSPredictor:
    """One-file predictor for later inference and baseline comparison."""
    def __init__(self, branchA=None, branchB=None, branchC=None, meta=None, residual=None):
        self.branchA = branchA
        self.branchB = branchB
        self.branchC = branchC
        self.meta = meta
        self.residual = residual

    @staticmethod
    def _sigmoid(z):
        return 1/(1+np.exp(-z))

    @staticmethod
    def _logit(p, eps=1e-6):
        p = np.clip(p, eps, 1-eps)
        return np.log(p/(1-p))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = []
        if self.branchA is not None:
            probs.append(self.branchA.predict_proba(X)[:,1])
        if self.branchB is not None:
            probs.append(self.branchB.predict_proba(X)[:,1])
        if self.branchC is not None:
            probs.append(self.branchC.predict_proba(X)[:,1])
        Z = np.column_stack(probs)
        p_stack = self.meta.predict_proba(Z)[:,1] if self.meta is not None else probs[0]
        if self.residual is not None:
            z = self._logit(p_stack)
            delta = self.residual.predict(X)
            return self._sigmoid(z + delta)
        return p_stack

# -----------------------
# Main training orchestration
# -----------------------
def train_pipeline(df: pd.DataFrame, y_col: str, outdir: Path, args):
    outdir.mkdir(parents=True, exist_ok=True)
    models_dir = outdir/"models"; models_dir.mkdir(exist_ok=True)

    y = df[y_col].astype(int).values
    X = df.drop(columns=[y_col])

    # Drop obvious IDs
    id_like = [c for c in X.columns if ID_PAT.search(c)]
    if id_like:
        X = X.drop(columns=id_like)

    # Coerce numeric where possible
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors='ignore')

    # Preprocessor
    pre, num_cols, cat_cols = build_preprocessor(X)

    # DRO-like sample weights
    groups_col = detect_groups(df, args.dro_group_regex)
    groups = df[groups_col].values if groups_col else None
    sw = group_sample_weights(groups, worst_group=None, upweight=args.dro_upweight) if args.enable_dro else None

    # --- Branches (OOF) ---
    oof_A, models_A = fit_branch_with_cv(X, y, pre, make_branch_A(num_cols), groups, n_splits=args.folds, random_state=42, sample_weight=sw, calibrate=True)
    oof_B = oof_C = None
    models_B = models_C = []
    if args.enable_branch_b:
        oof_B, models_B = fit_branch_with_cv(X, y, pre, make_branch_B(), groups, n_splits=args.folds, random_state=777, sample_weight=sw, calibrate=True)
    if args.enable_branch_c:
        baseC = make_branch_C()
        oof_C, models_C = fit_branch_with_cv(X, y, pre, baseC, groups, n_splits=args.folds, random_state=7, sample_weight=sw, calibrate=True)

    # Save branch OOF
    out_oof = {"A": oof_A, "y": y}
    if oof_B is not None: out_oof["B"] = oof_B
    if oof_C is not None: out_oof["C"] = oof_C
    pd.DataFrame(out_oof).to_csv(outdir/"oof_branches.csv", index=False)

    # Stacker & optional residual (OOF)
    oof_list = [oof_A] + ([oof_B] if oof_B is not None else []) + ([oof_C] if oof_C is not None else [])
    meta, oof_stack = fit_stacker(oof_list, y)

    oof_delta = None
    if args.enable_residual_head:
        oof_delta, oof_final, residual_models = add_residual_head_oof(oof_stack, X, y, pre, groups, reg_model=None, n_splits=args.folds)
    else:
        oof_final = oof_stack

    # Metrics
    roc_auc = roc_auc_score(y, oof_final)
    pr_auc = average_precision_score(y, oof_final)
    best_thr = tune_threshold(y, oof_final, goal=args.goal, min_precision=args.min_precision)

    # ROC/PR curves
    fpr, tpr, _ = roc_curve(y, oof_final)
    prec, rec, thr_pr = precision_recall_curve(y, oof_final)
    pd.DataFrame({"fpr":fpr, "tpr":tpr}).to_csv(outdir/"roc_curve.csv", index=False)
    pd.DataFrame({"precision":prec, "recall":rec}).to_csv(outdir/"pr_curve.csv", index=False)

    # DCA
    dca = decision_curve(y, oof_final)
    pd.DataFrame(dca, columns=["threshold","net_benefit"]).to_csv(outdir/"dca_curve.csv", index=False)

    # Bootstrap CIs
    rng = np.random.default_rng(2024)
    R = args.bootstrap
    aucs, prs, f1s = [], [], []
    n = len(y)
    for _ in range(R):
        idx = rng.integers(0, n, n)
        y_b = y[idx]; p_b = oof_final[idx]
        try:
            aucs.append(roc_auc_score(y_b, p_b))
            prs.append(average_precision_score(y_b, p_b))
            t = tune_threshold(y_b, p_b, goal=args.goal, min_precision=args.min_precision)
            f1s.append(t["f1"])
        except Exception:
            continue

    def ci(a):
        if not a: return (None, None, None)
        a = np.array(a)
        return float(np.mean(a)), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    auc_ci = ci(aucs)
    pr_ci  = ci(prs)
    f1_ci  = ci(f1s)

    with open(outdir/"bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump({"auc_mean_lo_hi": auc_ci, "pr_mean_lo_hi": pr_ci, "f1_mean_lo_hi": f1_ci}, f, indent=2)

    # --- Baselines (OOF) ---
    # Baseline 1: Logistic (no interactions), same pre
    base_log_clf = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=400)
    oof_log, _ = fit_branch_with_cv(X, y, pre, base_log_clf, groups, n_splits=args.folds, random_state=11, sample_weight=sw, calibrate=True)
    base_log_auc = roc_auc_score(y, oof_log); base_log_pr = average_precision_score(y, oof_log)

    # Baseline 2: LGBM (no monotone)
    base_lgbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.9, colsample_bytree=0.9,
                                   class_weight='balanced', random_state=22)
    oof_lgb, _ = fit_branch_with_cv(X, y, pre, base_lgbm, groups, n_splits=args.folds, random_state=22, sample_weight=sw, calibrate=True)
    base_lgb_auc = roc_auc_score(y, oof_lgb); base_lgb_pr = average_precision_score(y, oof_lgb)

    # --- Fit FULL models for deployment ---
    model_A_full = fit_full_branch(X, y, pre, make_branch_A(num_cols), sample_weight=sw, calibrate=True)
    model_B_full = fit_full_branch(X, y, pre, make_branch_B(), sample_weight=sw, calibrate=True) if args.enable_branch_b else None
    model_C_full = fit_full_branch(X, y, pre, make_branch_C(), sample_weight=sw, calibrate=True) if args.enable_branch_c else None

    # Get full-data branch probs for residual full-fit
    probs_full = []
    probs_full.append(model_A_full.predict_proba(X)[:,1])
    if model_B_full is not None:
        probs_full.append(model_B_full.predict_proba(X)[:,1])
    if model_C_full is not None:
        probs_full.append(model_C_full.predict_proba(X)[:,1])
    Z_full = np.column_stack(probs_full)

    # Meta is already fitted on OOF; use same meta for deployment
    # Residual full-fit
    residual_full = None
    if args.enable_residual_head:
        eps = 1e-6
        p_stack_full = meta.predict_proba(Z_full)[:,1]
        z_full = np.log(np.clip(p_stack_full, eps, 1-eps)/np.clip(1-p_stack_full, eps, 1-eps))
        r_full = y - 1/(1+np.exp(-z_full))
        residual_full = Pipeline([
            ("feat", SafeFeatureMaker()),
            ("miss", MissingPatternEncoder()),
            ("pre", pre),
            ("reg", make_residual_head()),
        ])
        residual_full.fit(X, r_full)

    # --- Save artifacts ---
    # Predictor object (one-file inference)
    predictor = PMRSPredictor(branchA=model_A_full, branchB=model_B_full, branchC=model_C_full, meta=meta, residual=residual_full)
    joblib.dump(predictor, models_dir/"pmrs_predictor.joblib")
    joblib.dump(meta, models_dir/"meta.joblib")
    joblib.dump(model_A_full, models_dir/"branch_A_full.joblib")
    if model_B_full is not None:
        joblib.dump(model_B_full, models_dir/"branch_B_full.joblib")
    if model_C_full is not None:
        joblib.dump(model_C_full, models_dir/"branch_C_full.joblib")
    if residual_full is not None:
        joblib.dump(residual_full, models_dir/"residual_full.joblib")

    # Baselines (full fit) for future reuse
    base_log_full = fit_full_branch(X, y, pre, LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=400), sample_weight=sw, calibrate=True)
    base_lgb_full = fit_full_branch(X, y, pre, lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.9, colsample_bytree=0.9, class_weight='balanced', random_state=22), sample_weight=sw, calibrate=True)
    joblib.dump(base_log_full, models_dir/"baseline_logistic.joblib")
    joblib.dump(base_lgb_full, models_dir/"baseline_lgbm.joblib")

    # Save OOF and metrics summaries
    pd.DataFrame({"y": y, "p_pmrs": oof_final, "p_log": oof_log, "p_lgb": oof_lgb}).to_csv(outdir/"oof_comparison.csv", index=False)

    # Threshold tuning & summary
    with open(outdir/"threshold_tuning.json", "w", encoding="utf-8") as f:
        json.dump(best_thr, f, indent=2)

    summary = {
        "dataset_rows": int(len(df)),
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
        "roc_auc_pmrs": float(roc_auc),
        "pr_auc_pmrs": float(pr_auc),
        "auc_mean_95ci_pmrs": {"mean": auc_ci[0], "lo": auc_ci[1], "hi": auc_ci[2]},
        "pr_mean_95ci_pmrs": {"mean": pr_ci[0], "lo": pr_ci[1], "hi": pr_ci[2]},
        "best_threshold_pmrs": best_thr,
        "pr_auc_baseline_log": float(base_log_pr),
        "auc_baseline_log": float(base_log_auc),
        "pr_auc_baseline_lgb": float(base_lgb_pr),
        "auc_baseline_lgb": float(base_lgb_auc),
        "components": {
            "branch_B": bool(args.enable_branch_b),
            "branch_C": bool(args.enable_branch_c),
            "residual": bool(args.enable_residual_head)
        }
    }
    with open(outdir/"metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # SHAP plots (Branch A)
    if _HAS_SHAP:
        try:
            # Use branch A full model's inner estimator
            pipe = model_A_full.base_estimator if isinstance(model_A_full, CalibratedClassifierCV) else model_A_full
            preA = pipe.named_steps['pre'] if isinstance(pipe, Pipeline) else None
            clfA = pipe.named_steps['clf'] if isinstance(pipe, Pipeline) else None
            if preA is not None and clfA is not None:
                Xs = X.sample(n=min(1000, len(X)), random_state=42)
                Xt = preA.transform(MissingPatternEncoder().fit_transform(SafeFeatureMaker().fit_transform(Xs)))
                expl = shap.TreeExplainer(clfA)
                sv = expl.shap_values(Xt)
                SV = sv[1] if isinstance(sv, list) else sv
                # Note: feature names for SHAP are best-effort; adjust as needed for your dataset.
                plt.figure(figsize=(9,6)); shap.summary_plot(SV, features=Xt, show=False)
                plt.tight_layout(); plt.savefig(outdir/"shap_beeswarm_A.png", dpi=400); plt.close()
                plt.figure(figsize=(9,6)); shap.summary_plot(SV, features=Xt, plot_type='bar', show=False)
                plt.tight_layout(); plt.savefig(outdir/"shap_bar_A.png", dpi=400); plt.close()
        except Exception as e:
            print("SHAP plotting failed:", e)

    # Conformal selective prediction
    if args.enable_conformal:
        rs = check_random_state(2025)
        idx = np.arange(len(y)); rs.shuffle(idx)
        k = int(0.2*len(y))
        cal_idx, dev_idx = idx[:k], idx[k:]
        q = conformal_thresholds(y[cal_idx], oof_final[cal_idx], alpha=args.conformal_alpha)
        pred_sets = apply_conformal(oof_final[dev_idx], q)
        amb = np.mean([len(s)==2 for s in pred_sets])
        with open(outdir/"conformal.json", "w", encoding="utf-8") as f:
            json.dump({"alpha": args.conformal_alpha, "q": q, "ambiguous_rate": float(amb), "calib_size": int(len(cal_idx))}, f, indent=2)

    print("Artifacts saved to:", models_dir)
    return summary

# -----------------------
# Public entry points (CLI & programmatic)
# -----------------------
def train_main(input_path: str, target_regex: str = r"(?i)^hf$", outdir: str = "./pmrs_pp_out",
               enable_branch_b: int = 1, enable_branch_c: int = 1, enable_residual_head: int = 1,
               enable_interactions: int = 0,  # placeholder toggle (not used in this core)
               enable_dro: int = 1, enable_conformal: int = 1, enable_subgroup_thresh: int = 0,
               top_interactions: int = 20, folds: int = 5, goal: str = "f1", min_precision: Optional[float] = None,
               bootstrap: int = 500, dro_group_regex: Optional[str] = None, dro_upweight: float = 2.0,
               conformal_alpha: float = 0.1):
    class Args: pass
    args = Args()
    args.input = input_path
    args.target_regex = target_regex
    args.outdir = outdir
    args.enable_branch_b = enable_branch_b
    args.enable_branch_c = enable_branch_c
    args.enable_residual_head = enable_residual_head
    args.enable_interactions = enable_interactions
    args.enable_dro = enable_dro
    args.enable_conformal = enable_conformal
    args.enable_subgroup_thresh = enable_subgroup_thresh
    args.top_interactions = top_interactions
    args.folds = folds
    args.goal = goal
    args.min_precision = min_precision
    args.bootstrap = bootstrap
    args.dro_group_regex = dro_group_regex
    args.dro_upweight = dro_upweight
    args.conformal_alpha = conformal_alpha

    outdir_path = Path(outdir); outdir_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, low_memory=False)
    y_col = find_target(df, target_regex)

    # Drop fully empty columns
    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # Ensure y in {0,1}
    if df[y_col].dtype != int and df[y_col].dtype != np.int64:
        y_map = {"yes":1, "y":1, "true":1, "t":1, "pos":1, "positive":1, "1":1,
                 "no":0, "n":0, "false":0, "f":0, "neg":0, "negative":0, "0":0}
        df[y_col] = df[y_col].map(lambda v: y_map.get(str(v).strip().lower(), v))
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce').fillna(0).astype(int)

    summary = train_pipeline(df, y_col, outdir_path, args)
    # registry manifest for future comparison
    manifest = {
        "input": input_path,
        "target_regex": target_regex,
        "outdir": str(outdir),
        "time": datetime.datetime.now().isoformat(timespec='seconds')
    }
    with open(outdir_path/"registry.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--target_regex", default=r"(?i)^hf$")
    ap.add_argument("--outdir", default="./pmrs_pp_out")

    # toggles
    ap.add_argument("--enable_branch_b", type=int, default=1)
    ap.add_argument("--enable_branch_c", type=int, default=1)
    ap.add_argument("--enable_residual_head", type=int, default=1)
    ap.add_argument("--enable_interactions", type=int, default=0)  # placeholder in this core
    ap.add_argument("--enable_dro", type=int, default=1)
    ap.add_argument("--enable_conformal", type=int, default=1)
    ap.add_argument("--enable_subgroup_thresh", type=int, default=0)
    ap.add_argument("--top_interactions", type=int, default=20)

    # CV/metrics
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--goal", choices=["f1", "recall"], default="f1")
    ap.add_argument("--min_precision", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=500)

    # DRO
    ap.add_argument("--dro_group_regex", type=str, default=None, help="Regex to detect group column; if not set, will auto-detect ID-like col.")
    ap.add_argument("--dro_upweight", type=float, default=2.0)

    # Conformal
    ap.add_argument("--conformal_alpha", type=float, default=0.1)

    args = ap.parse_args()

    train_main(input_path=args.input, target_regex=args.target_regex, outdir=args.outdir,
               enable_branch_b=args.enable_branch_b, enable_branch_c=args.enable_branch_c,
               enable_residual_head=args.enable_residual_head, enable_interactions=args.enable_interactions,
               enable_dro=args.enable_dro, enable_conformal=args.enable_conformal, enable_subgroup_thresh=args.enable_subgroup_thresh,
               top_interactions=args.top_interactions, folds=args.folds, goal=args.goal, min_precision=args.min_precision,
               bootstrap=args.bootstrap, dro_group_regex=args.dro_group_regex, dro_upweight=args.dro_upweight,
               conformal_alpha=args.conformal_alpha)

if __name__ == "__main__":
    main()
