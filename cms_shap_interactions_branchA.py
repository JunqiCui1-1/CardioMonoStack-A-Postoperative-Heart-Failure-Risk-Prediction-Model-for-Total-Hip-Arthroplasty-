#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================
# SHAP Pairwise Interactions · Branch A (Monotone LGBM)
# - Loads /content/final-analysis.csv
# - Builds feature pipeline (SafeFeature -> MPE -> ColumnTransformer)
# - Trains Branch A (monotone LightGBM)
# - Computes shap_interaction_values on a sample
# - Saves:
#   * shap_interaction_heatmap_top12.png   (400 dpi)
#   * shap_interaction_top_pairs.png       (400 dpi)
#   * shap_interaction_top_pairs.csv
# =========================================

import sys, subprocess, importlib, os, re, json, math, warnings, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

def _ensure(pkg):
    try: importlib.import_module(pkg)
    except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_ensure("lightgbm"); _ensure("shap"); _ensure("scikit-learn")

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
PHYS_LIMITS = {"na":(110,170),"sodium":(110,170),"k":(1.5,7.5),"potassium":(1.5,7.5),
               "sbp":(60,260),"systolic":(60,260),"dbp":(30,160),"diastolic":(30,160)}
MONO_PLUS  = [re.compile(p,re.I) for p in [r"^age$", r"creat|creatinine", r"bnp|nt.?pro.?bnp",
                                           r"troponin", r"lactate", r"nyha", r"prior.?hf|\bhf\b",
                                           r"atrial.?fib|afib"]]
MONO_MINUS = [re.compile(p,re.I) for p in [r"egfr", r"lvef|\bef\b", r"albumin"]]

def build_monotone_map(num_cols):
    cons=[];
    for c in num_cols:
        s=0
        if any(rx.search(c) for rx in MONO_PLUS): s=1
        elif any(rx.search(c) for rx in MONO_MINUS): s=-1
        cons.append(s)
    return cons

def _make_ohe():
    params = inspect.signature(OneHotEncoder.__init__).parameters
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False) if "sparse_output" in params \
           else OneHotEncoder(handle_unknown="ignore", sparse=False)

# ---------- Feature Makers ----------
class SafeFeatureMaker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X=X.copy(); cols=list(X.columns)
        def _find(pats):
            for p in pats:
                r=re.compile(p,re.I)
                for c in cols:
                    if r.search(c): return c
            return None
        # PP / MAP
        sbp=_find([r"\b(sbp|sys|systolic)\b"]); dbp=_find([r"\b(dbp|dia|diastolic)\b"])
        if sbp and dbp and sbp in X and dbp in X:
            s=pd.to_numeric(X[sbp], errors='coerce'); d=pd.to_numeric(X[dbp], errors='coerce')
            X["pp"]=s-d; X["map_calc"]=(2*d+s)/3.0
        # BUN/Cr
        bun=_find([r"\b(bun|urea)\b"]); cr=_find([r"\b(creat|creatinine)\b"])
        if bun and cr and bun in X and cr in X:
            with np.errstate(divide='ignore', invalid='ignore'):
                X["bun_cr_ratio"]=pd.to_numeric(X[bun],errors='coerce')/pd.to_numeric(X[cr],errors='coerce')
        # Na/K
        na=_find([r"\b(na|sodium)\b"]); k=_find([r"\b(k|potassium)\b"])
        if na and k and na in X and k in X:
            with np.errstate(divide='ignore', invalid='ignore'):
                X["na_k_ratio"]=pd.to_numeric(X[na],errors='coerce')/pd.to_numeric(X[k],errors='coerce')
        # QFlags
        for key,(lo,hi) in PHYS_LIMITS.items():
            for c in X.columns:
                if re.search(fr"\b{key}\b", c, re.I):
                    v=pd.to_numeric(X[c], errors='coerce')
                    X[c+"_qflag"]=((v<lo)|(v>hi)).astype(float)
        return X

class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,q_low=0.005,q_high=0.995): self.q_low=q_low; self.q_high=q_high; self.bounds_={}
    def fit(self,X,y=None):
        Xdf=X if hasattr(X,"columns") else pd.DataFrame(X)
        for c in Xdf.select_dtypes(include=[np.number]).columns:
            s=pd.to_numeric(Xdf[c],errors="coerce").dropna()
            if len(s)>=50:
                lo,hi=float(s.quantile(self.q_low)), float(s.quantile(self.q_high))
                if math.isfinite(lo) and math.isfinite(hi): self.bounds_[c]=(lo,hi)
        return self
    def transform(self,X):
        Xdf=X if hasattr(X,"columns") else pd.DataFrame(X)
        for c,(lo,hi) in self.bounds_.items():
            if c in Xdf.columns:
                Xdf[c]=np.clip(pd.to_numeric(Xdf[c],errors="coerce"),lo,hi)
        return Xdf.values

class MissingPatternEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,n_svd=5): self.n_svd=n_svd; self.svd_=None
    def fit(self,X,y=None):
        mask=X.isna().astype(float)
        from sklearn.decomposition import TruncatedSVD
        if mask.shape[1]>=5 and mask.shape[0]>=100:
            self.svd_=TruncatedSVD(n_components=min(self.n_svd, mask.shape[1]-1), random_state=42)
            self.svd_.fit(mask)
        return self
    def transform(self,X):
        X=X.copy(); mask=X.isna().astype(float)
        X["missing_count"]=mask.sum(axis=1)
        X["missing_frac"]=X["missing_count"]/max(1,mask.shape[1])
        if self.svd_ is not None:
            Z=self.svd_.transform(mask)
            for i in range(Z.shape[1]):
                X[f"miss_svd_{i+1}"]=Z[:,i]
        return X

def build_preprocessor(Xschema: pd.DataFrame):
    num_cols = Xschema.select_dtypes(include=[float,int,"float64","int64"]).columns.tolist()
    cat_cols = Xschema.select_dtypes(exclude=[float,int,"float64","int64"]).columns.tolist()
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

def get_feature_names_from_pre(pre):
    name_to_cols = {name: cols for name, _, cols in pre.transformers}
    num_cols = list(name_to_cols.get("num_vals", []))
    cat_cols = list(name_to_cols.get("cat", []))
    num_names  = list(num_cols)
    miss_names = [f"{c}_missing" for c in num_cols]
    cat_names=[]
    try:
        oh = pre.named_transformers_["cat"].named_steps["oh"]
        if hasattr(oh,"get_feature_names_out"):
            cat_names = list(oh.get_feature_names_out(cat_cols))
        else:
            cats = getattr(oh,"categories_", None)
            if cats is not None:
                for c, levels in zip(cat_cols, cats):
                    cat_names += [f"{c}_{str(v)}" for v in levels]
    except Exception:
        cat_names=[]
    return num_names + miss_names + cat_names, num_cols

def read_csv_smart(path):
    try: return pd.read_csv(path, low_memory=False)
    except Exception: return pd.read_csv(path, low_memory=False, encoding="latin1")

# ---------- Load & clean ----------
df = read_csv_smart(INPUT)
rx = re.compile(r"(?i)^hf$"); hits=[c for c in df.columns if rx.search(c)]
if not hits:
    for guess in [r"(?i)^heart.?failure$", r"(?i)hf_label$", r"(?i)label_hf$"]:
        rx2=re.compile(guess); hits=[c for c in df.columns if rx2.search(c)]
        if hits: break
if not hits: raise ValueError("Target column not found (try ^hf$).")
y_col = hits[0]

empty_cols=[c for c in df.columns if df[c].notna().sum()==0]
if empty_cols: df=df.drop(columns=empty_cols)

y_map={"yes":1,"y":1,"true":1,"t":1,"pos":1,"positive":1,"1":1,
       "no":0,"n":0,"false":0,"f":0,"neg":0,"negative":0,"0":0}
df[y_col] = df[y_col].map(lambda v: y_map.get(str(v).strip().lower(), v))
df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)

X = df.drop(columns=[y_col]).copy()
X = X.drop(columns=[c for c in X.columns if ID_PAT.search(c)], errors="ignore")
for c in X.columns:
    if X[c].dtype==object:
        X[c] = pd.to_numeric(X[c], errors="ignore")
y = df[y_col].astype(int).values

# ---------- Split ----------
X_train_df, X_valid_df, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42, shuffle=True
)
print(f"Split: train={len(y_train)}, valid={len(y_valid)} | pos_rate train={y_train.mean():.4f}, valid={y_valid.mean():.4f}")

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
# Fallback if feature names mismatch transformed matrix width
if len(feat_names) != Xt_train.shape[1]:
    feat_names = [f"f{i}" for i in range(Xt_train.shape[1])]
mono_base = build_monotone_map(num_cols_order)
oh_len = len(feat_names) - len(num_cols_order) - len(num_cols_order)
mono_vec = mono_base + [0]*len(num_cols_order) + [0]*max(0, oh_len)

clfA = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.5, reg_lambda=1.0,
    random_state=42, n_jobs=-1, class_weight="balanced",
    monotone_constraints=mono_vec if len(mono_vec)==Xt_train.shape[1] else None
)
clfA.fit(Xt_train, y_train)
print("Branch A trained. Monotone applied:", len(mono_vec)==Xt_train.shape[1])

# ---------- SHAP values (sample) ----------
n_samp = min(300, Xt_train.shape[0])  # limit for memory
rng = np.random.RandomState(42)
idx = rng.choice(Xt_train.shape[0], n_samp, replace=False)
Xt_sample = Xt_train[idx]

# Main-effect SHAP (for top features)
expl = shap.TreeExplainer(clfA)
sv = expl.shap_values(Xt_sample)
SV = sv[1] if isinstance(sv, list) and len(sv)>1 else (sv[0] if isinstance(sv, list) else sv)
imp = np.mean(np.abs(SV), axis=0)

# Interaction SHAP (can be heavy; we use the same small sample)
print("Computing shap_interaction_values on sample ...")
SI = expl.shap_interaction_values(Xt_sample)  # shape: (n_samp, d, d)
SI_mean = np.mean(np.abs(SI), axis=0)        # (d, d)

# ---------- Viz 1: Interaction heatmap (Top-12 features) ----------
TOPM = min(12, SI_mean.shape[0])
top_idx = np.argsort(-imp)[:TOPM]
SI_top = SI_mean[np.ix_(top_idx, top_idx)]
feat_top = [feat_names[i] for i in top_idx]

plt.figure(figsize=(8, 6), dpi=400)
plt.imshow(SI_top, cmap="YlGn", aspect="auto")
plt.colorbar(label="Mean |SHAP Interaction|")
plt.xticks(range(TOPM), feat_top, rotation=70, ha="right", fontsize=8)
plt.yticks(range(TOPM), feat_top, fontsize=8)
plt.title("SHAP Interaction Heatmap (Top-12 features)")
plt.tight_layout()
heatmap_path = OUTDIR / "shap_interaction_heatmap_top12.png"
plt.savefig(heatmap_path, dpi=400, bbox_inches="tight")
plt.close()
print("✅ Saved:", heatmap_path)

# ---------- Viz 2: Top-12 interacting pairs (bar) ----------
pairs = []
for i in range(SI_mean.shape[0]):
    for j in range(i+1, SI_mean.shape[1]):  # upper triangle only
        pairs.append((i, j, SI_mean[i, j]))
pairs.sort(key=lambda t: -t[2])
TOPP = min(12, len(pairs))
pairs_top = pairs[:TOPP]
names_pair = [f"{feat_names[i]} × {feat_names[j]}" for i, j, _ in pairs_top]
vals_pair  = [float(v) for _, _, v in pairs_top]

plt.figure(figsize=(8, 6), dpi=400)
ypos = np.arange(TOPP)[::-1]
plt.barh(ypos, vals_pair[::-1], color="#6BAF46")
plt.yticks(ypos, names_pair[::-1], fontsize=8)
plt.xlabel("Mean |SHAP Interaction|")
plt.title("Top-12 Feature Pairs by Interaction Strength")
plt.tight_layout()
pairs_bar_path = OUTDIR / "shap_interaction_top_pairs.png"
plt.savefig(pairs_bar_path, dpi=400, bbox_inches="tight")
plt.close()
print("✅ Saved:", pairs_bar_path)

# Save CSV
pairs_df = pd.DataFrame({
    "feat_i": [feat_names[i] for i, _, _ in pairs_top],
    "feat_j": [feat_names[j] for _, j, _ in pairs_top],
    "mean_abs_interaction": vals_pair
})
pairs_csv_path = OUTDIR / "shap_interaction_top_pairs.csv"
pairs_df.to_csv(pairs_csv_path, index=False)
print("✅ Saved:", pairs_csv_path)

# ---------- Auto-download (Colab) ----------
try:
    from google.colab import files
    for p in [heatmap_path, pairs_bar_path, pairs_csv_path]:
        files.download(str(p))
except Exception:
    pass

print("\nDone. Files in:", OUTDIR)
