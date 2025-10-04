#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D · CardioMonoStack Visualization (Yellow→Green, publication-grade PNG/HTML)
==============================================================================
Generates:
  - PDP probability surface:      cms_pdp_surface_<f1>_<f2>.html/.png
  - ICE single-instance surface:  cms_ice_surface_idx<id>_<f1>_<f2>.html/.png
  - Residual Δ(logit) surface:    cms_residual_delta_surface_<f1>_<f2>.html/.png (if residual exists)

Usage (Colab/local):
  %pip install -q plotly kaleido joblib numpy pandas
  !python cms_3d_surfaces.py \
      --input "/content/final-analysis.csv" \
      --target_regex "(?i)^hf$" \
      --outdir "/content/pmrspp_out/benchmarks" \
      --grid_n 30

Notes
- If run in the same session after bench_cardio_monostack.py, it will auto-grab:
    results["CardioMonoStack"]["fitted"] as the predictor and X/y splits if available.
- Otherwise, it tries to load PMRS++ predictor at:
    /content/pmrs_pp/models/pmrs_predictor.joblib
"""

import sys, subprocess, importlib, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

def _ensure(pkg):
    try: importlib.import_module(pkg)
    except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
_ensure("plotly"); _ensure("kaleido"); _ensure("joblib")

import joblib
import plotly.graph_objects as go

# ---------------- Paths / constants ----------------
DEFAULT_PMRS_PATH = "/content/pmrs_pp/models/pmrs_predictor.joblib"
ID_PAT = re.compile(r"(?:^id$|subject|hadm|icustay|patientunit|stayid|admission|visit|encounter|zid)$", re.I)

DPI_TARGET = 400
FIG_W_IN, FIG_H_IN = 7.0, 5.0
PNG_W, PNG_H = int(DPI_TARGET*FIG_W_IN), int(DPI_TARGET*FIG_H_IN)

# ---------------- Small utils ----------------
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
        raise ValueError("Target column not found (e.g., ^hf$). Use --target_regex.")
    return hits[0]

def normalize_binary_series(s: pd.Series) -> pd.Series:
    y_map = {"yes":1,"y":1,"true":1,"t":1,"pos":1,"positive":1,"1":1,
             "no":0,"n":0,"false":0,"f":0,"neg":0,"negative":0,"0":0}
    s2 = s.map(lambda v: y_map.get(str(v).strip().lower(), v))
    return pd.to_numeric(s2, errors="coerce").fillna(0).astype(int)

def _sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', str(name)).strip('_')

def _grid_from_percentiles(series: pd.Series, n: int = 30) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        raise ValueError(f"No numeric data to build grid for: {series.name}")
    qs = np.linspace(1, 99, n)
    return np.quantile(s, qs / 100.0)

def _make_reference_row(X_df: pd.DataFrame) -> pd.DataFrame:
    row = {}
    for c in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[c]):
            row[c] = pd.to_numeric(X_df[c], errors="coerce").median()
        else:
            m = X_df[c].mode(dropna=True)
            row[c] = (m.iloc[0] if len(m) else None)
    return pd.DataFrame([row])

def _pick_two_features(X_df: pd.DataFrame):
    pri = ['Creatinine','MAP','Sodium','Age','Glucose','Heart rate',
           'Blood urea nitrogen','Chloride','Hematocrit']
    present = [c for c in pri if c in X_df.columns and pd.api.types.is_numeric_dtype(X_df[c])]
    if len(present) >= 2:
        return present[:2]
    num = X_df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        raise ValueError("Not enough numeric features to make 3D surfaces.")
    var = num.var().sort_values(ascending=False)
    return var.index[:2].tolist()

def _surface_plot(x_vals, y_vals, Z, title, xlab, ylab, zlab, html_path: Path, png_path: Path,
                  zmin=None, zmax=None, colorscale="YlGn"):
    surf = go.Surface(
        x=x_vals, y=y_vals, z=Z.T,
        colorscale=colorscale,
        cmin=zmin if zmin is not None else np.nanmin(Z),
        cmax=zmax if zmax is not None else np.nanmax(Z),
        colorbar=dict(title=zlab)
    )
    fig = go.Figure(data=[surf])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=xlab, yaxis_title=ylab, zaxis_title=zlab),
        margin=dict(l=10, r=10, b=10, t=40)
    )
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), width=PNG_W, height=PNG_H, scale=1)
    except Exception as e:
        print("⚠️ PNG export needs kaleido; if it fails, keep the HTML:", e)
    fig.show()

# ---------------- Predictor adaptation ----------------
def _has_attrs(obj, names):
    return all(hasattr(obj, n) for n in names)

def get_predictor_and_mode():
    """
    Return (predictor, mode), where mode in {"pipeline", "pmrs"}.
    Priority:
      1) In-memory results["CardioMonoStack"]["fitted"] from bench_cardio_monostack.py
      2) PMRS++ predictor at DEFAULT_PMRS_PATH
    """
    # Try grabbing from a live notebook/global
    g = globals()
    if "results" in g and isinstance(g["results"], dict):
        try:
            pred = g["results"]["CardioMonoStack"]["fitted"]
            return pred, "pipeline"
        except Exception:
            pass

    # PMRS++ predictor
    p = Path(DEFAULT_PMRS_PATH)
    if p.exists():
        pred = joblib.load(p)
        # Heuristic for PMRSPredictor: has branchA/meta or residual attributes
        if _has_attrs(pred, ["predict_proba"]) and (hasattr(pred, "meta") or hasattr(pred, "residual")):
            return pred, "pmrs"

    raise RuntimeError(
        "No predictor found.\n"
        "- If you ran bench_cardio_monostack.py in the same session, keep 'results' in memory.\n"
        "- Otherwise provide PMRS++ predictor at /content/pmrs_pp/models/pmrs_predictor.joblib."
    )

def predict_parts(pred, mode, X_df: pd.DataFrame):
    """
    Returns (p_base, delta, p_final):
      - pipeline mode: p_base=None, delta=None, p_final = pipeline.predict_proba
      - pmrs mode:
         p_base = meta(stack(branches)), delta = residual(X) if exists, p_final = sigmoid(logit(p_base)+delta)
    """
    if mode == "pipeline":
        p = pred.predict_proba(X_df)[:, 1]
        return None, None, p

    # PMRS++ object with optional residual
    # Expected attrs: branchA (optional B/C), meta, residual (optional)
    parts = []
    if getattr(pred, "branchA", None) is not None:
        parts.append(pred.branchA.predict_proba(X_df)[:, 1])
    if getattr(pred, "branchB", None) is not None:
        parts.append(pred.branchB.predict_proba(X_df)[:, 1])
    if getattr(pred, "branchC", None) is not None:
        parts.append(pred.branchC.predict_proba(X_df)[:, 1])

    if len(parts) == 0:
        # fallback: whole predictor probability as final only
        p_final = pred.predict_proba(X_df)[:, 1]
        return None, None, p_final

    Z = np.column_stack(parts)
    p_base = pred.meta.predict_proba(Z)[:, 1] if getattr(pred, "meta", None) is not None else parts[0]

    if getattr(pred, "residual", None) is not None:
        delta = pred.residual.predict(X_df)
        eps = 1e-6
        z = np.log(np.clip(p_base, eps, 1-eps) / np.clip(1-p_base, eps, 1-eps))
        p_final = 1.0 / (1.0 + np.exp(-(z + delta)))
        return p_base, delta, p_final
    else:
        return p_base, None, p_base

# ---------------- Surfaces ----------------
def pdp_surface(pred, mode, X_train_df: pd.DataFrame,
                feat1: str = None, feat2: str = None, grid_n: int = 30,
                outdir: Path = Path("/content/pmrspp_out/benchmarks")):
    outdir.mkdir(parents=True, exist_ok=True)
    if feat1 is None or feat2 is None:
        feat1, feat2 = _pick_two_features(X_train_df)

    v1 = _grid_from_percentiles(X_train_df[feat1], grid_n)
    v2 = _grid_from_percentiles(X_train_df[feat2], grid_n)
    base = _make_reference_row(X_train_df)

    blocks = []
    for a in v1:
        r = pd.concat([base]*grid_n, ignore_index=True)
        r[feat1] = a
        r[feat2] = v2
        blocks.append(r)
    DF = pd.concat(blocks, ignore_index=True)

    _, _, p_final = predict_parts(pred, mode, DF)
    Z = p_final.reshape(grid_n, grid_n)

    tag = f"{_sanitize(feat1)}_{_sanitize(feat2)}"
    html_path = outdir / f"cms_pdp_surface_{tag}.html"
    png_path  = outdir / f"cms_pdp_surface_{tag}.png"

    _surface_plot(v1, v2, Z,
                  title=f"CardioMonoStack · PDP Surface: {feat1} × {feat2}",
                  xlab=feat1, ylab=feat2, zlab="Predicted probability",
                  html_path=html_path, png_path=png_path,
                  zmin=0.0, zmax=1.0, colorscale="YlGn")
    print("✅ PDP surface ->", html_path, "|", png_path)
    return {"feat1": feat1, "feat2": feat2, "grid_x": v1, "grid_y": v2, "Z": Z,
            "html": html_path, "png": png_path}

def ice_surface(pred, mode,
                X_row_df: pd.DataFrame, X_train_df: pd.DataFrame,
                feat1: str, feat2: str, grid_n: int = 30,
                outdir: Path = Path("/content/pmrspp_out/benchmarks")):
    outdir.mkdir(parents=True, exist_ok=True)

    v1 = _grid_from_percentiles(X_train_df[feat1], grid_n)
    v2 = _grid_from_percentiles(X_train_df[feat2], grid_n)

    blocks = []
    for a in v1:
        r = pd.concat([X_row_df]*grid_n, ignore_index=True)
        r[feat1] = a
        r[feat2] = v2
        blocks.append(r)
    DF = pd.concat(blocks, ignore_index=True)

    _, _, p_final = predict_parts(pred, mode, DF)
    Z = p_final.reshape(grid_n, grid_n)

    idx = int(X_row_df.index[0]) if X_row_df.index.size else 0
    tag = f"idx{idx}_{_sanitize(feat1)}_{_sanitize(feat2)}"
    html_path = outdir / f"cms_ice_surface_{tag}.html"
    png_path  = outdir / f"cms_ice_surface_{tag}.png"

    _surface_plot(v1, v2, Z,
                  title=f"CardioMonoStack · ICE Surface (idx={idx}): {feat1} × {feat2}",
                  xlab=feat1, ylab=feat2, zlab="Predicted probability",
                  html_path=html_path, png_path=png_path,
                  zmin=0.0, zmax=1.0, colorscale="YlGn")
    print("✅ ICE surface ->", html_path, "|", png_path)
    return {"feat1": feat1, "feat2": feat2, "grid_x": v1, "grid_y": v2, "Z": Z,
            "html": html_path, "png": png_path}

def residual_delta_surface(pred, mode, X_train_df: pd.DataFrame,
                           feat1: str, feat2: str, grid_n: int = 30,
                           outdir: Path = Path("/content/pmrspp_out/benchmarks")):
    outdir.mkdir(parents=True, exist_ok=True)
    if mode != "pmrs" or getattr(pred, "residual", None) is None:
        raise RuntimeError("Residual head not available — cannot draw Δ(logit) surface.")

    v1 = _grid_from_percentiles(X_train_df[feat1], grid_n)
    v2 = _grid_from_percentiles(X_train_df[feat2], grid_n)
    base = _make_reference_row(X_train_df)

    blocks = []
    for a in v1:
        r = pd.concat([base]*grid_n, ignore_index=True)
        r[feat1] = a
        r[feat2] = v2
        blocks.append(r)
    DF = pd.concat(blocks, ignore_index=True)

    # only delta
    _, delta, _ = predict_parts(pred, mode, DF)
    Z = delta.reshape(grid_n, grid_n)

    zabs = float(max(abs(np.nanmin(Z)), abs(np.nanmax(Z))))
    tag = f"{_sanitize(feat1)}_{_sanitize(feat2)}"
    html_path = outdir / f"cms_residual_delta_surface_{tag}.html"
    png_path  = outdir / f"cms_residual_delta_surface_{tag}.png"

    _surface_plot(v1, v2, Z,
                  title=f"CardioMonoStack · Residual Δ Surface: {feat1} × {feat2}",
                  xlab=feat1, ylab=feat2, zlab="Δ (logit)",
                  html_path=html_path, png_path=png_path,
                  zmin=-zabs, zmax=zabs, colorscale="YlGn")
    print("✅ Residual Δ surface ->", html_path, "|", png_path)
    return {"feat1": feat1, "feat2": feat2, "grid_x": v1, "grid_y": v2, "Z": Z,
            "html": html_path, "png": png_path}

# ---------------- CLI runner ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/content/final-analysis.csv")
    ap.add_argument("--target_regex", default=r"(?i)^hf$")
    ap.add_argument("--outdir", default="/content/pmrspp_out/benchmarks")
    ap.add_argument("--feat1", default=None)
    ap.add_argument("--feat2", default=None)
    ap.add_argument("--grid_n", type=int, default=30)
    ap.add_argument("--ice_index", type=int, default=None, help="Index for ICE (default: first positive if available)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Get predictor
    predictor, mode = get_predictor_and_mode()
    print(f"Using predictor mode: {mode}")

    # Load data + build X (raw feature frame like training time)
    df = read_csv_smart(args.input)
    y_col = find_target(df, args.target_regex)
    df[y_col] = normalize_binary_series(df[y_col])

    # drop empty & ID-like
    empty_cols = [c for c in df.columns if df[c].notna().sum()==0]
    if empty_cols: df = df.drop(columns=empty_cols)
    X_df = df.drop(columns=[y_col]).copy()
    X_df = X_df.drop(columns=[c for c in X_df.columns if ID_PAT.search(c)], errors="ignore")
    for c in X_df.columns:
        if X_df[c].dtype==object:
            X_df[c] = pd.to_numeric(X_df[c], errors="ignore")
    y = df[y_col].astype(int).values

    # Pick features
    f1, f2 = (args.feat1, args.feat2) if (args.feat1 and args.feat2) else _pick_two_features(X_df)
    print(f"3D features: {f1} × {f2}")

    # PDP
    out_pdp = pdp_surface(predictor, mode, X_df, feat1=f1, feat2=f2, grid_n=args.grid_n, outdir=outdir)

    # ICE: choose index
    if args.ice_index is not None and 0 <= args.ice_index < len(X_df):
        idx = int(args.ice_index)
    else:
        pos = np.where(y==1)[0]
        idx = int(pos[0]) if len(pos)>0 else 0
    row_df = X_df.iloc[[idx]].copy()
    out_ice = ice_surface(predictor, mode, row_df, X_df, feat1=f1, feat2=f2, grid_n=args.grid_n, outdir=outdir)

    # Residual Δ (only if PMRS++ with residual)
    try:
        out_res = residual_delta_surface(predictor, mode, X_df, feat1=f1, feat2=f2, grid_n=args.grid_n, outdir=outdir)
    except Exception as e:
        print("ℹ️ Skip residual Δ surface:", e)
        out_res = None

    # Optional: auto-download in Colab
    try:
        from google.colab import files
        paths = [out_pdp["html"], out_pdp["png"], out_ice["html"], out_ice["png"]]
        if out_res is not None: paths += [out_res["html"], out_res["png"]]
        for p in paths: files.download(str(p))
    except Exception:
        pass

if __name__ == "__main__":
    main()
