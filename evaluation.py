#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B · 3D Evaluation Surfaces (CardioMonoStack vs 6 Baselines)
===========================================================

Generates 3D evaluation plots from a previously trained benchmark run:
  1) Threshold–ROC 3D (all models in one figure)
  2) Threshold–Precision–Recall 3D (all models in one figure)
  3) Decision Curve (Net Benefit) 3D (one figure per model, auto subgroups)

Outputs:
  - Interactive Plotly HTML
  - High-resolution Matplotlib PNG (~400 dpi)

Expected globals (produced by the benchmark training script):
  - `results`: dict with per-model evaluation dicts.
      results[model_name]["eval"] contains:
        - "y":   1D ndarray of ground-truth labels on validation set
        - "p_cal": 1D ndarray of calibrated probabilities on validation set
  - `X_valid_df`: (optional) pandas DataFrame of validation features
      Used only for auto subgrouping (sex/gender or binned age). If absent,
      the script will fall back to a single "All" group.

Usage
-----
# After running the benchmark training script that creates `results` (+ optional X_valid_df):
python eval_3d_surfaces.py

# Or: import and call the plotting functions, passing `results` and `X_valid_df` if needed.
"""

import sys
import re
import importlib
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

# Lazy-install for Colab/ephemeral environments
def _ensure(pkg: str):
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_ensure("plotly")

import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------------------------------------------
# Output paths & figure settings
# ------------------------------------------------------------
try:
    BENCH  # provided by previous scripts
except NameError:
    BENCH = Path("/content/pmrspp_out/benchmarks")
BENCH.mkdir(parents=True, exist_ok=True)

DPI_TARGET = 400
FIG_W_IN, FIG_H_IN = 7.0, 5.0
CMAP = plt.get_cmap("YlGn")


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")


def _save_html(fig: go.Figure, path_html: Path):
    fig.write_html(str(path_html))


def _save_png(fig: plt.Figure, path_png: Path):
    fig.set_size_inches(FIG_W_IN, FIG_H_IN)
    fig.savefig(path_png, dpi=DPI_TARGET, bbox_inches="tight")
    plt.close(fig)


def _colors_for_models(n: int):
    """Return n RGBA colors sampled from YlGn."""
    return [CMAP(i / max(n - 1, 1)) for i in range(n)]


# ------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------
def confusion_matrix_safe(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return tn, fp, fn, tp."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def compute_threshold_curves(y_true: np.ndarray, p: np.ndarray, thr_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute FPR/TPR and Precision/Recall as functions of threshold."""
    y_true = y_true.astype(int)
    fpr, tpr, prec, rec = [], [], [], []
    for thr in thr_grid:
        yhat = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix_safe(y_true, yhat)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        prec.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        rec.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return dict(
        fpr=np.array(fpr), tpr=np.array(tpr),
        precision=np.array(prec), recall=np.array(rec)
    )


# ------------------------------------------------------------
# Decision curve analysis (Net Benefit)
# ------------------------------------------------------------
def net_benefit_from_counts(tp: int, fp: int, n: int, pt: float) -> float:
    pt = float(np.clip(pt, 1e-6, 1 - 1e-6))
    return (tp / n) - (fp / n) * (pt / (1 - pt))


def decision_curve_by_group(y_true: np.ndarray, p: np.ndarray, thr_grid: np.ndarray, groups: np.ndarray):
    """
    Compute DCA (net benefit) across an array of probability thresholds for each subgroup.
    Returns (labels, Z) where Z has shape (len(thr_grid), G).
    """
    labels = []
    cols = []
    uniques = [u for u in pd.unique(groups) if pd.notna(u)]
    if len(uniques) == 0:
        uniques = ["All"]
        groups = np.array(["All"] * len(y_true))

    for g in uniques:
        mask = (groups == g)
        if mask.sum() < 5:
            continue
        y_g = y_true[mask]
        p_g = p[mask]
        n = len(y_g)
        nb = []
        for pt in thr_grid:
            yhat = (p_g >= pt).astype(int)
            tn, fp, fn, tp = confusion_matrix_safe(y_g, yhat)
            nb.append(net_benefit_from_counts(tp, fp, n, pt))
        labels.append(str(g))
        cols.append(np.array(nb))
    if len(cols) == 0:
        # Fallback: single All group
        y_g, p_g = y_true, p
        n = len(y_g)
        nb = []
        for pt in thr_grid:
            yhat = (p_g >= pt).astype(int)
            tn, fp, fn, tp = confusion_matrix_safe(y_g, yhat)
            nb.append(net_benefit_from_counts(tp, fp, n, pt))
        labels = ["All"]
        cols = [np.array(nb)]

    Z = np.column_stack(cols)  # (len(thr), G)
    return labels, Z


def auto_groups(X_valid_df: pd.DataFrame):
    """
    Auto subgroup selection:
      1) Prefer 'sex'/'gender' if present.
      2) Else, bin numeric 'age' into 4 bins.
      3) Else, single 'All' group.
    """
    for cand in X_valid_df.columns:
        if re.search(r"^(sex|gender)$", str(cand), re.I):
            vals = X_valid_df[cand]
            return vals.astype(str).values, f"{cand}"

    # Age bins
    age_col = None
    for cand in X_valid_df.columns:
        if re.search(r"^age$", str(cand), re.I) and pd.api.types.is_numeric_dtype(X_valid_df[cand]):
            age_col = cand
            break
    if age_col is not None:
        edges = [-np.inf, 50, 65, 80, np.inf]
        labels = ["<50", "50–64", "65–79", "≥80"]
        bins = pd.cut(pd.to_numeric(X_valid_df[age_col], errors="coerce"),
                      edges, labels=labels, include_lowest=True)
        return bins.astype(str).fillna("NA").values, f"{age_col}_bins"

    return np.array(["All"] * len(X_valid_df)), "All"


# ------------------------------------------------------------
# Plotters
# ------------------------------------------------------------
def plot_threshold_roc_3d_all(results: dict, thr_grid=None, outdir: Path = None):
    """
    3D Threshold–ROC curves for all models in one figure.

    X-axis: Threshold
    Y-axis: FPR
    Z-axis: TPR
    """
    outdir = Path(outdir) if outdir is not None else BENCH
    outdir.mkdir(parents=True, exist_ok=True)
    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 61)

    # Plotly (interactive HTML)
    fig_p = go.Figure()
    names = list(results.keys())
    colors = _colors_for_models(len(names))
    for name, color in zip(names, colors):
        y = results[name]["eval"]["y"]
        p = results[name]["eval"]["p_cal"]
        C = compute_threshold_curves(y, p, thr_grid)
        r, g, b = int(255 * color[0]), int(255 * color[1]), int(255 * color[2])
        fig_p.add_trace(go.Scatter3d(
            x=thr_grid, y=C["fpr"], z=C["tpr"], mode="lines",
            name=name, line=dict(width=4, color=f"rgb({r},{g},{b})")
        ))
    fig_p.update_layout(
        title="Threshold–ROC 3D (All Models)",
        scene=dict(xaxis_title="Threshold", yaxis_title="FPR", zaxis_title="TPR"),
        margin=dict(l=10, r=10, b=10, t=40)
    )
    path_html = outdir / "surf_threshold_ROC_all.html"
    _save_html(fig_p, path_html)

    # Matplotlib (PNG)
    fig_m = plt.figure()
    ax = fig_m.add_subplot(111, projection="3d")
    for name, color in zip(names, colors):
        y = results[name]["eval"]["y"]
        p = results[name]["eval"]["p_cal"]
        C = compute_threshold_curves(y, p, thr_grid)
        ax.plot(thr_grid, C["fpr"], C["tpr"], label=name, color=color, linewidth=2.2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("FPR")
    ax.set_zlabel("TPR")
    ax.set_title("Threshold–ROC 3D (All Models)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    path_png = outdir / "surf_threshold_ROC_all.png"
    _save_png(fig_m, path_png)

    print("✅ Threshold–ROC 3D saved ->", path_html, "|", path_png)
    return path_html, path_png


def plot_threshold_pr_3d_all(results: dict, thr_grid=None, outdir: Path = None):
    """
    3D Threshold–Precision–Recall curves for all models in one figure.

    X-axis: Threshold
    Y-axis: Recall
    Z-axis: Precision
    """
    outdir = Path(outdir) if outdir is not None else BENCH
    outdir.mkdir(parents=True, exist_ok=True)
    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 61)

    # Plotly
    fig_p = go.Figure()
    names = list(results.keys())
    colors = _colors_for_models(len(names))
    for name, color in zip(names, colors):
        y = results[name]["eval"]["y"]
        p = results[name]["eval"]["p_cal"]
        C = compute_threshold_curves(y, p, thr_grid)
        r, g, b = int(255 * color[0]), int(255 * color[1]), int(255 * color[2])
        fig_p.add_trace(go.Scatter3d(
            x=thr_grid, y=C["recall"], z=C["precision"], mode="lines",
            name=name, line=dict(width=4, color=f"rgb({r},{g},{b})")
        ))
    fig_p.update_layout(
        title="Threshold–Precision–Recall 3D (All Models)",
        scene=dict(xaxis_title="Threshold", yaxis_title="Recall", zaxis_title="Precision"),
        margin=dict(l=10, r=10, b=10, t=40)
    )
    path_html = outdir / "surf_threshold_PR_all.html"
    _save_html(fig_p, path_html)

    # Matplotlib
    fig_m = plt.figure()
    ax = fig_m.add_subplot(111, projection="3d")
    for name, color in zip(names, colors):
        y = results[name]["eval"]["y"]
        p = results[name]["eval"]["p_cal"]
        C = compute_threshold_curves(y, p, thr_grid)
        ax.plot(thr_grid, C["recall"], C["precision"], label=name, color=color, linewidth=2.2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Recall")
    ax.set_zlabel("Precision")
    ax.set_title("Threshold–Precision–Recall 3D (All Models)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    path_png = outdir / "surf_threshold_PR_all.png"
    _save_png(fig_m, path_png)

    print("✅ Threshold–PR 3D saved ->", path_html, "|", path_png)
    return path_html, path_png


def _surface_plot_ylgn(x_vals: np.ndarray, y_ids: np.ndarray, Z: np.ndarray,
                       title: str, xlab: str, ylab: str, zlab: str,
                       y_ticklabels: List[str], html_path: Path, png_path: Path):
    """
    Helper to render a (threshold × subgroup) surface using Plotly and Matplotlib.
    Z is expected to be shape (len(x_vals), len(y_ids)).
    """
    # Plotly
    surf = go.Surface(
        x=x_vals, y=np.arange(len(y_ids)), z=Z.T,
        colorscale="YlGn", colorbar=dict(title=zlab)
    )
    fig_p = go.Figure(data=[surf])
    fig_p.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlab,
            yaxis_title=ylab,
            zaxis_title=zlab,
            yaxis=dict(tickmode="array",
                       tickvals=list(range(len(y_ids))),
                       ticktext=y_ticklabels)
        ),
        margin=dict(l=10, r=10, b=10, t=40)
    )
    _save_html(fig_p, html_path)

    # Matplotlib
    Xg, Yg = np.meshgrid(x_vals, np.arange(len(y_ids)), indexing="ij")
    fig_m = plt.figure()
    ax = fig_m.add_subplot(111, projection="3d")
    surf_m = ax.plot_surface(Xg, Yg, Z, cmap="YlGn",
                             linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    ax.set_title(title)
    ax.set_yticks(list(range(len(y_ids))))
    ax.set_yticklabels(y_ticklabels, rotation=0)
    cbar = fig_m.colorbar(surf_m, shrink=0.65, aspect=14, pad=0.08)
    cbar.set_label(zlab)
    _save_png(fig_m, png_path)


def plot_decision_curve_3d_per_model(name: str, y: np.ndarray, p: np.ndarray,
                                     X_valid_df: pd.DataFrame, thr_grid=None, outdir: Path = None):
    """
    Decision Curve 3D for a single model, auto subgrouped by sex/gender or age bins.
    """
    outdir = Path(outdir) if outdir is not None else BENCH
    outdir.mkdir(parents=True, exist_ok=True)
    if thr_grid is None:
        thr_grid = np.linspace(0.05, 0.95, 19)

    grp_vals, grp_name = auto_groups(X_valid_df)
    labels, Z = decision_curve_by_group(y, p, thr_grid, grp_vals)
    tag = f"{_sanitize(name)}_by_{_sanitize(grp_name)}"
    html_path = outdir / f"dca_3d_{tag}.html"
    png_path = outdir / f"dca_3d_{tag}.png"

    _surface_plot_ylgn(
        x_vals=thr_grid,
        y_ids=np.arange(len(labels)),
        Z=Z,  # (len(thr), G)
        title=f"Decision Curve 3D — {name} (by {grp_name})",
        xlab="Threshold probability (pt)",
        ylab=f"Subgroups ({grp_name})",
        zlab="Net Benefit",
        y_ticklabels=labels,
        html_path=html_path,
        png_path=png_path
    )
    print(f"✅ Decision Curve 3D saved for {name} ->", html_path, "|", png_path)
    return html_path, png_path


def download_existing(paths: List[Path]):
    """Convenience downloader for Colab."""
    try:
        from google.colab import files
    except Exception:
        print("⚠️ Not running in Google Colab. Please download from the file browser:")
        for p in paths:
            print(" -", p, "(exists)" if Path(p).exists() else "(missing)")
        return
    for p in paths:
        pth = Path(p)
        if pth.exists():
            try:
                files.download(str(pth))
            except Exception as e:
                print(f"Download failed: {pth} | {e}")
        else:
            print(f"(skip) Missing file: {pth}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Guard: results must exist
    if "results" not in globals():
        raise RuntimeError(
            "`results` not found. Run the benchmark training script first to populate `results`."
        )

    # Fallback for X_valid_df if not provided; DCA will then use a single "All" group
    try:
        X_valid_df  # type: ignore
    except NameError:
        any_key = next(iter(results))
        n_valid = len(results[any_key]["eval"]["y"])
        X_valid_df = pd.DataFrame(index=np.arange(n_valid))

    # 1) Threshold–ROC 3D (all models)
    html1, png1 = plot_threshold_roc_3d_all(results, thr_grid=None, outdir=BENCH)

    # 2) Threshold–PR 3D (all models)
    html2, png2 = plot_threshold_pr_3d_all(results, thr_grid=None, outdir=BENCH)

    # 3) Decision Curve 3D (one per model)
    dca_files: List[Path] = []
    for name in results.keys():
        yv = results[name]["eval"]["y"]
        pv = results[name]["eval"]["p_cal"]
        h, p = plot_decision_curve_3d_per_model(name, yv, pv, X_valid_df, thr_grid=None, outdir=BENCH)
        dca_files.extend([h, p])

    # Summary + optional download (Colab)
    all_out = [html1, png1, html2, png2] + dca_files
    print("\n=== File existence check ===")
    for p in all_out:
        print(f" - {p} : {Path(p).exists()}")
    print("\nAttempting downloads (Colab only)...")
    download_existing(all_out)
