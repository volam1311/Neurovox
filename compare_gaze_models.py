"""
Compare 5 gaze-mapping regression models on calibration data.

Models: Ridge, SVR (RBF), XGBoost, Gradient Boosting, Random Forest.

- Combines ALL available calibration CSVs (720 samples from 2 sessions).
- Fits each model on the FULL dataset.
- Evaluates with 5-fold Stratified-Group CV (stratified by point_index).
- Per-model screen plot shows: mean predicted position per point,
  std-deviation ellipse around predictions, and error lines to target.
- Summary plots: bar chart with mean/median/std, box-plot, and stats table.

Outputs in plots/:
  <model>.png          – screen-canvas plot (mean pred + variance ellipses)
  summary_bar.png      – bar chart comparing models
  summary_boxplot.png  – box plot of per-sample errors
  summary_table.png    – stats table
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor

FEATURE_COLS = [f"f{i}" for i in range(17)]
TARGET_COLS = ["target_x", "target_y"]
PLOT_DIR = Path("plots")

MODEL_COLORS_BGR = {
    "Ridge":             (180, 114, 76),
    "Stepwise Linear":   (150, 150, 70),
    "SVR":               (82, 132, 221),
    "XGBoost":           (104, 168, 85),
    "Gradient Boosting": (82, 78, 196),
    "Random Forest":     (179, 114, 129),
}
MODEL_COLORS_RGB = {
    "Ridge":             "#4C72B0",
    "Stepwise Linear":   "#4A90E2",
    "SVR":               "#DD8452",
    "XGBoost":           "#55A868",
    "Gradient Boosting": "#C44E52",
    "Random Forest":     "#8172B3",
}
MODEL_ORDER = list(MODEL_COLORS_RGB.keys())


def _make_models() -> dict[str, Pipeline]:
    return {
        "Ridge": Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Stepwise Linear": Pipeline([
            ("scale", StandardScaler()),
            ("stepwise", SequentialFeatureSelector(LinearRegression(), n_features_to_select="auto", tol=1e-3, cv=3)),
            ("model", LinearRegression()),
        ]),
        "SVR": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(SVR(kernel="rbf", C=10, epsilon=5, gamma="scale"))),
        ]),
        "XGBoost": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
            ))),
        ]),
        "Gradient Boosting": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.08,
                subsample=0.8, min_samples_leaf=3,
            ))),
        ]),
        "Random Forest": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=3,
                max_features="sqrt", random_state=42,
            ))),
        ]),
    }


# ── data ─────────────────────────────────────────────────────────────────

def load_and_combine() -> pd.DataFrame:
    csvs = list(Path(".").glob("*_cal_repeats.csv"))
    if not csvs:
        sys.exit("No *_cal_repeats.csv files found. Run calibration first.")
    frames = []
    for p in sorted(csvs):
        df = pd.read_csv(p)
        required = set(FEATURE_COLS) | set(TARGET_COLS) | {"point_index", "gaze_canvas_w", "gaze_canvas_h"}
        missing = required - set(df.columns)
        if missing:
            sys.exit(f"{p}: missing columns {sorted(missing)}")
        df["source"] = p.stem
        frames.append(df)
        print(f"  Loaded {p.name}: {len(df)} samples")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined)} total samples, "
          f"{combined['point_index'].nunique()} unique calibration points\n")
    return combined


# ── evaluation ───────────────────────────────────────────────────────────

def evaluate_models(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """5-fold GroupKFold CV. Returns {model_name: preds (N,2)}."""
    X = df[FEATURE_COLS].values
    Y = df[TARGET_COLS].values
    groups = df["point_index"].values

    gkf = GroupKFold(n_splits=5)
    models = _make_models()
    all_preds: dict[str, np.ndarray] = {}

    for name, pipe in models.items():
        print(f"  {name} … ", end="", flush=True)
        preds = np.full_like(Y, np.nan, dtype=float)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups)):
            p = clone(pipe)
            p.fit(X[train_idx], Y[train_idx])
            preds[test_idx] = p.predict(X[test_idx])
        errs = np.sqrt(np.sum((Y - preds) ** 2, axis=1))
        print(f"mean={errs.mean():.1f} px  median={np.median(errs):.1f} px")
        all_preds[name] = preds

    return all_preds


# ── screen-canvas plot per model ─────────────────────────────────────────

def _draw_ellipse(img, cx, cy, std_x, std_y, color, scale=2.0):
    """Draw a std-deviation ellipse (scale * sigma)."""
    axes = (max(int(std_x * scale), 2), max(int(std_y * scale), 2))
    cv2.ellipse(img, (int(cx), int(cy)), axes, 0, 0, 360, color, 1, cv2.LINE_AA)


def draw_screen_plot(
    name: str, df: pd.DataFrame, preds: np.ndarray,
    canvas_w: int, canvas_h: int, out: Path,
):
    pred_color = MODEL_COLORS_BGR[name]
    board = np.full((canvas_h, canvas_w, 3), (28, 28, 28), dtype=np.uint8)

    targets = df[TARGET_COLS].values
    point_ids = df["point_index"].values
    unique_pts = sorted(set(point_ids))

    all_errs = np.sqrt(np.sum((targets - preds) ** 2, axis=1))
    mean_err = all_errs.mean()
    median_err = float(np.median(all_errs))
    std_err = all_errs.std()
    var_err = all_errs.var()

    for pid in unique_pts:
        mask = point_ids == pid
        tx = targets[mask, 0].mean()
        ty = targets[mask, 1].mean()
        px_arr = preds[mask, 0]
        py_arr = preds[mask, 1]
        mean_px = px_arr.mean()
        mean_py = py_arr.mean()
        std_px = px_arr.std()
        std_py = py_arr.std()

        itx, ity = int(round(tx)), int(round(ty))
        ipx, ipy = int(round(mean_px)), int(round(mean_py))

        cv2.line(board, (itx, ity), (ipx, ipy), (255, 255, 0), 2, cv2.LINE_AA)

        _draw_ellipse(board, mean_px, mean_py, std_px, std_py, (200, 200, 200), scale=1.0)
        _draw_ellipse(board, mean_px, mean_py, std_px, std_py, (140, 140, 140), scale=2.0)

        cv2.circle(board, (itx, ity), 16, (0, 220, 0), 3, cv2.LINE_AA)
        cv2.circle(board, (ipx, ipy), 10, pred_color, -1, cv2.LINE_AA)
        cv2.circle(board, (ipx, ipy), 10, (255, 255, 255), 1, cv2.LINE_AA)

        pt_err = np.sqrt((tx - mean_px)**2 + (ty - mean_py)**2)
        cv2.putText(board, f"{pt_err:.0f}", (ipx + 14, ipy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    bar_h = 70
    cv2.rectangle(board, (0, 0), (canvas_w, bar_h), (45, 45, 45), -1)

    line1 = f"{name}"
    cv2.putText(board, line1, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)

    line2 = (f"Mean: {mean_err:.1f} px    Median: {median_err:.1f} px    "
             f"Std: {std_err:.1f} px    Variance: {var_err:.1f} px^2")
    cv2.putText(board, line2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (210, 210, 210), 1, cv2.LINE_AA)

    ly = bar_h + 28
    cv2.circle(board, (20, ly), 10, (0, 220, 0), 3, cv2.LINE_AA)
    cv2.putText(board, "Target", (36, ly + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(board, (130, ly), 8, pred_color, -1, cv2.LINE_AA)
    cv2.putText(board, "Mean Pred", (144, ly + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(board, (270, ly), (300, ly), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(board, "Error", (306, ly + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.ellipse(board, (390, ly), (18, 12), 0, 0, 360, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(board, "1-sigma / 2-sigma", (414, ly + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(out), board)
    print(f"  Saved {out}")


# ── matplotlib summary plots ─────────────────────────────────────────────

def draw_summary_bar(stats: dict[str, dict], out: Path):
    names = MODEL_ORDER
    means = [stats[n]["mean"] for n in names]
    medians = [stats[n]["median"] for n in names]
    stds = [stats[n]["std"] for n in names]
    colors = [MODEL_COLORS_RGB[n] for n in names]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(names))
    w = 0.35
    bars_mean = ax.bar(x - w/2, means, w, yerr=stds, capsize=4,
                       color=colors, edgecolor="white", linewidth=0.8, label="Mean ± Std")
    bars_med = ax.bar(x + w/2, medians, w,
                      color=colors, edgecolor="white", linewidth=0.8, alpha=0.55, label="Median")

    for bar, val in zip(bars_mean, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.12,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars_med, medians):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Pixel Error (px)", fontsize=12)
    ax.set_title("5-Fold GroupKFold CV — Mean (± Std) and Median Error", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def draw_summary_boxplot(all_errs: dict[str, np.ndarray], out: Path):
    names = MODEL_ORDER
    data = [all_errs[n] for n in names]
    colors = [MODEL_COLORS_RGB[n] for n in names]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6),
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    for i, (n, d) in enumerate(zip(names, data)):
        ax.text(i + 1, d.mean() + d.std() * 0.15,
                f"μ={d.mean():.1f}\nσ={d.std():.1f}", ha="center", va="bottom",
                fontsize=8, color="#333333")

    ax.set_ylabel("Pixel Error (px)", fontsize=12)
    ax.set_title("Error Distribution per Model (5-Fold GroupKFold CV)", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def draw_summary_table(stats: dict[str, dict], out: Path):
    names = MODEL_ORDER
    col_labels = ["Model", "Mean", "Median", "Std", "Variance", "Min", "Max"]
    rows = []
    for n in names:
        s = stats[n]
        rows.append([n, f"{s['mean']:.1f}", f"{s['median']:.1f}", f"{s['std']:.1f}",
                      f"{s['var']:.1f}", f"{s['min']:.1f}", f"{s['max']:.1f}"])

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#404040")
        table[0, j].set_text_props(color="white", fontweight="bold")

    best = min(names, key=lambda n: stats[n]["mean"])
    best_row = names.index(best) + 1
    for j in range(len(col_labels)):
        table[best_row, j].set_facecolor("#d4edda")

    fig.suptitle("All values in pixels (px).  Green row = best model.", fontsize=9, y=0.08, color="#666")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading calibration data …")
    df = load_and_combine()
    canvas_w = int(df["gaze_canvas_w"].iloc[0])
    canvas_h = int(df["gaze_canvas_h"].iloc[0])

    print("Running 5-fold GroupKFold CV …")
    all_preds = evaluate_models(df)

    PLOT_DIR.mkdir(exist_ok=True)
    targets = df[TARGET_COLS].values

    stats: dict[str, dict] = {}
    err_arrays: dict[str, np.ndarray] = {}

    print("\nDrawing screen plots …")
    for name in MODEL_ORDER:
        preds = all_preds[name]
        errs = np.sqrt(np.sum((targets - preds) ** 2, axis=1))
        err_arrays[name] = errs
        stats[name] = {
            "mean": errs.mean(),
            "median": float(np.median(errs)),
            "std": errs.std(),
            "var": errs.var(),
            "min": errs.min(),
            "max": errs.max(),
        }
        safe = name.lower().replace(" ", "_")
        draw_screen_plot(name, df, preds, canvas_w, canvas_h, PLOT_DIR / f"{safe}.png")

    print("\nDrawing summary charts …")
    draw_summary_bar(stats, PLOT_DIR / "summary_bar.png")
    draw_summary_boxplot(err_arrays, PLOT_DIR / "summary_boxplot.png")
    draw_summary_table(stats, PLOT_DIR / "summary_table.png")

    print("\n" + "=" * 78)
    print(f"  MODEL COMPARISON — 5-Fold GroupKFold CV on {len(df)} samples")
    print("=" * 78)
    print(f"  {'Model':20s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Var':>10s} {'Min':>8s} {'Max':>8s}")
    print("-" * 78)
    for n in MODEL_ORDER:
        s = stats[n]
        tag = " <<<" if n == min(MODEL_ORDER, key=lambda x: stats[x]["mean"]) else ""
        print(f"  {n:20s} {s['mean']:8.1f} {s['median']:8.1f} {s['std']:8.1f} "
              f"{s['var']:10.1f} {s['min']:8.1f} {s['max']:8.1f}{tag}")
    print("=" * 78)


if __name__ == "__main__":
    main()
