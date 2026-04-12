from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from stroke_eye_monitor.config import MonitorConfig, detect_screen_resolution
from stroke_eye_monitor.core.detector import FaceMeshEyeDetector
from stroke_eye_monitor.core.gaze_mapping import (
    GazeCalibration,
    _GAZE_CV_LOO_MAX_N,
    _build_candidates,
    _gaze_cv_mean_error,
    fit_gaze_model,
)
from stroke_eye_monitor.core.metrics import compute_eye_metrics, gaze_feature_vector
from stroke_eye_monitor.utils.opencv_canvas import sync_opencv_window_canvas

def _fixed_grid_norm_positions(
    *,
    n_per_side: int,
    margin: float = 0.06,
) -> list[tuple[float, float]]:
    """Evenly spaced normalized (0..1) dots, row-major (top→bottom, left→right)."""
    lo = float(margin)
    hi = 1.0 - float(margin)
    xs = np.linspace(lo, hi, int(n_per_side), dtype=np.float64)
    ys = np.linspace(lo, hi, int(n_per_side), dtype=np.float64)
    return [(float(x), float(y)) for y in ys for x in xs]


# Fixed 6x6 = 36 targets, inset from edges (reproducible; good screen coverage).
FIXED_GRID_36_TARGETS: list[tuple[float, float]] = _fixed_grid_norm_positions(n_per_side=6)


def _random_norm_targets(
    n: int,
    *,
    rng: np.random.Generator,
    margin: float = 0.06,
) -> list[tuple[float, float]]:
    """``n`` normalized (0..1) dot positions with margins and spacing (for diverse calibration)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    lo = margin
    hi = 1.0 - margin
    pts: list[tuple[float, float]] = []
    max_attempts = max(800, n * 600)
    attempts = 0
    # Spacing scales with sqrt(n) so ~36 dots remain spread across the canvas.
    cur_sep = max(0.035, 0.55 / math.sqrt(float(n)))
    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(lo, hi))
        y = float(rng.uniform(lo, hi))
        if all(math.hypot(x - px, y - py) >= cur_sep for px, py in pts):
            pts.append((x, y))
        if attempts % 400 == 0 and cur_sep > 0.035:
            cur_sep *= 0.88
    while len(pts) < n:
        pts.append((float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))))
    return pts


def run_calibration(
    *,
    cap: cv2.VideoCapture,
    detector: FaceMeshEyeDetector,
    proc_fn,
    gaze_width: int,
    gaze_height: int,
    out_path: Path,
    targets: list[tuple[float, float]] | None = None,
    samples_per_point: int = 10,
    ear_min: float = 0.17,
    ridge_lambda: float = 1e-2,
    gaze_model: str = "auto",
    use_fixed_grid: bool = False,
    n_calibration_points: int = 36,
    calibration_seed: int | None = None,
) -> GazeCalibration | None:
    """
    Show fixation dots on a gaze-sized canvas (fixed 6x6 grid by default, or random if disabled).
    User looks at each dot and presses SPACE. Iris features are regressed to dot positions.
    """
    if samples_per_point < 1:
        raise ValueError("samples_per_point must be at least 1")

    if targets is not None:
        t_list = targets
    elif use_fixed_grid:
        t_list = list(FIXED_GRID_36_TARGETS)
    else:
        if n_calibration_points < 3:
            raise ValueError("Need at least 3 calibration points (set --gaze-cal-points >= 3).")
        rng = np.random.default_rng(calibration_seed)
        t_list = _random_norm_targets(n_calibration_points, rng=rng)
    win = "Calibration — look at the white dot, press SPACE"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, gaze_width, gaze_height)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass

    gaze_width, gaze_height = sync_opencv_window_canvas(win, gaze_width, gaze_height)
    print(
        f"Calibration canvas (OpenCV window): {gaze_width} x {gaze_height}",
        flush=True,
    )
    layout = (
        "custom list"
        if targets is not None
        else ("6x6 fixed grid" if use_fixed_grid else "random")
    )
    print(f"Calibration layout: {layout} ({len(t_list)} points)", flush=True)

    feature_rows: list[np.ndarray] = []
    screen_xy: list[tuple[float, float]] = []
    all_samples: list[tuple[int, np.ndarray, float, float]] = []

    for idx, (fx, fy) in enumerate(t_list):
        tx = int(round(fx * (gaze_width - 1)))
        ty = int(round(fy * (gaze_height - 1)))
        print(
            f"Point {idx + 1}/{len(t_list)}: look at the dot, hold still, press SPACE",
            flush=True,
        )

        collecting = False
        collected: list[np.ndarray] = []

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            proc = proc_fn(frame)
            result = detector.process_bgr(proc)

            board = np.zeros((gaze_height, gaze_width, 3), dtype=np.uint8)
            r_dot = max(18, gaze_width // 40)
            cv2.circle(board, (tx, ty), r_dot, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(board, (tx, ty), r_dot + 4, (0, 0, 0), 3, cv2.LINE_AA)
            hint = f"{idx + 1}/{len(t_list)}  SPACE=capture  Q=abort"
            cv2.putText(
                board,
                hint,
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            if collecting and result.landmarks is not None:
                h0, w0 = result.image_shape
                m = compute_eye_metrics(result.landmarks, h0, w0)
                if m.left_ear >= ear_min and m.right_ear >= ear_min:
                    g = gaze_feature_vector(m, result.face_matrix)
                    if g is not None:
                        collected.append(g.copy())
                if len(collected) >= samples_per_point:
                    for sample in collected:
                        row = sample.copy()
                        all_samples.append((idx + 1, row, float(tx), float(ty)))
                        feature_rows.append(row)
                        screen_xy.append((float(tx), float(ty)))
                    print(
                        f"  captured {len(collected)} frames "
                        f"(each used as a training row; target=({tx},{ty}))",
                        flush=True,
                    )
                    break

            cv2.imshow(win, board)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyWindow(win)
                return None
            if key in (13, 32):  # Enter or Space
                collecting = True
                collected = []

    cv2.destroyWindow(win)

    # ── save ALL individual samples to CSV for later analysis / plotting ──
    csv_path = out_path.with_suffix(".csv")
    _save_calibration_csv(csv_path, all_samples, gaze_width, gaze_height)
    total = len(all_samples)
    n_dots = len(t_list)
    print(
        f"Saved {total} rows to {csv_path} (raw frames). "
        f"Fitting maps each of the {n_dots} fixation targets using a per-target median feature "
        f"(see messages below) so tree models do not overfit duplicate frames.",
        flush=True,
    )

    if not feature_rows:
        raise RuntimeError(
            "No gaze samples were collected (check camera, lighting, and --gaze-ear-min).",
        )

    # ── evaluate all models and let user choose ──
    cal = _select_and_fit_model(
        feature_rows, screen_xy, gaze_width, gaze_height,
        ridge_lambda=ridge_lambda, preferred_model=gaze_model,
    )
    cal.save(out_path)
    print(f"Saved calibration ({cal.model_type}) to {out_path}", flush=True)
    return cal


def _save_calibration_csv(
    csv_path: Path,
    all_samples: list[tuple[int, np.ndarray, float, float]],
    gaze_width: int,
    gaze_height: int,
) -> None:
    """Save every individual sample (not just medians) for later plotting / model comparison.

    Each row is one frame capture: point_index, sample_index, target, canvas size, features.
    """
    if not all_samples:
        return
    n_feat = all_samples[0][1].shape[0]
    header = (
        ["point_index", "sample_index", "target_x", "target_y",
         "gaze_canvas_w", "gaze_canvas_h"]
        + [f"f{i}" for i in range(n_feat)]
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        sample_counts: dict[int, int] = {}
        for point_idx, feat, tx, ty in all_samples:
            sample_counts[point_idx] = sample_counts.get(point_idx, 0) + 1
            si = sample_counts[point_idx]
            row = [point_idx, si, tx, ty, gaze_width, gaze_height] + feat.tolist()
            writer.writerow(row)


_MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge",
    "stepwise": "Stepwise LR",
    "poly": "Poly Ridge",
    "svr": "SVR",
    "gbr": "Gradient Boosting",
    "xgboost": "XGBoost",
    "rf": "Random Forest",
}


def _aggregate_samples_by_target_pixel(
    feature_rows: list[np.ndarray],
    screen_xy: list[tuple[float, float]],
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """One training row per on-screen target: median feature vector per (tx, ty) pixel.

    Many nearly-identical frames per fixation point confuse tree/boosting models: they
    memorize label noise and extrapolate wildly at runtime (gaze clamped to a screen corner).
    Ridge is less affected; aggregation still stabilizes CV and live behaviour.
    """
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for row, (sx, sy) in zip(feature_rows, screen_xy, strict=True):
        key = (int(round(sx)), int(round(sy)))
        buckets.setdefault(key, []).append(np.asarray(row, dtype=np.float64).reshape(-1))
    out_feat: list[np.ndarray] = []
    out_xy: list[tuple[float, float]] = []
    for key in sorted(buckets.keys()):
        arr = np.stack(buckets[key], axis=0)
        out_feat.append(np.median(arr, axis=0))
        out_xy.append((float(key[0]), float(key[1])))
    return out_feat, out_xy


def _select_and_fit_model(
    feature_rows: list[np.ndarray],
    screen_xy: list[tuple[float, float]],
    gaze_width: int,
    gaze_height: int,
    *,
    ridge_lambda: float = 1e-2,
    preferred_model: str = "auto",
) -> GazeCalibration:
    """Evaluate all models with CV, print a table, then fit the chosen one."""
    n_raw = len(feature_rows)
    feature_rows, screen_xy = _aggregate_samples_by_target_pixel(feature_rows, screen_xy)
    if len(feature_rows) < n_raw:
        print(
            f"Using {len(feature_rows)} unique fixation targets "
            f"(median of features over {n_raw} captured frames) for model fitting.",
            flush=True,
        )
    if len(feature_rows) < 3:
        raise RuntimeError(
            f"Need at least 3 distinct on-screen targets after aggregation; got {len(feature_rows)}. "
            "Try more calibration points or a larger gaze canvas.",
        )

    F = np.stack([np.asarray(r, dtype=np.float64).reshape(-1) for r in feature_rows], axis=0)
    Y = np.array(screen_xy, dtype=np.float64)
    candidates = _build_candidates(ridge_lambda, n_samples=len(feature_rows))

    results: list[tuple[str, float]] = []
    cv_kind = "LOO-CV" if F.shape[0] <= _GAZE_CV_LOO_MAX_N else "K-fold CV"
    print(f"\nEvaluating all models ({cv_kind}, n={F.shape[0]}) …", flush=True)
    err_hdr = "LOO err (px)" if F.shape[0] <= _GAZE_CV_LOO_MAX_N else "K-fold err (px)"
    print(f"  {'#':>2s}  {'Model':>20s}  {err_hdr:>18s}")
    print("  " + "-" * 46)

    for name in candidates:
        try:
            err = _gaze_cv_mean_error(candidates[name], F, Y)
            results.append((name, err))
        except Exception as exc:
            results.append((name, float("inf")))
            print(f"  {'':>2s}  {_MODEL_DISPLAY_NAMES.get(name, name):>20s}  FAILED ({exc})",
                  file=sys.stderr, flush=True)

    results.sort(key=lambda t: t[1])
    for i, (name, err) in enumerate(results):
        display = _MODEL_DISPLAY_NAMES.get(name, name)
        tag = " <<<" if i == 0 else ""
        if err == float("inf"):
            print(f"  {i + 1:>2d}  {display:>20s}  {'FAILED':>18s}")
        else:
            print(f"  {i + 1:>2d}  {display:>20s}  {err:>18.1f}{tag}")

    if preferred_model != "auto" and preferred_model in candidates:
        chosen_name = preferred_model
        print(f"\nUsing model from --gaze-model: {_MODEL_DISPLAY_NAMES.get(chosen_name, chosen_name)}", flush=True)
    else:
        if results[0][1] == float("inf"):
            raise RuntimeError(
                "All gaze model candidates failed during LOO-CV. "
                "Try more calibration samples or check the camera feed.",
            )
        chosen_name = results[0][0]
        print(
            f"\nAuto-selected best model: {_MODEL_DISPLAY_NAMES.get(chosen_name, chosen_name)} "
            f"(override with e.g. --gaze-model ridge)",
            flush=True,
        )

    print(f"Fitting {_MODEL_DISPLAY_NAMES.get(chosen_name, chosen_name)} on all data …", flush=True)
    return fit_gaze_model(
        feature_rows, screen_xy, gaze_width, gaze_height,
        ridge_lambda=ridge_lambda, model=chosen_name,
    )


def calibrate_cli(args: argparse.Namespace, proc_fn, cfg: MonitorConfig) -> int:
    gw = args.gaze_width
    gh = args.gaze_height

    screen = detect_screen_resolution()
    if screen is not None:
        gw, gh = screen
        print(f"Detected screen: {gw} x {gh}", flush=True)
    else:
        print(f"Could not detect screen; using {gw} x {gh}", flush=True)

    if os.name == "nt":
        cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera index {cfg.camera_index}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = FaceMeshEyeDetector(cfg, model_path=args.model)
    try:
        try:
            cal = run_calibration(
                cap=cap,
                detector=detector,
                proc_fn=proc_fn,
                gaze_width=gw,
                gaze_height=gh,
                out_path=Path(args.gaze_file),
                samples_per_point=args.gaze_samples,
                ear_min=args.gaze_ear_min,
                ridge_lambda=args.gaze_ridge,
                gaze_model=getattr(args, "gaze_model", "auto"),
                use_fixed_grid=not getattr(args, "gaze_cal_random", False),
                n_calibration_points=max(3, int(getattr(args, "gaze_cal_points", 36))),
                calibration_seed=getattr(args, "gaze_cal_seed", None),
            )
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if cal is None:
            print("Calibration aborted.", file=sys.stderr)
            return 1
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0
