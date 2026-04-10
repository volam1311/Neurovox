from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from stroke_eye_monitor.config import MonitorConfig
from stroke_eye_monitor.detector import FaceMeshEyeDetector
from stroke_eye_monitor.gaze_mapping import GazeCalibration, fit_affine_gaze
from stroke_eye_monitor.metrics import compute_eye_metrics, gaze_feature_vector

# Fractions of gaze canvas (helps cover the screen).
def _sync_canvas_to_opencv_window(win: str, gw: int, gh: int) -> tuple[int, int]:
    """Use the real OpenCV client size after fullscreen (fixes many Retina / scaling mismatches)."""
    board = np.zeros((max(gh, 1), max(gw, 1), 3), dtype=np.uint8)
    cv2.imshow(win, board)
    best_w, best_h = gw, gh
    for _ in range(60):
        cv2.waitKey(16)
        r = cv2.getWindowImageRect(win)
        cw, ch = int(r[2]), int(r[3])
        if cw >= 320 and ch >= 240:
            best_w, best_h = cw, ch
            break
    print(
        f"Calibration canvas (OpenCV window): {best_w} x {best_h}",
        flush=True,
    )
    return best_w, best_h


# 12-point grid: 4x3 (columns x rows), row-major.
_COL_FRAC = (0.08, 0.36, 0.64, 0.92)
_ROW_FRAC = (0.10, 0.50, 0.90)
DEFAULT_TARGETS: list[tuple[float, float]] = [
    (x, y) for y in _ROW_FRAC for x in _COL_FRAC
]


def run_calibration(
    *,
    cap: cv2.VideoCapture,
    detector: FaceMeshEyeDetector,
    proc_fn,
    gaze_width: int,
    gaze_height: int,
    out_path: Path,
    targets: list[tuple[float, float]] | None = None,
    samples_per_point: int = 45,
    ear_min: float = 0.17,
    ridge_lambda: float = 1e-2,
) -> GazeCalibration | None:
    """
    Show fixation dots on a gaze-sized canvas (12-point grid).
    User looks at each dot and presses SPACE. Iris offsets from MediaPipe are regressed to dot
    positions with ridge regression.
    """
    t_list = targets or DEFAULT_TARGETS
    win = "Calibration — look at the white dot, press SPACE"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, gaze_width, gaze_height)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass

    gaze_width, gaze_height = _sync_canvas_to_opencv_window(win, gaze_width, gaze_height)

    feature_rows: list[np.ndarray] = []
    screen_xy: list[tuple[float, float]] = []

    for idx, (fx, fy) in enumerate(t_list):
        tx = int(round(fx * (gaze_width - 1)))
        ty = int(round(fy * (gaze_height - 1)))
        print(f"Point {idx + 1}/{len(t_list)}: look at the dot, hold still, press SPACE", flush=True)

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
                # Skip blink/half-closed frames (a major source of jitter).
                if m.left_ear >= ear_min and m.right_ear >= ear_min:
                    g = gaze_feature_vector(m)
                    if g is not None:
                        collected.append(g.copy())
                if len(collected) >= samples_per_point:
                    # Median is more robust to occasional landmark spikes.
                    feat_med = np.median(np.stack(collected, axis=0), axis=0)
                    feature_rows.append(feat_med)
                    screen_xy.append((float(tx), float(ty)))
                    print(f"  captured mean feature, n={len(collected)}", flush=True)
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
    cal = fit_affine_gaze(
        feature_rows,
        screen_xy,
        gaze_width,
        gaze_height,
        ridge_lambda=ridge_lambda,
    )
    cal.save(out_path)
    print(f"Saved calibration to {out_path}", flush=True)
    return cal


def calibrate_cli(args: argparse.Namespace, proc_fn, cfg: MonitorConfig) -> int:
    from stroke_eye_monitor.config import detect_screen_resolution

    gw = args.gaze_width
    gh = args.gaze_height

    screen = detect_screen_resolution()
    if screen is not None:
        gw, gh = screen
        print(f"Detected screen: {gw} x {gh}", flush=True)
    else:
        print(f"Could not detect screen; using {gw} x {gh}", flush=True)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera index {cfg.camera_index}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = FaceMeshEyeDetector(cfg, model_path=args.model)
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
        )
        if cal is None:
            print("Calibration aborted.", file=sys.stderr)
            return 1
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0
