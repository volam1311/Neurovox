from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from stroke_eye_monitor.config import MonitorConfig, detect_screen_resolution
from stroke_eye_monitor.core.detector import FaceMeshEyeDetector
from stroke_eye_monitor.core.gaze_mapping import GazeCalibration, fit_affine_gaze
from stroke_eye_monitor.core.metrics import (
    BlinkDetector,
    compute_eye_metrics,
    gaze_feature_vector,
)
from stroke_eye_monitor.utils.opencv_canvas import sync_opencv_window_canvas

# 12-point grid: 4x3 (columns x rows), row-major.
_COL_FRAC = (0.08, 0.36, 0.64, 0.92)
_ROW_FRAC = (0.10, 0.50, 0.90)
DEFAULT_TARGETS: list[tuple[float, float]] = [
    (x, y) for y in _ROW_FRAC for x in _COL_FRAC
]


def _derive_blink_params_from_dwell(dwell_ms: int) -> tuple[float, float, float]:
    """Fallback burst timing when blink samples are unavailable (derived from dwell only)."""
    ds = dwell_ms / 1000.0
    commit = max(0.32, min(0.62, 0.18 + ds * 0.42))
    window = max(2.0, min(5.5, ds * 6.5 + 0.75))
    abort = max(2.0, min(6.5, window * 1.08))
    return window, commit, abort


def run_blink_calibration_phase(
    *,
    cap: cv2.VideoCapture,
    detector: FaceMeshEyeDetector,
    proc_fn,
    gaze_width: int,
    gaze_height: int,
    dwell_ms: int,
    blink_close: float,
    blink_open: float,
) -> tuple[float, float, float] | None:
    """Measure three blinks; derive burst window and commit pause. None = user aborted (Q)."""
    win = "Blink calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, gaze_width, gaze_height)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass

    gaze_width, gaze_height = sync_opencv_window_canvas(win, gaze_width, gaze_height)

    blink = BlinkDetector(
        close_threshold=blink_close,
        open_threshold=blink_open,
        min_closed_frames=1,
        cooldown_frames=4,
    )
    blink_times: list[float] = []
    deadline = time.monotonic() + 90.0

    print(
        "Blink calibration: blink 3 times at your natural pace (quick blinks are OK).",
        flush=True,
    )

    while len(blink_times) < 3:
        if time.monotonic() >= deadline:
            bw, cp, ia = _derive_blink_params_from_dwell(dwell_ms)
            print(
                "Blink calibration timed out; using estimates from dwell timing.",
                flush=True,
            )
            cv2.destroyWindow(win)
            return bw, cp, ia

        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        proc = proc_fn(frame)
        result = detector.process_bgr(proc)

        board = np.zeros((gaze_height, gaze_width, 3), dtype=np.uint8)
        hint = f"Blink 3 times ({len(blink_times)}/3)  Q=abort"
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
        cv2.putText(
            board,
            "Pause briefly after the third blink to confirm",
            (16, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        if result.landmarks is not None:
            h0, w0 = result.image_shape
            m = compute_eye_metrics(result.landmarks, h0, w0)
            avg_ear = (m.left_ear + m.right_ear) / 2.0
            if blink.feed(avg_ear):
                blink_times.append(time.monotonic())
                print(f"  blink captured {len(blink_times)}/3", flush=True)

        cv2.imshow(win, board)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(win)
            return None

    t1, t2, t3 = blink_times[0], blink_times[1], blink_times[2]
    g12 = t2 - t1
    g23 = t3 - t2
    gap_med = statistics.median([g12, g23])
    span = t3 - t1
    commit_pause = max(0.28, min(0.68, gap_med * 0.52 + 0.20))
    burst_window = max(2.0, min(6.0, span + 1.75 * commit_pause))
    idle_abort = max(2.5, min(7.0, burst_window + 0.35))
    print(
        f"Auto-calibrated blink timing: commit pause {commit_pause:.2f}s, "
        f"burst window {burst_window:.2f}s, idle abort {idle_abort:.2f}s",
        flush=True,
    )
    cv2.destroyWindow(win)
    return burst_window, commit_pause, idle_abort


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
    blink_close: float = 0.12,
    blink_open: float = 0.16,
) -> GazeCalibration | None:
    """
    Show fixation dots on a gaze-sized canvas (12-point grid).
    User looks at each dot and presses SPACE. Iris offsets from MediaPipe are regressed to dot
    positions with ridge regression.

    Then: blink calibration (three blinks) to personalize dwell-adjacent burst timing, or
    timeout/abort handling as documented in ``run_blink_calibration_phase``.
    """
    t_list = targets or DEFAULT_TARGETS
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

    feature_rows: list[np.ndarray] = []
    screen_xy: list[tuple[float, float]] = []
    fixation_durations: list[float] = []

    for idx, (fx, fy) in enumerate(t_list):
        tx = int(round(fx * (gaze_width - 1)))
        ty = int(round(fy * (gaze_height - 1)))
        print(
            f"Point {idx + 1}/{len(t_list)}: look at the dot, hold still, press SPACE",
            flush=True,
        )

        collecting = False
        collected: list[np.ndarray] = []
        fixation_start: float = 0.0

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
                board, hint, (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA,
            )

            if collecting and result.landmarks is not None:
                h0, w0 = result.image_shape
                m = compute_eye_metrics(result.landmarks, h0, w0)
                if m.left_ear >= ear_min and m.right_ear >= ear_min:
                    g = gaze_feature_vector(m, result.face_matrix)
                    if g is not None:
                        collected.append(g.copy())
                if len(collected) >= samples_per_point:
                    feat_med = np.median(np.stack(collected, axis=0), axis=0)
                    feature_rows.append(feat_med)
                    screen_xy.append((float(tx), float(ty)))
                    dur = time.monotonic() - fixation_start
                    fixation_durations.append(dur)
                    print(f"  captured mean feature, n={len(collected)}, fixation {dur:.2f}s", flush=True)
                    break

            cv2.imshow(win, board)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyWindow(win)
                return None
            if key in (13, 32):  # Enter or Space
                collecting = True
                collected = []
                fixation_start = time.monotonic()

    # Fixation duration here is mostly time to collect ``samples_per_point`` frames,
    # not a comfortable key dwell. Scale down and clamp to a short dwell range.
    if fixation_durations:
        median_s = statistics.median(fixation_durations)
        raw_ms = median_s * 1000.0
        dwell_ms = int(max(260, min(720, raw_ms * 0.36)))
    else:
        dwell_ms = 600
    print(f"Auto-calibrated dwell time: {dwell_ms}ms", flush=True)

    cv2.destroyWindow(win)

    blink_out = run_blink_calibration_phase(
        cap=cap,
        detector=detector,
        proc_fn=proc_fn,
        gaze_width=gaze_width,
        gaze_height=gaze_height,
        dwell_ms=dwell_ms,
        blink_close=blink_close,
        blink_open=blink_open,
    )
    if blink_out is None:
        return None
    bw, cp, ia = blink_out

    cal = fit_affine_gaze(
        feature_rows,
        screen_xy,
        gaze_width,
        gaze_height,
        ridge_lambda=ridge_lambda,
        dwell_ms=dwell_ms,
    )
    cal.blink_burst_window_s = bw
    cal.blink_commit_pause_s = cp
    cal.blink_idle_abort_s = ia
    cal.save(out_path)
    print(f"Saved calibration to {out_path}", flush=True)
    return cal


def calibrate_cli(args: argparse.Namespace, proc_fn, cfg: MonitorConfig) -> int:
    gw = args.gaze_width
    gh = args.gaze_height

    screen = detect_screen_resolution()
    if screen is not None:
        gw, gh = screen
        print(f"Detected screen: {gw} x {gh}", flush=True)
    else:
        print(f"Could not detect screen; using {gw} x {gh}", flush=True)

    print(f"Opening camera {cfg.camera_index} for calibration...", flush=True)
    import os

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
            blink_close=args.blink_close,
            blink_open=args.blink_open,
        )
        if cal is None:
            print("Calibration aborted.", file=sys.stderr)
            return 1
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0
