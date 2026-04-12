"""Raw iris-offset collection for analysis.

This mode saves **iris offsets + screen dot** per point. It does **not** write the same
11-D ``gaze_feature_vector`` as ``--calibrate`` (which adds head pose + bias). Do not
expect to drop ``--collect`` CSV into the gaze mapper; use ``--calibrate`` for the app.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from stroke_eye_monitor.config import MonitorConfig, detect_screen_resolution
from stroke_eye_monitor.core.detector import FaceMeshEyeDetector
from stroke_eye_monitor.core.metrics import compute_eye_metrics
from stroke_eye_monitor.utils.opencv_canvas import sync_opencv_window_canvas


def _generate_random_targets(n: int, margin: float = 0.08) -> list[tuple[float, float]]:
    """Generate n random (x_frac, y_frac) targets within [margin, 1-margin]."""
    lo = margin
    hi = 1.0 - margin
    return [(random.uniform(lo, hi), random.uniform(lo, hi)) for _ in range(n)]


def run_collection(
    *,
    cap: cv2.VideoCapture,
    detector: FaceMeshEyeDetector,
    proc_fn,
    canvas_width: int,
    canvas_height: int,
    out_csv: Path,
    num_points: int = 36,
    samples_per_point: int = 45,
    ear_min: float = 0.17,
) -> bool:
    """Show random dots on a fullscreen canvas; capture iris offsets at each.

    Saves a CSV with columns:
        point_id, screen_x, screen_y, left_nx, left_ny, right_nx, right_ny
    """
    targets = _generate_random_targets(num_points)

    win = "Data collection — look at the dot, press SPACE"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, canvas_width, canvas_height)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass

    canvas_width, canvas_height = sync_opencv_window_canvas(
        win, canvas_width, canvas_height
    )
    print(f"Canvas: {canvas_width} x {canvas_height}", flush=True)

    rows: list[dict[str, object]] = []

    for idx, (fx, fy) in enumerate(targets):
        tx = int(round(fx * (canvas_width - 1)))
        ty = int(round(fy * (canvas_height - 1)))
        print(
            f"Point {idx + 1}/{num_points}: look at the dot, hold still, press SPACE",
            flush=True,
        )

        collecting = False
        collected_l: list[tuple[float, float]] = []
        collected_r: list[tuple[float, float]] = []

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            proc = proc_fn(frame)
            result = detector.process_bgr(proc)

            board = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            r_dot = max(18, canvas_width // 40)
            cv2.circle(board, (tx, ty), r_dot, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(board, (tx, ty), r_dot + 4, (0, 0, 0), 3, cv2.LINE_AA)

            progress = ""
            if collecting:
                progress = f"  collecting {len(collected_l)}/{samples_per_point}"
            hint = f"{idx + 1}/{num_points}{progress}  SPACE=capture  Q=abort"
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
                if (
                    m.left_ear >= ear_min
                    and m.right_ear >= ear_min
                    and m.left_iris_offset is not None
                    and m.right_iris_offset is not None
                ):
                    collected_l.append(m.left_iris_offset)
                    collected_r.append(m.right_iris_offset)

                if len(collected_l) >= samples_per_point:
                    l_arr = np.array(collected_l)
                    r_arr = np.array(collected_r)
                    med_l = np.median(l_arr, axis=0)
                    med_r = np.median(r_arr, axis=0)
                    rows.append(
                        {
                            "point_id": idx + 1,
                            "screen_x": tx,
                            "screen_y": ty,
                            "left_nx": round(float(med_l[0]), 6),
                            "left_ny": round(float(med_l[1]), 6),
                            "right_nx": round(float(med_r[0]), 6),
                            "right_ny": round(float(med_r[1]), 6),
                        }
                    )
                    print(
                        f"  captured: L({med_l[0]:+.4f},{med_l[1]:+.4f}) "
                        f"R({med_r[0]:+.4f},{med_r[1]:+.4f})",
                        flush=True,
                    )
                    break

            cv2.imshow(win, board)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyWindow(win)
                if rows:
                    _write_csv(out_csv, rows)
                    print(f"Partial save ({len(rows)} points) to {out_csv}", flush=True)
                return False
            if key in (13, 32):
                collecting = True
                collected_l = []
                collected_r = []

    cv2.destroyWindow(win)
    _write_csv(out_csv, rows)
    print(f"Saved {len(rows)} points to {out_csv}", flush=True)
    return True


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "point_id",
        "screen_x",
        "screen_y",
        "left_nx",
        "left_ny",
        "right_nx",
        "right_ny",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_cli(args: argparse.Namespace, proc_fn, cfg: MonitorConfig) -> int:
    cw = args.gaze_width
    ch = args.gaze_height

    screen = detect_screen_resolution()
    if screen is not None:
        cw, ch = screen
        print(f"Detected screen: {cw} x {ch}", flush=True)
    else:
        print(f"Could not detect screen; using {cw} x {ch}", flush=True)

    print(f"Opening camera {cfg.camera_index} for data collection...", flush=True)
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
        ok = run_collection(
            cap=cap,
            detector=detector,
            proc_fn=proc_fn,
            canvas_width=cw,
            canvas_height=ch,
            out_csv=Path(args.collect_csv),
            num_points=args.collect_points,
            samples_per_point=args.collect_samples,
            ear_min=args.gaze_ear_min,
        )
        if not ok:
            print(
                "Collection aborted (partial data may have been saved).",
                file=sys.stderr,
            )
            return 1
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0
