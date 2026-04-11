from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stroke_eye_monitor.config import MonitorConfig, detect_screen_resolution
from stroke_eye_monitor.detector import FaceMeshEyeDetector
from stroke_eye_monitor.drawing import draw_face_mesh_eyes, draw_hud, DrawStyle
from stroke_eye_monitor.metrics import (
    BlinkDetector,
    compute_eye_metrics,
    gaze_feature_vector,
    smooth_exponential,
    smooth_vec2,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Live eye & iris tracking via MediaPipe Face Landmarker. "
            "Optional calibrated on-screen gaze (see --calibrate / --gaze)."
        )
    )
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument(
        "--width",
        type=int,
        default=640,
        help="Resize frame to this width before inference (height keeps aspect)",
    )
    p.add_argument("--no-mirror", action="store_true", help="Disable horizontal flip for display")
    p.add_argument(
        "--full-mesh",
        action="store_true",
        help="Draw full face tessellation (prettier, higher CPU draw cost)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to face_landmarker.task (default: download to ./models/)",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Run gaze calibration (saves --gaze-file), then exit",
    )
    p.add_argument(
        "--gaze",
        action="store_true",
        help="Load calibration and draw estimated gaze point on the video",
    )
    p.add_argument(
        "--gaze-file",
        type=str,
        default="gaze_calibration.json",
        help="Path to gaze calibration JSON (from --calibrate)",
    )
    p.add_argument("--gaze-width", type=int, default=1280, help="Calibration canvas width (pixels)")
    p.add_argument("--gaze-height", type=int, default=720, help="Calibration canvas height (pixels)")
    p.add_argument(
        "--gaze-samples",
        type=int,
        default=45,
        help="Frames averaged per calibration point after SPACE",
    )
    p.add_argument(
        "--gaze-alpha",
        type=float,
        default=0.25,
        help="Exponential smoothing for gaze position (0..1, higher = snappier)",
    )
    p.add_argument(
        "--gaze-ear-min",
        type=float,
        default=0.17,
        help="Reject gaze updates when either eye EAR is below this (blink filter)",
    )
    p.add_argument(
        "--gaze-ridge",
        type=float,
        default=1e-2,
        help="Ridge regularization used during calibration fit (only affects --calibrate)",
    )
    p.add_argument(
        "--keyboard",
        action="store_true",
        help="Show gaze keyboard overlay (implies --gaze); blink to select letters",
    )
    p.add_argument(
        "--collect",
        action="store_true",
        help="Show random dots, capture iris coordinates at each, save to CSV, then exit",
    )
    p.add_argument(
        "--collect-csv",
        type=str,
        default="gaze_data.csv",
        help="Output CSV path for --collect (default: gaze_data.csv)",
    )
    p.add_argument(
        "--collect-points",
        type=int,
        default=36,
        help="Number of random points to collect (default: 36)",
    )
    return p.parse_args(argv)


def _letterbox_width(frame: Any, target_w: int) -> Any:
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    scale = target_w / float(w)
    nh = int(round(h * scale))
    return cv2.resize(frame, (target_w, nh), interpolation=cv2.INTER_AREA)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.keyboard:
        args.gaze = True

    cfg = MonitorConfig(
        camera_index=args.camera,
        process_width=args.width,
        mirror_display=not args.no_mirror,
    )

    def proc_fn(frame: Any) -> Any:
        return _letterbox_width(frame, cfg.process_width)

    if args.calibrate:
        from stroke_eye_monitor.gaze_calibration import calibrate_cli

        return calibrate_cli(args, proc_fn, cfg)

    if args.collect:
        from stroke_eye_monitor.data_collection import collect_cli

        return collect_cli(args, proc_fn, cfg)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera index {cfg.camera_index}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = FaceMeshEyeDetector(cfg, model_path=args.model)
    fps_buf: collections.deque[float] = collections.deque(maxlen=30)
    last_tick = time.perf_counter()

    sm_l_ear = 0.25
    sm_r_ear = 0.25
    sm_asym = 0.0
    sm_li: tuple[float, float] | None = None
    sm_ri: tuple[float, float] | None = None
    alpha = 0.35

    gaze_cal = None
    if args.gaze:
        from stroke_eye_monitor.gaze_mapping import GazeCalibration

        gp = Path(args.gaze_file)
        if not gp.is_file():
            print(f"Gaze file not found: {gp}  (run with --calibrate first)", file=sys.stderr)
            detector.close()
            cap.release()
            return 1
        gaze_cal = GazeCalibration.load(gp)
        if gaze_cal.feature_dim != 5:
            print(
                "This gaze_calibration.json doesn't match the current eyes-only gaze model. "
                "Run --calibrate again to regenerate (expects 5 features: iris offsets + bias).",
                file=sys.stderr,
            )
            detector.close()
            cap.release()
            return 1
        gaze_w = gaze_cal.gaze_width
        gaze_h = gaze_cal.gaze_height

    sm_gx: float | None = None
    sm_gy: float | None = None

    keyboard = None
    blink_det = None
    kb_win = "Gaze Keyboard"
    kb_w, kb_h = 0, 0
    if args.keyboard and gaze_cal is not None:
        from stroke_eye_monitor.keyboard_overlay import GazeKeyboard

        screen = detect_screen_resolution()
        kb_w = screen[0] if screen else gaze_w
        kb_h = screen[1] if screen else gaze_h

        keyboard = GazeKeyboard()
        keyboard.layout(gaze_w, gaze_h)
        blink_det = BlinkDetector()

        cv2.namedWindow(kb_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(kb_win, kb_w, kb_h)
        try:
            cv2.setWindowProperty(kb_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except cv2.error:
            pass

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            now = time.perf_counter()
            fps_buf.append(1.0 / max(now - last_tick, 1e-6))
            last_tick = now
            fps = sum(fps_buf) / max(len(fps_buf), 1)

            proc = proc_fn(frame)
            result = detector.process_bgr(proc)

            display = proc
            hud_lines: list[str] = [
                "Research demo — not medical or security-grade gaze tracking.",
            ]
            if args.gaze:
                hud_lines.append(f"Gaze file: {args.gaze_file}")

            if result.landmarks is not None:
                h, w = result.image_shape
                m = compute_eye_metrics(result.landmarks, h, w)
                sm_l_ear = smooth_exponential(sm_l_ear, m.left_ear, alpha)
                sm_r_ear = smooth_exponential(sm_r_ear, m.right_ear, alpha)
                sm_asym = smooth_exponential(sm_asym, m.ear_asymmetry, alpha)
                sm_li = smooth_vec2(sm_li, m.left_iris_offset, alpha)
                sm_ri = smooth_vec2(sm_ri, m.right_iris_offset, alpha)

                style = DrawStyle.FULL if args.full_mesh else DrawStyle.EYES_ONLY
                draw_face_mesh_eyes(display, result.landmarks, style=style)

                hud_lines.append(f"EAR L {sm_l_ear:.3f}  R {sm_r_ear:.3f}  |d|: {sm_asym:.3f}")
                if sm_li and sm_ri:
                    hud_lines.append(
                        f"Iris offset L nx,ny {sm_li[0]:+.2f},{sm_li[1]:+.2f}  "
                        f"R {sm_ri[0]:+.2f},{sm_ri[1]:+.2f}"
                    )

                if gaze_cal is not None:
                    if m.left_ear >= args.gaze_ear_min and m.right_ear >= args.gaze_ear_min:
                        g = gaze_feature_vector(m)
                    else:
                        g = None
                    if g is not None:
                        rx, ry = gaze_cal.predict(g)
                        rx, ry = gaze_cal.clamp(rx, ry)
                        a = max(0.0, min(1.0, args.gaze_alpha))
                        if sm_gx is None:
                            sm_gx, sm_gy = rx, ry
                        else:
                            sm_gx = a * rx + (1.0 - a) * sm_gx
                            sm_gy = a * ry + (1.0 - a) * sm_gy

                    if sm_gx is not None and sm_gy is not None:
                        hud_lines.append(f"Gaze xy: ({sm_gx:.0f}, {sm_gy:.0f})")

                if keyboard is not None and blink_det is not None:
                    if sm_gx is not None and sm_gy is not None:
                        keyboard.update_gaze(sm_gx, sm_gy)

                    avg_ear = (m.left_ear + m.right_ear) / 2.0
                    if blink_det.feed(avg_ear):
                        letter = keyboard.select()
                        if letter is not None:
                            hud_lines.append(f"SELECTED: {letter}")

            if keyboard is not None:
                kb_canvas = np.zeros((kb_h, kb_w, 3), dtype=np.uint8)
                gaze_pt = (sm_gx, sm_gy) if sm_gx is not None else None
                keyboard.draw(
                    kb_canvas,
                    left_iris=sm_li,
                    right_iris=sm_ri,
                    gaze_xy=gaze_pt,
                )
                cv2.imshow(kb_win, kb_canvas)

            if cfg.mirror_display:
                display = cv2.flip(display, 1)

            draw_hud(
                display,
                fps=fps,
                process_ms=result.process_ms,
                face_ok=result.landmarks is not None,
                lines=hud_lines,
            )

            cv2.imshow(cfg.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d") and keyboard is not None and keyboard.typed:
                keyboard.typed.pop()
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    if keyboard is not None:
        typed = keyboard.typed_text
        if typed:
            print(f"Typed: {typed}")

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
