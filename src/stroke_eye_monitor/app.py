from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stroke_eye_monitor.cli_args import parse_args
from stroke_eye_monitor.config import MonitorConfig
from stroke_eye_monitor.core.detector import FaceMeshEyeDetector
from stroke_eye_monitor.core.gaze_mapping import GazeCalibration
from stroke_eye_monitor.pipeline.live import KeyboardSession, LiveEyePipeline
from stroke_eye_monitor.ui.drawing import draw_hud
from stroke_eye_monitor.utils.frame import letterbox_to_width
from stroke_eye_monitor.utils.fps import FpsMeter
from stroke_eye_monitor.utils.threaded_capture import ThreadedVideoCapture


def _load_gaze_calibration(
    args: argparse.Namespace,
) -> tuple[GazeCalibration | None, int]:
    """Load gaze JSON when ``--gaze``; return ``(calibration, exit_code)`` with ``exit_code`` 0 if ok."""
    if not args.gaze:
        return None, 0
    gp = Path(args.gaze_file)
    if not gp.is_file():
        print(
            f"Gaze file not found: {gp}  (run with --calibrate first)", file=sys.stderr
        )
        return None, 1
    gaze_cal = GazeCalibration.load(gp)
    if gaze_cal.feature_dim != 8:
        print(
            "This gaze_calibration.json doesn't match the current gaze model. "
            "Run --calibrate again to regenerate (expects 8 features: iris offsets, "
            "head rotation Rodrigues vector, bias).",
            file=sys.stderr,
        )
        return None, 1
    return gaze_cal, 0


def _build_live_pipeline(
    args: argparse.Namespace, gaze_cal: GazeCalibration | None
) -> LiveEyePipeline:
    kbd: KeyboardSession | None = None
    if args.keyboard and gaze_cal is not None:
        kbd = KeyboardSession.create_session(gaze_cal)

    return LiveEyePipeline(
        gaze_file_label=args.gaze_file if args.gaze else None,
        gaze_cal=gaze_cal,
        gaze_alpha=args.gaze_alpha,
        gaze_ear_min=args.gaze_ear_min,
        full_mesh=args.full_mesh,
        keyboard=kbd,
    )


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.keyboard:
        args.gaze = True

    from stroke_eye_monitor.config import detect_screen_resolution
    screen_res = detect_screen_resolution()
    target_width = screen_res[0] if screen_res else args.width

    cfg = MonitorConfig(
        camera_index=1,
        process_width=target_width,
        mirror_display=not args.no_mirror,
    )

    def proc_fn(frame: Any) -> Any:
        # Scale the frame down or up so it fits exactly inside the screen bounds
        if screen_res:
            sw, sh = screen_res
            fh, fw = frame.shape[:2]
            scale = min(sw / fw, sh / fh)
            nh, nw = int(round(fh * scale)), int(round(fw * scale))
            return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        return letterbox_to_width(frame, cfg.process_width)

    if args.calibrate:
        from stroke_eye_monitor.modes.gaze_calibration import calibrate_cli

        return calibrate_cli(args, proc_fn, cfg)

    if args.collect:
        from stroke_eye_monitor.modes.data_collection import collect_cli

        return collect_cli(args, proc_fn, cfg)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera index {cfg.camera_index}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = FaceMeshEyeDetector(cfg, model_path=args.model)
    gaze_cal, err = _load_gaze_calibration(args)
    if err != 0:
        detector.close()
        cap.release()
        return err

    pipeline = _build_live_pipeline(args, gaze_cal)
    fps_meter = FpsMeter()
    capture = ThreadedVideoCapture(cap)
    
    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    capture.start()

    try:
        while True:
            ok, frame = capture.read(timeout=0.5)
            if not ok or frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("d"):
                    pipeline.backspace_typed()
                continue

            fps = fps_meter.tick()
            proc = proc_fn(frame)
            result = detector.process_bgr(proc)

            display = proc
            hud_lines = pipeline.step(display, result)

            if cfg.mirror_display:
                display = cv2.flip(display, 1)

            # Center the camera frame on a full-screen canvas before drawing the UI
            if screen_res is not None:
                sw, sh = screen_res
                fh, fw = display.shape[:2]
                
                # Create a black background to match the OS resolution exactly
                canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
                
                # Center the camera image inside the canvas
                dx = (sw - fw) // 2
                dy = (sh - fh) // 2
                
                if dx >= 0 and dy >= 0:
                    canvas[dy:dy+fh, dx:dx+fw] = display
                else:
                    canvas = cv2.resize(display, (sw, sh))
                    
                display = canvas

            draw_hud(
                display,
                fps=fps,
                process_ms=result.process_ms,
                face_ok=result.landmarks is not None,
                lines=hud_lines,
            )

            # Overlay keyboard directly onto the webcam feed (after mirror flip!)
            if pipeline.keyboard_session is not None:
                pipeline.draw_keyboard(display)

            cv2.imshow(cfg.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                pipeline.backspace_typed()
    finally:
        capture.stop()
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    ks = pipeline.keyboard_session
    if ks is not None:
        typed = ks.keyboard.typed_text
        if typed:
            print(f"Typed: {typed}")

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
