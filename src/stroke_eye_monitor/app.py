from __future__ import annotations

import argparse
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stroke_eye_monitor.audio_voice import SpokenContextBuffer

import cv2
import numpy as np

from LLM.env import load_llm_env

load_llm_env()

from stroke_eye_monitor.cli_args import parse_args
from stroke_eye_monitor.config import MonitorConfig
from stroke_eye_monitor.core.detector import FaceMeshEyeDetector
from stroke_eye_monitor.core.gaze_mapping import GazeCalibration
from stroke_eye_monitor.pipeline.live import KeyboardSession, LiveEyePipeline
from stroke_eye_monitor.ui.drawing import draw_hud
from stroke_eye_monitor.utils.frame import letterbox_to_width
from stroke_eye_monitor.utils.fps import FpsMeter
from stroke_eye_monitor.utils.threaded_capture import ThreadedVideoCapture
from LLM.openai_backend import OpenAICompletion


def _load_gaze_calibration_from_path(path: Path) -> tuple[GazeCalibration | None, int]:
    """Load gaze JSON from ``path``. Return ``(calibration, exit_code)``; exit 1 on invalid file."""
    gaze_cal = GazeCalibration.load(path)
    if gaze_cal.feature_dim != 8:
        print(
            "This gaze_calibration.json doesn't match the current gaze model. "
            "Run calibration again to regenerate (expects 8 features: iris offsets, "
            "head rotation Rodrigues vector, bias).",
            file=sys.stderr,
        )
        return None, 1
    return gaze_cal, 0


def _build_live_pipeline(
    args: argparse.Namespace,
    gaze_cal: GazeCalibration | None,
    llm_backend=None,
    *,
    spoken_buffer: SpokenContextBuffer | None = None,
    on_sentence_chosen: Callable[[str], None] | None = None,
) -> LiveEyePipeline:
    kbd: KeyboardSession | None = None
    if args.keyboard and gaze_cal is not None:
        kbd = KeyboardSession.create_session(
            gaze_cal,
            on_sentence_chosen=on_sentence_chosen,
            blink_close=args.blink_close,
            blink_open=args.blink_open,
            spoken_buffer=spoken_buffer,
        )

    return LiveEyePipeline(
        gaze_file_label=args.gaze_file if args.gaze else None,
        gaze_cal=gaze_cal,
        gaze_alpha=args.gaze_alpha,
        gaze_ear_min=args.gaze_ear_min,
        full_mesh=args.full_mesh,
        keyboard=kbd,
        llm_backend=llm_backend,
        spoken_buffer=spoken_buffer,
    )


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.keyboard:
        args.gaze = True

    from stroke_eye_monitor.config import detect_screen_resolution

    screen_res = detect_screen_resolution()
    target_width = screen_res[0] if screen_res else args.width

    cfg = MonitorConfig(
        camera_index=args.camera,
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
            return cv2.resize(
                frame,
                (nw, nh),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )
        return letterbox_to_width(frame, cfg.process_width)

    if args.calibrate:
        from stroke_eye_monitor.modes.gaze_calibration import calibrate_cli

        return calibrate_cli(args, proc_fn, cfg)

    if args.collect:
        from stroke_eye_monitor.modes.data_collection import collect_cli

        return collect_cli(args, proc_fn, cfg)

    gaze_cal: GazeCalibration | None = None
    if args.gaze:
        gp = Path(args.gaze_file)
        if gp.is_file():
            gaze_cal, err = _load_gaze_calibration_from_path(gp)
            if err != 0:
                return err
        elif args.no_auto_calibrate:
            print(
                f"Gaze calibration file not found: {gp.resolve()}. "
                "Run without --no-auto-calibrate to calibrate once, or use --calibrate.",
                file=sys.stderr,
            )
            return 1

    if args.blink_open <= args.blink_close:
        print(
            f"Invalid blink thresholds: --blink-open ({args.blink_open}) must be greater than "
            f"--blink-close ({args.blink_close}).",
            file=sys.stderr,
        )
        return 1

    print(f"Opening camera {cfg.camera_index}...", flush=True)

    if os.name == "nt":
        cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera index {cfg.camera_index}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = FaceMeshEyeDetector(cfg, model_path=args.model)

    if args.gaze and gaze_cal is None:
        from stroke_eye_monitor.modes.gaze_calibration import run_calibration

        gw, gh = (
            (screen_res[0], screen_res[1])
            if screen_res is not None
            else (args.gaze_width, args.gaze_height)
        )
        print(
            "No gaze calibration file found; starting interactive calibration (12 gaze points "
            "+ blink timing), then continuing to the main view.",
            flush=True,
        )
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
            detector.close()
            cap.release()
            return 1
        gaze_cal = cal

    llm = OpenAICompletion()
    voice_listener = None
    spoken_buffer = None
    on_chosen: Callable[[str], None] | None = None
    pipeline_ref: list[LiveEyePipeline | None] = [None]
    voice_stt_wanted: list[bool] = [False]
    if args.keyboard and not args.no_voice:
        if args.voice_play_backend:
            os.environ["NEUROVOX_AUDIO_PLAY_BACKEND"] = args.voice_play_backend
        if args.voice_record_backend:
            os.environ["NEUROVOX_AUDIO_RECORD_BACKEND"] = args.voice_record_backend
        if getattr(args, "whisper_language", None):
            os.environ["OPENAI_WHISPER_LANGUAGE"] = args.whisper_language
        try:
            from LLM.audio_platform import describe_audio_stack
            from stroke_eye_monitor.audio_voice import SpokenContextBuffer, SttListener

            spoken_buffer = SpokenContextBuffer()
            voice_listener = SttListener(
                llm,
                spoken_buffer,
                chunk_seconds=float(args.audio_chunk_seconds),
                rms_threshold=args.stt_rms_threshold,
                peak_threshold=args.stt_peak_threshold,
            )
            voice_listener.start()
            print(
                "Voice: mic -> Whisper (context) | chosen phrase -> OpenAI TTS",
                flush=True,
            )
            print(describe_audio_stack(), flush=True)

            def _speak_async(text: str) -> None:
                """Block gaze + blink input and pause mic STT while TTS plays."""

                def run() -> None:
                    kbd = None
                    pl = pipeline_ref[0]
                    if pl is not None and pl.keyboard_session is not None:
                        kbd = pl.keyboard_session.keyboard
                    try:
                        if voice_listener is not None:
                            voice_listener.pause()
                        llm.speak(text)
                    finally:
                        # STT on/off is driven by keyboard.input_enabled in the main loop
                        # (idle = paused). Do not resume here or mic would run while idle.
                        if kbd is not None:
                            # Stay in typing mode after TTS so mic/STT can resume without
                            # another triple-blink unlock (idle only applies at app start).
                            kbd.block_input = False
                            kbd.block_overlay_text = "Ready to type"
                        if spoken_buffer is not None:
                            spoken_buffer.clear()

                pl = pipeline_ref[0]
                if pl is not None and pl.keyboard_session is not None:
                    kbd = pl.keyboard_session.keyboard
                    kbd.block_overlay_text = "Speaking..."
                    kbd.block_input = True
                threading.Thread(target=run, daemon=True).start()

            on_chosen = _speak_async
        except (ImportError, RuntimeError, OSError) as exc:
            print(f"Voice disabled: {exc}", flush=True)

    pipeline = _build_live_pipeline(
        args,
        gaze_cal,
        llm_backend=llm,
        spoken_buffer=spoken_buffer,
        on_sentence_chosen=on_chosen,
    )
    pipeline_ref[0] = pipeline
    if voice_listener is not None and pipeline.keyboard_session is not None:
        voice_listener.pause()

    fps_meter = FpsMeter()
    capture = ThreadedVideoCapture(cap)

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    capture.start()

    try:
        while True:
            ok, frame = capture.read(timeout=0.5)
            if not ok or frame is None:
                print("Waiting for camera frame...", end="\r", flush=True)
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

            if voice_listener is not None and pipeline.keyboard_session is not None:
                kb = pipeline.keyboard_session.keyboard
                # Listen whenever we are not in LLM/TTS (block_input) or picking a suggestion,
                # including while idle — so speech context is captured before triple-blink unlock.
                want_stt = (
                    not kb.block_input
                    and not kb.suggestions
                )
                if voice_stt_wanted[0] != want_stt:
                    voice_stt_wanted[0] = want_stt
                    if want_stt:
                        voice_listener.resume()
                    else:
                        voice_listener.pause()

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
                    canvas[dy : dy + fh, dx : dx + fw] = display
                else:
                    canvas = cv2.resize(display, (sw, sh))

                display = canvas

            if pipeline.keyboard_session is not None:
                # Do not draw debug tracking values that overlap with the keyboard UI
                hud_lines = []

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
        if voice_listener is not None:
            voice_listener.stop()
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
