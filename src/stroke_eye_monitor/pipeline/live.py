from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import time

import cv2
import numpy as np

from stroke_eye_monitor.audio_voice import SpokenContextBuffer

from stroke_eye_monitor.core.detector import FrameResult
from stroke_eye_monitor.core.gaze_mapping import GazeCalibration
from stroke_eye_monitor.core.metrics import (
    BlinkDetector,
    compute_eye_metrics,
    gaze_feature_vector,
    smooth_exponential,
    smooth_vec2,
)

from stroke_eye_monitor.ui.drawing import DrawStyle, draw_face_mesh_eyes
from stroke_eye_monitor.ui.keyboard_overlay import GazeKeyboard
from LLM.completion import CompletionBackend
import threading


@dataclass
class KeyboardSession:
    """Fullscreen keyboard window + blink detector (constructed once per run)."""

    window_name: str
    pixel_w: int
    pixel_h: int
    keyboard: GazeKeyboard
    blink: BlinkDetector

    @staticmethod
    def create_session(
        gaze_cal: GazeCalibration,
        *,
        on_sentence_chosen: Callable[[str], None] | None = None,
        blink_close: float = 0.12,
        blink_open: float = 0.16,
        spoken_buffer: SpokenContextBuffer | None = None,
        margin_top_frac: float | None = None,
        margin_bot_frac: float | None = None,
        infer_confirm_hold_s: float = 3.0,
    ) -> KeyboardSession:
        win = "Main Output"  # We'll just draw onto the main app window

        # Must match gaze (x,y) space from calibration — NOT physical screen size.
        kb_w, kb_h = gaze_cal.gaze_width, gaze_cal.gaze_height

        keyboard = GazeKeyboard(
            dwell_seconds=gaze_cal.dwell_ms / 1000.0,
            on_sentence_chosen=on_sentence_chosen,
            spoken_buffer=spoken_buffer,
        )
        keyboard.layout(
            kb_w,
            kb_h,
            gaze_model=getattr(gaze_cal, "model_type", None),
            margin_top_frac=margin_top_frac,
            margin_bot_frac=margin_bot_frac,
        )
        keyboard.infer_confirm_hold_s = max(0.5, float(infer_confirm_hold_s))
        blink = BlinkDetector(
            close_threshold=blink_close,
            open_threshold=blink_open,
            min_closed_frames=1,
            cooldown_frames=4,
        )

        return KeyboardSession(
            window_name=win,
            pixel_w=kb_w,
            pixel_h=kb_h,
            keyboard=keyboard,
            blink=blink,
        )


class LiveEyePipeline:
    """Per-frame processing: landmarks -> metrics -> optional gaze -> optional keyboard."""

    def __init__(
        self,
        *,
        gaze_file_label: str | None,
        gaze_cal: GazeCalibration | None,
        gaze_alpha: float,
        gaze_ear_min: float,
        full_mesh: bool,
        keyboard: KeyboardSession | None,
        metric_smooth_alpha: float = 0.35,
        llm_backend: CompletionBackend | None = None,
        spoken_buffer: SpokenContextBuffer | None = None,
        keyboard_gaze_median_n: int = 1,
        keyboard_gaze_gain: float = 1.0,
        blink_close_threshold: float = 0.12,
    ) -> None:
        self._gaze_file_label = gaze_file_label
        self._gaze_cal = gaze_cal
        self._gaze_alpha = gaze_alpha
        self._gaze_ear_min = gaze_ear_min
        self._full_mesh = full_mesh
        self._kbd = keyboard
        self._alpha = metric_smooth_alpha
        self.llm_backend = llm_backend
        self._spoken_buffer = spoken_buffer
        self._llm_lock = threading.Lock()
        self._kbd_median_n = max(0, int(keyboard_gaze_median_n))
        self._kbd_gain = max(0.01, float(keyboard_gaze_gain))
        self._blink_close_thresh = float(blink_close_threshold)
        self._last_step_monotonic: float | None = None
        if keyboard is not None and self._kbd_median_n >= 3:
            self._kbd_sm_buf: deque[tuple[float, float]] = deque(maxlen=self._kbd_median_n)
        else:
            self._kbd_sm_buf = deque(maxlen=1)

        self._sm_l_ear = 0.25
        self._sm_r_ear = 0.25
        self._sm_asym = 0.0
        self._sm_li: tuple[float, float] | None = None
        self._sm_ri: tuple[float, float] | None = None
        self._sm_gx: float | None = None
        self._sm_gy: float | None = None
        self._sm_gaze_sigma: float | None = None
        #: Last (x,y) passed to the keyboard hit test / gaze dot (smoothed; median if N>=3).
        self._pointer_gx: float | None = None
        self._pointer_gy: float | None = None

    def _draw_gaze_overlay(
        self,
        frame: np.ndarray,
        gx: float,
        gy: float,
        sigma_canvas: float | None,
    ) -> None:
        """Map calibration-canvas gaze to frame pixels and draw mean plus an uncertainty ring."""
        cal = self._gaze_cal
        if cal is None or cal.gaze_width < 1 or cal.gaze_height < 1:
            return
        dh, dw = frame.shape[:2]
        sx = dw / float(cal.gaze_width)
        sy = dh / float(cal.gaze_height)
        px = int(round(gx * sx))
        py = int(round(gy * sy))
        if sigma_canvas is not None and sigma_canvas > 0:
            r_sig = max(5, int(round(sigma_canvas * 0.5 * (sx + sy))))
            cv2.circle(frame, (px, py), r_sig, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, (0, 120, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

    @property
    def keyboard_session(self) -> KeyboardSession | None:
        return self._kbd

    def _fire_llm(self, abbreviated: str, prior_turns: list[str]) -> None:
        """Run LLM inference in a background thread with error recovery."""
        kb = self._kbd
        if kb is None:
            return
        try:
            assert self.llm_backend is not None
            spoken = None
            if self._spoken_buffer is not None:
                spoken = self._spoken_buffer.get_for_llm()
            suggestions = self.llm_backend.complete_ranked(
                abbreviated=abbreviated,
                history=prior_turns if prior_turns else None,
                spoken_context=spoken,
                k=3,
            )
            with self._llm_lock:
                kb.keyboard.set_suggestions(suggestions)
        except Exception as exc:
            print(f"LLM error: {exc}", flush=True)
        finally:
            with self._llm_lock:
                kb.keyboard.block_input = False

    def step(self, display, result: FrameResult) -> list[str]:
        hud: list[str] = [
            "Research demo -- not medical or security-grade gaze tracking.",
        ]
        if self._gaze_file_label is not None:
            hud.append(f"Gaze file: {self._gaze_file_label}")

        if result.landmarks is None:
            if self._kbd is not None:
                self._kbd.keyboard.reset_infer_confirm_accum()
            return hud

        now = time.monotonic()
        if self._last_step_monotonic is None:
            self._last_step_monotonic = now
        dt = max(0.0, min(0.25, now - self._last_step_monotonic))
        self._last_step_monotonic = now

        h, w = result.image_shape
        m = compute_eye_metrics(result.landmarks, h, w)
        self._sm_l_ear = smooth_exponential(self._sm_l_ear, m.left_ear, self._alpha)
        self._sm_r_ear = smooth_exponential(self._sm_r_ear, m.right_ear, self._alpha)
        self._sm_asym = smooth_exponential(self._sm_asym, m.ear_asymmetry, self._alpha)
        self._sm_li = smooth_vec2(self._sm_li, m.left_iris_offset, self._alpha)
        self._sm_ri = smooth_vec2(self._sm_ri, m.right_iris_offset, self._alpha)

        style = DrawStyle.FULL if self._full_mesh else DrawStyle.EYES_ONLY
        draw_face_mesh_eyes(display, result.landmarks, style=style)

        hud.append(
            f"EAR L {self._sm_l_ear:.3f}  R {self._sm_r_ear:.3f}  |d|: {self._sm_asym:.3f}"
        )
        if self._sm_li and self._sm_ri:
            hud.append(
                f"Iris offset L nx,ny {self._sm_li[0]:+.2f},{self._sm_li[1]:+.2f}  "
                f"R {self._sm_ri[0]:+.2f},{self._sm_ri[1]:+.2f}"
            )

        if self._gaze_cal is not None:
            if m.left_ear >= self._gaze_ear_min and m.right_ear >= self._gaze_ear_min:
                g = gaze_feature_vector(m, result.face_matrix)
            else:
                g = None
            if g is not None:
                rx, ry, sig = self._gaze_cal.predict_with_uncertainty(g)
                if np.isfinite(rx) and np.isfinite(ry) and np.isfinite(sig):
                    rx, ry = self._gaze_cal.clamp(rx, ry)
                    a = max(0.0, min(1.0, self._gaze_alpha))
                    if self._sm_gx is None:
                        self._sm_gx, self._sm_gy = rx, ry
                        self._sm_gaze_sigma = sig
                    else:
                        self._sm_gx = a * rx + (1.0 - a) * self._sm_gx
                        self._sm_gy = a * ry + (1.0 - a) * self._sm_gy
                        self._sm_gaze_sigma = a * sig + (1.0 - a) * (
                            self._sm_gaze_sigma or sig
                        )

                    if self._kbd is not None:
                        self._kbd_sm_buf.append(
                            (float(self._sm_gx), float(self._sm_gy))
                        )
                        if self._kbd_median_n >= 3:
                            arr = np.asarray(self._kbd_sm_buf, dtype=np.float64)
                            kbd_gx = float(np.median(arr[:, 0]))
                            kbd_gy = float(np.median(arr[:, 1]))
                        else:
                            kbd_gx = float(self._sm_gx)
                            kbd_gy = float(self._sm_gy)
                        b = self._kbd.keyboard.layout_bounds()
                        if (
                            b is not None
                            and self._gaze_cal is not None
                            and abs(self._kbd_gain - 1.0) > 1e-6
                        ):
                            minx, maxx, miny, maxy = b
                            cx = 0.5 * (float(minx) + float(maxx))
                            cy = 0.5 * (float(miny) + float(maxy))
                            kbd_gx = cx + self._kbd_gain * (kbd_gx - cx)
                            kbd_gy = cy + self._kbd_gain * (kbd_gy - cy)
                            gw = float(self._gaze_cal.gaze_width)
                            gh = float(self._gaze_cal.gaze_height)
                            kbd_gx = float(np.clip(kbd_gx, 0.0, gw - 1.0))
                            kbd_gy = float(np.clip(kbd_gy, 0.0, gh - 1.0))
                        self._kbd.keyboard.update_gaze(kbd_gx, kbd_gy)
                        self._pointer_gx, self._pointer_gy = kbd_gx, kbd_gy
                    else:
                        self._pointer_gx, self._pointer_gy = self._sm_gx, self._sm_gy

            if self._sm_gx is not None and self._sm_gy is not None:
                sg = self._sm_gaze_sigma
                if self._kbd is not None and self._pointer_gx is not None and self._pointer_gy is not None:
                    if sg is not None:
                        hud.append(
                            f"Gaze xy: ({self._pointer_gx:.0f}, {self._pointer_gy:.0f})  "
                            f"sigma~{sg:.0f}px"
                        )
                    else:
                        hud.append(f"Gaze xy: ({self._pointer_gx:.0f}, {self._pointer_gy:.0f})")
                elif sg is not None:
                    hud.append(f"Gaze xy: ({self._sm_gx:.0f}, {self._sm_gy:.0f})  sigma~{sg:.0f}px")
                else:
                    hud.append(f"Gaze xy: ({self._sm_gx:.0f}, {self._sm_gy:.0f})")

                if self._kbd is None:
                    self._draw_gaze_overlay(display, self._sm_gx, self._sm_gy, sg)

        if self._kbd is not None:
            kb = self._kbd.keyboard
            avg_ear = (m.left_ear + m.right_ear) / 2.0

            if not kb.block_input:
                if self._kbd.blink.feed(avg_ear):
                    letter = kb.select()
                    if letter is not None:
                        hud.append(f"SELECTED: {letter}")

            if kb.block_input:
                kb.reset_infer_confirm_accum()
            elif kb.input_enabled and not kb.suggestions:
                kb.feed_infer_confirm_closure(
                    dt, avg_ear, self._blink_close_thresh, kb.infer_confirm_hold_s
                )

            if not kb.block_input:
                if self.llm_backend is not None and kb.last_action == "PREDICT":
                    if kb.history:
                        kb.last_action = None
                        kb.block_overlay_text = "Running inference..."
                        kb.block_input = True
                        text = kb.history[-1]
                        prior = list(kb.history[:-1])
                        threading.Thread(
                            target=self._fire_llm,
                            args=(text, prior),
                            daemon=True,
                        ).start()

        return hud

    def draw_keyboard(self, kb_canvas: np.ndarray) -> None:
        if self._kbd is None:
            return
        self._kbd.keyboard.draw(
            kb_canvas,
            left_iris=self._sm_li,
            right_iris=self._sm_ri,
        )

    def draw_gaze_pointer_on_keyboard(self, frame: np.ndarray) -> None:
        """Draw gaze dot on top of the keyboard (same coords as ``update_gaze`` / hit test)."""
        if self._gaze_cal is None or self._kbd is None:
            return
        gx = self._pointer_gx if self._pointer_gx is not None else self._sm_gx
        gy = self._pointer_gy if self._pointer_gy is not None else self._sm_gy
        if gx is None or gy is None:
            return
        sg = self._sm_gaze_sigma
        self._draw_gaze_overlay(frame, float(gx), float(gy), sg)

    def keyboard_go_back(self) -> bool:
        """If keyboard is in letter stage, go back to row stage. Returns True if it went back."""
        if self._kbd is not None and self._kbd.keyboard.stage == 1:
            self._kbd.keyboard.go_back()
            return True
        return False

    def backspace_typed(self) -> None:
        if self._kbd is not None and self._kbd.keyboard.typed:
            self._kbd.keyboard.typed.pop()
