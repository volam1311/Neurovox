from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from stroke_eye_monitor.audio_voice import SpokenContextBuffer

from stroke_eye_monitor.config import detect_screen_resolution
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
    ) -> KeyboardSession:
        win = "Main Output"  # We'll just draw onto the main app window

        kb_w, kb_h = detect_screen_resolution() or (
            gaze_cal.gaze_width,
            gaze_cal.gaze_height,
        )

        keyboard = GazeKeyboard(
            dwell_seconds=gaze_cal.dwell_ms / 1000.0,
            on_sentence_chosen=on_sentence_chosen,
            spoken_buffer=spoken_buffer,
        )
        keyboard.layout(kb_w, kb_h)
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

        self._sm_l_ear = 0.25
        self._sm_r_ear = 0.25
        self._sm_asym = 0.0
        self._sm_li: tuple[float, float] | None = None
        self._sm_ri: tuple[float, float] | None = None
        self._sm_gx: float | None = None
        self._sm_gy: float | None = None

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
            return hud

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
                rx, ry = self._gaze_cal.predict(g)
                rx, ry = self._gaze_cal.clamp(rx, ry)
                a = max(0.0, min(1.0, self._gaze_alpha))
                if self._sm_gx is None:
                    self._sm_gx, self._sm_gy = rx, ry
                else:
                    self._sm_gx = a * rx + (1.0 - a) * self._sm_gx
                    self._sm_gy = a * ry + (1.0 - a) * self._sm_gy

            if self._sm_gx is not None and self._sm_gy is not None:
                hud.append(f"Gaze xy: ({self._sm_gx:.0f}, {self._sm_gy:.0f})")

        if self._kbd is not None:
            kb = self._kbd.keyboard

            # Block all gaze + blink input while LLM is running
            if not kb.block_input:
                if self._sm_gx is not None and self._sm_gy is not None:
                    kb.update_gaze(self._sm_gx, self._sm_gy)

                avg_ear = (m.left_ear + m.right_ear) / 2.0
                if self._kbd.blink.feed(avg_ear):
                    letter = kb.select()
                    if letter is not None:
                        hud.append(f"SELECTED: {letter}")

                # PREDICT must be consumed here; update_gaze must not clear last_action before this.
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

    def backspace_typed(self) -> None:
        if self._kbd is not None and self._kbd.keyboard.typed:
            self._kbd.keyboard.typed.pop()
