from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

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


@dataclass
class KeyboardSession:
    """Fullscreen keyboard window + blink detector (constructed once per run)."""

    window_name: str
    pixel_w: int
    pixel_h: int
    keyboard: GazeKeyboard
    blink: BlinkDetector

    @staticmethod
    def create_session(gaze_cal: GazeCalibration) -> KeyboardSession:
        win = "Main Output"  # We'll just draw onto the main app window

        # Must match gaze (x,y) space from calibration — NOT physical screen size.
        # Using monitor resolution here misaligned keys vs predicted gaze.
        kb_w, kb_h = gaze_cal.gaze_width, gaze_cal.gaze_height

        keyboard = GazeKeyboard()
        keyboard.layout(kb_w, kb_h, gaze_model=gaze_cal.model_type)
        blink = BlinkDetector(
            close_threshold=0.12,
            open_threshold=0.16,
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
    """Per-frame processing: landmarks → metrics → optional gaze → optional keyboard."""

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
        keyboard_gaze_median_n: int = 3,
    ) -> None:
        self._gaze_file_label = gaze_file_label
        self._gaze_cal = gaze_cal
        self._gaze_alpha = gaze_alpha
        self._gaze_ear_min = gaze_ear_min
        self._full_mesh = full_mesh
        self._kbd = keyboard
        self._alpha = metric_smooth_alpha
        self._kbd_median_n = max(0, int(keyboard_gaze_median_n))
        if keyboard is not None and self._kbd_median_n >= 3:
            self._kbd_raw_buf: deque[tuple[float, float]] = deque(maxlen=self._kbd_median_n)
        else:
            self._kbd_raw_buf = deque(maxlen=1)

        self._sm_l_ear = 0.25
        self._sm_r_ear = 0.25
        self._sm_asym = 0.0
        self._sm_li: tuple[float, float] | None = None
        self._sm_ri: tuple[float, float] | None = None
        self._sm_gx: float | None = None
        self._sm_gy: float | None = None
        self._sm_gaze_sigma: float | None = None

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

    def step(self, display, result: FrameResult) -> list[str]:
        hud: list[str] = [
            "Research demo — not medical or security-grade gaze tracking.",
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
                rx, ry, sig = self._gaze_cal.predict_with_uncertainty(g)
                rx, ry = self._gaze_cal.clamp(rx, ry)
                if self._kbd is not None:
                    self._kbd_raw_buf.append((rx, ry))
                    if self._kbd_median_n >= 3:
                        arr = np.asarray(self._kbd_raw_buf, dtype=np.float64)
                        kgx = float(np.median(arr[:, 0]))
                        kgy = float(np.median(arr[:, 1]))
                    else:
                        kgx, kgy = rx, ry
                    self._kbd.keyboard.update_gaze(kgx, kgy)
                a = max(0.0, min(1.0, self._gaze_alpha))
                if self._sm_gx is None:
                    self._sm_gx, self._sm_gy = rx, ry
                    self._sm_gaze_sigma = sig
                else:
                    self._sm_gx = a * rx + (1.0 - a) * self._sm_gx
                    self._sm_gy = a * ry + (1.0 - a) * self._sm_gy
                    self._sm_gaze_sigma = a * sig + (1.0 - a) * (self._sm_gaze_sigma or sig)

            if self._sm_gx is not None and self._sm_gy is not None:
                sg = self._sm_gaze_sigma
                if sg is not None:
                    hud.append(f"Gaze xy: ({self._sm_gx:.0f}, {self._sm_gy:.0f})  sigma~{sg:.0f}px")
                else:
                    hud.append(f"Gaze xy: ({self._sm_gx:.0f}, {self._sm_gy:.0f})")

                if self._kbd is None:
                    self._draw_gaze_overlay(display, self._sm_gx, self._sm_gy, sg)

        if self._kbd is not None:
            avg_ear = (m.left_ear + m.right_ear) / 2.0
            if self._kbd.blink.feed(avg_ear):
                letter = self._kbd.keyboard.select()
                if letter is not None:
                    hud.append(f"SELECTED: {letter}")

        return hud

    def draw_keyboard(self, kb_canvas: np.ndarray) -> None:
        if self._kbd is None:
            return
        gaze_pt = (self._sm_gx, self._sm_gy) if self._sm_gx is not None else None
        self._kbd.keyboard.draw(
            kb_canvas,
            left_iris=self._sm_li,
            right_iris=self._sm_ri,
        )

    def keyboard_go_back(self) -> bool:
        """If keyboard is in letter stage, go back to row stage. Returns True if it went back."""
        if self._kbd is not None and self._kbd.keyboard.stage == 1:
            self._kbd.keyboard.go_back()
            return True
        return False

    def backspace_typed(self) -> None:
        if self._kbd is not None and self._kbd.keyboard.typed:
            self._kbd.keyboard.typed.pop()
