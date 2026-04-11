from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Expand each key's hit box (not the drawn box) so small gaze errors still land.
_HIT_PAD_FRAC = 0.14

QWERTY_ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM "),
]
ROWS = len(QWERTY_ROWS)


@dataclass(frozen=True)
class KeyboardLayoutProfile:
    """Geometry and timing tuned to how different gaze regressors behave live."""

    margin_x_frac: float
    margin_top_frac: float
    margin_bot_frac: float
    dwell_seconds: float
    #: Second argument to ``cv2.addWeighted`` for the key overlay (higher = more opaque keys).
    overlay_key_alpha: float
    #: BGR for the active cell outline.
    active_outline_bgr: tuple[int, int, int]
    #: BGR for the dwell-fill on the active key.
    dwell_fill_bgr: tuple[int, int, int]


def keyboard_profile_for_gaze_model(model_type: str | None) -> KeyboardLayoutProfile:
    """Linear maps are often stable in a tight band; tree/boost models benefit from larger targets."""
    mt = (model_type or "").strip().lower()
    if mt in ("ridge", "poly"):
        return KeyboardLayoutProfile(
            margin_x_frac=0.14,
            margin_top_frac=0.22,
            margin_bot_frac=0.15,
            dwell_seconds=0.62,
            overlay_key_alpha=0.62,
            active_outline_bgr=(0, 220, 255),
            dwell_fill_bgr=(76, 175, 80),
        )
    if mt in ("rf", "xgboost", "gbr"):
        return KeyboardLayoutProfile(
            margin_x_frac=0.07,
            margin_top_frac=0.18,
            margin_bot_frac=0.12,
            dwell_seconds=0.88,
            overlay_key_alpha=0.55,
            active_outline_bgr=(80, 160, 255),
            dwell_fill_bgr=(70, 170, 100),
        )
    # svr, unknown, legacy
    return KeyboardLayoutProfile(
        margin_x_frac=0.10,
        margin_top_frac=0.20,
        margin_bot_frac=0.15,
        dwell_seconds=0.75,
        overlay_key_alpha=0.58,
        active_outline_bgr=(50, 200, 255),
        dwell_fill_bgr=(76, 175, 80),
    )


@dataclass
class KeyboardCell:
    row: int
    col: int
    letter: str
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class GazeKeyboard:
    """QWERTY gaze keyboard laid out from the gaze model used at calibration (see ``layout``)."""

    cells: list[KeyboardCell] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    _active_cell: int = -1
    _canvas_w: int = 0
    _canvas_h: int = 0
    _base_image: np.ndarray | None = field(default=None, repr=False, init=False)

    _cell_enter_time: float = 0.0
    _dwell_triggered: bool = False
    _blink_timestamps: list[float] = field(default_factory=list)

    _profile: KeyboardLayoutProfile = field(
        default_factory=lambda: keyboard_profile_for_gaze_model(None),
        repr=False,
    )
    _gaze_model_label: str = field(default="", repr=False)

    @property
    def stage(self) -> int:
        return 0

    def go_back(self) -> None:
        pass

    def layout(
        self,
        canvas_w: int,
        canvas_h: int,
        *,
        gaze_model: str | None = None,
    ) -> None:
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        self._base_image = None
        self._profile = keyboard_profile_for_gaze_model(gaze_model)
        raw = (gaze_model or "").strip()
        self._gaze_model_label = raw.upper()[:14] if raw else "DEFAULT"

        p = self._profile
        margin_x = int(round(canvas_w * p.margin_x_frac))
        top_y = int(round(canvas_h * p.margin_top_frac))
        bot_y = int(round(canvas_h * (1.0 - p.margin_bot_frac)))

        usable_w = canvas_w - 2 * margin_x
        usable_h = bot_y - top_y
        row_h = usable_h / ROWS

        max_keys = max(len(r) for r in QWERTY_ROWS)
        key_w = usable_w / max_keys

        self.cells = []
        for r, row_keys in enumerate(QWERTY_ROWS):
            y0 = int(round(top_y + r * row_h))
            y1 = int(round(top_y + (r + 1) * row_h))
            row_px_width = len(row_keys) * key_w
            start_x = margin_x + (usable_w - row_px_width) / 2
            for c, letter in enumerate(row_keys):
                x0 = int(round(start_x + c * key_w))
                x1 = int(round(start_x + (c + 1) * key_w))
                self.cells.append(KeyboardCell(
                    row=r, col=c, letter=letter,
                    x0=x0, y0=y0, x1=x1, y1=y1,
                ))

    @staticmethod
    def _cell_center(c: KeyboardCell) -> tuple[float, float]:
        return ((c.x0 + c.x1) * 0.5, (c.y0 + c.y1) * 0.5)

    @staticmethod
    def _padded_hit_rect(c: KeyboardCell) -> tuple[float, float, float, float]:
        w = max(1.0, float(c.x1 - c.x0))
        h = max(1.0, float(c.y1 - c.y0))
        p = _HIT_PAD_FRAC * min(w, h)
        return (float(c.x0 - p), float(c.x1 + p), float(c.y0 - p), float(c.y1 + p))

    def _nearest_center_index(self, gx: float, gy: float) -> int:
        best_i = -1
        min_d = float("inf")
        for i, c in enumerate(self.cells):
            cx, cy = self._cell_center(c)
            d = (gx - cx) ** 2 + (gy - cy) ** 2
            if d < min_d:
                min_d = d
                best_i = i
        return best_i

    def hit_test(self, gx: float, gy: float) -> int:
        """Padded key rectangles first (easier to acquire), then strict, then nearest center."""
        if not self.cells:
            return -1
        loose: list[int] = []
        for i, c in enumerate(self.cells):
            x0, x1, y0, y1 = self._padded_hit_rect(c)
            if x0 <= gx <= x1 and y0 <= gy <= y1:
                loose.append(i)
        if len(loose) == 1:
            return loose[0]
        if len(loose) > 1:
            best_i = loose[0]
            cx, cy = self._cell_center(self.cells[best_i])
            best_d = (gx - cx) ** 2 + (gy - cy) ** 2
            for j in loose[1:]:
                cx, cy = self._cell_center(self.cells[j])
                d = (gx - cx) ** 2 + (gy - cy) ** 2
                if d < best_d:
                    best_d = d
                    best_i = j
            return best_i

        strict: list[int] = []
        for i, c in enumerate(self.cells):
            if c.x0 <= gx <= c.x1 and c.y0 <= gy <= c.y1:
                strict.append(i)
        if len(strict) == 1:
            return strict[0]
        if len(strict) > 1:
            best_i = strict[0]
            cx, cy = self._cell_center(self.cells[best_i])
            best_d = (gx - cx) ** 2 + (gy - cy) ** 2
            for j in strict[1:]:
                cx, cy = self._cell_center(self.cells[j])
                d = (gx - cx) ** 2 + (gy - cy) ** 2
                if d < best_d:
                    best_d = d
                    best_i = j
            return best_i

        return self._nearest_center_index(gx, gy)

    def update_gaze(self, gx: float, gy: float) -> None:
        new_cell = self.hit_test(gx, gy)
        if new_cell != self._active_cell:
            self._active_cell = new_cell
            self._cell_enter_time = time.time()
            self._dwell_triggered = False
        elif self._active_cell >= 0 and not self._dwell_triggered:
            if time.time() - self._cell_enter_time >= self._profile.dwell_seconds:
                self._dwell_triggered = True
                letter = self.cells[self._active_cell].letter
                if letter == " ":
                    self.typed.append(" ")
                else:
                    self.typed.append(letter)

    def select(self) -> str | None:
        """Called on blink. Triple-blink within 2.5s triggers PREDICT."""
        now = time.time()
        self._blink_timestamps.append(now)
        self._blink_timestamps = [t for t in self._blink_timestamps if now - t <= 2.5]

        if len(self._blink_timestamps) >= 3:
            self._blink_timestamps.clear()
            sentence = "".join(self.typed).strip()
            print(f"\n>>> PREDICT TRIGGERED! <<<")
            print(f"=== CONFIRMED SENTENCE: {sentence} ===\n")
            if sentence:
                self.history.append(sentence)
                if len(self.history) > 4:
                    self.history.pop(0)
            self.typed.clear()
            return "PREDICT"
        return None

    @property
    def typed_text(self) -> str:
        return "".join(self.typed)

    @property
    def active_cell_index(self) -> int:
        return self._active_cell

    def draw(
        self,
        frame: np.ndarray,
        left_iris: tuple[float, float] | None = None,
        right_iris: tuple[float, float] | None = None,
    ) -> None:
        if not self.cells or self._canvas_w < 1 or self._canvas_h < 1:
            return
        h, w = frame.shape[:2]
        sx = w / self._canvas_w
        sy = h / self._canvas_h

        if self._base_image is None or self._base_image.shape[:2] != (h, w):
            self._base_image = np.zeros((h, w, 3), dtype=np.uint8)
            for cell in self.cells:
                dx0 = int(round(cell.x0 * sx))
                dy0 = int(round(cell.y0 * sy))
                dx1 = int(round(cell.x1 * sx))
                dy1 = int(round(cell.y1 * sy))
                cv2.rectangle(self._base_image, (dx0, dy0), (dx1, dy1), (30, 30, 30), -1)
                cv2.rectangle(self._base_image, (dx0, dy0), (dx1, dy1), (100, 100, 100), 2)

                label = "SPC" if cell.letter == " " else cell.letter
                font = cv2.FONT_HERSHEY_DUPLEX
                cell_px_w = dx1 - dx0
                scale = max(0.8, min(2.5, cell_px_w / 50.0))
                thickness = max(2, min(6, int(round(cell_px_w / 25.0))))
                (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
                tx = (dx0 + dx1) // 2 - tw // 2
                ty = (dy0 + dy1) // 2 + th // 2
                cv2.putText(
                    self._base_image, label, (tx, ty),
                    font, scale, (235, 235, 235), thickness, cv2.LINE_AA,
                )

        overlay = self._base_image.copy()

        if 0 <= self._active_cell < len(self.cells):
            cell = self.cells[self._active_cell]
            dx0 = int(round(cell.x0 * sx))
            dy0 = int(round(cell.y0 * sy))
            dx1 = int(round(cell.x1 * sx))
            dy1 = int(round(cell.y1 * sy))

            ol_b, ol_g, ol_r = self._profile.active_outline_bgr
            cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), (80, 60, 20), -1)
            cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), (ol_b, ol_g, ol_r), 3)

            label = "SPC" if cell.letter == " " else cell.letter
            font = cv2.FONT_HERSHEY_DUPLEX
            cell_px_w = dx1 - dx0
            scale = max(0.8, min(2.5, cell_px_w / 50.0))
            thickness = min((max(2, min(6, int(round(cell_px_w / 25.0))))) + 1, 7)
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            tx = (dx0 + dx1) // 2 - tw // 2
            ty = (dy0 + dy1) // 2 + th // 2
            cv2.putText(
                overlay, label, (tx, ty),
                font, scale, (0, 255, 255), thickness, cv2.LINE_AA,
            )

            if not self._dwell_triggered:
                elapsed = time.time() - self._cell_enter_time
                dw = self._profile.dwell_seconds
                progress = max(0.0, min(1.0, elapsed / dw))
                if progress > 0:
                    bar_h = int(round((dy1 - dy0) * progress))
                    df_b, df_g, df_r = self._profile.dwell_fill_bgr
                    cv2.rectangle(overlay, (dx0, dy1 - bar_h), (dx1, dy1), (df_b, df_g, df_r), -1)

        ka = self._profile.overlay_key_alpha
        cv2.addWeighted(overlay, ka, frame, 1.0 - ka, 0, dst=frame)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Top bar: typed text + history
        typed_str = self.typed_text
        lines_to_draw: list[tuple[str, tuple[int, int, int], float, int]] = []
        for hist_str in self.history:
            lines_to_draw.append((f"> {hist_str}", (150, 150, 150), 1.4, 3))
        if typed_str:
            lines_to_draw.append((typed_str + "_", (0, 255, 200), 1.8, 4))

        if lines_to_draw:
            line_height = 55
            total_height = len(lines_to_draw) * line_height + 15
            if total_height <= h:
                msg_bg = frame[0:total_height, 0:w].copy()
                cv2.rectangle(msg_bg, (0, 0), (w, total_height), (10, 10, 10), -1)
                cv2.addWeighted(msg_bg, 0.85, frame[0:total_height, 0:w], 0.15, 0,
                                dst=frame[0:total_height, 0:w])
            current_y = 45
            for text, color, scale, thickness in lines_to_draw:
                cv2.putText(frame, text, (15, current_y), font, scale, color, thickness, cv2.LINE_AA)
                current_y += line_height

        # Bottom bar: hints + iris coords
        bot_bar_h = 45
        cv2.rectangle(frame, (0, h - bot_bar_h), (w, h), (20, 20, 20), -1)

        dwell_ms = int(round(self._profile.dwell_seconds * 1000))
        hint = (
            f"Dwell {dwell_ms}ms = type  |  Blink 3x = PREDICT  |  D = backspace"
            f"  |  gaze: {self._gaze_model_label}"
        )
        cv2.putText(frame, hint, (15, h - 14), font, 0.55, (180, 180, 180), 2, cv2.LINE_AA)

        coords_parts: list[str] = []
        if left_iris is not None:
            coords_parts.append(f"L:{left_iris[0]:+.2f},{left_iris[1]:+.2f}")
        if right_iris is not None:
            coords_parts.append(f"R:{right_iris[0]:+.2f},{right_iris[1]:+.2f}")
        if coords_parts:
            coords_text = "  ".join(coords_parts)
            (tw, _), _ = cv2.getTextSize(coords_text, font, 0.5, 1)
            cv2.putText(frame, coords_text, (w - tw - 15, h - 14), font, 0.5,
                        (180, 255, 180), 1, cv2.LINE_AA)
