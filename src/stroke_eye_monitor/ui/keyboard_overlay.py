from __future__ import annotations

import string
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Standard QWERTY layout
QWERTY_ROWS = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM")]
ROWS = len(QWERTY_ROWS)


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
    """Familiar QWERTY layout with automatic dwell-typing and blink-to-predict."""

    cells: list[KeyboardCell] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)
    _active_cell: int = -1
    _canvas_w: int = 0
    _canvas_h: int = 0
    _base_image: np.ndarray | None = field(default=None, repr=False, init=False)

    # Dwell parameters
    _cell_enter_time: float = 0.0
    _dwell_triggered: bool = False

    # Blink prediction tracking
    _blink_timestamps: list[float] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    def layout(self, canvas_w: int, canvas_h: int) -> None:
        """Compute cell pixel bounds for the full canvas."""
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        self._base_image = None
        row_h = canvas_h / ROWS
        self.cells = []

        # Base key width on the longest row (QWERTYUIOP = 10 keys)
        max_keys = max(len(r) for r in QWERTY_ROWS[:-1])
        key_w = canvas_w / max_keys

        for r, row_keys in enumerate(QWERTY_ROWS):
            y0 = int(round(r * row_h))
            y1 = int(round((r + 1) * row_h))

            # Center the row based on specific key count
            row_px_width = len(row_keys) * key_w
            start_x = (canvas_w - row_px_width) / 2
            for c, letter in enumerate(row_keys):
                x0 = int(round(start_x + c * key_w))
                x1 = int(round(start_x + (c + 1) * key_w))
                self.cells.append(
                    KeyboardCell(
                        row=r, col=c, letter=letter, x0=x0, y0=y0, x1=x1, y1=y1
                    )
                )

    def hit_test(self, gx: float, gy: float) -> int:
        """Return cell index nearest to the gaze coordinate (center-snapping)."""
        if not self.cells:
            return -1

        best_index = -1
        min_dist = float("inf")

        for i, c in enumerate(self.cells):
            # Calculate the physical center of the cell
            cx = (c.x0 + c.x1) / 2.0
            cy = (c.y0 + c.y1) / 2.0

            # Calculate raw Euclidean distance to gaze coordinate
            dist_sq = (gx - cx) ** 2 + (gy - cy) ** 2

            if dist_sq < min_dist:
                min_dist = dist_sq
                best_index = i

        return best_index

    def update_gaze(self, gx: float, gy: float) -> None:
        new_cell = self.hit_test(gx, gy)

        # Reset the dwell timer if the gaze moves to a completely new cell
        if new_cell != self._active_cell:
            self._active_cell = new_cell
            self._cell_enter_time = time.time()
            self._dwell_triggered = False
        else:
            # If staying on the same cell, check if we've dwelled long enough
            if self._active_cell >= 0 and not self._dwell_triggered:
                cell = self.cells[self._active_cell]
                target_dwell = 0.8  # 800ms to type

                if time.time() - self._cell_enter_time >= target_dwell:
                    self._dwell_triggered = True
                    self.typed.append(cell.letter)

    def select(self) -> str | None:
        """Called by the pipeline when a blink is detected. Tracks blinks to trigger prediction."""
        now = time.time()
        self._blink_timestamps.append(now)

        # Keep only blinks from the last 2.5 seconds
        self._blink_timestamps = [t for t in self._blink_timestamps if now - t <= 2.5]

        # If 3 blinks happened quickly, trigger predict!
        if len(self._blink_timestamps) >= 3:
            self._blink_timestamps.clear()  # reset

            sentence = "".join(self.typed).strip()
            print("\n>>> PREDICT TRIGGERED! <<<")
            print(f"=== CONFIRMED SENTENCE: {sentence} ===\n")
            
            if sentence:
                self.history.append(sentence)
                # Keep only the last 3 sentences so it doesn't clutter the screen
                if len(self.history) > 3:
                    self.history.pop(0)

            # Clear text for the next sentence
            self.typed.clear()

            return "PREDICT"


        # A single blink does nothing on its own (typing is handled purely by the 800ms dwell)
        return None

    @property
    def typed_text(self) -> str:
        return "".join(self.typed)

    def draw(
        self,
        frame: np.ndarray,
        left_iris: tuple[float, float] | None = None,
        right_iris: tuple[float, float] | None = None,
    ) -> None:
        """Overlay the keyboard grid and highlights onto the frame."""
        h, w = frame.shape[:2]
        sx = w / self._canvas_w
        sy = h / self._canvas_h

        # 1. Create a cached base image of the keyboard grid (only done once)
        if self._base_image is None or self._base_image.shape[:2] != (h, w):
            self._base_image = np.zeros((h, w, 3), dtype=np.uint8)
            for cell in self.cells:
                dx0 = int(round(cell.x0 * sx))
                dy0 = int(round(cell.y0 * sy))
                dx1 = int(round(cell.x1 * sx))
                dy1 = int(round(cell.y1 * sy))

                cv2.rectangle(
                    self._base_image, (dx0, dy0), (dx1, dy1), (30, 30, 30), -1
                )
                cv2.rectangle(
                    self._base_image, (dx0, dy0), (dx1, dy1), (100, 100, 100), 2
                )

                font = cv2.FONT_HERSHEY_DUPLEX
                cell_px_w = dx1 - dx0
                scale = max(0.55, min(1.5, cell_px_w / 72.0))
                thickness = max(3, min(6, int(round(cell_px_w / 28.0))))
                (tw, th), _ = cv2.getTextSize(cell.letter, font, scale, thickness)
                tx = (dx0 + dx1) // 2 - tw // 2
                ty = (dy0 + dy1) // 2 + th // 2
                cv2.putText(
                    self._base_image,
                    cell.letter,
                    (tx, ty),
                    font,
                    scale,
                    (235, 235, 235),
                    thickness,
                    cv2.LINE_AA,
                )

        overlay = self._base_image.copy()

        # 2. Draw only the active cell over the overlay
        if 0 <= self._active_cell < len(self.cells):
            cell = self.cells[self._active_cell]
            dx0 = int(round(cell.x0 * sx))
            dy0 = int(round(cell.y0 * sy))
            dx1 = int(round(cell.x1 * sx))
            dy1 = int(round(cell.y1 * sy))

            cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), (80, 60, 20), -1)
            cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), (0, 220, 255), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            cell_px_w = dx1 - dx0
            scale = max(0.55, min(1.5, cell_px_w / 72.0))
            thickness = min((max(3, min(6, int(round(cell_px_w / 28.0))))) + 1, 7)
            (tw, th), _ = cv2.getTextSize(cell.letter, font, scale, thickness)
            tx = (dx0 + dx1) // 2 - tw // 2
            ty = (dy0 + dy1) // 2 + th // 2
            cv2.putText(
                overlay,
                cell.letter,
                (tx, ty),
                font,
                scale,
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            # Draw the Progress Bar (Dwell fill effect)
            if not self._dwell_triggered:
                target_dwell = 0.8
                elapsed = time.time() - self._cell_enter_time
                progress = max(0.0, min(1.0, elapsed / target_dwell))

                if progress > 0:
                    bar_h = int(round((dy1 - dy0) * progress))  # Grow vertically
                    cv2.rectangle(
                        overlay, (dx0, dy1 - bar_h), (dx1, dy1), (76, 175, 80), -1
                    )

        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)

        bar_h = 40
        font = cv2.FONT_HERSHEY_SIMPLEX

        hint = "Gaze at a letter for 800ms   |   Blink 3 times fast to PREDICT"
        cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.putText(frame, hint, (10, 26), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw confirmed sentences history and current typing together in a dark panel
        typed_str = self.typed_text
        lines_to_draw = []
        for i, hist_str in enumerate(self.history):
            lines_to_draw.append((f"> {hist_str}", (150, 150, 150), 0.7, 2))  # Gray for history
            
        if typed_str:
            lines_to_draw.append((typed_str + "_", (0, 255, 200), 0.9, 3))    # Bright Cyan for active typing
            
        if lines_to_draw:
            line_height = 36
            total_height = len(lines_to_draw) * line_height + 10
            y2 = bar_h + total_height
            
            # Semi-transparent dark background for readability
            msg_bg = frame[bar_h:y2, 0:w].copy()
            cv2.rectangle(msg_bg, (0, 0), (w, total_height), (10, 10, 10), -1)
            cv2.addWeighted(msg_bg, 0.85, frame[bar_h:y2, 0:w], 0.15, 0, dst=frame[bar_h:y2, 0:w])
            
            current_y = bar_h + 30
            for text, color, scale, thickness in lines_to_draw:
                cv2.putText(
                    frame,
                    text,
                    (15, current_y),
                    font,
                    scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                current_y += line_height

        coords_parts: list[str] = []
        if left_iris is not None:
            coords_parts.append(f"L:{left_iris[0]:+.2f},{left_iris[1]:+.2f}")
        if right_iris is not None:
            coords_parts.append(f"R:{right_iris[0]:+.2f},{right_iris[1]:+.2f}")

        if coords_parts:
            coords_text = "  ".join(coords_parts)
            cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 20), -1)
            cv2.putText(
                frame,
                coords_text,
                (10, h - 12),
                font,
                0.55,
                (180, 255, 180),
                1,
                cv2.LINE_AA,
            )
