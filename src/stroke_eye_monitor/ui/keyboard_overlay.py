from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# 26 single-letter cells across 4 rows: 7 + 7 + 7 + 5.
ROW_COLS = (7, 7, 7, 5)
ROWS = len(ROW_COLS)

LETTERS: list[str] = list(string.ascii_uppercase)


@dataclass
class KeyboardCell:
    row: int
    col: int
    letter: str
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0


@dataclass
class GazeKeyboard:
    """26-cell alphabetical gaze keyboard: gaze highlights a cell, blink commits the letter."""

    cells: list[KeyboardCell] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)

    _active_cell: int = -1
    _canvas_w: int = 0
    _canvas_h: int = 0
    _base_image: np.ndarray | None = field(default=None, repr=False, init=False)

    def layout(self, canvas_w: int, canvas_h: int) -> None:
        """Compute cell pixel bounds for the full canvas."""
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        self._base_image = None
        row_h = canvas_h / ROWS
        self.cells = []
        idx = 0
        for r, ncols in enumerate(ROW_COLS):
            cell_w = canvas_w / ncols
            y0 = int(round(r * row_h))
            y1 = int(round((r + 1) * row_h))
            for c in range(ncols):
                x0 = int(round(c * cell_w))
                x1 = int(round((c + 1) * cell_w))
                self.cells.append(KeyboardCell(
                    row=r, col=c, letter=LETTERS[idx],
                    x0=x0, y0=y0, x1=x1, y1=y1,
                ))
                idx += 1

    def hit_test(self, gx: float, gy: float) -> int:
        """Return cell index the gaze point falls in, or -1."""
        if not self.cells:
            return -1
        ix = int(gx)
        iy = int(gy)
        for i, c in enumerate(self.cells):
            if c.x0 <= ix < c.x1 and c.y0 <= iy < c.y1:
                return i
        return -1

    def update_gaze(self, gx: float, gy: float) -> None:
        self._active_cell = self.hit_test(gx, gy)

    def select(self) -> str | None:
        """Called on blink. Returns the selected letter or None."""
        if self._active_cell < 0 or self._active_cell >= len(self.cells):
            return None
        letter = self.cells[self._active_cell].letter
        self.typed.append(letter)
        return letter

    @property
    def typed_text(self) -> str:
        return "".join(self.typed)

    @property
    def active_cell_index(self) -> int:
        return self._active_cell

    def draw(
        self,
        frame: Any,
        *,
        left_iris: tuple[float, float] | None = None,
        right_iris: tuple[float, float] | None = None,
        gaze_xy: tuple[float, float] | None = None,
    ) -> None:
        """Draw the keyboard grid on the frame (semi-transparent).

        Cell coordinates are in calibration-canvas space; they are rescaled
        to fit the actual frame dimensions.
        """
        if not self.cells or self._canvas_w < 1 or self._canvas_h < 1:
            return
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

                cv2.rectangle(self._base_image, (dx0, dy0), (dx1, dy1), (30, 30, 30), -1)
                cv2.rectangle(self._base_image, (dx0, dy0), (dx1, dy1), (100, 100, 100), 2)

                font = cv2.FONT_HERSHEY_DUPLEX
                cell_px_w = dx1 - dx0
                scale = max(0.55, min(1.5, cell_px_w / 72.0))
                thickness = max(3, min(6, int(round(cell_px_w / 28.0))))
                (tw, th), _ = cv2.getTextSize(cell.letter, font, scale, thickness)
                tx = (dx0 + dx1) // 2 - tw // 2
                ty = (dy0 + dy1) // 2 + th // 2
                cv2.putText(self._base_image, cell.letter, (tx, ty), font, scale, (235, 235, 235), thickness, cv2.LINE_AA)

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
            cv2.putText(overlay, cell.letter, (tx, ty), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)

        bar_h = 40
        font = cv2.FONT_HERSHEY_SIMPLEX

        hint = "Blink to select letter   |   D = backspace   |   Esc = quit"
        cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.putText(frame, hint, (10, 26), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        typed_str = self.typed_text
        if typed_str:
            y2 = bar_h + 36
            cv2.rectangle(frame, (0, bar_h), (w, y2), (15, 15, 15), -1)
            cv2.putText(
                frame, typed_str, (10, y2 - 8),
                font, 0.9, (0, 255, 200), 3, cv2.LINE_AA,
            )

        coords_parts: list[str] = []
        if left_iris is not None:
            coords_parts.append(f"L({left_iris[0]:+.2f},{left_iris[1]:+.2f})")
        if right_iris is not None:
            coords_parts.append(f"R({right_iris[0]:+.2f},{right_iris[1]:+.2f})")
        if gaze_xy is not None:
            coords_parts.append(f"Gaze({gaze_xy[0]:.0f},{gaze_xy[1]:.0f})")
        if coords_parts:
            coords_text = "  ".join(coords_parts)
            cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 20), -1)
            cv2.putText(
                frame, coords_text, (10, h - 12),
                font, 0.55, (180, 255, 180), 1, cv2.LINE_AA,
            )
