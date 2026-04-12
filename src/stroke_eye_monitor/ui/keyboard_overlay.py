from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Standard QWERTY layout; last key is backspace.
QWERTY_ROWS = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM") + ["BKSP"]]

# Session transcript for LLM context (sliding window is applied server-side)
_MAX_HISTORY_TURNS = 32

# BGR — modern slate + cyan accent (readable on camera)
_C_BAR = (42, 48, 58)
_C_KEY_BG = (38, 44, 54)
_C_KEY_EDGE = (85, 95, 110)
_C_PANEL = (34, 40, 50)
_C_CARD = (48, 56, 68)
_C_BORDER = (75, 95, 110)
_C_CARD_HI = (58, 68, 88)
_C_ACCENT = (255, 190, 80)
_C_ACCENT2 = (255, 220, 120)
_C_TEXT = (245, 245, 250)
_C_MUTED = (155, 165, 180)
_C_TYPED = (220, 230, 255)
_C_GREEN = (80, 175, 76)

# Original blink UX: count blinks in a short window; confirm as soon as 3 are seen (no extra pause).
_BLINK_WINDOW_S = 2.5
# After last blink, quiet time before committing a phrase pick (suggestions only).
_SUGGEST_COMMIT_PAUSE_S = 0.55
# If only 1 blink so far, wait longer before choosing option 1 so 2nd/3rd blink can land in time.
_SUGGEST_SINGLE_BLINK_CONFIRM_S = 1.35


def _key_face_label(letter: str) -> str:
    """Short label for on-screen key cap (OpenCV font)."""
    if letter == "BKSP":
        return "DEL"
    return letter


def _truncate(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "..."


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
    """QWERTY with dwell-typing; triple-blink unlock / predict like the original demo."""

    cells: list[KeyboardCell] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)
    input_enabled: bool = False
    _active_cell: int = -1
    _canvas_w: int = 0
    _canvas_h: int = 0
    _base_image: np.ndarray | None = field(default=None, repr=False, init=False)

    _cell_enter_time: float = 0.0
    _dwell_triggered: bool = False

    _blink_timestamps: list[float] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    block_input: bool = False
    block_overlay_text: str = "Idle - 3 blinks to start typing"
    suggestions: list[Any] = field(default_factory=list)
    last_action: str | None = None

    dwell_seconds: float = 0.8

    # After LLM: blink 1x / 2x / 3x to pick option; 4x = dismiss
    _suggest_blink_ts: list[float] = field(default_factory=list)

    on_sentence_chosen: Callable[[str], None] | None = None
    spoken_buffer: Any = None

    def set_suggestions(self, suggestions: list[Any]) -> None:
        self.suggestions = list(suggestions)[:3]
        self._suggest_blink_ts.clear()
        self._blink_timestamps.clear()

    def _dismiss_suggestions(self) -> None:
        self.suggestions = []
        self._suggest_blink_ts.clear()
        self._active_cell = -1
        self._dwell_triggered = False

    def _apply_pick(self, index: int) -> None:
        if not self.suggestions or index < 0 or index >= len(self.suggestions):
            self._dismiss_suggestions()
            return
        item = self.suggestions[index]
        chosen = str(getattr(item, "text", str(item))).strip()
        if self.history:
            self.history[-1] = chosen
        else:
            self.history.append(chosen)
        if len(self.history) > _MAX_HISTORY_TURNS:
            self.history.pop(0)
        print(f"Chosen response: {chosen}", flush=True)
        if self.on_sentence_chosen is not None:
            try:
                self.on_sentence_chosen(chosen)
            except Exception as exc:
                print(f"on_sentence_chosen error: {exc}", flush=True)
        self.suggestions = []
        self._suggest_blink_ts.clear()
        self._active_cell = -1
        self._dwell_triggered = False

    def _tick_suggestion_blink_resolve(self) -> None:
        if not self.suggestions or self.block_input:
            return
        now = time.time()
        self._suggest_blink_ts = [t for t in self._suggest_blink_ts if now - t <= 2.5]
        if not self._suggest_blink_ts:
            return
        last = max(self._suggest_blink_ts)
        n = len(self._suggest_blink_ts)
        # Do not lock in option 1 after the first blink's short pause; give time for 2nd/3rd blink.
        if n == 1:
            if now - last < _SUGGEST_SINGLE_BLINK_CONFIRM_S:
                return
        elif now - last < _SUGGEST_COMMIT_PAUSE_S:
            return
        self._suggest_blink_ts.clear()
        if n >= 4:
            self._dismiss_suggestions()
            return
        idx = min(n - 1, len(self.suggestions) - 1)
        self._apply_pick(idx)

    def layout(self, canvas_w: int, canvas_h: int) -> None:
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        self._base_image = None

        self._hint_h = max(44, int(canvas_h * 0.042))
        self._suggest_h = max(150, int(canvas_h * 0.17))
        self._status_h = max(28, int(canvas_h * 0.028))
        self._bottom_h = max(30, int(canvas_h * 0.03))
        kbd_h = int(canvas_h * 0.44)
        self._kbd_top = canvas_h - kbd_h - self._bottom_h
        self._kbd_bottom = canvas_h - self._bottom_h
        self._text_top = self._hint_h + self._suggest_h + self._status_h
        self._text_bottom = self._kbd_top

        rows = QWERTY_ROWS
        nrows = len(rows)
        row_h = kbd_h / nrows
        self.cells = []
        max_keys = max(len(r) for r in rows)
        margin_x = canvas_w * 0.05
        key_w = (canvas_w - 2 * margin_x) / max_keys

        for r, row_keys in enumerate(rows):
            y0 = int(round(self._kbd_top + r * row_h))
            y1 = int(round(self._kbd_top + (r + 1) * row_h))
            if len(row_keys) == 1:
                x0 = int(round(margin_x))
                x1 = int(round(canvas_w - margin_x))
                self.cells.append(
                    KeyboardCell(
                        row=r, col=0, letter=row_keys[0], x0=x0, y0=y0, x1=x1, y1=y1
                    )
                )
            else:
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
        if not self.cells:
            return -1
        best_index = -1
        min_dist = float("inf")
        for i, c in enumerate(self.cells):
            cx = (c.x0 + c.x1) / 2.0
            cy = (c.y0 + c.y1) / 2.0
            dist_sq = (gx - cx) ** 2 + (gy - cy) ** 2
            if dist_sq < min_dist:
                min_dist = dist_sq
                best_index = i
        return best_index

    def update_gaze(self, gx: float, gy: float) -> None:
        if self.suggestions and not self.block_input:
            self._tick_suggestion_blink_resolve()
            self._active_cell = -1
            return

        if not self.input_enabled:
            self._active_cell = -1
            self._dwell_triggered = False
            return

        new_cell = self.hit_test(gx, gy)
        if new_cell != self._active_cell:
            self._active_cell = new_cell
            self._cell_enter_time = time.time()
            self._dwell_triggered = False
        else:
            if self._active_cell >= 0 and not self._dwell_triggered:
                cell = self.cells[self._active_cell]
                if time.time() - self._cell_enter_time >= self.dwell_seconds:
                    self._dwell_triggered = True
                    if cell.letter == "BKSP":
                        if self.typed:
                            self.typed.pop()
                    else:
                        self.typed.append(cell.letter)

    def select(self) -> str | None:
        """Blink: suggestion mode = count blinks; else triple-blink = unlock (idle) or PREDICT."""
        if self.suggestions and not self.block_input:
            now = time.time()
            self._suggest_blink_ts.append(now)
            self._suggest_blink_ts = [t for t in self._suggest_blink_ts if now - t <= 2.5]
            return None
        if self.suggestions:
            return None
        if self.block_input:
            return None

        now = time.time()
        self._blink_timestamps.append(now)
        self._blink_timestamps = [
            t for t in self._blink_timestamps if now - t <= _BLINK_WINDOW_S
        ]
        if len(self._blink_timestamps) < 3:
            return None
        self._blink_timestamps.clear()

        if not self.input_enabled:
            self.input_enabled = True
            print(">>> Typing unlocked (3 blinks) <<<", flush=True)
            return None

        self._trigger_predict()
        return "PREDICT"

    def _trigger_predict(self) -> None:
        sentence = "".join(self.typed).strip()
        print("\n>>> PREDICT TRIGGERED! <<<")
        print(f"=== CONFIRMED SENTENCE: {sentence} ===\n")
        self.typed.clear()
        if not sentence:
            self.last_action = None
            return
        self.history.append(sentence)
        if len(self.history) > _MAX_HISTORY_TURNS:
            self.history.pop(0)
        self.last_action = "PREDICT"

    @property
    def typed_text(self) -> str:
        return "".join(self.typed)

    @property
    def pending_suggest_blink_count(self) -> int:
        now = time.time()
        return len([t for t in self._suggest_blink_ts if now - t <= 2.5])

    def _scaled(self, val: int, sy: float) -> int:
        return int(round(val * sy))

    def draw(
        self,
        frame: np.ndarray,
        left_iris: tuple[float, float] | None = None,
        right_iris: tuple[float, float] | None = None,
    ) -> None:
        h, w = frame.shape[:2]
        sx = w / self._canvas_w
        sy = h / self._canvas_h

        hint_h = self._scaled(self._hint_h, sy)
        suggest_h = self._scaled(self._suggest_h, sy)
        status_h = self._scaled(self._status_h, sy)
        bottom_h = self._scaled(self._bottom_h, sy)
        kbd_top = self._scaled(self._kbd_top, sy)
        text_top = self._scaled(self._text_top, sy)
        text_bot = kbd_top

        if self._base_image is None or self._base_image.shape[:2] != (h, w):
            self._base_image = np.zeros((h, w, 3), dtype=np.uint8)
            for cell in self.cells:
                dx0, dy0 = int(round(cell.x0 * sx)), int(round(cell.y0 * sy))
                dx1, dy1 = int(round(cell.x1 * sx)), int(round(cell.y1 * sy))
                cv2.rectangle(self._base_image, (dx0, dy0), (dx1, dy1), _C_KEY_BG, -1)
                cv2.rectangle(self._base_image, (dx0, dy0), (dx1, dy1), _C_KEY_EDGE, 1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cell_px_w = dx1 - dx0
                scale = max(1.0, min(3.0, cell_px_w / 40.0))
                thick = max(2, min(7, int(round(cell_px_w / 22.0))))
                cap_label = _key_face_label(cell.letter)
                (tw, th_t), _ = cv2.getTextSize(cap_label, font, scale, thick)
                tx = (dx0 + dx1) // 2 - tw // 2
                ty = (dy0 + dy1) // 2 + th_t // 2
                cv2.putText(
                    self._base_image,
                    cap_label,
                    (tx, ty),
                    font,
                    scale,
                    _C_TEXT,
                    thick,
                    cv2.LINE_AA,
                )

        overlay = self._base_image.copy()

        if (
            self.input_enabled
            and not self.block_input
            and 0 <= self._active_cell < len(self.cells)
        ):
            cell = self.cells[self._active_cell]
            dx0, dy0 = int(round(cell.x0 * sx)), int(round(cell.y0 * sy))
            dx1, dy1 = int(round(cell.x1 * sx)), int(round(cell.y1 * sy))
            cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), _C_CARD_HI, -1)
            cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), _C_ACCENT, 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cell_px_w = dx1 - dx0
            scale = max(1.0, min(3.0, cell_px_w / 40.0))
            thick = min(max(3, min(8, int(round(cell_px_w / 20.0)))) + 1, 9)
            cap_label = _key_face_label(cell.letter)
            (tw, th_t), _ = cv2.getTextSize(cap_label, font, scale, thick)
            tx = (dx0 + dx1) // 2 - tw // 2
            ty = (dy0 + dy1) // 2 + th_t // 2
            cv2.putText(overlay, cap_label, (tx, ty), font, scale, _C_ACCENT2, thick, cv2.LINE_AA)
            if not self._dwell_triggered:
                elapsed = time.time() - self._cell_enter_time
                progress = max(0.0, min(1.0, elapsed / self.dwell_seconds))
                if progress > 0:
                    fill_h = int(round((dy1 - dy0) * progress))
                    cv2.rectangle(overlay, (dx0, dy1 - fill_h), (dx1, dy1), _C_GREEN, -1)

        if self.block_input:
            dim = np.zeros_like(overlay)
            cv2.addWeighted(overlay, 0.32, dim, 0.68, 0, dst=overlay)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, dst=frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        sf = max(0.45, h / 1080.0)
        dwell_ms = int(self.dwell_seconds * 1000)

        cv2.rectangle(frame, (0, 0), (w, hint_h), _C_BAR, -1)
        cv2.line(frame, (0, hint_h - 1), (w, hint_h - 1), _C_KEY_EDGE, 1)
        if self.suggestions:
            hint = f"Choose with blinks: 1 / 2 / 3  =  options 1-3   |   4 blinks = cancel   |   Type: {dwell_ms}ms dwell"
        elif not self.input_enabled:
            if self.spoken_buffer is not None:
                hint = (
                    f"IDLE - mic ON for context  |  3 blinks = start typing  |  {dwell_ms}ms dwell when typing"
                )
            else:
                hint = (
                    f"IDLE  |  3 blinks = start typing  |  {dwell_ms}ms dwell when typing"
                )
        elif self.spoken_buffer is not None:
            hint = (
                f"Typing - mic ON  |  dwell {dwell_ms}ms  |  gaze DEL = delete  |  3 blinks = run inference"
            )
        else:
            hint = (
                f"Typing  |  dwell {dwell_ms}ms  |  gaze DEL = delete  |  3 blinks = run inference"
            )
        cv2.putText(
            frame,
            hint,
            (int(16 * sf), int(hint_h * 0.68)),
            font,
            0.62 * sf,
            _C_MUTED,
            max(1, int(2 * sf)),
            cv2.LINE_AA,
        )

        suggest_top = hint_h
        suggest_bot = hint_h + suggest_h
        if self.suggestions:
            cv2.rectangle(frame, (0, suggest_top), (w, suggest_bot), _C_PANEL, -1)
            cv2.line(frame, (0, suggest_bot - 1), (w, suggest_bot - 1), _C_KEY_EDGE, 1)
            title = "Pick your phrase"
            cv2.putText(
                frame,
                title,
                (int(18 * sf), int(suggest_top + 26 * sf)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.72 * sf,
                _C_TEXT,
                max(1, int(2 * sf)),
                cv2.LINE_AA,
            )
            pad = int(14 * sf)
            footer_h = int(36 * sf)
            card_top = int(suggest_top + 38 * sf)
            card_bot = suggest_bot - footer_h
            n_cards = min(3, len(self.suggestions))
            gap = int(10 * sf)
            inner_w = w - 2 * pad
            card_w = (inner_w - gap * (n_cards - 1)) // max(1, n_cards)
            pending = self.pending_suggest_blink_count

            labels = ("1 blink", "2 blinks", "3 blinks")
            for i in range(n_cards):
                sug = self.suggestions[i]
                text = getattr(sug, "text", str(sug))
                text = _truncate(text, 42)
                x0 = pad + i * (card_w + gap)
                x1 = x0 + card_w
                cv2.rectangle(frame, (x0, card_top), (x1, card_bot), _C_CARD, -1)
                cv2.rectangle(frame, (x0, card_top), (x1, card_bot), _C_BORDER, 1)
                badge_cx = x0 + int(card_w * 0.22)
                badge_cy = card_top + int((card_bot - card_top) * 0.28)
                r = int(min(22 * sf, card_w * 0.14))
                cv2.circle(frame, (badge_cx, badge_cy), r, _C_ACCENT, -1, cv2.LINE_AA)
                cv2.circle(frame, (badge_cx, badge_cy), r, (200, 230, 255), 1, cv2.LINE_AA)
                dig = str(i + 1)
                (tw, th_t), _ = cv2.getTextSize(dig, cv2.FONT_HERSHEY_DUPLEX, 1.0 * sf, 2)
                cv2.putText(
                    frame,
                    dig,
                    (badge_cx - tw // 2, badge_cy + th_t // 3),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sf,
                    (30, 30, 35),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    labels[i],
                    (x0 + int(12 * sf), card_top + int((card_bot - card_top) * 0.52)),
                    font,
                    0.48 * sf,
                    _C_MUTED,
                    1,
                    cv2.LINE_AA,
                )
                y_txt = card_top + int((card_bot - card_top) * 0.72)
                for line in _wrap_text_lines(text, max_chars=max(12, card_w // int(7 * sf))):
                    cv2.putText(
                        frame,
                        line,
                        (x0 + int(12 * sf), y_txt),
                        font,
                        0.52 * sf,
                        _C_TEXT,
                        max(1, int(2 * sf)),
                        cv2.LINE_AA,
                    )
                    y_txt += int(22 * sf)

            foot = (
                "4 quick blinks = cancel  |  pause after last blink: "
                "~0.55s for 2-3 blinks, ~1.35s if you only blink once (option 1)"
            )
            cv2.putText(
                frame,
                foot,
                (pad, suggest_bot - int(12 * sf)),
                font,
                0.48 * sf,
                _C_MUTED,
                1,
                cv2.LINE_AA,
            )
            if pending > 0:
                pulse = f"Blinks detected: {pending}"
                cv2.putText(
                    frame,
                    pulse,
                    (w - int(220 * sf), int(suggest_top + 26 * sf)),
                    font,
                    0.55 * sf,
                    _C_ACCENT2,
                    max(1, int(2 * sf)),
                    cv2.LINE_AA,
                )
        elif self.block_input:
            cv2.rectangle(frame, (0, suggest_top), (w, suggest_bot), _C_PANEL, -1)
            loading_text = self.block_overlay_text or "Please wait..."
            (tw, _), _ = cv2.getTextSize(loading_text, font, 0.85 * sf, max(1, int(2 * sf)))
            tx = (w - tw) // 2
            ty = suggest_top + int(suggest_h * 0.55)
            cv2.putText(
                frame,
                loading_text,
                (tx, ty),
                font,
                0.85 * sf,
                _C_ACCENT2,
                max(1, int(2 * sf)),
                cv2.LINE_AA,
            )
        elif self.spoken_buffer is not None:
            cv2.rectangle(frame, (0, suggest_top), (w, suggest_bot), _C_PANEL, -1)
            cv2.line(frame, (0, suggest_bot - 1), (w, suggest_bot - 1), _C_KEY_EDGE, 1)
            if self.input_enabled:
                title = "Microphone (sent with next inference)"
                sub = "Included once when you run inference (3 blinks)"
            else:
                title = "Microphone (listening — idle)"
                sub = "Speak anytime; 3 blinks unlock typing; inference uses speech + letters"
            cv2.putText(
                frame,
                title,
                (int(18 * sf), int(suggest_top + 22 * sf)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.58 * sf,
                _C_TEXT,
                max(1, int(2 * sf)),
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                sub,
                (int(18 * sf), int(suggest_top + 48 * sf)),
                font,
                0.5 * sf,
                _C_MUTED,
                1,
                cv2.LINE_AA,
            )
            y_line = int(suggest_top + 72 * sf)
            for line in self.spoken_buffer.snapshot_lines_for_ui(4):
                safe = line.encode("ascii", errors="replace").decode("ascii")
                safe = _truncate(safe, 96)
                cv2.putText(
                    frame,
                    safe,
                    (int(18 * sf), y_line),
                    font,
                    0.46 * sf,
                    _C_MUTED,
                    1,
                    cv2.LINE_AA,
                )
                y_line += int(20 * sf)

        status_top = suggest_bot
        cv2.rectangle(frame, (0, status_top), (w, status_top + status_h), (30, 36, 44), -1)

        typed_str = self.typed_text
        lines_to_draw: list[tuple[str, tuple[int, int, int]]] = []
        avail_h = text_bot - text_top
        line_h_px = max(20, int(avail_h * 0.22))
        max_lines = max(1, avail_h // line_h_px)

        for hist_str in self.history[-max(1, max_lines - 1) :]:
            # ASCII only: OpenCV Hershey fonts cannot render Unicode bullets (e.g. U+2022).
            lines_to_draw.append((f"> {hist_str}", _C_MUTED))
        if typed_str:
            lines_to_draw.append((typed_str + "_", _C_TYPED))

        if lines_to_draw:
            total_h = len(lines_to_draw) * line_h_px + 10
            panel_top = text_top
            panel_bot = min(text_top + total_h, text_bot)
            if panel_bot > panel_top:
                sub = frame[panel_top:panel_bot, 0:w]
                dark = np.zeros_like(sub)
                cv2.addWeighted(sub, 0.22, dark, 0.78, 0, dst=sub)
            cur_y = text_top + int(line_h_px * 0.8)
            for txt, color in lines_to_draw:
                is_typed = color == _C_TYPED
                s = 0.88 * sf if is_typed else 0.62 * sf
                t = max(1, int(3 * sf)) if is_typed else max(1, int(2 * sf))
                cv2.putText(frame, txt, (int(16 * sf), cur_y), font, s, color, t, cv2.LINE_AA)
                cur_y += line_h_px

        coords_parts: list[str] = []
        if left_iris is not None:
            coords_parts.append(f"L {left_iris[0]:+.2f},{left_iris[1]:+.2f}")
        if right_iris is not None:
            coords_parts.append(f"R {right_iris[0]:+.2f},{right_iris[1]:+.2f}")
        if coords_parts:
            cv2.rectangle(frame, (0, h - bottom_h), (w, h), _C_BAR, -1)
            cv2.putText(
                frame,
                "   ".join(coords_parts),
                (int(12 * sf), h - int(9 * sf)),
                font,
                0.42 * sf,
                (160, 210, 180),
                1,
                cv2.LINE_AA,
            )


def _wrap_text_lines(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= max_chars:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines[:3]
