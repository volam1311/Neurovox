from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from LLM.audio_platform import play_infer_confirm_chime_async

from stroke_eye_monitor.ui.brand_theme import (
    BRAND_ACCENT_SOFT,
    BRAND_TEAL,
    CHAT_CARD_BORDER,
    CHAT_CARD_HI,
    CHAT_DIVIDER,
    CHAT_HEADER,
    CHAT_KEY_BG,
    CHAT_KEY_EDGE,
    CHAT_KEY_INNER,
    CHAT_MUTED,
    CHAT_PANEL,
    CHAT_SUCCESS,
    CHAT_SURFACE,
    CHAT_TEXT,
    CHAT_ACCENT,
    CHAT_ACCENT_SOFT,
    CHAT_ON_ACCENT,
)

# From origin/main: gaze-model-aware geometry + robust hit testing (eye_dimension stack).
_HIT_PAD_FRAC = 0.14
_KEY_MAX_WIDTH_OVER_HEIGHT = 1.32
_KEY_GAP_FRAC = 0.09

# Standard QWERTY; last key is backspace (AAC / inference flow).
QWERTY_ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    ["RESET"] + list("ZXCVBNM") + [",", ".", "BKSP"],
]
ROWS = len(QWERTY_ROWS)


@dataclass(frozen=True)
class KeyboardLayoutProfile:
    """Geometry and timing tuned to how different gaze regressors behave live."""

    margin_x_frac: float
    margin_top_frac: float
    margin_bot_frac: float
    dwell_seconds: float
    overlay_key_alpha: float
    active_outline_bgr: tuple[int, int, int]
    dwell_fill_bgr: tuple[int, int, int]


def keyboard_profile_for_gaze_model(model_type: str | None) -> KeyboardLayoutProfile:
    """Linear maps are often stable in a tight band; tree/boost models benefit from larger targets."""
    mt = (model_type or "").strip().lower()
    if mt in ("ridge", "poly"):
        return KeyboardLayoutProfile(
            margin_x_frac=0.10,
            margin_top_frac=0.06,
            margin_bot_frac=0.20,
            dwell_seconds=0.62,
            overlay_key_alpha=0.62,
            active_outline_bgr=BRAND_TEAL,
            dwell_fill_bgr=BRAND_ACCENT_SOFT,
        )
    if mt in ("rf", "xgboost", "gbr"):
        return KeyboardLayoutProfile(
            margin_x_frac=0.06,
            margin_top_frac=0.06,
            margin_bot_frac=0.20,
            dwell_seconds=0.88,
            overlay_key_alpha=0.55,
            active_outline_bgr=BRAND_TEAL,
            dwell_fill_bgr=BRAND_ACCENT_SOFT,
        )
    return KeyboardLayoutProfile(
        margin_x_frac=0.08,
        margin_top_frac=0.06,
        margin_bot_frac=0.20,
        dwell_seconds=0.75,
        overlay_key_alpha=0.58,
        active_outline_bgr=BRAND_TEAL,
        dwell_fill_bgr=BRAND_ACCENT_SOFT,
    )

# Session transcript for LLM context (sliding window is applied server-side)
_MAX_HISTORY_TURNS = 32

# BGR — chat-style neutrals (see brand_theme.py)
_C_BAR = CHAT_DIVIDER
_C_HEADER = CHAT_HEADER
_C_KEY_BG = CHAT_KEY_BG
_C_KEY_EDGE = CHAT_KEY_EDGE
_C_KEY_INNER = CHAT_KEY_INNER
_C_PANEL = CHAT_PANEL
_C_CARD = CHAT_SURFACE
_C_BORDER = CHAT_CARD_BORDER
_C_CARD_HI = CHAT_CARD_HI
_C_ACCENT = CHAT_ACCENT
_C_ACCENT2 = CHAT_ACCENT_SOFT
_C_TEXT = CHAT_TEXT
_C_MUTED = CHAT_MUTED
_C_TYPED = CHAT_TEXT
_C_GREEN = CHAT_SUCCESS
# Inner padding for key face (fraction of min key side)
_KEY_FACE_PAD_FRAC = 0.10

# Original blink UX: count blinks in a short window; confirm as soon as 3 are seen (no extra pause).
_BLINK_WINDOW_S = 2.5
# After last blink, quiet time before committing a phrase pick (suggestions only).
_SUGGEST_COMMIT_PAUSE_S = 0.55
# If only 1 blink so far, wait this long after the blink before locking option 1 (time for 2nd/3rd).
_SUGGEST_SINGLE_BLINK_CONFIRM_S = 2.0
# Ignore blink edges for this long after suggestions appear (avoids spurious pick from UI transition).
_SUGGEST_BLINK_ARM_DELAY_S = 1.6
# If the user does not pick a suggestion (or reset), return to typing after this many seconds.
_SUGGEST_AUTO_DISMISS_S = 75.0


def _key_face_label(letter: str) -> str:
    """Label for on-screen key cap (OpenCV font)."""
    if letter == "BKSP":
        return "DEL"
    if letter == "RESET":
        return "RESET"
    return letter


def _key_face_scale_for_label(
    label: str,
    font: int,
    thick: int,
    max_text_w: int,
    base_scale: float,
) -> float:
    """Shrink scale so the full label fits within max_text_w (e.g. RESET on a narrow key)."""
    s = float(base_scale)
    for _ in range(14):
        tw, _ = cv2.getTextSize(label, font, s, thick)[0]
        if tw <= max_text_w and s >= 0.42:
            return s
        s *= 0.9
    return max(0.42, s)


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
    """QWERTY: 3 blinks unlock typing; blink on key types; eyes closed ~3s confirms to inference."""

    cells: list[KeyboardCell] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)
    input_enabled: bool = False
    _active_cell: int = -1
    _canvas_w: int = 0
    _canvas_h: int = 0
    _base_image: np.ndarray | None = field(default=None, repr=False, init=False)

    _last_gx: float | None = field(default=None, repr=False)
    _last_gy: float | None = field(default=None, repr=False)

    _blink_timestamps: list[float] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    block_input: bool = False
    block_overlay_text: str = "Idle - 3 blinks to start typing"
    suggestions: list[Any] = field(default_factory=list)
    last_action: str | None = None

    dwell_seconds: float = 0.8
    # When True, gaze dot is drawn at the key center (hit test still uses raw gaze).
    gravity_snap: bool = True
    # After unlock: cumulative seconds eyes closed to confirm line -> inference (no gaze rule).
    infer_confirm_hold_s: float = 3.0
    _infer_confirm_accum_s: float = field(default=0.0, repr=False)
    # Shown while TTS reads a chosen phrase (main thread draws; worker sets text).
    tts_spoken_text: str | None = None

    # Voice STT: right-eye wink ~hold_s arms mic; left-eye wink disarms (no capture otherwise).
    mic_capture_enabled: bool = False
    wink_mic_hold_s: float = 1.0
    _wink_r_hold_s: float = field(default=0.0, repr=False)
    _wink_l_hold_s: float = field(default=0.0, repr=False)

    # After LLM: blink 1x / 2x / 3x to pick option; 4x = dismiss
    _suggest_blink_ts: list[float] = field(default_factory=list)
    _suggest_armed_at: float = field(default=0.0, repr=False)
    _suggest_deadline: float = field(default=0.0, repr=False)

    on_sentence_chosen: Callable[[str], None] | None = None
    spoken_buffer: Any = None

    _profile: KeyboardLayoutProfile = field(
        default_factory=lambda: keyboard_profile_for_gaze_model(None),
        repr=False,
    )
    _gaze_model_label: str = field(default="", repr=False)
    _blink: Any = field(default=None, repr=False, init=False)

    def attach_blink(self, blink: Any) -> None:
        """Optional blink detector ref so Reset can clear edge state without importing here."""
        self._blink = blink

    def reset_key_active(self) -> bool:
        if not (0 <= self._active_cell < len(self.cells)):
            return False
        return self.cells[self._active_cell].letter == "RESET"

    def reset_infer_confirm_accum(self) -> None:
        self._infer_confirm_accum_s = 0.0

    def reset_mic_wink_accums(self) -> None:
        self._wink_r_hold_s = 0.0
        self._wink_l_hold_s = 0.0

    def feed_mic_wink_gate(
        self,
        dt: float,
        l_ear: float,
        r_ear: float,
        close_t: float,
        open_t: float,
        hold_s: float,
    ) -> None:
        """Arm/disarm mic capture: right-eye wink (R closed, L open) vs left-eye wink to stop."""
        hold = max(0.25, float(hold_s))
        ct = float(close_t)
        ot = float(open_t)
        # One-sided winks rarely match global blink thresholds: the open eye often sits
        # between close_t and open_t. Use peer_min + asymmetry instead of l>=open_t only.
        peer_min = max(ct * 0.95, (ct + ot) * 0.5 - 0.02)
        closed_max = ct + 0.03
        asym = 0.018
        # Left wink (disarm): L-side closure is often weaker on EAR; R peer eye may read lower.
        peer_min_for_left_wink_peer = max(ct * 0.82, peer_min - 0.035)
        closed_max_l = ct + 0.055
        asym_l = 0.01

        if l_ear <= ct and r_ear <= ct:
            self._wink_r_hold_s = 0.0
            self._wink_l_hold_s = 0.0
            return
        if not self.mic_capture_enabled:
            # Right wink: R lower than L, R near closed, L open enough vs R
            if (
                r_ear <= closed_max
                and l_ear >= peer_min
                and (l_ear - r_ear) >= asym
            ):
                self._wink_r_hold_s += float(dt)
                self._wink_l_hold_s = 0.0
                if self._wink_r_hold_s >= hold:
                    self.mic_capture_enabled = True
                    self._wink_r_hold_s = 0.0
                    print(">>> Mic capture ON (right-eye wink) <<<", flush=True)
            else:
                self._wink_r_hold_s = 0.0
        else:
            # Left wink: primary = relaxed (typical). Fallback = same shape as right-wink arm.
            left_wink_loose = (
                l_ear <= closed_max_l
                and r_ear >= peer_min_for_left_wink_peer
                and (r_ear - l_ear) >= asym_l
            )
            left_wink_tight = (
                l_ear <= closed_max
                and r_ear >= peer_min
                and (r_ear - l_ear) >= asym
            )
            if left_wink_loose or left_wink_tight:
                self._wink_l_hold_s += float(dt)
                self._wink_r_hold_s = 0.0
                if self._wink_l_hold_s >= hold:
                    self.mic_capture_enabled = False
                    self._wink_l_hold_s = 0.0
                    print(">>> Mic capture OFF (left-eye wink) <<<", flush=True)
            else:
                self._wink_l_hold_s = 0.0

    @property
    def mic_wink_r_progress_s(self) -> float:
        return float(self._wink_r_hold_s)

    @property
    def mic_wink_l_progress_s(self) -> float:
        return float(self._wink_l_hold_s)

    def feed_infer_confirm_closure(
        self,
        dt: float,
        avg_ear: float,
        close_thresh: float,
        hold_s: float,
    ) -> None:
        """Accumulate continuous eyes-closed time, then run inference (no gaze-to-chat rule)."""
        if not self.input_enabled or self.block_input or self.suggestions:
            self._infer_confirm_accum_s = 0.0
            return
        hold = max(0.5, float(hold_s))
        if avg_ear <= close_thresh:
            self._infer_confirm_accum_s += float(dt)
            if self._infer_confirm_accum_s >= hold:
                self._infer_confirm_accum_s = 0.0
                play_infer_confirm_chime_async()
                self._trigger_predict()
        else:
            self._infer_confirm_accum_s = 0.0

    @property
    def infer_confirm_accum_s(self) -> float:
        return float(self._infer_confirm_accum_s)

    def set_suggestions(self, suggestions: list[Any]) -> None:
        gathered: list[Any] = []
        for item in list(suggestions)[:3]:
            t = str(getattr(item, "text", str(item))).strip()
            if t:
                gathered.append(item)
        self.suggestions = gathered
        self._suggest_blink_ts.clear()
        self._blink_timestamps.clear()
        self._infer_confirm_accum_s = 0.0
        self._suggest_armed_at = time.time()
        self._suggest_deadline = (
            time.time() + _SUGGEST_AUTO_DISMISS_S if self.suggestions else 0.0
        )

    def _dismiss_suggestions(self) -> None:
        self.suggestions = []
        self._suggest_blink_ts.clear()
        self._active_cell = -1
        self._suggest_deadline = 0.0

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
        self._suggest_deadline = 0.0

    def _tick_suggestion_blink_resolve(self) -> None:
        if not self.suggestions or self.block_input:
            return
        now = time.time()
        if now < self._suggest_armed_at + _SUGGEST_BLINK_ARM_DELAY_S:
            return
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

    @property
    def stage(self) -> int:
        """Reserved for multi-stage keyboards; LLM keyboard is single-stage."""
        return 0

    def go_back(self) -> None:
        """No-op (single-stage QWERTY)."""
        return None

    def layout_bounds(self) -> tuple[int, int, int, int] | None:
        """Axis-aligned bounds of all keys in layout pixels (same space as ``hit_test``)."""
        if not self.cells:
            return None
        return (
            min(c.x0 for c in self.cells),
            max(c.x1 for c in self.cells),
            min(c.y0 for c in self.cells),
            max(c.y1 for c in self.cells),
        )

    def layout(
        self,
        canvas_w: int,
        canvas_h: int,
        *,
        gaze_model: str | None = None,
        margin_top_frac: float | None = None,
        margin_bot_frac: float | None = None,
    ) -> None:
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        self._base_image = None
        self._profile = keyboard_profile_for_gaze_model(gaze_model)
        raw = (gaze_model or "").strip()
        self._gaze_model_label = raw.upper()[:14] if raw else "DEFAULT"

        # Large keyboard band for gaze tolerance; single chat column uses the rest.
        self._bottom_h = max(26, int(canvas_h * 0.024))
        kbd_h = int(canvas_h * 0.60)
        self._kbd_top = canvas_h - kbd_h - self._bottom_h
        self._kbd_bottom = canvas_h - self._bottom_h
        self._chat_title_h = max(36, int(canvas_h * 0.034))
        self._text_top = self._chat_title_h + int(canvas_h * 0.012)
        self._text_bottom = self._kbd_top

        p = self._profile
        top_y = int(self._kbd_top)
        bot_y = int(self._kbd_bottom)
        cw = int(canvas_w)
        margin_x = int(round(cw * p.margin_x_frac))
        usable_w = cw - 2 * margin_x
        usable_h = max(1, bot_y - top_y)
        row_h = usable_h / ROWS

        max_keys = max(len(r) for r in QWERTY_ROWS)
        key_w_full = usable_w / float(max_keys)
        key_w_cap = float(row_h) * _KEY_MAX_WIDTH_OVER_HEIGHT
        key_w = min(key_w_full, key_w_cap)
        gap = max(0.0, key_w * _KEY_GAP_FRAC)

        self.cells = []
        for r, row_keys in enumerate(QWERTY_ROWS):
            y0 = int(round(top_y + r * row_h))
            y1 = int(round(top_y + (r + 1) * row_h))
            n = len(row_keys)
            row_inner_w = n * key_w + max(0, n - 1) * gap
            start_x = margin_x + max(0.0, (usable_w - row_inner_w) * 0.5)
            for c, letter in enumerate(row_keys):
                x0f = start_x + c * (key_w + gap)
                x1f = x0f + key_w
                x0 = int(round(x0f))
                x1 = int(round(x1f))
                if x1 <= x0:
                    x1 = x0 + 1
                self.cells.append(
                    KeyboardCell(
                        row=r, col=c, letter=letter, x0=x0, y0=y0, x1=x1, y1=y1
                    )
                )

    @staticmethod
    def _cell_center(c: KeyboardCell) -> tuple[float, float]:
        return ((c.x0 + c.x1) * 0.5, (c.y0 + c.y1) * 0.5)

    @staticmethod
    def _padded_hit_rect(c: KeyboardCell) -> tuple[float, float, float, float]:
        w = max(1.0, float(c.x1 - c.x0))
        h = max(1.0, float(c.y1 - c.y0))
        pad = _HIT_PAD_FRAC * min(w, h)
        return (float(c.x0 - pad), float(c.x1 + pad), float(c.y0 - pad), float(c.y1 + pad))

    def _nearest_key_row_first(self, gx: float, gy: float) -> int:
        if not self.cells:
            return -1
        by_row: dict[int, list[tuple[int, KeyboardCell]]] = defaultdict(list)
        for i, c in enumerate(self.cells):
            by_row[c.row].append((i, c))

        row_metrics: list[tuple[float, float, int]] = []
        for r, lst in by_row.items():
            y0 = min(c.y0 for _, c in lst)
            y1 = max(c.y1 for _, c in lst)
            yc = 0.5 * (y0 + y1)
            if y0 <= gy <= y1:
                dy = 0.0
            else:
                dy = min(abs(gy - y0), abs(gy - y1))
            row_metrics.append((dy, abs(gy - yc), r))
        row_metrics.sort(key=lambda t: (t[0], t[1]))
        best_row = row_metrics[0][2]

        lst = by_row[best_row]
        best_i = lst[0][0]
        best_dx = float("inf")
        for i, c in lst:
            cx, _ = self._cell_center(c)
            dx = abs(gx - cx)
            if dx < best_dx:
                best_dx = dx
                best_i = i
        return best_i

    def hit_test(self, gx: float, gy: float) -> int:
        if not self.cells:
            return -1
        # Gaze above the keyboard band is conversation/header — do not snap to a key.
        if float(gy) < float(self._kbd_top):
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

        return self._nearest_key_row_first(gx, gy)

    def update_gaze(self, gx: float, gy: float) -> None:
        self._last_gx = float(gx)
        self._last_gy = float(gy)
        if self.suggestions and not self.block_input:
            if self._suggest_deadline > 0.0 and time.time() >= self._suggest_deadline:
                self._dismiss_suggestions()
            self._tick_suggestion_blink_resolve()
            if self.suggestions and not self.block_input:
                idx = self.hit_test(gx, gy)
                if 0 <= idx < len(self.cells) and self.cells[idx].letter == "RESET":
                    self._active_cell = idx
                else:
                    self._active_cell = -1
                return

        # Locked (before 3-blink unlock): do not latch gaze to any key — otherwise RESET
        # stays active and select() handles reset before the unlock blink count.
        if not self.input_enabled:
            self._active_cell = -1
            return

        self._active_cell = self.hit_test(gx, gy)

    def pointer_gaze_for_display(self) -> tuple[float, float]:
        """Point for gaze overlay: snapped to the active key center when gravity is on."""
        gx = float(self._last_gx if self._last_gx is not None else 0.0)
        gy = float(self._last_gy if self._last_gy is not None else 0.0)
        if not self.gravity_snap:
            return gx, gy
        if self.suggestions:
            if self.reset_key_active():
                return self._cell_center(self.cells[self._active_cell])
            return gx, gy
        # Before unlock: show raw gaze (no snap to RESET / keys) for reliable 3-blink unlock.
        if not self.input_enabled:
            return gx, gy
        if 0 <= self._active_cell < len(self.cells):
            return self._cell_center(self.cells[self._active_cell])
        return gx, gy

    def user_reset_interface(self) -> None:
        """Leave suggestion / wait states and return to typing (large target + blink)."""
        self.suggestions = []
        self._suggest_blink_ts.clear()
        self._suggest_armed_at = 0.0
        self._suggest_deadline = 0.0
        self._active_cell = -1
        self.block_input = False
        self.block_overlay_text = "Ready to type"
        self.last_action = None
        self.tts_spoken_text = None
        self._infer_confirm_accum_s = 0.0
        self.typed.clear()
        if self._blink is not None:
            try:
                self._blink.reset()
            except Exception:
                pass

    def select(self) -> str | None:
        """Suggestions: blink count. Else: 3 blinks unlock; blink on key types; infer uses eye closure."""
        if self.suggestions and not self.block_input:
            now = time.time()
            if now < self._suggest_armed_at + _SUGGEST_BLINK_ARM_DELAY_S:
                return None
            self._suggest_blink_ts.append(now)
            self._suggest_blink_ts = [t for t in self._suggest_blink_ts if now - t <= 2.5]
            return None
        if self.suggestions:
            return None
        if self.block_input:
            return None

        now = time.time()

        if not self.input_enabled:
            self._blink_timestamps.append(now)
            self._blink_timestamps = [
                t for t in self._blink_timestamps if now - t <= _BLINK_WINDOW_S
            ]
            if len(self._blink_timestamps) < 3:
                return None
            self._blink_timestamps.clear()
            self._infer_confirm_accum_s = 0.0
            self.input_enabled = True
            print(">>> Typing unlocked (3 blinks) <<<", flush=True)
            return None

        if self.reset_key_active():
            self.user_reset_interface()
            print(">>> UI reset (Ready to type) <<<", flush=True)
            return None

        # hit_test already returns -1 when gaze is above the keyboard; no extra ly guard.
        # Keyboard: one blink commits the highlighted key.
        self._infer_confirm_accum_s = 0.0
        if 0 <= self._active_cell < len(self.cells):
            cell = self.cells[self._active_cell]
            if cell.letter == "BKSP":
                if self.typed:
                    self.typed.pop()
                return "BKSP"
            if cell.letter == "RESET":
                return None
            self.typed.append(cell.letter)
            return cell.letter
        return None

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

        bottom_h = self._scaled(self._bottom_h, sy)

        if self._base_image is None or self._base_image.shape[:2] != (h, w):
            self._base_image = np.zeros((h, w, 3), dtype=np.uint8)
            for cell in self.cells:
                dx0, dy0 = int(round(cell.x0 * sx)), int(round(cell.y0 * sy))
                dx1, dy1 = int(round(cell.x1 * sx)), int(round(cell.y1 * sy))
                cw = max(1, dx1 - dx0)
                ch = max(1, dy1 - dy0)
                pad = max(2, int(min(cw, ch) * _KEY_FACE_PAD_FRAC))
                ix0, iy0 = dx0 + pad, dy0 + pad
                ix1, iy1 = dx1 - pad, dy1 - pad
                if ix1 <= ix0:
                    ix1 = ix0 + 1
                if iy1 <= iy0:
                    iy1 = iy0 + 1
                cv2.rectangle(self._base_image, (dx0, dy0), (dx1, dy1), _C_KEY_BG, -1)
                cv2.rectangle(self._base_image, (dx0, dy0), (dx1, dy1), _C_KEY_EDGE, 2)
                cv2.rectangle(self._base_image, (ix0, iy0), (ix1, iy1), _C_KEY_INNER, -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cell_px_w = ix1 - ix0
                inner_w = min(ix1 - ix0, iy1 - iy0)
                max_tw = max(8, int(inner_w * 0.88))
                base_scale = max(1.0, min(3.0, cell_px_w / 40.0))
                thick = max(2, min(7, int(round(cell_px_w / 22.0))))
                cap_label = _key_face_label(cell.letter)
                scale = (
                    _key_face_scale_for_label(cap_label, font, thick, max_tw, base_scale)
                    if len(cap_label) > 3
                    else base_scale
                )
                (tw, th_t), _ = cv2.getTextSize(cap_label, font, scale, thick)
                tx = (ix0 + ix1) // 2 - tw // 2
                ty = (iy0 + iy1) // 2 + th_t // 2
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

        _key_hl = False
        if 0 <= self._active_cell < len(self.cells):
            _ac = self.cells[self._active_cell]
            if _ac.letter == "RESET":
                _key_hl = True
            elif self.input_enabled and not self.block_input and not self.suggestions:
                _key_hl = True
        if _key_hl:
            cell = self.cells[self._active_cell]
            dx0, dy0 = int(round(cell.x0 * sx)), int(round(cell.y0 * sy))
            dx1, dy1 = int(round(cell.x1 * sx)), int(round(cell.y1 * sy))
            cw = max(1, dx1 - dx0)
            ch = max(1, dy1 - dy0)
            pad = max(2, int(min(cw, ch) * _KEY_FACE_PAD_FRAC))
            ix0, iy0 = dx0 + pad, dy0 + pad
            ix1, iy1 = dx1 - pad, dy1 - pad
            if ix1 <= ix0:
                ix1 = ix0 + 1
            if iy1 <= iy0:
                iy1 = iy0 + 1
            ol_b, ol_g, ol_r = self._profile.active_outline_bgr
            cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), _C_CARD_HI, -1)
            cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), (ol_b, ol_g, ol_r), 2)
            cv2.rectangle(overlay, (ix0, iy0), (ix1, iy1), _C_KEY_INNER, -1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cell_px_w = ix1 - ix0
            inner_w = min(ix1 - ix0, iy1 - iy0)
            max_tw = max(8, int(inner_w * 0.88))
            base_scale = max(1.0, min(3.0, cell_px_w / 40.0))
            thick = min(max(3, min(8, int(round(cell_px_w / 20.0)))) + 1, 9)
            cap_label = _key_face_label(cell.letter)
            scale = (
                _key_face_scale_for_label(cap_label, font, thick, max_tw, base_scale)
                if len(cap_label) > 3
                else base_scale
            )
            (tw, th_t), _ = cv2.getTextSize(cap_label, font, scale, thick)
            tx = (ix0 + ix1) // 2 - tw // 2
            ty = (iy0 + iy1) // 2 + th_t // 2
            cv2.putText(overlay, cap_label, (tx, ty), font, scale, _C_ACCENT2, thick, cv2.LINE_AA)
            if cell.letter == "BKSP":
                type_lbl = "del"
            elif cell.letter == "RESET":
                type_lbl = "blink"
            else:
                type_lbl = "type"
            hint_scale = 0.38 * scale if cell.letter != "RESET" else max(0.28, 0.32 * scale)
            cv2.putText(
                overlay,
                type_lbl,
                (ix0 + 4, iy1 - 6),
                font,
                hint_scale,
                _C_MUTED,
                1,
                cv2.LINE_AA,
            )

        if self.block_input:
            dim = np.zeros_like(overlay)
            cv2.addWeighted(overlay, 0.32, dim, 0.68, 0, dst=overlay)
        ka = self._profile.overlay_key_alpha
        cv2.addWeighted(overlay, ka, frame, 1.0 - ka, 0, dst=frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        sf = max(0.45, h / 1080.0)
        # Modest bump over f327888 baseline (was non-clipping); scale line steps by the same factor.
        _ds = 1.1
        # Bring status / hints closer to the main typed line size (was visually tiny vs. input).
        _bal = 1.22

        kbd_top_px = self._scaled(self._kbd_top, sy)
        title_h_px = self._scaled(self._chat_title_h, sy)

        # Single chat-style panel: mic + history + your typing (+ suggestions inside same box).
        cv2.rectangle(frame, (0, 0), (w, kbd_top_px), _C_PANEL, -1)
        cv2.line(frame, (0, kbd_top_px - 1), (w, kbd_top_px - 1), _C_KEY_EDGE, 1)
        cv2.rectangle(frame, (0, 0), (w, title_h_px), _C_HEADER, -1)
        cv2.line(frame, (0, title_h_px - 1), (w, title_h_px - 1), _C_BAR, 1)
        cv2.putText(
            frame,
            "Conversation",
            (int(18 * sf), int(title_h_px * 0.72)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.72 * sf * _ds * _bal,
            _C_TEXT,
            max(1, int(2 * sf)),
            cv2.LINE_AA,
        )
        if self.suggestions:
            subhint = (
                "1/2/3 blinks = pick reply | 4 blinks = cancel | "
                "RESET (bottom row) + blink"
            )
        elif not self.input_enabled:
            subhint = (
                "3 blinks = unlock | Mic: hold right wink ~{:.0f}s | RESET + blink"
            ).format(self.wink_mic_hold_s)
        else:
            subhint = (
                "Blink keys to type | Close eyes ~{:.0f}s to send | "
                "Mic | DEL | RESET"
            ).format(self.infer_confirm_hold_s)
        cv2.putText(
            frame,
            subhint,
            (int(14 * sf), int(title_h_px + 18 * sf)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.58 * sf * _ds * _bal,
            _C_TEXT,
            max(1, int(2 * sf)),
            cv2.LINE_AA,
        )

        body_y0 = int(title_h_px + 42 * sf * _ds)

        if self.block_input:
            pad = int(14 * sf)
            status = self.block_overlay_text or "Please wait..."
            y = body_y0 + int(22 * sf * _ds)
            cv2.putText(
                frame,
                status,
                (pad, y),
                font,
                0.62 * sf * _ds * _bal,
                _C_ACCENT2,
                max(1, int(2 * sf)),
                cv2.LINE_AA,
            )
            y += int(30 * sf * _ds)
            if self.tts_spoken_text:
                spoken = self.tts_spoken_text.encode("ascii", errors="replace").decode("ascii")
                max_c = max(24, int((w - 2 * pad) / (7 * sf)))
                for line in _wrap_text_lines(spoken, max_chars=max_c):
                    cv2.putText(
                        frame,
                        line,
                        (pad, y),
                        font,
                        0.58 * sf * _ds * _bal,
                        _C_TYPED,
                        max(1, int(2 * sf)),
                        cv2.LINE_AA,
                    )
                    y += int(26 * sf * _ds)
                    if y > kbd_top_px - int(12 * sf):
                        break
        elif self.suggestions:
            pad = int(20 * sf)
            suggest_top = body_y0
            suggest_bot = kbd_top_px - int(20 * sf)
            card_top = int(suggest_top + 14 * sf)
            card_bot = suggest_bot - int(40 * sf)
            n_cards = min(3, len(self.suggestions))
            gap = int(18 * sf)
            inner_w = w - 2 * pad
            card_w = (inner_w - gap * (n_cards - 1)) // max(1, n_cards)
            pending = self.pending_suggest_blink_count
            cv2.putText(
                frame,
                "Assistant suggestions",
                (pad, int(suggest_top + 6 * sf)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.58 * sf * _ds * _bal,
                _C_ACCENT2,
                max(1, int(2 * sf)),
                cv2.LINE_AA,
            )
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
                dig = str(i + 1)
                (tw, th_t), _ = cv2.getTextSize(dig, cv2.FONT_HERSHEY_DUPLEX, 1.0 * sf, 2)
                cv2.putText(
                    frame,
                    dig,
                    (badge_cx - tw // 2, badge_cy + th_t // 3),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sf,
                    CHAT_ON_ACCENT,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    labels[i],
                    (x0 + int(14 * sf), card_top + int((card_bot - card_top) * 0.52)),
                    font,
                    0.48 * sf * _ds * _bal,
                    _C_MUTED,
                    1,
                    cv2.LINE_AA,
                )
                y_txt = card_top + int((card_bot - card_top) * 0.72)
                for line in _wrap_text_lines(text, max_chars=max(12, card_w // int(7 * sf))):
                    cv2.putText(
                        frame,
                        line,
                        (x0 + int(14 * sf), y_txt),
                        font,
                        0.52 * sf * _ds * _bal,
                        _C_TEXT,
                        max(1, int(2 * sf)),
                        cv2.LINE_AA,
                    )
                    y_txt += int(26 * sf)
            foot = (
                "4 blinks = cancel | RESET + blink | "
                f"back to typing in ~{_SUGGEST_AUTO_DISMISS_S:.0f}s if no pick"
            )
            cv2.putText(
                frame,
                foot,
                (pad, suggest_bot - int(12 * sf)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52 * sf * _ds * _bal,
                _C_TEXT,
                max(1, int(2 * sf)),
                cv2.LINE_AA,
            )
            if pending > 0:
                cv2.putText(
                    frame,
                    f"Blinks: {pending}",
                    (w - int(160 * sf), int(suggest_top + 6 * sf)),
                    font,
                    0.52 * sf * _ds * _bal,
                    _C_ACCENT2,
                    max(1, int(2 * sf)),
                    cv2.LINE_AA,
                )
        else:
            # --- Layout: status strip (top) | left-aligned entry block (vertically centered) | history ---
            panel_pad = int(12 * sf)
            panel_bot = kbd_top_px - panel_pad
            font_dup = cv2.FONT_HERSHEY_DUPLEX
            x_entry = int(16 * sf)

            y_line = float(body_y0)
            if self.input_enabled and not self.block_input and not self.suggestions:
                hold = max(0.5, float(self.infer_confirm_hold_s))
                acc = min(hold, self.infer_confirm_accum_s)
                cv2.putText(
                    frame,
                    f"Send: keep eyes closed {acc:.1f} / {hold:.0f}s",
                    (int(14 * sf), int(y_line + 14 * sf)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.58 * sf * _ds * _bal,
                    _C_GREEN,
                    max(1, int(2 * sf)),
                    cv2.LINE_AA,
                )
                y_line += 30 * sf * _ds
            if self.spoken_buffer is not None:
                wh = max(0.25, float(self.wink_mic_hold_s))
                rprog = min(wh, self.mic_wink_r_progress_s)
                lprog = min(wh, self.mic_wink_l_progress_s)
                mic_lbl = (
                    f"Mic {'on' if self.mic_capture_enabled else 'off'} - "
                    f"right wink ~{wh:.0f}s on / left wink off | "
                    f"R {rprog:.1f}/{wh:.1f}s L {lprog:.1f}/{wh:.1f}s"
                )
                cv2.putText(
                    frame,
                    mic_lbl,
                    (int(16 * sf), int(y_line + 16 * sf)),
                    font_dup,
                    0.58 * sf * _ds * _bal,
                    _C_ACCENT,
                    max(1, int(2 * sf)),
                    cv2.LINE_AA,
                )
                y_line += 34 * sf * _ds
                for line in self.spoken_buffer.snapshot_lines_for_ui(4):
                    safe = line.encode("ascii", errors="replace").decode("ascii")
                    safe = _truncate(safe, 96)
                    cv2.putText(
                        frame,
                        safe,
                        (int(22 * sf), int(y_line)),
                        font,
                        0.5 * sf * _ds * _bal,
                        _C_MUTED,
                        1,
                        cv2.LINE_AA,
                    )
                    y_line += 22 * sf * _ds
                y_line += 10 * sf * _ds

            y_status_end = y_line
            typed_str = self.typed_text
            hist_lines: list[str] = []
            for hist_str in self.history[-3:]:
                hsafe = hist_str.encode("ascii", errors="replace").decode("ascii")
                hist_lines.append(f"Last: {hsafe}")
            hist_scale = 0.52 * sf * _ds * _bal
            line_h_hist = int(24 * sf * _ds)

            you_lbl = "You (typed)"
            you_scale = 0.56 * sf * _ds * _bal
            you_th = max(1, int(2 * sf))
            (tw_u, th_u), bl_u = cv2.getTextSize(you_lbl, font_dup, you_scale, you_th)
            visual_gap = int(18 * sf * _ds)

            typed_scale = 0.78 * sf * _ds
            typed_th = max(1, int(2 * sf))
            if typed_str:
                typed_line = typed_str + "_"
                (tw_t, th_t), bl_t = cv2.getTextSize(typed_line, font, typed_scale, typed_th)
                block_h = float(visual_gap + th_u + th_t)
            else:
                (tw_t, th_t), bl_t = (0, 0), 0
                typed_line = ""
                block_h = float(th_u)

            margin_after_status = int(22 * sf * _ds)
            hist_total_h = len(hist_lines) * line_h_hist + int(10 * sf)
            avail_top = float(y_status_end + margin_after_status)
            avail_bot = float(panel_bot - hist_total_h - int(8 * sf))
            region_h = max(0.0, avail_bot - avail_top)

            if region_h < block_h + 4:
                block_top = avail_top
            else:
                block_top = avail_top + (region_h - block_h) / 2.0

            baseline_you = int(block_top + (th_u - bl_u))
            cv2.putText(
                frame,
                you_lbl,
                (x_entry, baseline_you),
                font_dup,
                you_scale,
                _C_ACCENT,
                you_th,
                cv2.LINE_AA,
            )

            if typed_str:
                baseline_typed = int(
                    baseline_you + bl_u + visual_gap + (th_t - bl_t)
                )
                cv2.putText(
                    frame,
                    typed_line,
                    (x_entry, baseline_typed),
                    font,
                    typed_scale,
                    _C_TYPED,
                    typed_th,
                    cv2.LINE_AA,
                )
                bottom_entry = float(baseline_typed + bl_t)
            else:
                bottom_entry = float(baseline_you + bl_u)

            n_hist = len(hist_lines)
            if n_hist > 0:
                y_hist = float(panel_bot - n_hist * line_h_hist - int(8 * sf))
                y_hist = max(y_hist, bottom_entry + int(14 * sf * _ds))
                y_last = y_hist + float(n_hist - 1) * float(line_h_hist)
                max_last = float(panel_bot - int(4 * sf))
                if y_last > max_last:
                    y_hist -= y_last - max_last
                    y_hist = max(y_hist, bottom_entry + int(14 * sf * _ds))

            for hline in hist_lines:
                cv2.putText(
                    frame,
                    hline,
                    (int(22 * sf), int(y_hist)),
                    font,
                    hist_scale,
                    _C_MUTED,
                    1,
                    cv2.LINE_AA,
                )
                y_hist += line_h_hist

        coords_parts: list[str] = []
        if left_iris is not None:
            coords_parts.append(f"L {left_iris[0]:+.2f},{left_iris[1]:+.2f}")
        if right_iris is not None:
            coords_parts.append(f"R {right_iris[0]:+.2f},{right_iris[1]:+.2f}")
        if coords_parts:
            cv2.rectangle(frame, (0, h - bottom_h), (w, h), _C_BAR, -1)
            bar_txt = "   ".join(coords_parts) + f"  |  gaze: {self._gaze_model_label}"
            cv2.putText(
                frame,
                bar_txt,
                (int(12 * sf), h - int(9 * sf)),
                font,
                0.42 * sf,
                _C_MUTED,
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
