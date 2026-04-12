"""Dark, high-contrast UI for readability (BGR for OpenCV).

Neurovox accent: teal #36D1C1 (logo).
"""

from __future__ import annotations

# App / letterbox background
CHAT_APP_BG: tuple[int, int, int] = (28, 30, 34)

# Shell
CHAT_PANEL: tuple[int, int, int] = (32, 34, 40)
CHAT_HEADER: tuple[int, int, int] = (26, 28, 34)
CHAT_DIVIDER: tuple[int, int, int] = (58, 60, 66)

# Keyboard keys
CHAT_KEY_BG: tuple[int, int, int] = (42, 44, 50)
CHAT_KEY_INNER: tuple[int, int, int] = (36, 38, 44)
CHAT_KEY_EDGE: tuple[int, int, int] = (72, 74, 80)

# Text
CHAT_TEXT: tuple[int, int, int] = (248, 248, 250)
CHAT_MUTED: tuple[int, int, int] = (148, 150, 158)

# Primary accent: #36D1C1 → BGR (193, 209, 54)
CHAT_ACCENT: tuple[int, int, int] = (193, 209, 54)
# Softer teal for fills / secondary emphasis
CHAT_ACCENT_SOFT: tuple[int, int, int] = (120, 175, 85)

# Text on accent fills (badges on bright teal)
CHAT_ON_ACCENT: tuple[int, int, int] = (28, 30, 34)

# Suggestion cards
CHAT_SURFACE: tuple[int, int, int] = (38, 40, 46)
CHAT_CARD_BORDER: tuple[int, int, int] = (68, 70, 78)
CHAT_CARD_HI: tuple[int, int, int] = (48, 50, 58)

# Status / progress
CHAT_SUCCESS: tuple[int, int, int] = (140, 220, 160)

# Optional key highlights (match accent)
RESET_BTN_FILL: tuple[int, int, int] = (70, 120, 55)
RESET_BTN_BORDER: tuple[int, int, int] = CHAT_ACCENT
RESET_BTN_TEXT: tuple[int, int, int] = CHAT_TEXT
RESET_BTN_HI: tuple[int, int, int] = (90, 145, 70)

# Back-compat for imports (legacy Neurovox names → chat palette)
BRAND_OFF_WHITE = CHAT_APP_BG
BRAND_TEAL = CHAT_ACCENT
BRAND_TEXT_DARK = CHAT_TEXT
BRAND_ACCENT_SOFT = CHAT_ACCENT_SOFT
BRAND_BAR = CHAT_DIVIDER
BRAND_MINT = CHAT_KEY_BG
BRAND_PANEL = CHAT_PANEL
