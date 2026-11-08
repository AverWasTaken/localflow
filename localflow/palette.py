"""LocalFlow colour tokens: the single source for the windows (theme.py), the overlay pill and the tray icon.

Plain hex strings, no Qt import, so anything can use them before a QApplication exists.

Neutral graphite surfaces with one accent: a calm "signal blue". Blue keeps the meaning "ready / listening"
(the same family Windows uses for its own mic-in-use cue), and it stays clear of the state colours the app
already relies on: red = recording, amber = busy, teal = game mode.
"""

# surfaces, darkest to lightest
BG = "#16171a"            # window background
SIDEBAR = "#1b1c20"       # sidebar + bottom bar
SURFACE = "#202125"       # panels, setting groups
SURFACE2 = "#2a2b30"      # controls (buttons, combos)
SURFACE3 = "#34353b"      # control hover
BORDER = "#2f3035"        # hairlines
BORDER_STRONG = "#3f4148" # control borders on hover, popups

# text
TEXT = "#e8e9ec"          # primary
TEXT2 = "#a4a7af"         # secondary: descriptions, labels
TEXT3 = "#80838b"         # tertiary: timing, meta (still >= 4.5:1 on BG)

# the one accent
ACCENT = "#2f72e0"        # fills: primary buttons, switches, checkboxes (white text 4.8:1)
ACCENT_HOVER = "#3d80ec"
ACCENT_PRESSED = "#2862c4"
ACCENT_TEXT = "#7aaaff"   # accent used as text on dark surfaces
ACCENT_SOFT = "rgba(47,114,224,0.18)"   # selection / selected nav item

# states
DANGER = "#f06b73"
SUCCESS = "#4cc38a"
WARN = "#e2a53b"          # busy / loading
REC = "#e5484d"           # meeting recording, recording dot
ERROR = "#ef4444"         # tray icon error state
OVERLAY_ACCENT = "#4d8df5"   # overlay pill: hands-free wave start, info dot (a step brighter than ACCENT)
OVERLAY_LIGHT = "#96c4ff"    # overlay pill: hands-free wave end, processing dots
GAME = "#2bb3a3"          # game mode
PAUSED = "#6b6e76"

# meeting speakers: "You" uses the accent text colour, "Others" a warm sand so the two voices never blur
YOU = ACCENT_TEXT
OTHERS = "#d6ae72"


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
