"""LocalFlow colour tokens: the single source for the windows (theme.py), the overlay pill and the tray icon.

Plain hex strings, no Qt import, so anything can use them before a QApplication exists.

Neutral graphite surfaces with one accent: a calm "signal blue". Blue keeps the meaning "ready / listening"
(the same family Windows uses for its own mic-in-use cue), and it stays clear of the state colours the app
already relies on: red = recording, amber = busy, teal = game mode.
"""

# surfaces, darkest to lightest
BG = "#0f1013"            # window background (content area)
SIDEBAR = "#131418"       # sidebar
SURFACE = "#17181c"       # panels, setting groups, reader
SURFACE2 = "#1f2025"      # controls (buttons, combos), selected nav item
SURFACE3 = "#282a30"      # control hover
BORDER = "#24262b"        # hairlines, card outlines
BORDER_STRONG = "#31333a" # control borders, popups

# text
TEXT = "#ececef"          # primary
TEXT2 = "#a2a5ae"         # secondary: descriptions, labels (7.2:1 on SURFACE)
TEXT3 = "#868993"         # tertiary: timing, meta (>= 4.6:1 on BG, SURFACE and SURFACE2)

# the one accent
ACCENT = "#2d6be0"        # fills: primary buttons, switches, checkboxes (white text 4.9:1)
ACCENT_HOVER = "#3877ea"
ACCENT_PRESSED = "#285fc6"
ACCENT_TEXT = "#86b0ff"   # accent used as text / icons on dark surfaces
ACCENT_SOFT = "rgba(45,107,224,0.20)"   # selection, checked key chips

# states
DANGER = "#f2727a"
DANGER_FILL = "#cc3b44"   # solid destructive button (white text 4.9:1)
DANGER_FILL_HOVER = "#d6444d"
SUCCESS = "#4cc38a"
WARN = "#e2a53b"          # busy / loading
REC = "#e5484d"           # meeting recording, recording dot
ERROR = "#ef4444"         # tray icon error state
OVERLAY_ACCENT = "#4d8df5"   # overlay pill: hands-free wave start, info dot (a step brighter than ACCENT)
OVERLAY_LIGHT = "#96c4ff"    # overlay pill: hands-free wave end, processing dots
GAME = "#2bb3a3"          # game mode
PAUSED = "#6b6e76"

# shape + size tokens (px), used by theme.py
RADIUS_S = 6              # chips, small buttons inside rows
RADIUS_M = 8              # controls: buttons, inputs, combos
RADIUS_L = 12             # cards, panels, dialogs
CONTROL_H = 32            # one height for buttons, inputs and combos
CONTROL_W = 260           # right-column width for combos / line edits on settings rows

# meeting speakers: "You" uses the accent text colour, "Others" a warm sand so the two voices never blur
YOU = ACCENT_TEXT
OTHERS = "#d6ae72"


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
