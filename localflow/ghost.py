"""Gray "ghost" text drawn right of the caret for Tab autocomplete (autocomplete.py).

A frameless, translucent, click-through, never-activating topmost window on the Qt GUI thread. Callers on other
threads use ``show_ghost`` / ``hide_ghost`` (queued signals). Screen positions come in as PHYSICAL pixels (UI
Automation, per-monitor DPI aware) and are mapped to Qt's logical coordinates per monitor.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from typing import Callable, Optional, Sequence

from PySide6.QtCore import QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

GHOST_RGB = (128, 128, 128)
GHOST_ALPHA = 0.6                 # text opacity
FONT_FAMILY = "Segoe UI"
FONT_FROM_CARET = 0.72            # fallback font pixel size = caret (line box) height * this
MONO_FAMILY = "Consolas"
MONO_HINTS = ("mono", "consol", "courier", "cascadia", "menlo", "fira code", "jetbrains", "source code",
              "lucida console")
FONT_TRUST = (0.35, 1.0)          # the field's font size is used when it is this fraction of the caret height
TALL_RECT = 1.6                   # caret rect taller than this x the font height: sit on the bottom of it
MIN_FIT_CHARS = 1

_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_TOPMOST = 0x00000008
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000
GHOST_EX_STYLE = _WS_EX_NOACTIVATE | _WS_EX_TRANSPARENT | _WS_EX_LAYERED | _WS_EX_TOOLWINDOW
_SWP_NOSIZE, _SWP_NOMOVE, _SWP_NOZORDER, _SWP_NOACTIVATE, _SWP_FRAMECHANGED = 0x1, 0x2, 0x4, 0x10, 0x20


# ----------------------------------------------------------------------------- pure helpers (unit tested)
def screen_for_phys(px: float, py: float, screens: Sequence[tuple]) -> Optional[tuple]:
    """screens: (x, y, w, h, dpr) in Qt logical units; Qt keeps each screen's top-left at its native position and
    scales only the size, so the native rect is (x, y, w*dpr, h*dpr)."""
    for s in screens:
        x, y, wd, h, dpr = s
        if x <= px < x + wd * dpr and y <= py < y + h * dpr:
            return s
    return None


def phys_to_logical(px: float, py: float, screens: Sequence[tuple]) -> Optional[tuple[float, float, float, tuple]]:
    """(logical x, logical y, dpr, screen) for a physical screen point, None when it is on no screen."""
    s = screen_for_phys(px, py, screens)
    if s is None:
        return None
    x, y, _w, _h, dpr = s
    return x + (px - x) / dpr, y + (py - y) / dpr, dpr, s


def fit_text(text: str, width: Callable[[str], float], avail: float) -> str:
    """Longest word-boundary prefix of ``text`` whose width fits ``avail`` ('' if not even MIN_FIT_CHARS fit).
    What is shown is exactly what Tab types."""
    if avail <= 0 or not text:
        return ""
    if width(text) <= avail:
        return text
    best = ""
    i = 0
    while True:
        j = text.find(" ", i + 1)
        cand = text if j < 0 else text[:j]
        if width(cand) > avail:
            break
        best = cand
        if j < 0:
            break
        i = j
    best = best.rstrip()
    return best if len(best.strip()) >= MIN_FIT_CHARS else ""


def first_word(text: str) -> str:
    """The guess's first word with its leading space ('' if none)."""
    m = len(text) - len(text.lstrip())
    j = text.find(" ", m)
    return (text if j < 0 else text[:j]).rstrip() if text.strip() else ""


def font_px(caret_h_logical: float) -> int:
    return max(9, min(48, int(round(caret_h_logical * FONT_FROM_CARET))))


def ghost_font_px(caret_h_logical: float, font_pt: float) -> tuple[int, bool]:
    """(pixel size, from the field's font?). The field's point size maps to Qt logical pixels at 96 / 72 on every
    monitor (Qt applies the monitor scale itself); it is trusted only when it fits the caret height, else the
    caret-height estimate is used."""
    if font_pt and font_pt > 0:
        px = font_pt * 96.0 / 72.0
        if FONT_TRUST[0] * caret_h_logical <= px <= FONT_TRUST[1] * caret_h_logical:
            return max(6, min(72, int(round(px)))), True
    return font_px(caret_h_logical), False


def pick_family(name: str, installed) -> str:
    """The field's font family when installed (first match of a CSS-style list), a monospace stand-in for
    monospace fields, else Segoe UI."""
    inst = {str(f).lower(): str(f) for f in installed or ()}
    cands = [c.strip().strip("'\"") for c in (name or "").split(",")]
    for c in cands:
        if c and c.lower() in inst:
            return inst[c.lower()]
    if any(h in c.lower() for c in cands for h in MONO_HINTS):
        return MONO_FAMILY
    return FONT_FAMILY


_families_cache: Optional[frozenset] = None


def _installed_families() -> frozenset:
    global _families_cache
    if _families_cache is None:
        try:
            _families_cache = frozenset(QFontDatabase.families())
        except Exception:
            _families_cache = frozenset()
    return _families_cache


def _qt_screens() -> list[tuple]:
    out = []
    for s in QGuiApplication.screens():
        g = s.geometry()
        out.append((g.x(), g.y(), g.width(), g.height(), float(s.devicePixelRatio() or 1.0)))
    return out


# ----------------------------------------------------------------------------- widget
class GhostOverlay(QWidget):
    _sig_show = Signal(int, str, object, object, float, object)
    _sig_hide = Signal(int)

    def __init__(self, screens: Optional[Callable[[], list]] = None):
        super().__init__(None)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.WindowDoesNotAcceptFocus | Qt.NoDropShadowWindowHint
                            | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self._screens = screens or _qt_screens
        self._listener = None
        self._token = 0
        self._text = ""
        self._font = QFont(FONT_FAMILY)
        self._font.setPixelSize(14)
        self._ttl = QTimer(self)
        self._ttl.setSingleShot(True)
        self._ttl.timeout.connect(lambda: self._on_hide(self._token))
        self._sig_show.connect(self._on_show, Qt.QueuedConnection)
        self._sig_hide.connect(self._on_hide, Qt.QueuedConnection)
        self.native_ex_style = 0          # what _apply_native_styles set (tests)
        self.fail_reason = ""             # why the last layout_for returned None (diagnostics)

    # ---- thread-safe API
    def set_listener(self, listener) -> None:
        self._listener = listener

    def show_ghost(self, token: int, text: str, caret: tuple, field_rect: Optional[tuple], ttl_s: float,
                   font: Optional[tuple] = None) -> None:
        """font: (size in pt, family) of the field text at the caret, or None."""
        self._sig_show.emit(int(token), str(text), tuple(caret), tuple(field_rect) if field_rect else None,
                            float(ttl_s), tuple(font) if font else None)

    def hide_ghost(self, token: int = 0) -> None:
        self._sig_hide.emit(int(token))

    @property
    def shown_text(self) -> str:
        return self._text if self.isVisible() else ""

    # ---- Qt thread
    def layout_for(self, text: str, caret: tuple, field_rect: Optional[tuple], font_spec: Optional[tuple] = None):
        """(logical x, y, w, h, font, shown text) or None. Pure geometry + font metrics; no window changes."""
        cx, cy, ch = caret
        self.fail_reason = ""
        m = phys_to_logical(cx, cy, self._screens())
        if m is None:
            self.fail_reason = "caret off screen"
            return None
        lx, ly, dpr, scr = m
        h_l = ch / dpr
        pt, fam = (font_spec or (0.0, ""))[:2]
        px, trusted = ghost_font_px(h_l, float(pt or 0.0))
        font = QFont(pick_family(fam or "", _installed_families()) if fam else FONT_FAMILY)
        font.setPixelSize(px)
        fm = QFontMetricsF(font)
        sx, sy, sw, sh, _ = scr
        right = sx + sw - 2                                   # monitor right edge (logical)
        if field_rect:
            fr = phys_to_logical(field_rect[2] - 1, cy, self._screens())
            if fr is not None and fr[3] == scr:
                right = min(right, fr[0] - 2)
        x = lx + 1
        shown = fit_text(text, fm.horizontalAdvance, right - x)
        if not shown:
            # caret at the field's right edge (the app will wrap): show the first word past it, up to the monitor edge
            first = first_word(text)
            if first and fm.horizontalAdvance(first) <= sx + sw - 2 - x:
                shown = first
        if not shown:
            self.fail_reason = f"no room ({int(right - x)} px left)"
            return None
        w = int(fm.horizontalAdvance(shown)) + 4
        fh = int(fm.height()) + 1
        if trusted and h_l > TALL_RECT * fm.height():
            h, y = fh, ly + h_l - fh               # tall line box: text sits at its bottom
        else:
            h, y = max(int(round(h_l)), fh), ly    # text centred in the line box (paintEvent)
        return int(round(x)), int(round(y)), w, h, font, shown

    def _on_show(self, token: int, text: str, caret, field_rect, ttl_s: float, font_spec=None) -> None:
        lay = None
        try:
            lay = self.layout_for(text, caret, field_rect, font_spec)
        except Exception:
            log.exception("ghost layout failed")
            self.fail_reason = "error"
        lst = self._listener
        if lay is None:
            if lst is not None:
                lst.ghost_failed(token, getattr(self, "fail_reason", "") or "?")
            return
        x, y, w, h, font, shown = lay
        self._token, self._text, self._font = token, shown, font
        self.setGeometry(x, y, w, h)
        self._apply_native_styles()         # winId() creates the native window: styles go on before it shows
        if not self.isVisible():
            self.show()
        self.update()
        self._ttl.start(int(max(0.5, ttl_s) * 1000))
        if lst is not None and not lst.ghost_shown(token, shown, ttl_s):
            self._on_hide(token)

    def _on_hide(self, token: int) -> None:
        if token and token != self._token:
            return
        self._ttl.stop()
        self._text = ""
        if self.isVisible():
            self.hide()

    def paintEvent(self, _ev) -> None:
        if not self._text:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.setFont(self._font)
        c = QColor(*GHOST_RGB)
        c.setAlphaF(GHOST_ALPHA)
        p.setPen(c)
        fm = QFontMetricsF(self._font)
        base = (self.height() - fm.height()) / 2.0 + fm.ascent()
        p.drawText(QPoint(1, int(round(base))), self._text)
        p.end()

    def _apply_native_styles(self) -> None:
        if sys.platform != "win32" or QGuiApplication.platformName() != "windows":
            self.native_ex_style = GHOST_EX_STYLE         # offscreen tests: record the intent
            return
        try:
            u = ctypes.windll.user32
            get = u.GetWindowLongPtrW
            put = u.SetWindowLongPtrW
            get.restype = ctypes.c_ssize_t
            get.argtypes = [ctypes.c_void_p, ctypes.c_int]
            put.restype = ctypes.c_ssize_t
            put.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
            h = int(self.winId())
            ex = get(h, _GWL_EXSTYLE)
            want = ex | GHOST_EX_STYLE
            if want != ex:
                put(h, _GWL_EXSTYLE, want)
                u.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_uint]
                u.SetWindowPos(h, None, 0, 0, 0, 0,
                               _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_NOACTIVATE | _SWP_FRAMECHANGED)
            self.native_ex_style = want
        except Exception:
            log.debug("ghost native styles failed", exc_info=True)
