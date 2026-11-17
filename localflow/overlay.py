"""Floating dictation overlay: a small dark pill with a live waveform.

Public methods (set_state / push_level / reposition) are safe to call from any
thread. The widget never takes focus or mouse input (Win32 NOACTIVATE +
TRANSPARENT extended styles on top of the Qt window flags).
"""
from __future__ import annotations

import collections
import ctypes
import math
import sys
import time

from PySide6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, QTimer,
                            QVariantAnimation, Signal)
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPainterPath,
                           QPen, QTextLayout, QTextOption)
from PySide6.QtWidgets import QApplication, QWidget

try:  # Windows only; everything degrades gracefully without it
    import win32api
    import win32con
    import win32gui
except Exception:  # pragma: no cover
    win32api = win32con = win32gui = None

VALID_STATES = ("hidden", "recording", "handsfree", "processing", "error", "info", "answer")

PILL_H = 38.0
MARGIN = 22.0            # transparent canvas margin around the pill (room for the shadow)
MAX_PILL_W = 360.0
BOTTOM_OFFSET = 48       # px between the resting pill's bottom and the bottom of the work area

# answer card (command mode questions): sits under the pill; the pill lifts to make room
CARD_MIN_W = 220.0
CARD_MAX_W = 440.0
CARD_MAX_H = 176.0       # padding + question line + 6 answer lines (Segoe needs ~170) + headroom
CARD_R = 14.0
CARD_PAD_X = 16.0
CARD_PAD_T = 12.0
CARD_PAD_B = 14.0
CARD_Q_GAP = 5.0         # question line -> answer
CARD_LINE_EXTRA = 2.0    # extra leading between answer lines
CARD_MAX_LINES = 6
CARD_Q_MAX_W = 280.0     # a long question may widen the card only this far (text area)
CARD_GAP = 8.0           # pill bottom -> card top when lifted
CARD_RISE = 8.0          # the card rises this many px into place
LIFT_IN_MS = 280
LIFT_OUT_MS = 200
SWAP_MS = 240
ANSWER_LABEL = "Answer"
ANSWER_DEFAULT_MS = 6000

# fixed transparent canvas; anchored by its bottom so the resting pill never moves on screen
CANVAS_W = int(max(MAX_PILL_W, CARD_MAX_W) + 2 * MARGIN)
CANVAS_H = int(MARGIN + PILL_H + CARD_GAP + CARD_MAX_H + MARGIN)
REST_Y = CANVAS_H - MARGIN - PILL_H     # pill top (canvas coords) when not lifted
N_BARS = 24
SCROLL_PERIOD = 0.045    # seconds per bar step

WIDTHS = {"recording": 150.0, "handsfree": 178.0, "processing": 86.0}
LABEL_MAX_W = 150.0      # style label inside the recording pill ("casual · Discord")
ERROR_SECS = 2.5
INFO_SECS = 1.8

# win32 extended styles
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000
_HWND_TOPMOST = -1
_SWP_NOSIZE, _SWP_NOMOVE, _SWP_NOZORDER, _SWP_NOACTIVATE, _SWP_FRAMECHANGED = 0x1, 0x2, 0x4, 0x10, 0x20

# colours follow palette.py (accent blue, light accent, recording red)
from .palette import OVERLAY_ACCENT, OVERLAY_LIGHT, REC, TEXT, TEXT2  # noqa: E402

ACCENT_C = QColor(OVERLAY_ACCENT)   # hands-free wave start, info dot
LIGHT_C = QColor(OVERLAY_LIGHT)     # hands-free wave end, processing dots
VIOLET = ACCENT_C                  # older names
BLUE = LIGHT_C
REC_RED = QColor(REC)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _out_cubic(x: float) -> float:
    return 1.0 - (1.0 - x) ** 3


def _in_out_cubic(x: float) -> float:
    return 4.0 * x ** 3 if x < 0.5 else 1.0 - (-2.0 * x + 2.0) ** 3 / 2.0


def _compress(raw: float) -> float:
    """Raw RMS (speech ~0.02-0.2) -> 0..1 bar height using a dB-ish curve."""
    if raw <= 1e-5:
        return 0.0
    db = 20.0 * math.log10(raw)
    v = (db + 52.0) / 40.0          # -52 dB -> 0, -12 dB -> 1
    return max(0.0, min(1.0, v)) ** 1.1


class _User32:
    """Thin ctypes wrapper with correct 64-bit signatures."""

    def __init__(self):
        u = ctypes.windll.user32
        self.get = u.GetWindowLongPtrW if hasattr(u, "GetWindowLongPtrW") else u.GetWindowLongW
        self.set = u.SetWindowLongPtrW if hasattr(u, "SetWindowLongPtrW") else u.SetWindowLongW
        self.get.restype = ctypes.c_ssize_t
        self.get.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.set.restype = ctypes.c_ssize_t
        self.set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
        self.pos = u.SetWindowPos
        self.pos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        self.pos.restype = ctypes.c_int


class _Card:
    """Laid-out answer card, kept as plain strings so an outgoing card paints from its own state."""
    __slots__ = ("question", "answer", "w", "h", "q_line", "lines", "text_a")

    def __init__(self, question: str, answer: str, w: float, h: float, q_line: str, lines: list,
                 text_a: float = 1.0):
        self.question, self.answer = question, answer
        self.w, self.h = w, h
        self.q_line, self.lines = q_line, lines
        self.text_a = text_a       # < 1 only for a card frozen mid-swap


class Overlay(QWidget):
    _sig_state = Signal(str, str)
    _sig_answer = Signal(str, str, int)
    _sig_reposition = Signal()

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.WindowDoesNotAcceptFocus | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFixedSize(CANVAS_W, CANVAS_H)
        self.setWindowOpacity(0.0)

        # --- state ---
        self._state = "hidden"
        self._mode = "recording"       # what is painted (kept while fading out)
        self._text = ""
        self._hiding = False
        self._pw = WIDTHS["recording"]  # current pill width
        self._levels = collections.deque(maxlen=512)   # filled from any thread
        self._cur = 0.0
        self._target = 0.0
        self._hist = collections.deque([0.0] * N_BARS, maxlen=N_BARS)
        self._acc = 0.0
        self._last_tick = time.perf_counter()
        self._t0 = time.perf_counter()
        self._native_done = False
        self._u32 = None

        self._font = QFont("Segoe UI Variable Text")
        self._font.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Arial"])
        self._font.setPointSizeF(9.5)
        self._font.setWeight(QFont.Weight.Medium)
        self._label_font = QFont(self._font)
        self._label_font.setPointSizeF(8.5)
        self._label_font.setWeight(QFont.Weight.Normal)
        self._q_font = QFont(self._label_font)          # answer card: question echo
        self._a_font = QFont(self._font)                # answer card: the answer
        self._a_font.setPointSizeF(10.5)
        self._a_font.setWeight(QFont.Weight.Normal)

        # answer card
        self._card = None          # current _Card (kept while it collapses / fades out)
        self._card_prev = None     # outgoing _Card during a swap
        self._lift = 0.0           # pill lift: 0 = at rest, 1 = raised above the card
        self._card_a = 0.0         # card opacity
        self._card_rise = CARD_RISE  # px the card still sits below its resting place
        self._lift_dir = "in"
        self._lift_from = (0.0, 0.0, CARD_RISE)
        self._swap = 1.0           # 0 -> 1 swap from _card_prev to _card (linear progress)

        # --- timers / animations ---
        self._frame = QTimer(self)
        self._frame.setInterval(16)
        self._frame.setTimerType(Qt.PreciseTimer)
        self._frame.timeout.connect(self._tick)

        self._auto = QTimer(self)
        self._auto.setSingleShot(True)
        self._auto.timeout.connect(self._begin_hide)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.finished.connect(self._fade_finished)

        self._wanim = QVariantAnimation(self)
        self._wanim.setDuration(200)
        self._wanim.setEasingCurve(QEasingCurve.OutCubic)
        self._wanim.valueChanged.connect(self._on_width)

        self._lanim = QVariantAnimation(self)    # pill lift + card reveal / collapse
        self._lanim.valueChanged.connect(self._on_lift)
        self._lanim.finished.connect(self._lift_finished)

        self._xanim = QVariantAnimation(self)    # answer -> answer swap
        self._xanim.setDuration(SWAP_MS)
        self._xanim.setEasingCurve(QEasingCurve.Linear)     # curves applied in _card_size / _swap_alphas
        self._xanim.valueChanged.connect(self._on_swap)
        self._xanim.finished.connect(self._swap_finished)

        self._sig_state.connect(self._apply_state)
        self._sig_answer.connect(self._apply_answer)
        self._sig_reposition.connect(self._do_reposition)

        # create the native window now so we can set the Win32 styles before first show
        self.winId()
        self._apply_native_styles()

    # ------------------------------------------------------------------ public API
    def set_state(self, state: str, text: str = "") -> None:
        """Thread-safe. state: hidden|recording|handsfree|processing|error|info|answer.
        For recording/handsfree/processing, ``text`` is an optional small label shown in the pill ("casual · Discord",
        "Command", "Working…"). Any state other than "answer" collapses a visible answer card."""
        self._sig_state.emit(str(state), str(text))

    def show_answer(self, question: str, answer: str, duration_ms: int) -> None:
        """Thread-safe. Show ``answer`` in a card under the pill (``question`` as a muted line above it) and hide
        after ``duration_ms``. Works from hidden, and replaces a card that is already showing."""
        try:
            ms = int(duration_ms)
        except Exception:
            ms = ANSWER_DEFAULT_MS
        self._sig_answer.emit(str(question or ""), str(answer or ""), ms)

    def push_level(self, level: float) -> None:
        """Thread-safe, lock-free. level = raw RMS (0..1) of the latest audio block."""
        try:
            self._levels.append(float(level))
        except Exception:
            pass

    def reposition(self) -> None:
        """Thread-safe. Move to the bottom-center of the foreground window's monitor."""
        self._sig_reposition.emit()

    # ------------------------------------------------------------------ native bits
    def _hwnd(self) -> int:
        return int(self.winId())

    def _apply_native_styles(self) -> None:
        if sys.platform != "win32":
            return
        try:
            if self._u32 is None:
                self._u32 = _User32()
            u = self._u32
            h = self._hwnd()
            ex = u.get(h, _GWL_EXSTYLE)
            want = ex | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW | _WS_EX_TRANSPARENT | _WS_EX_LAYERED
            if want != ex:
                u.set(h, _GWL_EXSTYLE, want)
                u.pos(h, None, 0, 0, 0, 0,
                      _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_NOACTIVATE | _SWP_FRAMECHANGED)
            self._native_done = True
        except Exception:
            pass

    def _assert_topmost(self) -> None:
        if sys.platform != "win32":
            return
        try:
            if self._u32 is None:
                self._u32 = _User32()
            self._u32.pos(self._hwnd(), _HWND_TOPMOST, 0, 0, 0, 0,
                          _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE)
        except Exception:
            pass

    # ------------------------------------------------------------------ positioning
    def _target_screen(self):
        app = QApplication.instance()
        screens = app.screens()
        if win32gui is not None and win32api is not None:
            try:
                fg = win32gui.GetForegroundWindow()
                if fg:
                    mon = win32api.MonitorFromWindow(fg, win32con.MONITOR_DEFAULTTONEAREST)
                    l, t, r, b = win32api.GetMonitorInfo(mon)["Monitor"]
                    for s in screens:                       # Qt keeps native top-left origins
                        g = s.geometry()
                        if g.x() == l and g.y() == t:
                            return s
                    cx, cy = (l + r) // 2, (t + b) // 2
                    for s in screens:
                        if s.geometry().contains(cx, cy):
                            return s
            except Exception:
                pass
        return app.primaryScreen()

    def _do_reposition(self) -> None:
        scr = self._target_screen()
        if scr is None:
            return
        av = scr.availableGeometry()
        x = av.x() + (av.width() - CANVAS_W) // 2
        y = av.y() + av.height() - BOTTOM_OFFSET + int(MARGIN) - CANVAS_H   # resting pill bottom 48px up
        if self.x() != x or self.y() != y:
            self.move(x, y)

    # ------------------------------------------------------------------ state machine
    def _text_width(self, text: str) -> float:
        return QFontMetricsF(self._font).horizontalAdvance(text)

    def _elided(self, text: str, width: float) -> str:
        return QFontMetricsF(self._font).elidedText(text, Qt.ElideRight, int(width))

    def _label_width(self, text: str) -> float:
        if not text:
            return 0.0
        return min(LABEL_MAX_W, self._label_metrics_width(text)) + 12.0

    def _label_metrics_width(self, text: str) -> float:
        return QFontMetricsF(self._label_font).horizontalAdvance(text)

    def _target_width(self, state: str, text: str) -> float:
        if state in ("recording", "handsfree", "processing"):    # optional label ("casual · Discord", "Working…")
            return WIDTHS[state] + self._label_width(text)
        if state in WIDTHS:
            return WIDTHS[state]
        # error/info: icon area (34) + text + right pad (18)
        w = 34.0 + self._text_width(text) + 18.0
        return max(110.0, min(MAX_PILL_W, w))

    def _apply_state(self, state: str, text: str) -> None:
        if state not in VALID_STATES:
            return
        if state == "answer":                    # bare set_state("answer", text): a card without a question line
            self._apply_answer("", text, ANSWER_DEFAULT_MS)
            return
        self._auto.stop()
        if state == "hidden":
            self._begin_hide()                   # a card fades out with the window
            return
        if state in ("error", "info") and not text:
            text = "Error" if state == "error" else ""
        self._collapse_card()
        self._enter(state, text)
        if state == "error":
            self._auto.start(int(ERROR_SECS * 1000))
        elif state == "info":
            self._auto.start(int(INFO_SECS * 1000))

    def _apply_answer(self, question: str, answer: str, duration_ms: int) -> None:
        answer = " ".join(str(answer).split())
        question = " ".join(str(question).split())
        if not answer:
            return
        self._auto.stop()
        card = self._layout_card(question, answer)
        if self.isVisible() and self._card is not None and (self._lift > 0.01 or self._card_a > 0.01):
            # replace: old text fades out, the card (and the pill above it) eases to the new size, new text fades in
            self._card_prev = self._snapshot_card()
            self._card = card
            self._xanim.stop()
            self._swap = 0.0
            self._xanim.setStartValue(0.0)
            self._xanim.setEndValue(1.0)
            self._xanim.start()
        else:
            self._xanim.stop()
            self._card_prev = None
            self._card = card
            self._swap = 1.0
            self._lift, self._card_a, self._card_rise = 0.0, 0.0, CARD_RISE
        self._enter("answer", ANSWER_LABEL)
        if self._lift < 1.0 or self._card_a < 1.0 or self._lanim.state() == QVariantAnimation.Running:
            self._start_lift("in")
        self._auto.start(max(1000, int(duration_ms)))

    def _collapse_card(self) -> None:
        """A non-answer state arrived: fade the card out and glide the pill back down."""
        if self._card is not None and self._swap < 1.0:
            self._xanim.stop()
            self._card = self._snapshot_card()   # freeze what is on screen right now
            self._card_prev = None
            self._swap = 1.0
        running = self._lanim.state() == QVariantAnimation.Running
        if self._lift <= 0.0 and self._card_a <= 0.0 and not running:
            return
        if running and self._lift_dir == "out":
            return
        self._start_lift("out")

    def _start_lift(self, direction: str) -> None:
        """One linear 0->1 progress drives the pill lift and the card fade/rise (curves in _on_lift).
        Starts from whatever is on screen, so reversing mid-way never jumps."""
        self._lanim.stop()
        self._lift_dir = direction
        self._lift_from = (self._lift, self._card_a, self._card_rise)
        self._lanim.setDuration(LIFT_IN_MS if direction == "in" else LIFT_OUT_MS)
        self._lanim.setEasingCurve(QEasingCurve.Linear)
        self._lanim.setStartValue(0.0)
        self._lanim.setEndValue(1.0)
        self._lanim.start()

    def _enter(self, state: str, text: str) -> None:
        """Show the pill in ``state``, fading the window in if needed."""
        was_visible = self.isVisible() and not self._hiding and self.windowOpacity() > 0.02
        prev = self._mode if was_visible else None
        self._state = state
        self._mode = state
        self._text = text
        self._hiding = False
        tw = self._target_width(state, text)
        self._do_reposition()

        if state in ("recording", "handsfree") and prev not in ("recording", "handsfree"):
            self._reset_wave()

        if not was_visible:
            self._wanim.stop()
            self._pw = tw
            self._fade.stop()
            self._fade.setDuration(140)
            self._fade.setEasingCurve(QEasingCurve.OutCubic)
            self._fade.setStartValue(self.windowOpacity())
            self._fade.setEndValue(1.0)
            self.show()                    # never activate()
            self._assert_topmost()
            self._fade.start()
        else:
            self._fade.stop()
            if self.windowOpacity() < 1.0:
                self._fade.setDuration(100)
                self._fade.setStartValue(self.windowOpacity())
                self._fade.setEndValue(1.0)
                self._fade.start()
            self._wanim.stop()
            if abs(tw - self._pw) > 0.5:
                self._wanim.setStartValue(float(self._pw))
                self._wanim.setEndValue(float(tw))
                self._wanim.start()
            self._assert_topmost()

        if state in ("recording", "handsfree", "processing"):
            self._last_tick = time.perf_counter()
            if not self._frame.isActive():
                self._frame.start()
        else:
            self._frame.stop()
        self.update()

    def _begin_hide(self) -> None:
        self._auto.stop()
        self._state = "hidden"
        if not self.isVisible():
            self._frame.stop()
            return
        self._hiding = True
        self._fade.stop()
        self._fade.setDuration(180)
        self._fade.setEasingCurve(QEasingCurve.InCubic)
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _fade_finished(self) -> None:
        if self._hiding and self._state == "hidden":
            self._hiding = False
            self._frame.stop()
            self._wanim.stop()
            self.hide()
            self.setWindowOpacity(0.0)
            self._levels.clear()
            self._lanim.stop()
            self._xanim.stop()
            self._lift, self._card_a, self._card_rise = 0.0, 0.0, CARD_RISE
            self._swap = 1.0
            self._card = self._card_prev = None

    def _on_width(self, v) -> None:
        self._pw = float(v)
        self.update()

    def _on_lift(self, v) -> None:
        """Sequenced so the pill never slides across a visible card.
        in:  the pill glides up over the first 60% (OutCubic); the card fades in and rises CARD_RISE px from 35%.
        out: the card fades and sinks over the first 40%; the pill glides down from 25% (InOutCubic)."""
        u = max(0.0, min(1.0, float(v)))
        l0, a0, r0 = self._lift_from
        if self._lift_dir == "in":
            self._lift = l0 + (1.0 - l0) * _out_cubic(min(1.0, u / 0.6))
            delay = 0.35 * (1.0 - a0)
            c = max(0.0, min(1.0, (u - delay) / (1.0 - delay)))
            self._card_a = a0 + (1.0 - a0) * c
            self._card_rise = r0 * (1.0 - _out_cubic(c))
        else:
            f = min(1.0, u / 0.4)
            self._card_a = a0 * (1.0 - f)
            self._card_rise = r0 + (CARD_RISE - r0) * f
            delay = 0.25 * a0
            m = max(0.0, min(1.0, (u - delay) / (1.0 - delay)))
            self._lift = l0 * (1.0 - _in_out_cubic(m))
        self.update()

    def _lift_finished(self) -> None:
        if self._lift_dir == "out" and self._mode != "answer":
            self._card = self._card_prev = None
            self.update()

    def _on_swap(self, v) -> None:
        self._swap = float(v)
        self.update()

    def _swap_finished(self) -> None:
        self._swap = 1.0
        self._card_prev = None
        self.update()

    # ------------------------------------------------------------------ answer card layout
    def _layout_card(self, question: str, answer: str) -> _Card:
        qfm = QFontMetricsF(self._q_font)
        afm = QFontMetricsF(self._a_font)
        max_in = CARD_MAX_W - 2 * CARD_PAD_X
        min_in = CARD_MIN_W - 2 * CARD_PAD_X
        q_want = min(CARD_Q_MAX_W, qfm.horizontalAdvance(question) + 1.0) if question else 0.0
        inner = max(min_in, min(max_in, math.ceil(max(q_want, afm.horizontalAdvance(answer) + 1.0))))

        # as many answer lines as the fixed canvas allows (never more than CARD_MAX_LINES)
        line_h = afm.lineSpacing() + CARD_LINE_EXTRA
        q_block = (qfm.height() + CARD_Q_GAP) if question else 0.0
        room = CARD_MAX_H - CARD_PAD_T - CARD_PAD_B - q_block
        max_lines = max(1, min(CARD_MAX_LINES, int((room - afm.height()) // line_h) + 1))
        lines = self._wrap(answer, inner, max_lines)
        q_line = qfm.elidedText(question, Qt.ElideRight, int(inner)) if question else ""
        h = CARD_PAD_T + q_block + afm.height() + (len(lines) - 1) * line_h + CARD_PAD_B
        return _Card(question, answer, inner + 2 * CARD_PAD_X, min(CARD_MAX_H, float(math.ceil(h))), q_line, lines)

    def _wrap(self, text: str, width: float, max_lines: int) -> list:
        afm = QFontMetricsF(self._a_font)
        layout = QTextLayout(text, self._a_font)
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(opt)
        u16 = text.encode("utf-16-le", "surrogatepass")  # QTextLayout offsets count UTF-16 units (emoji = 2)

        def part(start: int, n: int | None = None) -> str:
            end = len(u16) if n is None else 2 * (start + n)
            return u16[2 * start:end].decode("utf-16-le", "surrogatepass")

        layout.beginLayout()
        out = []
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            start = line.textStart()
            if len(out) == max_lines - 1:          # last allowed line: take the rest, elide it
                out.append(afm.elidedText(part(start).strip(), Qt.ElideRight, int(width)))
                break
            out.append(part(start, line.textLength()).rstrip())
        layout.endLayout()
        return [ln for ln in out if ln] or [afm.elidedText(text, Qt.ElideRight, int(width))]

    def _snapshot_card(self):
        """The card as drawn right now: mid-swap, the more visible text at the interpolated size."""
        c = self._card
        if c is None or self._card_prev is None or self._swap >= 1.0:
            return c
        w, h = self._card_size()
        old_a, new_a = self._swap_alphas()
        t, a = (c, new_a) if new_a >= old_a else (self._card_prev, old_a)
        return _Card(t.question, t.answer, w, h, t.q_line, t.lines, a * t.text_a)

    def _swap_alphas(self) -> tuple:
        """Old text fades out over the first 40% of a swap, new text fades in over the last 65%."""
        s = self._swap
        return 1.0 - min(1.0, s / 0.4), max(0.0, min(1.0, (s - 0.35) / 0.65))

    def _card_size(self) -> tuple:
        c, pv = self._card, self._card_prev
        if c is None:
            return 0.0, 0.0
        if pv is None or self._swap >= 1.0:
            return c.w, c.h
        k = _out_cubic(self._swap)
        return _lerp(pv.w, c.w, k), _lerp(pv.h, c.h, k)

    def _pill_y(self) -> float:
        if self._card is None or self._lift <= 0.0:
            return float(REST_Y)
        return REST_Y - self._lift * (self._card_size()[1] + CARD_GAP)

    def _reset_wave(self) -> None:
        self._hist = collections.deque([0.0] * N_BARS, maxlen=N_BARS)
        self._cur = self._target = 0.0
        self._acc = 0.0
        self._levels.clear()

    # ------------------------------------------------------------------ animation
    def _tick(self) -> None:
        now = time.perf_counter()
        dt = max(0.001, min(0.1, now - self._last_tick))
        self._last_tick = now
        if self._mode in ("recording", "handsfree"):
            peak = 0.0
            total = 0.0
            n = 0
            while True:
                try:
                    v = self._levels.popleft()
                except IndexError:
                    break
                peak = max(peak, v)
                total += v
                n += 1
            if n:
                raw = 0.65 * peak + 0.35 * (total / n)
                self._target = _compress(raw)
            else:
                self._target *= math.exp(-dt * 6.0)
            # exponential smoothing: fast attack, slower release
            rate = 28.0 if self._target > self._cur else 9.0
            self._cur += (self._target - self._cur) * (1.0 - math.exp(-rate * dt))
            self._acc += dt
            while self._acc >= SCROLL_PERIOD:
                self._acc -= SCROLL_PERIOD
                self._hist.append(self._cur)
        self.update()

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.fillRect(self.rect(), Qt.transparent)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        mode = self._mode
        pw = self._pw
        x0 = (CANVAS_W - pw) / 2.0
        rect = QRectF(x0, self._pill_y(), pw, PILL_H)
        t = time.perf_counter() - self._t0

        card_rect, card_a = self._card_geometry()
        if card_rect is not None:              # both shadows first, so neither smudges the other body
            p.setOpacity(card_a)
            self._paint_shadow(p, card_rect, mode, CARD_R)
            p.setOpacity(1.0)
        self._paint_shadow(p, rect, mode)
        if card_rect is not None:
            p.setOpacity(card_a)
            self._paint_card(p, card_rect)
            p.setOpacity(1.0)
        self._paint_pill(p, rect, mode)

        if mode in ("recording", "handsfree"):
            lw = self._label_width(self._text)
            if mode == "recording":
                self._paint_rec_dot(p, rect, t)
            else:
                self._paint_lock(p, rect)
            self._paint_wave(p, QRectF(rect.left() + 36, rect.top(), rect.width() - 36 - 16 - lw, rect.height()),
                             t, QColor(244, 244, 248) if mode == "recording" else VIOLET,
                             None if mode == "recording" else BLUE)
            if lw:
                self._paint_label(p, rect, lw)
        elif mode == "processing":
            lw = self._label_width(self._text)
            if lw:      # dots on the left, label on the right (command mode: "Working…")
                self._paint_dots(p, QRectF(rect.left(), rect.top(), WIDTHS["processing"], rect.height()), t)
                self._paint_label(p, rect, lw)
            else:
                self._paint_dots(p, rect, t)
        elif mode == "error":
            self._paint_error(p, rect)
        elif mode in ("info", "answer"):
            self._paint_info(p, rect)
        p.end()

    def _paint_shadow(self, p: QPainter, rect: QRectF, mode: str, radius: float = PILL_H / 2) -> None:
        p.setPen(Qt.NoPen)
        layers = 9
        for i in range(layers):
            grow = (layers - i) * 1.5
            a = 3 + i * 0.6
            r = rect.adjusted(-grow, -grow + 3, grow, grow + 3)
            p.setBrush(QColor(0, 0, 0, int(a)))
            p.drawRoundedRect(r, radius + grow, radius + grow)

    def _paint_pill(self, p: QPainter, rect: QRectF, mode: str) -> None:
        if mode == "error":
            top, bot, border = QColor(84, 26, 32, 240), QColor(62, 18, 24, 240), QColor(255, 110, 110, 120)
        elif mode == "handsfree":
            top, bot, border = QColor(24, 31, 46, 240), QColor(17, 21, 32, 240), QColor(122, 170, 255, 95)
        else:
            top, bot, border = QColor(32, 33, 37, 238), QColor(22, 23, 26, 238), QColor(255, 255, 255, 36)
        g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        g.setColorAt(0, top)
        g.setColorAt(1, bot)
        p.setBrush(g)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(rect, PILL_H / 2, PILL_H / 2)
        # 1px border + faint top highlight
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(border, 1.0))
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), PILL_H / 2 - 0.5, PILL_H / 2 - 0.5)

    # ------------------------------------------------------------------ answer card painting
    def _card_geometry(self) -> tuple:
        """Card rect (canvas coords) and opacity for the current lift, or (None, 0) when nothing shows."""
        alpha = max(0.0, min(1.0, self._card_a))
        if self._card is None or alpha <= 0.001:
            return None, 0.0
        w, h = self._card_size()
        bottom = CANVAS_H - MARGIN + self._card_rise
        return QRectF((CANVAS_W - w) / 2.0, bottom - h, w, h), alpha

    def _paint_card(self, p: QPainter, rect: QRectF) -> None:
        # same surface as the neutral pill: graphite gradient + hairline
        g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        g.setColorAt(0, QColor(32, 33, 37, 238))
        g.setColorAt(1, QColor(22, 23, 26, 238))
        p.setBrush(g)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(rect, CARD_R, CARD_R)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 36), 1.0))
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), CARD_R - 0.5, CARD_R - 0.5)

        clip = QPainterPath()
        clip.addRoundedRect(rect.adjusted(1, 1, -1, -1), CARD_R - 1, CARD_R - 1)
        p.save()
        p.setClipPath(clip)
        base = p.opacity()
        new_a = 1.0
        if self._card_prev is not None and self._swap < 1.0:
            old_a, new_a = self._swap_alphas()
            if old_a > 0.0:
                p.setOpacity(base * old_a * self._card_prev.text_a)
                self._paint_card_text(p, rect, self._card_prev)
        p.setOpacity(base * new_a * self._card.text_a)
        self._paint_card_text(p, rect, self._card)
        p.restore()

    def _paint_card_text(self, p: QPainter, rect: QRectF, card: _Card) -> None:
        qfm = QFontMetricsF(self._q_font)
        afm = QFontMetricsF(self._a_font)
        x = rect.left() + CARD_PAD_X
        y = rect.top() + CARD_PAD_T
        if card.q_line:
            p.setFont(self._q_font)
            p.setPen(QColor(TEXT2))
            p.drawText(QPointF(x, y + qfm.ascent()), card.q_line)
            y += qfm.height() + CARD_Q_GAP
        p.setFont(self._a_font)
        p.setPen(QColor(TEXT))
        line_h = afm.lineSpacing() + CARD_LINE_EXTRA
        for i, ln in enumerate(card.lines):
            p.drawText(QPointF(x, y + afm.ascent() + i * line_h), ln)

    def _paint_rec_dot(self, p: QPainter, rect: QRectF, t: float) -> None:
        c = QPointF(rect.left() + 20, rect.center().y())
        pulse = 0.5 + 0.5 * math.sin(t * 3.2)
        glow = QColor(REC_RED)
        glow.setAlpha(int(40 + 40 * pulse))
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(c, 7.5 + 1.2 * pulse, 7.5 + 1.2 * pulse)
        p.setBrush(REC_RED)
        p.drawEllipse(c, 4.4, 4.4)
        p.setBrush(QColor(255, 255, 255, 90))
        p.drawEllipse(QPointF(c.x() - 1.1, c.y() - 1.3), 1.3, 1.3)

    def _paint_lock(self, p: QPainter, rect: QRectF) -> None:
        cx, cy = rect.left() + 20, rect.center().y()
        col = QColor(160, 196, 255)
        pen = QPen(col, 1.7)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        # shackle
        sh = QPainterPath()
        sh.moveTo(cx - 3.2, cy - 0.5)
        sh.lineTo(cx - 3.2, cy - 3.0)
        sh.arcTo(QRectF(cx - 3.2, cy - 7.4, 6.4, 6.8), 180, -180)
        sh.lineTo(cx + 3.2, cy - 0.5)
        p.drawPath(sh)
        # body
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawRoundedRect(QRectF(cx - 5.6, cy - 1.0, 11.2, 8.6), 2.2, 2.2)
        p.setBrush(QColor(24, 31, 46))
        p.drawEllipse(QPointF(cx, cy + 2.6), 1.2, 1.2)
        p.drawRect(QRectF(cx - 0.55, cy + 2.8, 1.1, 2.4))

    def _bar_values(self, t: float) -> list[float]:
        frac = max(0.0, min(1.0, self._acc / SCROLL_PERIOD))
        pts = list(self._hist) + [self._cur]
        out = []
        for i in range(N_BARS):
            v = _lerp(pts[i], pts[i + 1], frac)
            breath = 0.075 + 0.05 * math.sin(t * 2.3 + i * 0.5)
            out.append(breath + v * (1.0 - breath))
        return out

    def _paint_wave(self, p: QPainter, area: QRectF, t: float, c1: QColor, c2: QColor | None) -> None:
        vals = self._bar_values(t)
        pitch = area.width() / N_BARS
        bw = pitch * 0.56
        max_h = PILL_H - 14
        min_h = 3.0
        cy = area.center().y()
        p.setPen(Qt.NoPen)
        for i, v in enumerate(vals):
            taper = 0.55 + 0.45 * math.sin(math.pi * (i + 0.5) / N_BARS)
            h = min_h + (max_h - min_h) * min(1.0, v * taper * 1.05)
            x = area.left() + i * pitch + (pitch - bw) / 2
            if c2 is None:
                col = QColor(c1)
                col.setAlpha(int(120 + 135 * min(1.0, v * 1.2)))
            else:
                k = i / (N_BARS - 1)
                col = QColor(int(_lerp(c1.red(), c2.red(), k)), int(_lerp(c1.green(), c2.green(), k)),
                             int(_lerp(c1.blue(), c2.blue(), k)), int(150 + 105 * min(1.0, v * 1.2)))
            p.setBrush(col)
            p.drawRoundedRect(QRectF(x, cy - h / 2, bw, h), bw / 2, bw / 2)

    def _paint_dots(self, p: QPainter, rect: QRectF, t: float) -> None:
        cy = rect.center().y()
        cx = rect.center().x()
        p.setPen(Qt.NoPen)
        for i in range(3):
            ph = 0.5 + 0.5 * math.sin(t * 6.0 - i * 0.95)
            r = 2.6 + 1.7 * ph
            col = QColor(LIGHT_C)
            col.setAlpha(int(90 + 165 * ph))
            p.setBrush(col)
            p.drawEllipse(QPointF(cx + (i - 1) * 15, cy - 1.6 * (ph - 0.5)), r, r)

    def _paint_error(self, p: QPainter, rect: QRectF) -> None:
        c = QPointF(rect.left() + 19, rect.center().y())
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 107, 107))
        p.drawEllipse(c, 7.2, 7.2)
        p.setPen(QPen(QColor(62, 18, 24), 1.8, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(c.x(), c.y() - 3.3), QPointF(c.x(), c.y() + 0.9))
        p.drawPoint(QPointF(c.x(), c.y() + 3.6))
        self._paint_text(p, rect, QColor(255, 226, 226))

    def _paint_info(self, p: QPainter, rect: QRectF) -> None:
        c = QPointF(rect.left() + 19, rect.center().y())
        p.setPen(Qt.NoPen)
        glow = QColor(VIOLET)
        glow.setAlpha(60)
        p.setBrush(glow)
        p.drawEllipse(c, 7, 7)
        p.setBrush(VIOLET)
        p.drawEllipse(c, 3.6, 3.6)
        self._paint_text(p, rect, QColor(236, 236, 242))

    def _paint_label(self, p: QPainter, rect: QRectF, lw: float) -> None:
        """Small muted style label at the right end of the recording pill."""
        p.setFont(self._label_font)
        p.setPen(QColor(222, 224, 232))
        box = QRectF(rect.right() - 16 - (lw - 12.0), rect.top(), lw - 12.0, rect.height())
        txt = QFontMetricsF(self._label_font).elidedText(self._text, Qt.ElideRight, int(box.width()))
        p.drawText(box, int(Qt.AlignVCenter | Qt.AlignRight), txt)

    def _paint_text(self, p: QPainter, rect: QRectF, color: QColor) -> None:
        p.setFont(self._font)
        p.setPen(color)
        tr = QRectF(rect.left() + 34, rect.top(), rect.width() - 34 - 16, rect.height())
        txt = self._elided(self._text, tr.width())
        p.drawText(tr, Qt.AlignVCenter | Qt.AlignLeft, txt)


# ---------------------------------------------------------------------- demo
if __name__ == "__main__":
    import threading

    app = QApplication(sys.argv)
    ov = Overlay()
    seq = [("recording", ""), ("handsfree", ""), ("processing", ""), ("error", "Microphone not found"),
           ("info", "Paused"), ("recording", "Command"), ("processing", "Working…"),
           ("answer", "How many hours is it from 7:30am to 5pm?", "9 hours 30 minutes (9.5 hours)"),
           ("answer", "Who wrote Pride and Prejudice?",
            "Jane Austen. It was published in 1813 as her second novel, after Sense and Sensibility."),
           ("recording", ""), ("hidden", "")]
    stop = False

    def feeder():  # simulates the audio thread
        t0 = time.time()
        while not stop:
            t = time.time() - t0
            env = max(0.0, math.sin(t * 1.3)) ** 2
            ov.push_level(0.005 + 0.18 * env * (0.6 + 0.4 * abs(math.sin(t * 17))))
            time.sleep(1 / 50)

    threading.Thread(target=feeder, daemon=True).start()
    idx = [0]

    def step():
        if idx[0] >= len(seq):
            global stop
            stop = True
            app.quit()
            return
        s, *args = seq[idx[0]]
        print("state:", s)
        if s == "answer":
            q, a = args
            threading.Thread(target=ov.show_answer, args=(q, a, 6000), daemon=True).start()
        else:
            threading.Thread(target=ov.set_state, args=(s, args[0]), daemon=True).start()
        idx[0] += 1

    timer = QTimer()
    timer.timeout.connect(step)
    timer.start(3000)
    step()
    app.exec()
