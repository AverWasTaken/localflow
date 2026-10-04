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
                           QPen)
from PySide6.QtWidgets import QApplication, QWidget

try:  # Windows only; everything degrades gracefully without it
    import win32api
    import win32con
    import win32gui
except Exception:  # pragma: no cover
    win32api = win32con = win32gui = None

VALID_STATES = ("hidden", "recording", "handsfree", "processing", "error", "info")

PILL_H = 38.0
MARGIN = 22.0            # transparent canvas margin around the pill (room for the shadow)
MAX_PILL_W = 360.0
CANVAS_W = int(MAX_PILL_W + 2 * MARGIN)
CANVAS_H = int(PILL_H + 2 * MARGIN)
BOTTOM_OFFSET = 48       # px between pill bottom and bottom of the work area
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

VIOLET = QColor(150, 122, 255)
BLUE = QColor(86, 170, 255)
REC_RED = QColor(255, 69, 58)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


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


class Overlay(QWidget):
    _sig_state = Signal(str, str)
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

        self._sig_state.connect(self._apply_state)
        self._sig_reposition.connect(self._do_reposition)

        # create the native window now so we can set the Win32 styles before first show
        self.winId()
        self._apply_native_styles()

    # ------------------------------------------------------------------ public API
    def set_state(self, state: str, text: str = "") -> None:
        """Thread-safe. state: hidden|recording|handsfree|processing|error|info.
        For recording/handsfree, ``text`` is an optional small style label shown in the pill ("casual · Discord")."""
        self._sig_state.emit(str(state), str(text))

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
        y = av.y() + av.height() - BOTTOM_OFFSET - int(PILL_H) - int(MARGIN)
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
        if state in ("recording", "handsfree"):
            return WIDTHS[state] + self._label_width(text)
        if state in WIDTHS:
            return WIDTHS[state]
        # error/info: icon area (34) + text + right pad (18)
        w = 34.0 + self._text_width(text) + 18.0
        return max(110.0, min(MAX_PILL_W, w))

    def _apply_state(self, state: str, text: str) -> None:
        if state not in VALID_STATES:
            return
        self._auto.stop()
        if state == "hidden":
            self._begin_hide()
            return
        if state in ("error", "info") and not text:
            text = "Error" if state == "error" else ""
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
        if state == "error":
            self._auto.start(int(ERROR_SECS * 1000))
        elif state == "info":
            self._auto.start(int(INFO_SECS * 1000))
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

    def _on_width(self, v) -> None:
        self._pw = float(v)
        self.update()

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
        rect = QRectF(x0, MARGIN, pw, PILL_H)
        t = time.perf_counter() - self._t0

        self._paint_shadow(p, rect, mode)
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
            self._paint_dots(p, rect, t)
        elif mode == "error":
            self._paint_error(p, rect)
        elif mode == "info":
            self._paint_info(p, rect)
        p.end()

    def _paint_shadow(self, p: QPainter, rect: QRectF, mode: str) -> None:
        p.setPen(Qt.NoPen)
        layers = 9
        for i in range(layers):
            grow = (layers - i) * 1.5
            a = 3 + i * 0.6
            r = rect.adjusted(-grow, -grow + 3, grow, grow + 3)
            p.setBrush(QColor(0, 0, 0, int(a)))
            p.drawRoundedRect(r, PILL_H / 2 + grow, PILL_H / 2 + grow)

    def _paint_pill(self, p: QPainter, rect: QRectF, mode: str) -> None:
        if mode == "error":
            top, bot, border = QColor(84, 26, 32, 240), QColor(62, 18, 24, 240), QColor(255, 110, 110, 120)
        elif mode == "handsfree":
            top, bot, border = QColor(30, 27, 46, 240), QColor(20, 19, 32, 240), QColor(160, 140, 255, 90)
        else:
            top, bot, border = QColor(30, 30, 36, 238), QColor(18, 18, 22, 238), QColor(255, 255, 255, 38)
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
        col = QColor(190, 170, 255)
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
        p.setBrush(QColor(30, 27, 46))
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
            col = QColor(205, 195, 255)
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
        p.setPen(QColor(190, 190, 205))
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
           ("info", "Paused"), ("recording", ""), ("hidden", "")]
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
        s, tx = seq[idx[0]]
        print("state:", s)
        threading.Thread(target=ov.set_state, args=(s, tx), daemon=True).start()
        idx[0] += 1

    timer = QTimer()
    timer.timeout.connect(step)
    timer.start(3000)
    step()
    app.exec()
