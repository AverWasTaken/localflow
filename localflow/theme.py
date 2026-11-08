"""The shared dark theme: QSS, palette, dark title bar and the small widgets every LocalFlow window uses.

Colours come from palette.py. settings_ui re-exports the public names (DARK_QSS, apply_theme, ...) for older
imports."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (QAbstractButton, QApplication, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QStyleFactory, QVBoxLayout, QWidget)

from .palette import (ACCENT, ACCENT_HOVER, ACCENT_PRESSED, ACCENT_SOFT, ACCENT_TEXT, BG, BORDER, BORDER_STRONG,
                      DANGER, OTHERS, REC, SIDEBAR, SUCCESS, SURFACE, SURFACE2, SURFACE3, TEXT, TEXT2, TEXT3, WARN)

MUTED = TEXT2                     # older name
FONT = '"Segoe UI Variable Text", "Segoe UI", sans-serif'
FONT_DISPLAY = '"Segoe UI Variable Display", "Segoe UI", sans-serif'


def _make_icons() -> dict[str, str]:
    """Render the few glyphs QSS needs (check mark, arrows) to PNGs; returns name -> forward-slash path."""
    d = Path(tempfile.gettempdir()) / "localflow_ui_icons2"
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    def draw(name: str, painter_fn, size: int = 16):
        img = QImage(size * 2, size * 2, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.scale(2, 2)
        painter_fn(p, size)
        p.end()
        f = d / f"{name}.png"
        img.save(str(f))
        out[name] = f.as_posix()

    def check(p, s):
        p.setPen(QPen(QColor("white"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPolyline([QPointF(s * .25, s * .52), QPointF(s * .43, s * .70), QPointF(s * .76, s * .32)])

    def arrow(up: bool):
        def f(p, s):
            p.setPen(QPen(QColor(TEXT2), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            y0, y1 = (s * .62, s * .38) if up else (s * .38, s * .62)
            p.drawPolyline([QPointF(s * .30, y0), QPointF(s * .5, y1), QPointF(s * .70, y0)])
        return f

    try:
        draw("check", check)
        draw("down", arrow(False))
        draw("up", arrow(True))
    except Exception:
        pass
    return out


_I = _make_icons()


def _url(name: str) -> str:
    return f"url({_I[name]})" if name in _I else "none"


# Radii: 10 = panels, 6 = controls, 4 = chips. Spacing steps: 4 / 8 / 12 / 16 / 24.
DARK_QSS = f"""
* {{ font-family: {FONT}; font-size: 13px; color: {TEXT}; }}
QWidget {{ background: transparent; }}
QWidget#root, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; }}
QWidget#sidebarBox {{ background: {SIDEBAR}; border-right: 1px solid {BORDER}; }}
QWidget#bottomBar {{ background: {SIDEBAR}; border-top: 1px solid {BORDER}; }}
QToolTip {{ background: {SURFACE2}; color: {TEXT}; border: 1px solid {BORDER_STRONG}; padding: 4px 6px; }}

QLabel {{ background: transparent; }}
QLabel#pageTitle {{ font-family: {FONT_DISPLAY}; font-size: 24px; font-weight: 600; }}
QLabel#pageSub, QLabel[role="muted"] {{ color: {TEXT2}; }}
QLabel[role="faint"] {{ color: {TEXT3}; font-size: 12px; }}
QLabel#brand {{ font-family: {FONT_DISPLAY}; font-size: 16px; font-weight: 600; }}
QLabel#navGroup {{ color: {TEXT3}; font-size: 12px; }}
QLabel#cardTitle {{ font-size: 14px; font-weight: 600; color: {TEXT}; }}
QLabel#sectionTitle {{ font-size: 15px; font-weight: 600; color: {TEXT}; }}
QLabel#rowTitle {{ font-weight: 500; }}
QLabel#rowDesc {{ color: {TEXT2}; font-size: 12px; }}
QLabel#fieldError {{ color: {DANGER}; font-size: 12px; }}
QLabel#saved {{ color: {SUCCESS}; font-weight: 600; }}
QLabel#savedErr {{ color: {DANGER}; font-weight: 600; }}
QLabel#chip {{ background: {SURFACE2}; color: {TEXT2}; border-radius: 4px; padding: 1px 7px; font-size: 11px; }}
QLabel#chipAccent {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; border-radius: 4px; padding: 1px 7px;
                    font-size: 11px; font-weight: 600; }}
QLabel#chipRec {{ background: rgba(229,72,77,0.16); color: #ff8a8e; border-radius: 4px; padding: 1px 7px;
                 font-size: 11px; font-weight: 600; }}
QLabel#chipWarn {{ background: rgba(226,165,59,0.15); color: {WARN}; border-radius: 4px; padding: 1px 7px;
                  font-size: 11px; font-weight: 600; }}
QLabel#chipDanger {{ background: rgba(240,107,115,0.15); color: {DANGER}; border-radius: 4px; padding: 1px 7px;
                    font-size: 11px; font-weight: 600; }}
QLabel#statValue {{ font-family: {FONT_DISPLAY}; font-size: 22px; font-weight: 600; }}
QLabel#statLabel {{ color: {TEXT2}; font-size: 12px; }}

QFrame#card, QFrame#panel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#entry {{ background: transparent; border: none; border-bottom: 1px solid {BORDER}; }}
QFrame#entry:hover {{ background: rgba(255,255,255,0.018); }}
QFrame#sep {{ background: {BORDER}; border: none; max-height: 1px; min-height: 1px; }}
QFrame#vsep {{ background: {BORDER}; border: none; max-width: 1px; min-width: 1px; }}
QFrame#banner {{ background: rgba(226,165,59,0.10); border: 1px solid rgba(226,165,59,0.35); border-radius: 8px; }}

QListWidget#sidebar {{ background: transparent; border: none; outline: none; padding: 0 10px; }}
QListWidget#sidebar::item {{ height: 36px; padding-left: 12px; border-radius: 6px; color: {TEXT2}; margin: 1px 0;
                            border-left: 2px solid transparent; }}
QListWidget#sidebar::item:hover {{ background: {SURFACE}; color: {TEXT}; }}
QListWidget#sidebar::item:disabled {{ color: {TEXT3}; font-size: 12px; background: transparent; height: 30px;
                                     padding-top: 10px; }}
QListWidget#sidebar::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; border-left: 2px solid {ACCENT_TEXT}; }}

QPushButton {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: 6px; padding: 6px 14px;
              color: {TEXT}; }}
QPushButton:hover {{ background: {SURFACE3}; border-color: #4a4c54; }}
QPushButton:pressed {{ background: #25262b; }}
QPushButton:disabled {{ color: #5f6168; border-color: {BORDER}; }}
QPushButton:focus {{ border-color: {ACCENT_TEXT}; }}
QPushButton[role="primary"] {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: white; font-weight: 600; }}
QPushButton[role="primary"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[role="primary"]:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton[role="primary"]:disabled {{ background: #26364f; border-color: #26364f; color: #8190a8; }}
QPushButton[role="primary"]:focus {{ border-color: #a9c6ff; }}
QPushButton[role="danger"] {{ color: {DANGER}; }}
QPushButton[role="danger"]:hover {{ background: rgba(240,107,115,0.14); border-color: rgba(240,107,115,0.5); }}
QPushButton[role="ghost"] {{ background: transparent; border: 1px solid transparent; color: {TEXT2}; padding: 4px 8px; }}
QPushButton[role="ghost"]:hover {{ background: {SURFACE2}; color: {TEXT}; }}
QPushButton[role="ghostDanger"] {{ background: transparent; border: 1px solid transparent; color: {TEXT2};
                                  padding: 4px 8px; font-size: 12px; }}
QPushButton[role="ghostDanger"]:hover {{ background: rgba(240,107,115,0.14); color: {DANGER}; }}
QPushButton[role="mini"] {{ padding: 4px 10px; font-size: 12px; }}
QPushButton[role="chipBtn"] {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: 14px;
                              padding: 5px 13px; font-size: 12px; }}
QPushButton[role="chipBtn"]:hover {{ border-color: {ACCENT_TEXT}; color: {ACCENT_TEXT}; }}
QPushButton[role="record"] {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: 15px;
                             padding: 0 16px 0 13px; font-weight: 600; }}
QPushButton[role="record"]:hover {{ border-color: rgba(229,72,77,0.7); }}
QPushButton[role="recording"] {{ background: {REC}; border: 1px solid {REC}; border-radius: 15px;
                                padding: 0 16px 0 13px; font-weight: 600; color: white; }}
QPushButton[role="recording"]:hover {{ background: #ee5a5f; }}
QFrame#segBox {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 7px; }}
QPushButton[role="seg"] {{ background: transparent; border: 1px solid transparent; border-radius: 5px;
                          padding: 4px 14px; color: {TEXT2}; }}
QPushButton[role="seg"]:hover {{ color: {TEXT}; }}
QPushButton[role="seg"]:checked {{ background: {SURFACE3}; color: {TEXT}; border-color: #4a4c54; }}
QPushButton::menu-indicator {{ image: {_url("down")}; subcontrol-position: right center; subcontrol-origin: padding;
                               width: 12px; right: 6px; }}
QPushButton[menu="1"] {{ padding-right: 24px; }}

QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QSpinBox {{ background: {BG}; border: 1px solid {BORDER_STRONG};
    border-radius: 6px; padding: 6px 10px; selection-background-color: {ACCENT}; selection-color: white; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus {{ border-color: {ACCENT_TEXT}; }}
QLineEdit:read-only {{ color: {TEXT2}; background: {SURFACE}; }}
QLineEdit:disabled, QPlainTextEdit:disabled {{ color: #5f6168; }}
QLineEdit#titleEdit {{ background: transparent; border: 1px solid transparent; font-family: {FONT_DISPLAY};
    font-size: 19px; font-weight: 600; padding: 2px 6px; }}
QLineEdit#titleEdit:hover {{ border-color: {BORDER_STRONG}; }}
QLineEdit#titleEdit:focus {{ border-color: {ACCENT_TEXT}; background: {BG}; }}
QPlainTextEdit, QTextEdit {{ padding: 8px 10px; }}
QTextBrowser#reader {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 4px; }}

QSpinBox {{ padding-right: 24px; }}
QSpinBox::up-button, QSpinBox::down-button {{ subcontrol-origin: border; width: 20px; border: none; background: transparent; }}
QSpinBox::up-button {{ subcontrol-position: top right; margin: 3px 3px 0 0; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0 3px 3px 0; }}
QSpinBox::up-arrow {{ image: {_url("up")}; width: 12px; height: 12px; }}
QSpinBox::down-arrow {{ image: {_url("down")}; width: 12px; height: 12px; }}

QComboBox {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: 6px; padding: 5px 30px 5px 10px;
            min-height: 20px; }}
QComboBox:hover {{ border-color: #4a4c54; }}
QComboBox:focus, QComboBox:on {{ border-color: {ACCENT_TEXT}; }}
QComboBox:disabled {{ color: #5f6168; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 26px; border: none; }}
QComboBox::down-arrow {{ image: {_url("down")}; width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: 6px; padding: 4px;
    outline: none; selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT}; }}
QComboBox QAbstractItemView::item {{ min-height: 26px; padding-left: 8px; border-radius: 4px; }}

QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #50525a; background: {BG}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT_TEXT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: {_url("check")}; }}
QCheckBox::indicator:disabled {{ border-color: {BORDER}; }}
QCheckBox:disabled {{ color: #5f6168; }}

QSlider::groove:horizontal {{ height: 4px; background: #34363c; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: white; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: #dfe9ff; }}

QProgressBar {{ background: {SURFACE2}; border: none; border-radius: 2px; max-height: 4px; min-height: 4px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

QListWidget#plain, QTableWidget {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px; outline: none;
    gridline-color: transparent; selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT}; }}
QListWidget#plain::item {{ padding: 6px 10px; border-radius: 4px; }}
QListWidget#plain::item:selected {{ background: {ACCENT_SOFT}; }}
QListWidget#trList {{ background: transparent; border: none; outline: none; }}
QListWidget#trList::item {{ border-radius: 8px; margin: 1px 0; border: 1px solid transparent; }}
QListWidget#trList::item:hover {{ background: {SURFACE}; }}
QListWidget#trList::item:selected {{ background: {ACCENT_SOFT}; border: 1px solid rgba(122,170,255,0.35); }}
QTableWidget::item {{ padding: 4px 8px; border-bottom: 1px solid #222327; }}
QTableWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QTableWidget QLineEdit {{ border-radius: 4px; padding: 2px 6px; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: {SURFACE}; color: {TEXT2}; border: none; border-bottom: 1px solid {BORDER};
    padding: 7px 10px; font-size: 12px; font-weight: 600; }}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #3a3c42; border-radius: 4px; min-height: 36px; margin: 0 2px; }}
QScrollBar::handle:vertical:hover {{ background: #4d5058; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #3a3c42; border-radius: 4px; min-width: 36px; margin: 2px 0; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {BORDER}; }}
QMenu {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; }}
QMenu::item:disabled {{ color: #6a6c73; }}
QMenu::separator {{ height: 1px; background: {BORDER_STRONG}; margin: 4px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 4px; }}
QMenu::indicator:checked {{ image: {_url("check")}; }}

QMessageBox, QDialog {{ background: {SURFACE}; }}
QMessageBox QLabel {{ color: {TEXT}; }}
"""


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor(BG))
    p.setColor(QPalette.WindowText, QColor(TEXT))
    p.setColor(QPalette.Base, QColor(BG))
    p.setColor(QPalette.AlternateBase, QColor(SURFACE))
    p.setColor(QPalette.Text, QColor(TEXT))
    p.setColor(QPalette.Button, QColor(SURFACE2))
    p.setColor(QPalette.ButtonText, QColor(TEXT))
    p.setColor(QPalette.Highlight, QColor(ACCENT))
    p.setColor(QPalette.HighlightedText, QColor("white"))
    p.setColor(QPalette.ToolTipBase, QColor(SURFACE2))
    p.setColor(QPalette.ToolTipText, QColor(TEXT))
    p.setColor(QPalette.PlaceholderText, QColor(TEXT3))
    p.setColor(QPalette.Link, QColor(ACCENT_TEXT))
    return p


def apply_theme(app: QApplication) -> None:
    """Fusion style + dark palette + DARK_QSS for the whole app (message boxes, menus, ...)."""
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setPalette(_dark_palette())
    app.setStyleSheet(DARK_QSS)


def style_window(w: QWidget) -> None:
    """Make a top-level window look right even if apply_theme() was never called."""
    w.setObjectName("root")
    w.setAttribute(Qt.WA_StyledBackground, True)
    w.setStyle(QStyleFactory.create("Fusion"))
    w.setPalette(_dark_palette())
    w.setStyleSheet(DARK_QSS)


def enable_dark_titlebar(w: QWidget) -> None:
    """Ask DWM for an immersive dark title bar (Win10 1809+/Win11)."""
    try:
        import ctypes
        hwnd = int(w.winId())
        val = ctypes.c_int(1)
        for attr in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), attr, ctypes.byref(val), 4) == 0:
                break
    except Exception:
        pass


def repolish(w: QWidget) -> None:
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


# ---------------------------------------------------------------- small widgets
class Switch(QAbstractButton):
    """Toggle switch (checkable button)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(40, 22)
        self.setFocusPolicy(Qt.StrongFocus)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        on = self.isChecked()
        track = QColor(ACCENT) if on else QColor("#3a3c43")
        if not self.isEnabled():
            track.setAlpha(110)
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0.5, 1, 39, 20), 10, 10)
        p.setBrush(QColor("white") if self.isEnabled() else QColor(170, 172, 178))
        x = 29.0 if on else 11.0
        p.drawEllipse(QPointF(x, 11), 7.5, 7.5)
        if self.hasFocus():
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(ACCENT_TEXT), 1.2))
            p.drawRoundedRect(QRectF(-0.5, 0, 41, 22), 11, 11)


class Card(QFrame):
    """Settings group: a titled panel made of rows [title + description] ........ [control]."""

    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("card")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(20, 16, 20, 6)
        self._lay.setSpacing(0)
        self._n = 0
        if title:
            t = QLabel(title)
            t.setObjectName("cardTitle")
            self._lay.addWidget(t)
            self._lay.addSpacing(4)

    def _sep(self):
        sep = QFrame()
        sep.setObjectName("sep")
        sep.setFrameShape(QFrame.NoFrame)
        self._lay.addWidget(sep)

    def add_row(self, title: str, desc: str = "", control: QWidget | None = None, stack: bool = False):
        if self._n:
            self._sep()
        self._n += 1
        row = QWidget()
        outer = QVBoxLayout(row) if stack else QHBoxLayout(row)
        outer.setContentsMargins(0, 12, 0, 12)
        outer.setSpacing(10 if stack else 20)
        txt = QVBoxLayout()
        txt.setSpacing(2)
        tl = QLabel(title)
        tl.setObjectName("rowTitle")
        txt.addWidget(tl)
        if desc:
            dl = QLabel(desc)
            dl.setObjectName("rowDesc")
            dl.setWordWrap(True)
            txt.addWidget(dl)
        if stack:
            outer.addLayout(txt)
            if control is not None:
                outer.addWidget(control)
        else:
            outer.addLayout(txt, 1)
            if control is not None:
                outer.addWidget(control, 0, Qt.AlignVCenter | Qt.AlignRight)
        self._lay.addWidget(row)
        return row

    def add_widget(self, w: QWidget):
        if self._n:
            self._sep()
        self._n += 1
        self._lay.addWidget(w)


def page(title: str, subtitle: str = "") -> tuple[QScrollArea, QVBoxLayout]:
    """A scrollable page; returns (scroll area to put in the stack, vbox to fill)."""
    inner = QWidget()
    inner.setObjectName("pageInner")
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(32, 26, 32, 28)
    lay.setSpacing(16)
    h = QLabel(title)
    h.setObjectName("pageTitle")
    lay.addWidget(h)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("pageSub")
        s.setWordWrap(True)
        lay.addWidget(s)
        lay.addSpacing(2)
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(inner)
    return sa, lay


def btn(text: str, role: str = "", w: int | None = None) -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    if role:
        b.setProperty("role", role)
    if w:
        b.setFixedWidth(w)
    return b


def label(text: str = "", obj: str = "", role: str = "", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    if role:
        lb.setProperty("role", role)
    lb.setWordWrap(wrap)
    return lb


def hsep(vertical: bool = False) -> QFrame:
    f = QFrame()
    f.setObjectName("vsep" if vertical else "sep")
    f.setFrameShape(QFrame.NoFrame)
    return f


__all__ = ["ACCENT", "ACCENT_HOVER", "ACCENT_PRESSED", "ACCENT_SOFT", "ACCENT_TEXT", "BG", "BORDER", "BORDER_STRONG",
           "DANGER", "MUTED", "OTHERS", "REC", "SIDEBAR", "SUCCESS", "SURFACE", "SURFACE2", "SURFACE3", "TEXT", "TEXT2",
           "TEXT3", "WARN", "DARK_QSS", "apply_theme", "style_window", "enable_dark_titlebar", "Switch", "Card",
           "page", "btn", "label", "hsep", "repolish"]
