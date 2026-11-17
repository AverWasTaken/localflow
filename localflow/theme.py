"""The shared dark theme: QSS, palette, dark title bar and the small widgets every LocalFlow window uses.

Colours and sizes come from palette.py. settings_ui re-exports the public names (DARK_QSS, apply_theme, ...) for older
imports."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QStyleFactory, QVBoxLayout, QWidget)

from .palette import (ACCENT, ACCENT_HOVER, ACCENT_PRESSED, ACCENT_SOFT, ACCENT_TEXT, BG, BORDER, BORDER_STRONG,
                      CONTROL_H, DANGER, DANGER_FILL, DANGER_FILL_HOVER, OTHERS, RADIUS_L, RADIUS_M, RADIUS_S, REC,
                      SIDEBAR, SUCCESS, SURFACE, SURFACE2, SURFACE3, TEXT, TEXT2, TEXT3, WARN)

MUTED = TEXT2                     # older name
FONT = '"Segoe UI Variable Text", "Segoe UI", sans-serif'
FONT_DISPLAY = '"Segoe UI Variable Display", "Segoe UI", sans-serif'

# derived shades that only the QSS needs
_HOVER_ROW = "#1b1c21"            # hover fill for list / table rows
_SELECT = "#1a2640"               # selected row in tables / lists (accent, but quiet)
_BORDER_HOVER = "#3d4048"         # control border on hover
_DISABLED = "#5d6068"             # disabled text
_TRACK_OFF = "#33353c"            # switch track off, slider groove
_HANDLE = "#2e3036"               # scrollbar handle
_HANDLE_HOVER = "#444750"
_REC_SOFT = "rgba(229,72,77,0.16)"
_REC_TEXT = "#ff8f93"
_WARN_SOFT = "rgba(226,165,59,0.15)"
_DANGER_SOFT = "rgba(242,114,122,0.14)"


def _make_icons() -> dict[str, str]:
    """Render the few glyphs QSS needs (check mark, arrows) to PNGs; returns name -> forward-slash path."""
    d = Path(tempfile.gettempdir()) / "localflow_ui_icons3"
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
            p.setPen(QPen(QColor(TEXT2), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            y0, y1 = (s * .60, s * .40) if up else (s * .40, s * .60)
            p.drawPolyline([QPointF(s * .32, y0), QPointF(s * .5, y1), QPointF(s * .68, y0)])
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


# Radii: RADIUS_L = cards / panels / dialogs, RADIUS_M = controls, RADIUS_S = chips and row buttons.
# Controls are CONTROL_H (32 px) tall: 13 px text + 6 px padding + 1 px border, never a min-height on bare
# QPushButton / QLineEdit (in-table editors must stay short). Spacing steps: 4 / 8 / 12 / 16 / 24.
DARK_QSS = f"""
* {{ font-family: {FONT}; font-size: 13px; color: {TEXT}; }}
QWidget {{ background: transparent; }}
QWidget#root, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; }}
QWidget#sidebarBox {{ background: {SIDEBAR}; border-right: 1px solid {BORDER}; }}
QWidget#bottomBar {{ background: {BG}; border-top: 1px solid {BORDER}; }}
QToolTip {{ background: {SURFACE2}; color: {TEXT}; border: 1px solid {BORDER_STRONG}; padding: 5px 8px;
    border-radius: {RADIUS_S}px; }}

QLabel {{ background: transparent; }}
QLabel#pageTitle {{ font-family: {FONT_DISPLAY}; font-size: 22px; font-weight: 600; }}
QLabel#pageSub, QLabel[role="muted"] {{ color: {TEXT2}; }}
QLabel[role="faint"] {{ color: {TEXT3}; font-size: 12px; }}
QLabel#brand {{ font-family: {FONT_DISPLAY}; font-size: 15px; font-weight: 600; }}
QLabel#navGroup {{ color: {TEXT3}; font-size: 12px; }}
QLabel#cardTitle {{ font-family: {FONT_DISPLAY}; font-size: 15px; font-weight: 600; color: {TEXT}; }}
QLabel#sectionTitle {{ font-family: {FONT_DISPLAY}; font-size: 15px; font-weight: 600; color: {TEXT}; }}
QLabel#rowTitle {{ font-weight: 400; color: {TEXT}; }}
QLabel#rowDesc {{ color: {TEXT2}; font-size: 12px; }}
QLabel#fieldError {{ color: {DANGER}; font-size: 12px; }}
QLabel#saved {{ color: {SUCCESS}; font-weight: 600; }}
QLabel#savedErr {{ color: {DANGER}; font-weight: 600; }}
QLabel#chip {{ background: {SURFACE2}; color: {TEXT2}; border: 1px solid {BORDER_STRONG}; border-radius: {RADIUS_S}px;
    padding: 0 6px; font-size: 11px; }}
QLabel#chipAccent {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; border-radius: {RADIUS_S}px; padding: 1px 7px;
    font-size: 11px; font-weight: 600; }}
QLabel#chipRec {{ background: {_REC_SOFT}; color: {_REC_TEXT}; border-radius: {RADIUS_S}px; padding: 1px 7px;
    font-size: 11px; font-weight: 600; }}
QLabel#chipWarn {{ background: {_WARN_SOFT}; color: {WARN}; border-radius: {RADIUS_S}px; padding: 1px 7px;
    font-size: 11px; font-weight: 600; }}
QLabel#chipDanger {{ background: {_DANGER_SOFT}; color: {DANGER}; border-radius: {RADIUS_S}px; padding: 1px 7px;
    font-size: 11px; font-weight: 600; }}
QLabel#statValue {{ font-family: {FONT_DISPLAY}; font-size: 20px; font-weight: 600; }}
QLabel#statLabel {{ color: {TEXT2}; font-size: 12px; }}
QLabel#statIcon {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: {RADIUS_M}px; }}
QLabel#entryApp {{ font-weight: 600; }}
QLabel#entryText {{ font-size: 14px; }}
QLabel#entryRaw {{ color: {TEXT3}; }}
QLabel#recTime {{ font-family: {FONT_DISPLAY}; font-size: 16px; font-weight: 600; color: {_REC_TEXT}; }}
QLabel#emptyState {{ color: {TEXT2}; font-size: 14px; padding: 48px 12px; }}
QLabel#listEmpty {{ color: {TEXT2}; padding: 30px 10px; }}
QLabel#trTitle {{ font-weight: 600; }}
QLabel#noteTitle {{ font-weight: 600; }}
QLabel#statusFailed {{ color: {DANGER}; font-size: 12px; }}
QLabel#statusWarn {{ color: {WARN}; font-size: 12px; }}
QLabel#testOk {{ color: {ACCENT_TEXT}; font-weight: 600; }}
QLabel#testIdle {{ color: {TEXT2}; }}
QLabel#dialogTitle {{ font-family: {FONT_DISPLAY}; font-size: 16px; font-weight: 600; }}
QLabel#dialogText {{ color: {TEXT2}; }}

QFrame#card, QFrame#panel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS_L}px; }}
QFrame#liveAnswer {{ background: {SURFACE}; border: 1px solid rgba(134,176,255,0.40); border-radius: {RADIUS_L}px; }}
QFrame#entry {{ background: transparent; border: none; border-bottom: 1px solid {BORDER}; }}
QFrame#sep {{ background: {BORDER}; border: none; max-height: 1px; min-height: 1px; }}
QFrame#vsep {{ background: {BORDER}; border: none; max-width: 1px; min-width: 1px; }}
QFrame#banner {{ background: rgba(226,165,59,0.10); border: 1px solid rgba(226,165,59,0.35);
    border-radius: {RADIUS_M}px; }}
QScrollArea#listPanel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS_L}px; }}
QScrollArea#listPanel > QWidget, QScrollArea#listPanel > QWidget > QWidget {{ background: transparent; }}
QScrollArea#listPanel QScrollBar:vertical {{ margin: 8px 3px 8px 0; }}

QListWidget#sidebar {{ background: transparent; border: none; outline: none; padding: 0 10px; }}
QListWidget#sidebar::item {{ height: 36px; padding-left: 10px; border-radius: {RADIUS_M}px; color: {TEXT2};
    margin: 1px 0; border: none; }}
QListWidget#sidebar::item:hover {{ background: {_HOVER_ROW}; color: {TEXT}; }}
QListWidget#sidebar::item:disabled {{ color: {TEXT3}; font-size: 12px; background: transparent; height: 26px;
    padding-top: 14px; padding-left: 12px; }}
QListWidget#sidebar::item:selected {{ background: {SURFACE2}; color: {TEXT}; }}

QPushButton {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: {RADIUS_M}px; padding: 6px 14px;
    color: {TEXT}; }}
QPushButton:hover {{ background: {SURFACE3}; border-color: {_BORDER_HOVER}; }}
QPushButton:pressed {{ background: {SURFACE}; }}
QPushButton:disabled {{ color: {_DISABLED}; border-color: {BORDER}; background: {SURFACE}; }}
QPushButton:focus {{ border-color: {ACCENT_TEXT}; }}
QPushButton[role="primary"] {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: white; font-weight: 600; }}
QPushButton[role="primary"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[role="primary"]:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton[role="primary"]:disabled {{ background: #1f2c45; border-color: #1f2c45; color: #7c8aa3; }}
QPushButton[role="primary"]:focus {{ border-color: #b9d0ff; }}
QPushButton[role="danger"] {{ color: {DANGER}; }}
QPushButton[role="danger"]:hover {{ background: {_DANGER_SOFT}; border-color: rgba(242,114,122,0.45); }}
QPushButton[role="dangerSolid"] {{ background: {DANGER_FILL}; border: 1px solid {DANGER_FILL}; color: white;
    font-weight: 600; }}
QPushButton[role="dangerSolid"]:hover {{ background: {DANGER_FILL_HOVER}; border-color: {DANGER_FILL_HOVER}; }}
QPushButton[role="dangerSolid"]:focus {{ border-color: #ffc2c5; }}
QPushButton[role="ghost"] {{ background: transparent; border: 1px solid transparent; color: {TEXT2}; }}
QPushButton[role="ghost"]:hover {{ background: {SURFACE2}; color: {TEXT}; }}
QPushButton[role="ghost"]:focus {{ border-color: {ACCENT_TEXT}; }}
QPushButton[role="ghostDanger"] {{ background: transparent; border: 1px solid transparent; color: {TEXT2}; }}
QPushButton[role="ghostDanger"]:hover {{ background: {_DANGER_SOFT}; color: {DANGER}; }}
QPushButton[role="ghostDanger"]:focus {{ border-color: {ACCENT_TEXT}; }}
QPushButton[role="ghost"]:disabled, QPushButton[role="ghostDanger"]:disabled, QPushButton[role="icon"]:disabled,
QPushButton[role="iconDanger"]:disabled {{ color: {_DISABLED}; background: transparent; border-color: transparent; }}
QPushButton[role="mini"] {{ padding: 6px 12px; }}
QPushButton[role="iconBox"] {{ padding: 0; }}
QPushButton[role="icon"] {{ background: transparent; border: 1px solid transparent; border-radius: {RADIUS_S}px;
    padding: 5px; }}
QPushButton[role="icon"]:hover {{ background: {SURFACE3}; }}
QPushButton[role="icon"]:focus {{ border-color: {ACCENT_TEXT}; }}
QPushButton[role="iconDanger"] {{ background: transparent; border: 1px solid transparent; border-radius: {RADIUS_S}px;
    padding: 5px; }}
QPushButton[role="iconDanger"]:hover {{ background: {_DANGER_SOFT}; }}
QPushButton[role="iconDanger"]:focus {{ border-color: {ACCENT_TEXT}; }}
QPushButton[role="chipBtn"] {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: {RADIUS_M}px;
    padding: 5px 13px; font-size: 12px; }}
QPushButton[role="chipBtn"]:hover {{ border-color: {ACCENT_TEXT}; color: {ACCENT_TEXT}; }}
QPushButton[role="chipBtn"]:disabled {{ color: {_DISABLED}; border-color: {BORDER}; background: transparent; }}
QPushButton[role="record"] {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: {RADIUS_M}px;
    padding: 0 14px 0 12px; font-weight: 600; }}
QPushButton[role="record"]:hover {{ background: {SURFACE3}; border-color: rgba(229,72,77,0.65); }}
QPushButton[role="record"]:focus {{ border-color: {ACCENT_TEXT}; }}
QPushButton[role="recording"] {{ background: {DANGER_FILL}; border: 1px solid {DANGER_FILL}; border-radius: {RADIUS_M}px;
    padding: 0 14px 0 12px; font-weight: 600; color: white; }}
QPushButton[role="recording"]:hover {{ background: {DANGER_FILL_HOVER}; }}
QFrame#segBox {{ background: {BG}; border: 1px solid {BORDER}; border-radius: {RADIUS_M}px; }}
QPushButton[role="seg"] {{ background: transparent; border: 1px solid transparent; border-radius: {RADIUS_S}px;
    padding: 4px 14px; color: {TEXT2}; }}
QPushButton[role="seg"]:hover {{ color: {TEXT}; }}
QPushButton[role="seg"]:checked {{ background: {SURFACE3}; color: {TEXT}; border-color: {BORDER_STRONG}; }}
QPushButton[role="seg"]:focus {{ border-color: {ACCENT_TEXT}; }}
QPushButton::menu-indicator {{ image: {_url("down")}; subcontrol-position: right center; subcontrol-origin: padding;
    width: 12px; right: 8px; }}
QPushButton[menu="1"] {{ padding-right: 28px; }}

QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QSpinBox {{ background: {BG}; border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_M}px; padding: 6px 10px; selection-background-color: {ACCENT}; selection-color: white; }}
QLineEdit:hover, QPlainTextEdit:hover, QSpinBox:hover {{ border-color: {_BORDER_HOVER}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus {{ border-color: {ACCENT_TEXT}; }}
QLineEdit:read-only {{ color: {TEXT2}; background: {SURFACE2}; border-color: {BORDER}; }}
QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled {{ color: {_DISABLED}; border-color: {BORDER}; }}
QLineEdit#titleEdit {{ background: transparent; border: 1px solid transparent; font-family: {FONT_DISPLAY};
    font-size: 19px; font-weight: 600; padding: 2px 6px; }}
QLineEdit#titleEdit:hover {{ border-color: {BORDER_STRONG}; }}
QLineEdit#titleEdit:focus {{ border-color: {ACCENT_TEXT}; background: {BG}; }}
QPlainTextEdit, QTextEdit {{ padding: 8px 10px; }}
QTextBrowser#reader {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS_L}px; padding: 8px 8px; }}

QSpinBox {{ padding-right: 26px; }}
QSpinBox::up-button, QSpinBox::down-button {{ subcontrol-origin: border; width: 20px; border: none;
    background: transparent; }}
QSpinBox::up-button {{ subcontrol-position: top right; margin: 4px 4px 0 0; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0 4px 4px 0; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {SURFACE3}; border-radius: 4px; }}
QSpinBox::up-arrow {{ image: {_url("up")}; width: 12px; height: 12px; }}
QSpinBox::down-arrow {{ image: {_url("down")}; width: 12px; height: 12px; }}

QComboBox {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: {RADIUS_M}px;
    padding: 5px 30px 5px 10px; min-height: 20px; }}
QComboBox:hover {{ border-color: {_BORDER_HOVER}; background: {SURFACE3}; }}
QComboBox:focus, QComboBox:on {{ border-color: {ACCENT_TEXT}; }}
QComboBox:disabled {{ color: {_DISABLED}; border-color: {BORDER}; background: {SURFACE}; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 28px; border: none; }}
QComboBox::down-arrow {{ image: {_url("down")}; width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: {RADIUS_M}px;
    padding: 4px; outline: none; selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT}; }}
QComboBox QAbstractItemView::item {{ min-height: 28px; padding-left: 8px; border-radius: 4px; }}
QComboBox QAbstractItemView::item:hover {{ background: {SURFACE3}; }}

QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #4d5058; background: {BG}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT_TEXT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: {_url("check")}; }}
QCheckBox::indicator:disabled {{ border-color: {BORDER}; }}
QCheckBox:disabled {{ color: {_DISABLED}; }}
QCheckBox:focus {{ color: {TEXT}; }}
QCheckBox#keyChip {{ spacing: 0; background: {SURFACE2}; border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_M}px; padding: 6px 14px; color: {TEXT2}; }}
QCheckBox#keyChip::indicator {{ width: 0; height: 0; border: none; background: transparent; image: none; }}
QCheckBox#keyChip:hover {{ border-color: {_BORDER_HOVER}; color: {TEXT}; }}
QCheckBox#keyChip:checked {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {TEXT}; font-weight: 600; }}
QCheckBox#keyChip:focus {{ border-color: {ACCENT_TEXT}; }}

QSlider::groove:horizontal {{ height: 4px; background: {_TRACK_OFF}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: white; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: #dfe9ff; }}
QSlider::handle:horizontal:focus {{ border: 2px solid {ACCENT_TEXT}; width: 12px; height: 12px; }}

QProgressBar {{ background: {SURFACE3}; border: none; border-radius: 2px; max-height: 4px; min-height: 4px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

QListWidget#plain {{ background: {BG}; border: 1px solid {BORDER}; border-radius: {RADIUS_M}px; outline: none;
    padding: 4px; selection-background-color: {_SELECT}; selection-color: {TEXT}; }}
QListWidget#plain::item {{ padding: 6px 10px; border-radius: {RADIUS_S}px; }}
QListWidget#plain::item:hover {{ background: {_HOVER_ROW}; }}
QListWidget#plain::item:selected {{ background: {_SELECT}; color: {TEXT}; }}
QListWidget#trList {{ background: transparent; border: none; outline: none; }}
QListWidget#trList::item {{ border-radius: {RADIUS_M}px; margin: 1px 0; border: 1px solid transparent; }}
QListWidget#trList::item:hover {{ background: {_HOVER_ROW}; }}
QListWidget#trList::item:selected {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; }}

QTableWidget {{ background: transparent; border: none; outline: none; gridline-color: transparent;
    selection-background-color: {_SELECT}; selection-color: {TEXT}; }}
QTableWidget::item {{ padding: 0 10px; border-bottom: 1px solid {BORDER}; }}
QTableWidget::item:hover {{ background: {_HOVER_ROW}; }}
QTableWidget::item:selected {{ background: {_SELECT}; color: {TEXT}; }}
QTableWidget QLineEdit {{ min-height: 0; border-radius: 4px; padding: 0 6px; background: {BG};
    border: 1px solid {ACCENT_TEXT}; }}
QTableWidget QComboBox {{ background: transparent; border: 1px solid transparent; border-radius: {RADIUS_S}px;
    padding: 2px 26px 2px 8px; min-height: 0; margin: 4px 0; }}
QTableWidget QComboBox:hover {{ background: {SURFACE2}; border-color: {BORDER_STRONG}; }}
QTableWidget QComboBox:focus, QTableWidget QComboBox:on {{ border-color: {ACCENT_TEXT}; }}
QTableWidget QComboBox QLineEdit {{ background: transparent; border: none; padding: 0; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: transparent; color: {TEXT3}; border: none; border-bottom: 1px solid {BORDER_STRONG};
    padding: 8px 10px; font-size: 12px; font-weight: 600; }}
QTableCornerButton::section {{ background: transparent; border: none; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {_HANDLE}; border-radius: 3px; min-height: 36px; margin: 0 1px; }}
QScrollBar::handle:vertical:hover {{ background: {_HANDLE_HOVER}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {_HANDLE}; border-radius: 3px; min-width: 36px; margin: 1px 0; }}
QScrollBar::handle:horizontal:hover {{ background: {_HANDLE_HOVER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {BORDER}; }}
QMenu {{ background: {SURFACE2}; border: 1px solid {BORDER_STRONG}; border-radius: {RADIUS_M}px; padding: 4px; }}
QMenu::item {{ padding: 7px 24px 7px 12px; border-radius: {RADIUS_S}px; }}
QMenu::item:selected {{ background: {SURFACE3}; }}
QMenu::item:disabled {{ color: #6a6c73; }}
QMenu::separator {{ height: 1px; background: {BORDER_STRONG}; margin: 4px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 4px; }}
QMenu::indicator:checked {{ image: {_url("check")}; }}

QMessageBox, QDialog {{ background: {SURFACE}; }}
QDialog#confirm {{ background: {SURFACE}; }}
QWidget#dialogButtons {{ background: {BG}; border-top: 1px solid {BORDER}; }}
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


def _set_window_icon(w: QWidget) -> None:
    try:
        from .icons import app_icon
        w.setWindowIcon(app_icon())
    except Exception:
        pass


_STYLE: list = []


def _shared_style():
    """One Fusion style for every styled window, owned by the QApplication so it outlives them all (a style freed
    before its widget makes the widget's destructor crash)."""
    if not _STYLE:
        st = QStyleFactory.create("Fusion")
        app = QApplication.instance()
        if app is not None:
            st.setParent(app)
        _STYLE.append(st)
    return _STYLE[0]


def style_window(w: QWidget) -> None:
    """Make a top-level window look right even if apply_theme() was never called."""
    w.setObjectName("root")
    w.setAttribute(Qt.WA_StyledBackground, True)
    w.setStyle(_shared_style())
    w.setPalette(_dark_palette())
    w.setStyleSheet(DARK_QSS)
    _set_window_icon(w)


def _colorref(hex_color: str) -> int:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b << 16) | (g << 8) | r


def enable_dark_titlebar(w: QWidget, caption: str = BG) -> None:
    """Dark title bar (Win10 1809+/Win11) and, on Win11, a caption colour that matches the window. Silent elsewhere."""
    try:
        import ctypes
        hwnd = ctypes.c_void_p(int(w.winId()))
        dwm = ctypes.windll.dwmapi
        val = ctypes.c_int(1)
        for attr in (20, 19):                       # DWMWA_USE_IMMERSIVE_DARK_MODE (new id, then the pre-20H1 id)
            if dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), 4) == 0:
                break
        col = ctypes.c_uint(_colorref(caption))
        dwm.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(col), 4)          # DWMWA_CAPTION_COLOR (Win11 22000+)
        border = ctypes.c_uint(_colorref(BORDER_STRONG))
        dwm.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(border), 4)       # DWMWA_BORDER_COLOR
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
        track = QColor(ACCENT) if on else QColor(_TRACK_OFF)
        if not self.isEnabled():
            track.setAlpha(110)
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(1, 2, 38, 18), 9, 9)
        p.setBrush(QColor("white") if self.isEnabled() else QColor(150, 152, 158))
        x = 29.0 if on else 11.0
        p.drawEllipse(QPointF(x, 11), 7, 7)
        if self.hasFocus():
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(ACCENT_TEXT), 1.5))
            p.drawRoundedRect(QRectF(0.75, 0.75, 38.5, 20.5), 10.25, 10.25)


class KeyChip(QCheckBox):
    """Checkbox drawn as a key chip (QSS hides the indicator). The whole chip is clickable, not just the text."""

    def hitButton(self, pos):
        return self.rect().contains(pos)


class Card(QFrame):
    """Settings group: a titled panel made of rows [title + description] ........ [control]."""

    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("card")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(20, 16, 20, 4)
        self._lay.setSpacing(0)
        self._n = 0
        if title:
            t = QLabel(title)
            t.setObjectName("cardTitle")
            self._lay.addWidget(t)
            self._lay.addSpacing(6)

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
        outer.setContentsMargins(0, 14, 0, 14)
        outer.setSpacing(10 if stack else 24)
        txt = QVBoxLayout()
        txt.setSpacing(3)
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
    lay.setContentsMargins(36, 28, 36, 32)
    lay.setSpacing(16)
    head = QVBoxLayout()
    head.setSpacing(4)
    h = QLabel(title)
    h.setObjectName("pageTitle")
    head.addWidget(h)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("pageSub")
        s.setWordWrap(True)
        head.addWidget(s)
    lay.addLayout(head)
    lay.addSpacing(4)
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(inner)
    return sa, lay


def btn(text: str, role: str = "", w: int | None = None, icon: str = "", tip: str = "") -> QPushButton:
    """Push button. ``icon`` names an icons.py glyph (tinted to match the role); an icon button with no text gets
    the "icon" role unless ``role`` says otherwise."""
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    if icon and not text and not role:
        role = "icon"
    if role:
        b.setProperty("role", role)
    if icon:
        set_btn_icon(b, icon)
    if tip:
        b.setToolTip(tip)
    if w:
        b.setFixedWidth(w)
    return b


ICON_GAP = 6


def set_btn_icon(b: QPushButton, name: str) -> None:
    from PySide6.QtCore import QSize

    from .icons import icon
    role = b.property("role") or ""
    color = {"primary": "#ffffff", "dangerSolid": "#ffffff", "danger": DANGER, "iconDanger": TEXT2}.get(role, TEXT2)
    if role in ("icon", "iconDanger", "iconBox") or not b.text():
        b.setIcon(icon(name, color, 16))
        b.setIconSize(QSize(16, 16))
    else:                                   # icon + text: 6 px transparent gap baked into the pixmap
        b.setIcon(icon(name, color, 15, pad=ICON_GAP))
        b.setIconSize(QSize(15 + ICON_GAP, 15))
    b.setProperty("iconName", name)


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


def _break_long_words(text: str, limit: int = 34) -> str:
    """Shorten unbroken words (long file names) in the middle so a wrapping label never gets clipped."""
    out = []
    for w in text.split(" "):
        if len(w) > limit:
            w = w[: limit // 2] + "…" + w[-(limit // 2 - 2):]
        out.append(w)
    return " ".join(out)


class ConfirmDialog(QDialog):
    """Themed yes/no dialog: a title line, the question, an optional detail line, Cancel + the action button.

    Cancel is the default button (Enter cancels), Esc rejects. ``exec()`` returns QDialog.Accepted only when the
    action button was pressed."""

    def __init__(self, parent=None, title: str = "", text: str = "", ok_text: str = "OK", info: str = "",
                 danger: bool = True):
        super().__init__(parent)
        self.setObjectName("confirm")
        self.setWindowTitle(title or "LocalFlow")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        if parent is None:                      # app style already applies; just the palette and the QSS
            self.setPalette(_dark_palette())
            self.setStyleSheet(DARK_QSS)
        _set_window_icon(self)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumWidth(420)
        self.setMaximumWidth(520)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(24, 22, 24, 22)
        bl.setSpacing(8)
        self.text_label = label(_break_long_words(text), "dialogTitle", wrap=True)
        bl.addWidget(self.text_label)
        self.info_label = label(info, "dialogText", wrap=True)
        self.info_label.setVisible(bool(info))
        bl.addWidget(self.info_label)
        outer.addWidget(body)
        bar = QWidget()
        bar.setObjectName("dialogButtons")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(24, 14, 24, 14)
        hl.setSpacing(8)
        hl.addStretch(1)
        self.cancel_btn = btn("Cancel")
        self.ok_btn = btn(ok_text, "dangerSolid" if danger else "primary")
        self.cancel_btn.setMinimumWidth(88)
        self.ok_btn.setMinimumWidth(88)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setAutoDefault(True)          # Enter on the focused action button confirms
        self.cancel_btn.setAutoDefault(True)
        hl.addWidget(self.cancel_btn)
        hl.addWidget(self.ok_btn)
        outer.addWidget(bar)
        self.cancel_btn.setDefault(True)          # after it is inside the dialog, or Qt ignores it
        self._dark_done = False

    def showEvent(self, e):
        if not self._dark_done:
            enable_dark_titlebar(self, SURFACE)
            self._dark_done = True
        super().showEvent(e)
        self.cancel_btn.setFocus()


def confirm(parent, title: str, text: str, ok_text: str, info: str = "", danger: bool = True) -> bool:
    """Ask a yes/no question with the themed dialog. True only when the action button was pressed."""
    return ConfirmDialog(parent, title, text, ok_text, info, danger).exec() == QDialog.Accepted


__all__ = ["ACCENT", "ACCENT_HOVER", "ACCENT_PRESSED", "ACCENT_SOFT", "ACCENT_TEXT", "BG", "BORDER", "BORDER_STRONG",
           "DANGER", "MUTED", "OTHERS", "REC", "SIDEBAR", "SUCCESS", "SURFACE", "SURFACE2", "SURFACE3", "TEXT", "TEXT2",
           "TEXT3", "WARN", "DARK_QSS", "apply_theme", "style_window", "enable_dark_titlebar", "Switch", "Card",
           "page", "btn", "label", "hsep", "repolish", "ConfirmDialog", "KeyChip", "confirm", "set_btn_icon", "CONTROL_H",
           "RADIUS_S", "RADIUS_M", "RADIUS_L"]
