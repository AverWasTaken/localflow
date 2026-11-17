"""Small stroke icon set for the LocalFlow window, plus the app mark (waveform tile).

Icon paths are from Lucide (https://lucide.dev, ISC license), 24x24 viewBox, 2px round strokes. Nothing is rendered at
import time: a QPixmap needs a QApplication, so every icon is built on first use and cached.
"""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

from .palette import ACCENT, ACCENT_TEXT, SUCCESS, TEXT2, TEXT3

_P = {
    "home": '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/>'
            '<path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5'
            'a2 2 0 0 1-2-2z"/>',
    "transcripts": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
                   '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>',
    "general": '<path d="M20 7h-9"/><path d="M14 17H5"/><circle cx="17" cy="17" r="3"/><circle cx="7" cy="7" r="3"/>',
    "dictionary": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "snippets": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 '
                '1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    "styles": '<path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838'
              'a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/>',
    "about": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "copy": '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>'
            '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    "mic": '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/>'
           '<path d="M12 19v3"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>'
             '<path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
    "up": '<path d="m18 15-6-6-6 6"/>',
    "down": '<path d="m6 9 6 6 6-6"/>',
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0'
              '-2 2v13a2 2 0 0 0 2 2Z"/>',
    "calendar": '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>',
    "words": '<path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/>'
             '<path d="M22 10v3"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 '
             '3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
}

NAMES = tuple(_P)
_SCALE = 3          # render at 3x so 125 % / 150 % screens stay sharp


def _svg(name: str, color: str, stroke: float) -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
            f'{_P[name]}</svg>').encode()


@lru_cache(maxsize=256)
def pixmap(name: str, size: int = 16, color: str = TEXT2, stroke: float = 2.0, pad: int = 0) -> QPixmap:
    """One tinted icon at ``size`` logical px (device pixel ratio set, so it paints crisp on HiDPI). ``pad`` adds
    transparent space on the right: the gap between an icon and the text of a push button."""
    from PySide6.QtSvg import QSvgRenderer
    px = size * _SCALE
    pm = QPixmap(px + pad * _SCALE, px)
    pm.fill(Qt.transparent)
    r = QSvgRenderer(QByteArray(_svg(name, color, stroke)))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    r.render(p, QRectF(0, 0, px, px))
    p.end()
    pm.setDevicePixelRatio(_SCALE)
    return pm


@lru_cache(maxsize=128)
def icon(name: str, color: str = TEXT2, size: int = 16, selected: str | None = None,
         active: str | None = None, pad: int = 0) -> QIcon:
    """QIcon with explicit Normal / Selected / Active / Disabled pixmaps (Qt would otherwise grey or tint them)."""
    ic = QIcon()
    ic.addPixmap(pixmap(name, size, color, pad=pad), QIcon.Normal, QIcon.Off)
    ic.addPixmap(pixmap(name, size, selected or color, pad=pad), QIcon.Selected, QIcon.Off)
    ic.addPixmap(pixmap(name, size, active or color, pad=pad), QIcon.Active, QIcon.Off)
    ic.addPixmap(pixmap(name, size, "#4a4d55", pad=pad), QIcon.Disabled, QIcon.Off)
    return ic


def nav_icon(name: str) -> QIcon:
    """Sidebar icon: muted normally, accent when its page is selected."""
    name = {"transcriptions": "transcripts"}.get(name, name)
    return icon(name, TEXT2, 18, selected=ACCENT_TEXT)


def check_icon(size: int = 16, pad: int = 0) -> QIcon:
    return icon("check", SUCCESS, size, pad=pad)


def muted_icon(name: str, size: int = 16) -> QIcon:
    return icon(name, TEXT3, size)


# ---------------------------------------------------------------- app mark
BARS = (14, 28, 40, 28, 18, 30, 12)       # same silhouette as the tray icon (app.make_icon)


def draw_mark(p: QPainter, rect: QRectF, color: str = ACCENT) -> None:
    """The LocalFlow mark: an accent tile with seven white waveform bars, drawn into ``rect`` (square)."""
    s = rect.width() / 64.0
    p.save()
    p.setRenderHint(QPainter.Antialiasing, True)
    p.translate(rect.topLeft())
    p.scale(s, s)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawRoundedRect(QRectF(2, 2, 60, 60), 16, 16)
    p.setBrush(QColor("white"))
    for i, h in enumerate(BARS):
        p.drawRoundedRect(QRectF(9 + i * 7.6, 32 - h / 2, 4, h), 2, 2)
    p.restore()


@lru_cache(maxsize=16)
def mark_pixmap(size: int = 22) -> QPixmap:
    px = size * _SCALE
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    draw_mark(p, QRectF(0, 0, px, px))
    p.end()
    pm.setDevicePixelRatio(_SCALE)
    return pm


_APP_ICON: list = []


def app_icon() -> QIcon:
    """Window / taskbar icon in the sizes Windows asks for."""
    if not _APP_ICON:
        ic = QIcon()
        for n in (16, 20, 24, 32, 40, 48, 64, 128, 256):
            pm = QPixmap(n, n)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            draw_mark(p, QRectF(0, 0, n, n))
            p.end()
            ic.addPixmap(pm)
        _APP_ICON.append(ic)
    return _APP_ICON[0]
