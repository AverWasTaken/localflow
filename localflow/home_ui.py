"""Home page: usage numbers in one strip + the dictation history list."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .history_ui import HistoryList
from .icons import pixmap
from .palette import TEXT2
from .theme import label
from .uifmt import fmt_int, fmt_saved, fmt_streak, fmt_wpm

STATS = (("words_7d", "Words this week", fmt_int), ("total_words", "Words in total", fmt_int),
         ("avg_wpm", "Words per minute", fmt_wpm), ("time_saved_s", "Saved vs typing", fmt_saved),
         ("streak_days", "Day streak", fmt_streak))


def stat_values(stats: dict | None) -> list[tuple[str, str, str]]:
    """[(key, label, formatted value)] in display order; missing numbers show as zero."""
    s = stats or {}
    return [(k, lbl, f(s.get(k, 0))) for k, lbl, f in STATS]


_ICONS = {"words_7d": "calendar", "total_words": "words", "avg_wpm": "gauge", "time_saved_s": "clock",
          "streak_days": "flame"}


class StatsStrip(QFrame):
    """One panel with the five usage numbers, each with a small icon (not five separate cards)."""

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)
        self.values: dict[str, QLabel] = {}
        self._icons: list[QLabel] = []
        for i, (key, lbl, _f) in enumerate(STATS):
            cell = QWidget()
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(12)
            ic = QLabel()
            ic.setObjectName("statIcon")
            ic.setFixedSize(36, 36)
            ic.setAlignment(Qt.AlignCenter)
            ic.setPixmap(pixmap(_ICONS.get(key, "words"), 18, TEXT2))
            cl.addWidget(ic, 0, Qt.AlignVCenter)
            self._icons.append(ic)
            txt = QVBoxLayout()
            txt.setSpacing(0)
            v = label("0", "statValue")
            txt.addWidget(v)
            sl = label(lbl, "statLabel")
            txt.addWidget(sl)
            cl.addLayout(txt, 1)
            lay.addWidget(cell, 1)
            self.values[key] = v
        self.setToolTip("Time saved compares your dictated words with typing them at 40 words per minute.")

    ICONS_MIN_W = 810       # below this the icon tiles go, so the numbers and labels never clip

    def resizeEvent(self, e):
        show = self.width() >= self.ICONS_MIN_W
        for ic in self._icons:
            if ic.isHidden() == show:
                ic.setVisible(show)
        super().resizeEvent(e)

    def set_stats(self, stats: dict | None) -> None:
        for key, _lbl, val in stat_values(stats):
            self.values[key].setText(val)


class HomePage(QWidget):
    def __init__(self, history):
        super().__init__()
        self.history = history
        lay = QVBoxLayout(self)
        lay.setContentsMargins(36, 28, 36, 24)
        lay.setSpacing(16)
        lay.addWidget(label("Home", "pageTitle"))
        self.stats = StatsStrip()
        lay.addWidget(self.stats)
        lay.addSpacing(8)
        self.list = HistoryList(history, "History")
        lay.addWidget(self.list, 1)
        self.list.changed.connect(self.refresh_stats)
        self.list._auto.timeout.connect(self.refresh_stats)     # numbers move even when the shown rows do not

    def refresh_stats(self) -> None:
        try:
            s = self.history.stats() if self.history is not None else None
        except Exception:
            s = None
        self.stats.set_stats(s)

    def showEvent(self, e):
        self.refresh_stats()
        super().showEvent(e)
