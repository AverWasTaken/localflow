"""Home page: usage numbers in one strip + the dictation history list."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .history_ui import HistoryList
from .theme import hsep, label
from .uifmt import fmt_int, fmt_saved, fmt_streak, fmt_wpm

STATS = (("words_7d", "Words this week", fmt_int), ("total_words", "Words in total", fmt_int),
         ("avg_wpm", "Words per minute", fmt_wpm), ("time_saved_s", "Saved vs typing", fmt_saved),
         ("streak_days", "Day streak", fmt_streak))


def stat_values(stats: dict | None) -> list[tuple[str, str, str]]:
    """[(key, label, formatted value)] in display order; missing numbers show as zero."""
    s = stats or {}
    return [(k, lbl, f(s.get(k, 0))) for k, lbl, f in STATS]


class StatsStrip(QFrame):
    """One panel, five numbers separated by hairlines (not five cards)."""

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 14, 4, 14)
        lay.setSpacing(0)
        self.values: dict[str, QLabel] = {}
        for i, (key, lbl, _f) in enumerate(STATS):
            if i:
                lay.addWidget(hsep(vertical=True))
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(18, 0, 18, 0)
            cl.setSpacing(2)
            v = label("0", "statValue")
            cl.addWidget(v)
            cl.addWidget(label(lbl, "statLabel"))
            lay.addWidget(cell, 1)
            self.values[key] = v
        self.setToolTip("Time saved compares your dictated words with typing them at 40 words per minute.")

    def set_stats(self, stats: dict | None) -> None:
        for key, _lbl, val in stat_values(stats):
            self.values[key].setText(val)


class HomePage(QWidget):
    def __init__(self, history):
        super().__init__()
        self.history = history
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 26, 32, 20)
        lay.setSpacing(16)
        lay.addWidget(label("Home", "pageTitle"))
        self.stats = StatsStrip()
        lay.addWidget(self.stats)
        lay.addSpacing(6)
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
