"""Update prompt + controller: checks GitHub in the background, offers the new version, downloads its setup
program with a progress bar, then starts it and quits so setup can replace this copy (see updater.py)."""
from __future__ import annotations

import logging
import threading
from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QHBoxLayout, QProgressBar, QSystemTrayIcon, QVBoxLayout, QWidget

from . import updater
from .theme import (DARK_QSS, SURFACE, _dark_palette, _set_window_icon, btn, enable_dark_titlebar, label)

log = logging.getLogger(__name__)

LAUNCH_DELAY_MS = 15_000            # first check after start
RECHECK_MS = 24 * 3600 * 1000       # then once a day while running
DEFER_MS = 60_000                   # launch-check prompt postponed while dictating / recording / in game mode


class UpdateDialog(QDialog):
    """One themed window for every update state: offer, downloading, waiting, installing, error, message."""

    def __init__(self, ctl: "UpdateController"):
        super().__init__(None)
        self.ctl = ctl
        self.setObjectName("confirm")
        self.setWindowTitle("LocalFlow update")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setPalette(_dark_palette())
        self.setStyleSheet(DARK_QSS)
        _set_window_icon(self)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumWidth(440)
        self.setMaximumWidth(540)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(24, 22, 24, 22)
        bl.setSpacing(8)
        self.title = label("", "dialogTitle", wrap=True)
        self.text = label("", "dialogText", wrap=True)
        self.notes = label("", role="faint", wrap=True)
        self.notes.setTextFormat(Qt.PlainText)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.status = label("", role="muted", wrap=True)
        for w in (self.title, self.text, self.notes, self.bar, self.status):
            bl.addWidget(w)
        outer.addWidget(body)
        bar = QWidget()
        bar.setObjectName("dialogButtons")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(24, 14, 24, 14)
        hl.setSpacing(8)
        self.skip_btn = btn("Skip this version", "ghost")
        self.release_btn = btn("Open release page", "ghost")
        self.later_btn = btn("Later")
        self.cancel_btn = btn("Cancel")
        self.close_btn = btn("Close")
        self.ok_btn = btn("Download and install", "primary")
        hl.addWidget(self.skip_btn)
        hl.addWidget(self.release_btn)
        hl.addStretch(1)
        for b in (self.later_btn, self.cancel_btn, self.close_btn, self.ok_btn):
            b.setMinimumWidth(88)
            b.setAutoDefault(True)
            hl.addWidget(b)
        outer.addWidget(bar)
        self.skip_btn.clicked.connect(ctl.skip)
        self.release_btn.clicked.connect(ctl.open_release_page)
        self.later_btn.clicked.connect(self.close)
        self.close_btn.clicked.connect(self.close)
        self.cancel_btn.clicked.connect(ctl.cancel)
        self.ok_btn.clicked.connect(ctl.download_and_install)
        self._dark_done = False

    def show_state(self, title: str, text: str, buttons: tuple[str, ...], notes: str = "", progress: bool = False,
                   status: str = "") -> None:
        self.title.setText(title)
        self.text.setText(text)
        self.text.setVisible(bool(text))
        self.notes.setText(notes)
        self.notes.setVisible(bool(notes))
        self.bar.setVisible(progress)
        self.status.setText(status)
        self.status.setVisible(bool(status))
        names = {"skip": self.skip_btn, "release": self.release_btn, "later": self.later_btn,
                 "cancel": self.cancel_btn, "close": self.close_btn, "ok": self.ok_btn}
        for k, b in names.items():
            b.setVisible(k in buttons)
        for k in ("ok", "close", "cancel", "later"):       # Enter: the main action of this state
            if k in buttons:
                names[k].setDefault(True)
                names[k].setFocus()
                break
        self.adjustSize()

    def set_progress(self, done: int, total: int) -> None:
        self.bar.setValue(int(1000 * done / total) if total > 0 else 0)
        self.status.setText(f"{updater.mb(done)} of {updater.mb(total)} MB")
        self.status.setVisible(True)

    def showEvent(self, e):
        if not self._dark_done:
            enable_dark_titlebar(self, SURFACE)
            self._dark_done = True
        super().showEvent(e)

    def reject(self):
        """Esc, the title bar X and close() all end here (QDialog.closeEvent calls reject). Downloading or waiting
        to install: closing the window cancels that (nothing carries on unseen)."""
        if self.ctl.busy:
            self.ctl.cancel()
        super().reject()


class UpdateController(QObject):
    """Owns the update check timers, the dialog and the download thread. ``app`` is the LocalFlowApp."""
    _checked = Signal(object, str, bool)        # UpdateInfo or None, error ("" = ok), manual
    _progress = Signal(int, int)
    _downloaded = Signal(str, str)              # path, error ("" = ok; "cancelled" = user cancelled)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.cfg = app.cfg
        self.info: Optional[updater.UpdateInfo] = None
        self.dialog: Optional[UpdateDialog] = None
        self._checking = False
        self._manual_waiting = False            # a manual check asked while a launch check was already running
        self._cancel: Optional[threading.Event] = None
        self._state = "idle"                    # idle | downloading | waiting | installing
        self._installer = ""
        self._pending: Optional[updater.UpdateInfo] = None      # launch-check offer waiting for a quiet moment
        self._wait_timer = QTimer(self)
        self._wait_timer.setInterval(1000)
        self._wait_timer.timeout.connect(self._install_when_free)
        self._daily = QTimer(self)
        self._daily.setInterval(RECHECK_MS)
        self._daily.timeout.connect(lambda: self.check(manual=False))
        self._checked.connect(self._on_checked)
        self._progress.connect(self._on_progress)
        self._downloaded.connect(self._on_downloaded)

    @property
    def busy(self) -> bool:
        return self._state != "idle"

    def start(self) -> None:
        """Schedule the launch check and the daily re-check (both skipped while updates.check_on_launch is off)."""
        QTimer.singleShot(LAUNCH_DELAY_MS, lambda: self.check(manual=False))
        self._daily.start()
        tray = getattr(self.app, "tray", None)
        if tray is not None:
            tray.messageClicked.connect(self._raise)        # the "update available" balloon

    # ----- check -----
    def check(self, manual: bool = False) -> None:
        """Look for an update on a background thread. Manual checks always report back; launch checks only speak up
        when there is something to offer."""
        if not manual and not self.cfg.get("updates.check_on_launch", True):
            return
        if self.busy or (self.dialog is not None and self.dialog.isVisible() and self.info is not None):
            if manual:
                self._raise()
            return
        if self._checking:
            self._manual_waiting = self._manual_waiting or manual
            return
        self._checking = True
        self._manual_waiting = manual

        def run():
            info, err = None, ""
            try:
                info = updater.check_latest()
            except updater.UpdateError as e:
                err = str(e)
            except Exception as e:      # never let a check take anything down
                log.exception("update check failed")
                err = f"{type(e).__name__}: {e}"
            self._checked.emit(info, err, False)
        threading.Thread(target=run, name="update-check", daemon=True).start()

    def _on_checked(self, info, err: str, _manual: bool) -> None:
        self._checking = False
        manual, self._manual_waiting = self._manual_waiting, False
        if err:
            log.info("update check: %s", err)
            if manual:
                self._show("error", err=f"Couldn't check for updates: {err}")
            return
        if info is None:
            log.info("update check: %s is the latest version", updater.current_version())
            if manual:
                self._show("latest")
            return
        log.info("update check: %s available", info.tag)
        if not updater.should_offer(info, str(self.cfg.get("updates.skipped_version", "") or ""), manual):
            log.info("update check: %s was skipped by the user", info.version)
            return
        self.info = info
        if manual:
            self._pending = None        # this offer replaces one a launch check is holding back
            self._show("offer", activate=True)
        else:
            self._pending = info
            self._offer_when_quiet()

    def _quiet(self) -> bool:
        return not (getattr(self.app, "game_active", False) or self.app.update_blocker())

    def _offer_when_quiet(self) -> None:
        """Launch check: don't pop a window up mid-dictation, mid-recording or over a game."""
        if self._pending is None or self.busy:
            return
        skipped = str(self.cfg.get("updates.skipped_version", "") or "")
        if not self.cfg.get("updates.check_on_launch", True) or not updater.should_offer(self._pending, skipped):
            self._pending = None        # turned off or skipped while it waited
            return
        if not self._quiet():
            QTimer.singleShot(DEFER_MS, self._offer_when_quiet)
            return
        self.info, self._pending = self._pending, None
        self._show("offer", activate=False)
        tray = getattr(self.app, "tray", None)
        if tray is not None and QSystemTrayIcon.supportsMessages():
            tray.showMessage("LocalFlow update", f"LocalFlow {self.info.version} is available.",
                             QSystemTrayIcon.Information, 8000)

    # ----- dialog -----
    def _dialog(self) -> UpdateDialog:
        if self.dialog is None:
            self.dialog = UpdateDialog(self)
        return self.dialog

    def _raise(self) -> None:
        d = self.dialog
        if d is not None:
            if d.isMinimized():
                d.showNormal()
            d.show()
            d.raise_()
            d.activateWindow()

    def _show(self, state: str, err: str = "", activate: bool = True) -> None:
        d = self._dialog()
        cur = updater.current_version()
        if state == "latest":
            d.show_state("You're on the latest version", f"LocalFlow {cur} is the newest release.", ("close",))
        elif state == "error":
            d.show_state("Update problem", err, ("release", "close"))
        elif state == "offer":
            i = self.info
            how = ("LocalFlow closes, installs the update and starts again."
                   if updater.can_self_update() else
                   "This copy wasn't installed with the setup program, so Download opens the release page.")
            d.show_state(f"Update available ({i.tag})", f"You have {cur}. {how}",
                         ("skip", "later", "ok"), notes=updater.trim_notes(i.notes))
            d.ok_btn.setText("Download and install" if updater.can_self_update() else "Download")
        if not d.isVisible():
            d.setAttribute(Qt.WA_ShowWithoutActivating, not activate)
            d.show()
        if activate:
            self._raise()

    # ----- actions -----
    def open_release_page(self) -> None:
        url = self.info.html_url if self.info is not None else updater.RELEASES_URL
        QDesktopServices.openUrl(QUrl(url))

    def skip(self) -> None:
        if self.info is not None:
            self.cfg.set("updates.skipped_version", self.info.version)
            log.info("update: user skipped %s", self.info.version)
        if self.dialog is not None:
            self.dialog.close()

    def download_and_install(self) -> None:
        if self.info is None or self.busy:
            return
        if not updater.can_self_update():
            self.open_release_page()
            self.dialog.close()
            return
        info = self.info
        self._state = "downloading"
        self._cancel = threading.Event()
        cancel = self._cancel
        d = self._dialog()
        d.show_state(f"Downloading LocalFlow {info.version}", "", ("cancel",), progress=True,
                     status=f"0.0 of {updater.mb(info.size)} MB")
        d.bar.setValue(0)

        def run():
            try:
                path = updater.download(info, progress=self._progress.emit, cancel=cancel)
                self._downloaded.emit(str(path), "")
            except updater.UpdateCancelled:
                self._downloaded.emit("", "cancelled")
            except updater.UpdateError as e:
                self._downloaded.emit("", str(e))
            except Exception as e:
                log.exception("update download failed")
                self._downloaded.emit("", f"{type(e).__name__}: {e}")
        threading.Thread(target=run, name="update-download", daemon=True).start()

    def cancel(self) -> None:
        if self._state == "downloading" and self._cancel is not None:
            self._cancel.set()          # _on_downloaded resets the state
        elif self._state == "waiting":
            self._wait_timer.stop()
            self._state = "idle"
            log.info("update: install cancelled while waiting")
            if self.dialog is not None:
                self.dialog.close()

    def _on_progress(self, done: int, total: int) -> None:
        if self._state == "downloading" and self.dialog is not None:
            self.dialog.set_progress(done, total)

    def _on_downloaded(self, path: str, err: str) -> None:
        if self._cancel is not None and self._cancel.is_set():
            err = "cancelled"           # cancelled after the download had already finished: don't install
        if err == "cancelled":
            self._state = "idle"
            log.info("update: download cancelled")
            if self.dialog is not None and self.dialog.isVisible():
                self.dialog.close()
            return
        if err:
            self._state = "idle"
            log.warning("update: download failed: %s", err)
            self._show("error", err=f"The download failed: {err}\n\nYou can download the new version from the "
                                    "release page instead.")
            return
        self._installer = path
        self._state = "waiting"
        self._install_when_free()

    def _install_when_free(self) -> None:
        """Start setup once no dictation / meeting recording is running, then quit so setup can replace us."""
        if self._state != "waiting":
            self._wait_timer.stop()
            return
        why = self.app.update_blocker()
        if not why:
            self.app.installing_update = True  # no new dictation / meeting from here on...
            why = self.app.update_blocker()    # ...then check again, so nothing starts between check and quit
            if why:
                self.app.installing_update = False
        if why:
            if not self._wait_timer.isActive():
                self._wait_timer.start()
                self.dialog.show_state("Update downloaded", f"The update installs when {why} finishes.",
                                       ("cancel",), progress=True)
                self.dialog.bar.setValue(1000)
            return
        self._wait_timer.stop()
        try:
            updater.launch_installer(self._installer)
        except Exception as e:
            self._state = "idle"
            self.app.installing_update = False
            log.exception("update: could not start setup")
            self._show("error", err=f"Couldn't start the setup program: {e}")
            return
        self._state = "installing"
        self.dialog.show_state("Installing the update",
                               "LocalFlow is closing now. It starts again when the update is installed.", (),
                               progress=True)
        self.dialog.bar.setValue(1000)
        QTimer.singleShot(800, self.app.quit)
