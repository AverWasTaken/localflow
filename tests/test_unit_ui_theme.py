"""Theme widgets from the UI redesign: themed confirm dialog, icon set, copy feedback."""
import unittest


class ThemeWidgets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.qapp = QApplication.instance() or QApplication([])

    def test_confirm_dialog_defaults_to_cancel(self):
        from PySide6.QtWidgets import QDialog
        from localflow.theme import ConfirmDialog
        d = ConfirmDialog(None, "Clear history", "Delete all?", "Delete all", "Can't be undone.")
        self.assertEqual(d.windowTitle(), "Clear history")
        self.assertEqual(d.ok_btn.text(), "Delete all")
        self.assertEqual(d.info_label.text(), "Can't be undone.")
        self.assertTrue(d.cancel_btn.isDefault())
        self.assertFalse(d.ok_btn.isDefault())
        d.reject()                                  # what Esc does
        self.assertEqual(d.result(), QDialog.Rejected)
        d2 = ConfirmDialog(None, "t", "q", "OK")
        self.assertTrue(d2.info_label.isHidden())
        d2.ok_btn.click()
        self.assertEqual(d2.result(), QDialog.Accepted)

    def test_key_chip_whole_area_clickable(self):
        from PySide6.QtCore import QPoint
        from localflow.theme import KeyChip
        c = KeyChip("Ctrl")
        c.resize(56, 32)
        self.assertTrue(c.hitButton(QPoint(2, 2)))
        self.assertTrue(c.hitButton(QPoint(54, 30)))
        self.assertFalse(c.hitButton(QPoint(60, 10)))

    def test_icons_render(self):
        from localflow import icons
        for name in icons.NAMES:
            pm = icons.pixmap(name, 16)
            self.assertFalse(pm.isNull(), name)
            self.assertFalse(pm.toImage().isNull())
        self.assertFalse(icons.app_icon().isNull())
        self.assertFalse(icons.nav_icon("transcriptions").isNull())

    def test_flash_copied_keeps_label(self):
        from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
        from localflow.history_ui import flash_copied
        from localflow.theme import btn
        b = btn("Copy", "mini", icon="copy")
        flash_copied(b, "Copy")
        self.assertEqual(b.text(), "Copied")
        loop = QEventLoop()
        QTimer.singleShot(1300, loop.quit)
        loop.exec()
        QCoreApplication.processEvents()
        self.assertEqual(b.text(), "Copy")
        ib = btn("", icon="copy", tip="Copy the text")
        flash_copied(ib, "")
        self.assertEqual(ib.toolTip(), "Copied")
        loop = QEventLoop()
        QTimer.singleShot(1300, loop.quit)
        loop.exec()
        self.assertEqual(ib.toolTip(), "Copy the text")
        self.assertEqual(ib.text(), "")


if __name__ == "__main__":
    unittest.main()
