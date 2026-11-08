"""Offline tests for autostart: only report, create or delete the LocalFlow task when it runs this copy
(schtasks is faked; nothing is registered or deleted)."""
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import autostart  # noqa: E402
from localflow.paths import PROJECT_DIR  # noqa: E402

OURS = str(PROJECT_DIR / ".venv" / "Scripts" / "pythonw.exe")
OTHER = r"C:\Somewhere Else\LocalFlow\.venv\Scripts\pythonw.exe"
SIBLING = str(PROJECT_DIR) + "2\\.venv\\Scripts\\pythonw.exe"   # prefix of our path, but a different folder


def task_xml(cmd: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-16"?>\n<Task><Actions Context="Author"><Exec>\n'
            f'      <Command>{escape(cmd)}</Command>\n      <Arguments>-m localflow</Arguments>\n'
            '</Exec></Actions></Task>\n')


class FakeSchtasks:
    """Answers /Query with the given task (or "not found") and records every call."""

    def __init__(self, cmd):
        self.cmd = cmd
        self.calls = []

    def __call__(self, args, timeout=60):
        self.calls.append(args)
        if args[:2] == ["schtasks", "/Query"]:
            if self.cmd is None:
                return subprocess.CompletedProcess(args, 1, "", "ERROR: The system cannot find the file specified.")
            return subprocess.CompletedProcess(args, 0, task_xml(self.cmd), "")
        return subprocess.CompletedProcess(args, 0, "", "")

    def changed(self):
        return [c for c in self.calls if c[:2] != ["schtasks", "/Query"]]


class Autostart(unittest.TestCase):
    def run_with(self, cmd, fn, *a):
        fake = FakeSchtasks(cmd)
        with mock.patch.object(autostart, "_run", fake):
            return fn(*a), fake

    def test_reads_command(self):
        exe, _ = self.run_with(OURS, autostart.task_executable)
        self.assertEqual(exe, OURS)
        exe, _ = self.run_with(f'"{OTHER}"', autostart.task_executable)
        self.assertEqual(exe, OTHER)
        exe, _ = self.run_with(None, autostart.task_executable)
        self.assertIsNone(exe)

    def test_enabled_only_for_this_copy(self):
        self.assertTrue(self.run_with(OURS, autostart.is_enabled)[0])
        self.assertFalse(self.run_with(OTHER, autostart.is_enabled)[0])
        self.assertFalse(self.run_with(SIBLING, autostart.is_enabled)[0])
        self.assertFalse(self.run_with(None, autostart.is_enabled)[0])

    def test_xml_escaped_path(self):
        odd = r"C:\Users\A&B <x>\LocalFlow\.venv\Scripts\pythonw.exe"
        exe, _ = self.run_with(odd, autostart.task_executable)
        self.assertEqual(exe, odd)
        with mock.patch.object(autostart, "PROJECT_DIR", Path(r"C:\Users\A&B <x>\LocalFlow")):
            self.assertTrue(self.run_with(odd, autostart.is_enabled)[0])

    def test_case_insensitive_path(self):
        self.assertTrue(self.run_with(OURS.upper(), autostart.is_enabled)[0])

    def test_enable_refuses_other_copy(self):
        ok, fake = self.run_with(OTHER, autostart.set_enabled, True)
        self.assertFalse(ok)
        self.assertEqual(fake.changed(), [])

    def test_disable_leaves_other_copy(self):
        ok, fake = self.run_with(OTHER, autostart.set_enabled, False)
        self.assertTrue(ok)
        self.assertEqual(fake.changed(), [])

    def test_disable_without_task_is_noop(self):
        ok, fake = self.run_with(None, autostart.set_enabled, False)
        self.assertTrue(ok)
        self.assertEqual(fake.changed(), [])

    def test_disable_ours_deletes(self):
        ok, fake = self.run_with(OURS, autostart.set_enabled, False)
        self.assertTrue(ok)
        self.assertEqual(fake.changed(), [["schtasks", "/Delete", "/TN", "LocalFlow", "/F"]])

    def test_enable_runs_register_script(self):
        for cmd in (None, OURS):
            ok, fake = self.run_with(cmd, autostart.set_enabled, True)
            self.assertTrue(ok)
            (call,) = fake.changed()
            self.assertEqual(call[0], "powershell")
            self.assertTrue(call[-1].endswith("register-task.ps1"))
            self.assertNotIn("-Force", call)


if __name__ == "__main__":
    unittest.main()
