"""Is a text box focused right now? Checked right before a dictation is inserted (pipeline -> app's inserter).

Three answers: "field" (paste as usual), "none" (only on positive evidence that nothing typeable is focused: copy to
the clipboard and show a toast) and "unknown" (everything else: behave exactly as before). Read-only-ness comes from
ValuePattern / LegacyIAccessible state, never from TextPattern (a browser page with nothing focused still has a
TextPattern and a caret range).

Win32 facts are read on the calling thread (microseconds); UIA facts on the shared UIA thread (uia.py) with a short
deadline. Nothing here logs or reads field text.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import uia as U
from . import winutil

log = logging.getLogger(__name__)

FIELD, NONE, UNKNOWN = "field", "none", "unknown"

# UIA control type ids (UIAutomationClient.h)
CT_BUTTON, CT_EDIT, CT_HYPERLINK, CT_LISTITEM, CT_MENUITEM = 50000, 50004, 50005, 50007, 50011
CT_TABITEM, CT_TREEITEM, CT_DOCUMENT = 50019, 50024, 50030
CT_NAMES = {50000: "Button", 50001: "Calendar", 50002: "CheckBox", 50003: "ComboBox", 50004: "Edit",
            50005: "Hyperlink", 50006: "Image", 50007: "ListItem", 50008: "List", 50009: "Menu", 50010: "MenuBar",
            50011: "MenuItem", 50012: "ProgressBar", 50013: "RadioButton", 50014: "ScrollBar", 50015: "Slider",
            50016: "Spinner", 50017: "StatusBar", 50018: "Tab", 50019: "TabItem", 50020: "Text", 50021: "ToolBar",
            50022: "ToolTip", 50023: "Tree", 50024: "TreeItem", 50025: "Custom", 50026: "Group", 50027: "Thumb",
            50028: "DataGrid", 50029: "DataItem", 50030: "Document", 50031: "SplitButton", 50032: "Window",
            50033: "Pane", 50034: "Header", 50035: "HeaderItem", 50036: "Table", 50037: "TitleBar",
            50038: "Separator"}
NON_TEXT_TYPES = {CT_BUTTON, CT_LISTITEM, CT_TREEITEM, CT_HYPERLINK, CT_MENUITEM, CT_TABITEM}
UIA_VALUE_PATTERN, UIA_LEGACY_PATTERN = 10002, 10018
STATE_SYSTEM_READONLY = 0x40

DESKTOP_CLASSES = {"progman", "workerw"}
TASKBAR_CLASSES = {"shell_traywnd", "shell_secondarytraywnd"}
# windows that take typed text without exposing a text box (consoles / terminals)
TERMINAL_CLASSES = {"consolewindowclass", "cascadia_hosting_window_class", "mintty", "putty", "virtualconsoleclass"}
BROWSER_FRAMEWORKS = {"chrome", "gecko"}
GECKO_WINDOW_CLASSES = {"mozillawindowclass"}      # Gecko through the MSAA bridge may report another framework


def _edit_class(cls: str) -> bool:
    c = cls.lower()
    return c == "edit" or c.startswith("richedit") or c.startswith("scintilla") or c in TERMINAL_CLASSES


@dataclass
class Facts:
    exe: str = ""
    fg_class: str = ""
    focus_class: str = ""
    caret: bool = False             # GetGUIThreadInfo hwndCaret of the foreground thread
    own: bool = False               # the foreground window is ours
    no_window: bool = False         # no foreground window at all
    uia: str = ""                   # "" = read | skipped | timeout | busy | error:<Type> | no focus
    uia_own: bool = False           # the UIA focus is in our own process
    ctype: int = 0
    framework: str = ""
    el_class: str = ""
    value_ro: Optional[bool] = None     # ValuePattern.IsReadOnly (None: no ValuePattern)
    legacy_ro: Optional[bool] = None    # LegacyIAccessible state has READONLY (None: no pattern)
    children: Optional[int] = None      # documents only: 1 = has a child, 0 = none, None = not checked

    def describe(self) -> str:
        """Log line: control type / framework / classes, never text."""
        parts = [f"exe={self.exe or '?'}", f"fg={self.fg_class or '-'}"]
        if self.focus_class:
            parts.append(f"focus={self.focus_class}")
        if self.caret:
            parts.append("caret")
        if self.uia:
            parts.append(f"uia={self.uia}")
        if self.ctype:
            parts.append(f"type={CT_NAMES.get(self.ctype, self.ctype)}")
        if self.framework:
            parts.append(f"fw={self.framework}")
        if self.el_class:
            parts.append(f"cls={self.el_class[:40]}")
        if self.value_ro is not None:
            parts.append(f"value_ro={int(self.value_ro)}")
        if self.legacy_ro is not None:
            parts.append(f"legacy_ro={int(self.legacy_ro)}")
        if self.children is not None:
            parts.append(f"children={self.children}")
        return " ".join(parts)


@dataclass
class Decision:
    kind: str                       # field | none | unknown
    reason: str
    shell: bool = False             # desktop / taskbar / override: no keystroke at all, just copy
    facts: Facts = field(default_factory=Facts)
    ms: int = 0


def decide(f: Facts, overrides: Optional[dict] = None) -> Decision:
    """Pure decision table (see module docstring). Field evidence wins; "none" only on positive evidence."""
    def d(kind, reason, shell=False):
        return Decision(kind, reason, shell, f)

    if f.no_window:
        return d(UNKNOWN, "no foreground window")
    if f.own:
        return d(UNKNOWN, "own process")
    ov = (overrides or {}).get((f.exe or "").lower()) if f.exe else None
    if ov in (FIELD, NONE):
        return d(ov, "override", shell=(ov == NONE))
    # ---- field (wins)
    if f.caret:
        return d(FIELD, "system caret")
    if f.focus_class and _edit_class(f.focus_class):
        return d(FIELD, f"focus class {f.focus_class}")
    if f.fg_class.lower() in TERMINAL_CLASSES:
        return d(FIELD, f"terminal {f.fg_class}")
    if f.uia_own:
        return d(UNKNOWN, "UIA focus in own process")
    if f.ctype == CT_EDIT:
        return d(FIELD, "UIA Edit")
    if f.value_ro is False:
        return d(FIELD, f"editable value ({CT_NAMES.get(f.ctype, f.ctype)})")
    # ---- none (positive evidence only)
    fg = f.fg_class.lower()
    if fg in DESKTOP_CLASSES:
        return d(NONE, "desktop", shell=True)
    if fg in TASKBAR_CLASSES:
        return d(NONE, "taskbar", shell=True)
    if f.uia:                       # UIA not read (timeout / error / busy ...): no evidence either way
        return d(UNKNOWN, f"UIA {f.uia}")
    if f.ctype in NON_TEXT_TYPES:
        return d(NONE, f"UIA {CT_NAMES[f.ctype]}")
    browser = f.framework.lower() in BROWSER_FRAMEWORKS or fg in GECKO_WINDOW_CLASSES
    if f.ctype == CT_DOCUMENT and browser and (f.value_ro is True or f.legacy_ro is True) and (f.children or 0) >= 1:
        return d(NONE, f"read-only {f.framework or 'browser'} page")
    return d(UNKNOWN, f"UIA {CT_NAMES.get(f.ctype, f.ctype or '?')}")


# ----------------------------------------------------------------------------- fact gathering
def _uia_job(ctx) -> dict:
    """Runs on the UIA thread. Returns plain values only (no COM pointers leave this function)."""
    el = ctx.uia.GetFocusedElement()
    if not el:
        return {"uia": "no focus"}
    try:
        pid = int(el.CurrentProcessId)
    except Exception:
        pid = 0
    if pid == ctx.own_pid:
        return {"uia_own": True}
    out: dict = {}
    try:
        out["ctype"] = int(el.CurrentControlType)
    except Exception:
        out["ctype"] = 0
    for key, prop in (("framework", "CurrentFrameworkId"), ("el_class", "CurrentClassName")):
        try:
            out[key] = str(getattr(el, prop) or "")
        except Exception:
            pass
    Um = ctx.U
    try:
        vp = el.GetCurrentPattern(UIA_VALUE_PATTERN)
        if vp:
            out["value_ro"] = bool(vp.QueryInterface(Um.IUIAutomationValuePattern).CurrentIsReadOnly)
    except Exception:
        pass
    try:
        lp = el.GetCurrentPattern(UIA_LEGACY_PATTERN)
        if lp:
            st = int(lp.QueryInterface(Um.IUIAutomationLegacyIAccessiblePattern).CurrentState)
            out["legacy_ro"] = bool(st & STATE_SYSTEM_READONLY)
    except Exception:
        pass
    if out["ctype"] == CT_DOCUMENT:
        try:
            out["children"] = 1 if ctx.walker.GetFirstChildElement(el) else 0
        except Exception:
            pass
    return out


def _win32_facts() -> tuple[Facts, int]:
    f = Facts()
    hwnd = U._fg_hwnd()
    if not hwnd:
        f.no_window = True
        return f, 0
    if U._window_pid(hwnd) == os.getpid():
        f.own = True
        return f, hwnd
    f.fg_class = U.class_name(hwnd)
    try:
        f.exe = winutil.window_info(hwnd)[0]
    except Exception:
        pass
    g = U.gui_thread_info(hwnd)
    if g is not None:
        f.caret = bool(g.hwndCaret)
        if g.hwndFocus:
            f.focus_class = U.class_name(int(g.hwndFocus))
    return f, hwnd


def gather(timeout: float = 0.15, thread: Optional[U.UiaThread] = None) -> Facts:
    t0 = time.perf_counter()
    f, _hwnd = _win32_facts()
    if f.no_window or f.own:
        f.uia = "skipped"
        return f
    pre = decide(f)
    if pre.kind == FIELD:           # Win32 already proved a text box: no UIA round trip
        f.uia = "skipped"
        return f
    left = timeout - (time.perf_counter() - t0)
    if left <= 0.005:
        f.uia = "timeout"
        return f
    try:
        res = (thread or U.shared()).call(_uia_job, left)
    except TimeoutError as e:
        f.uia = "busy" if "busy" in str(e) else "timeout"
        return f
    except Exception as e:
        f.uia = f"error:{type(e).__name__}"
        return f
    for k, v in (res or {}).items():
        setattr(f, k, v)
    return f


def probe(cfg=None, timeout: float = 0.15, gather_fn: Optional[Callable[..., Facts]] = None) -> Decision:
    """Decide for whatever is focused now. Never raises; logs the decision and why (no text)."""
    t0 = time.perf_counter()
    overrides = {}
    try:
        overrides = {str(k).lower(): str(v).lower()
                     for k, v in dict((cfg.get("insert.no_field_override", {}) if cfg is not None else {}) or {}).items()}
    except Exception:
        overrides = {}
    try:
        f = (gather_fn or gather)(timeout)
        dec = decide(f, overrides)
    except Exception as e:
        log.exception("field probe failed")
        dec = Decision(UNKNOWN, f"probe error {type(e).__name__}")
    dec.ms = int((time.perf_counter() - t0) * 1000)
    log.info("field probe: %s (%s) %d ms | %s", dec.kind, dec.reason, dec.ms, dec.facts.describe())
    return dec


def warm() -> None:
    """Start the UIA thread and touch the focused app's UIA provider (recording start). Never raises/blocks."""
    try:
        hwnd = U._fg_hwnd()
        if not hwnd or U._window_pid(hwnd) == os.getpid():     # never query our own windows over UIA
            return
        U.shared().post(_warm_job)
    except Exception:
        log.debug("field probe warm-up failed", exc_info=True)


def _warm_job(ctx) -> None:
    el = ctx.uia.GetFocusedElement()
    if el:
        try:
            el.CurrentControlType
        except Exception:
            pass


# ----------------------------------------------------------------------------- insert-or-copy
COPIED_TOAST = "Copied to clipboard"
COPY_FAILED_TOAST = "Couldn't copy to clipboard"


def deliver(text: str, *, cfg, insert: Callable[..., Optional[dict]], copy: Callable[..., bool],
            probe_fn: Optional[Callable[..., Decision]] = None, mode: str = "paste", restore: bool = True,
            restore_delay_ms: int = 300) -> Optional[dict]:
    """The dictation inserter: insert as before unless the probe says no text box is focused.

    "none" in paste mode (UIA evidence): Ctrl+V still goes out (apps that route a paste into their message box keep
    working) but the clipboard is NOT restored, so the text stays there; desktop / taskbar / override "none", or type
    mode: no keystroke, just copy. Adds ``copied`` / ``copy_failed`` to the returned info."""
    if not cfg.get("insert.copy_when_no_field", True):
        return insert(text, mode=mode, restore=restore, restore_delay_ms=restore_delay_ms)
    dec = (probe_fn or probe)(cfg)
    if dec.kind != NONE:
        return insert(text, mode=mode, restore=restore, restore_delay_ms=restore_delay_ms)
    if mode == "paste" and not dec.shell:
        info = dict(insert(text, mode="paste", restore=False, restore_delay_ms=restore_delay_ms,
                           clipboard_history=True, fallback_type=False) or {})
        ok = info.get("ok", True) is not False
    else:
        try:
            ok = bool(copy(text, history=True))
        except Exception:
            log.exception("copy to clipboard failed")
            ok = False
        info = {"mode": "copy", "ok": ok, "reason": "" if ok else "clipboard busy"}
    info["no_field"] = dec.reason
    info["copied" if ok else "copy_failed"] = True
    if not ok:
        what = "COPY FAILED"
    elif info.get("mode") == "paste":
        what = "copied to clipboard, Ctrl+V sent, clipboard kept"
    else:
        what = "copied to clipboard, no keystroke"
    log.info("no text box (%s): %s", dec.reason, what)
    return info
