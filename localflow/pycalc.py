"""Command mode calculations: decide whether a spoken request is a calculation (unit conversion, percentage,
arithmetic, date maths), pull the Python script the LLM wrote for it out of the reply, check it, and run it in a
locked-down child interpreter. The script's printed output is the answer, so numbers never come from the model's
mental maths.

Layers (each one alone should stop a hostile script; together they are defence in depth):
  1. check_code(): AST allowlist (only maths / date stdlib modules, no dunders, no introspection builtins, no
     module-valued attributes that lead to sys / os / ...).
  2. the child prelude: restricted builtins, an __import__ that only hands out the allowlist, a capped stdout and
     a deny-everything sys.addaudithook (blocks open, os.*, subprocess, socket, ctypes, winreg, new imports, ...).
  3. the child process: base interpreter in isolated mode (-I), fresh empty temp dir as cwd, no console window,
     a Windows job object (256 MB, one process, no clipboard / desktop access, killed with the job) and a timeout.
"""
from __future__ import annotations

import ast
import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ALLOWED_MODULES = ("math", "cmath", "fractions", "decimal", "datetime", "calendar", "statistics")
MAX_CODE_CHARS = 4000
MAX_OUTPUT_CHARS = 2000
MEMORY_LIMIT_MB = 256

_BANNED_NAMES = frozenset({
    "sys", "os", "modules", "builtins", "operator", "re", "functools", "importlib", "subprocess", "open", "exec",
    "eval", "compile", "input", "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "exit", "quit", "help", "memoryview", "type", "object", "super", "classmethod", "staticmethod", "property",
    "codecs", "io", "socket", "ctypes", "winreg", "shutil", "pathlib", "signal", "threading", "multiprocessing",
    "pickle", "marshal", "inspect", "gc", "weakref", "locale", "encodings", "mro", "format_map",
})
# frame / generator / traceback / code object attributes lead back to globals and the real builtins
_BANNED_ATTR_PREFIXES = ("_", "gi_", "f_", "tb_", "cr_", "ag_", "co_", "func_")


def _module_attrs() -> frozenset:
    """Attribute names on the allowlisted modules that hold OTHER modules (statistics.sys, calendar.locale,
    statistics.random, ...): reaching through them escapes the allowlist. Names that are themselves allowlisted
    (calendar.datetime, statistics.math) are fine, and so is datetime.datetime (a class)."""
    out = set()
    for name in ALLOWED_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        out.update(k for k, v in vars(mod).items() if isinstance(v, types.ModuleType))
    return frozenset(out - set(ALLOWED_MODULES))


_MODULE_ATTRS = _module_attrs()

# ----------------------------------------------------------------------------- extracting the code
_FENCE_CODE_RE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)(?:```|\Z)", re.S | re.I)
_CLOSED_FENCE_RE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)
NO_CALC = "NO_CALC"


def is_no_calc(reply: str) -> bool:
    """The model said the request is not a calculation."""
    s = (reply or "").strip().strip("`'\".").strip()
    return s.upper().startswith(NO_CALC) and "```" not in (reply or "")


def extract_code(reply: str, allow_bare: bool = True, closed_only: bool = False) -> Optional[str]:
    """The Python script in an LLM reply: the first fenced block; with ``allow_bare``, an unfenced reply that
    parses as Python and calls print() also counts. ``closed_only`` (a reply cut off at the token limit): only a
    fenced block that was finished counts. None when there is no script."""
    from .cleanup import strip_think
    text = strip_think(reply or "")
    m = (_CLOSED_FENCE_RE if closed_only else _FENCE_CODE_RE).search(text)
    if m:
        code = m.group(1).strip("\n")
        return code if code.strip() else None
    if not allow_bare or closed_only:
        return None
    s = text.strip()
    if not s or "print" not in s:
        return None
    try:
        ast.parse(s)
    except SyntaxError:
        return None
    return s


# ----------------------------------------------------------------------------- static check
def check_code(code: str) -> str:
    """'' when the script may run, else why it is rejected."""
    if not code or not code.strip():
        return "empty script"
    if len(code) > MAX_CODE_CHARS:
        return f"script too long ({len(code)} chars)"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    allowed = set(ALLOWED_MODULES)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name not in allowed:       # also rejects dotted imports (datetime.x) - none are needed
                    return f"import {a.name} not allowed"
        elif isinstance(n, ast.ImportFrom):
            if n.level or n.module not in allowed:
                return f"from {'.' * (n.level or 0)}{n.module or ''} import not allowed"
            mod = importlib.import_module(n.module)
            for a in n.names:
                if a.name == "*":
                    return "from-import * not allowed"
                if a.name in _BANNED_NAMES or a.name in _MODULE_ATTRS or a.name.startswith("_"):
                    return f"from {n.module} import {a.name} not allowed"
                if isinstance(getattr(mod, a.name, None), types.ModuleType):
                    return f"from {n.module} import {a.name} not allowed (module)"
                if a.asname and (a.asname in _BANNED_NAMES or a.asname.startswith("_")):
                    return f"import alias {a.asname} not allowed"
        elif isinstance(n, ast.alias):
            if n.asname and (n.asname in _BANNED_NAMES or n.asname.startswith("_")):
                return f"import alias {n.asname} not allowed"
        elif isinstance(n, ast.Name):
            if n.id in _BANNED_NAMES or (n.id.startswith("_") and n.id != "_"):     # "for _ in ..." is fine
                return f"name {n.id} not allowed"
        elif isinstance(n, ast.Attribute):
            if n.attr in _BANNED_NAMES or n.attr in _MODULE_ATTRS or n.attr.startswith(_BANNED_ATTR_PREFIXES):
                return f"attribute {n.attr} not allowed"
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name.startswith("_"):
                return f"name {n.name} not allowed"
        elif isinstance(n, ast.arg):
            if n.arg.startswith("_") and n.arg != "_":
                return f"name {n.arg} not allowed"
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            if any(x.startswith("_") or x in _BANNED_NAMES for x in n.names):
                return "name not allowed"
        elif isinstance(n, (ast.AsyncFor, ast.AsyncWith, ast.Await)):
            return "async code not allowed"
        elif isinstance(n, ast.Match):
            return "match statements not allowed"   # class patterns read attributes by name (kwd_attrs)
        elif isinstance(n, ast.ExceptHandler):
            if n.name and (n.name.startswith("_") or n.name in _BANNED_NAMES):
                return f"name {n.name} not allowed"
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and "__" in n.value:
            return "dunder in a string not allowed"     # "{0.__class__}".format(x) reaches attributes too
    if not _computes(tree):
        return "the script only prints a fixed value - compute the answer in the code"
    return ""


_WRAPPERS = frozenset({"print", "round", "float", "int", "str", "abs", "format"})


def _computes(tree: ast.AST) -> bool:
    """The script does some work: an operator, a comparison, a subscript, an attribute, a real function call or a
    computed variable. print(45), print("12 inches"), print(round(14, 2)) and x = 15.96; print(x) would just pass
    the model's mental answer through."""
    assigned: dict[str, bool] = {}      # name -> every assignment is a constant
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            const = isinstance(n.value, ast.Constant) or (isinstance(n.value, ast.UnaryOp)
                                                          and isinstance(n.value.operand, ast.Constant))
            for t in n.targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        assigned[x.id] = assigned.get(x.id, True) and const
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.comprehension, ast.NamedExpr)):
            tgt = n.target
            for x in ast.walk(tgt):
                if isinstance(x, ast.Name):
                    assigned[x.id] = False
    for n in ast.walk(tree):
        if isinstance(n, (ast.BinOp, ast.BoolOp, ast.Compare, ast.Subscript, ast.Attribute, ast.comprehension)):
            return True
        if isinstance(n, ast.UnaryOp) and not isinstance(n.operand, ast.Constant):
            return True
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in _WRAPPERS):
            return True
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in _WRAPPERS \
                and not assigned.get(n.id, False):
            return True
    return False


_DIGITS_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def request_has_numbers(request: str, code: str) -> bool:
    """Every number the script uses (round()'s digits aside) is written in the request; 0.15 counts as 15%."""
    want = set()
    for m in _DIGITS_RE.findall(request or ""):
        try:
            want.add(float(m.replace(",", "")))
        except ValueError:
            pass
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    skip = {id(n.args[1]) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "round" and len(n.args) > 1}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) \
                and id(n) not in skip:
            v = float(n.value)
            if not any(abs(v - w) < 1e-9 or abs(v * 100 - w) < 1e-7 for w in want):
                return False
    return True


def same_script(a: str, b: str) -> bool:
    """The two scripts are the same program (formatting and comments aside)."""
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return False


# ----------------------------------------------------------------------------- running it
# runs in the child before the script: reads the script from stdin, compiles it, then locks the interpreter down.
# Normal maths / date / print code raises no audit events at all, so everything after the script's own exec is
# denied.
_PRELUDE = r'''
import sys, builtins, math, cmath, fractions, decimal, datetime, calendar, statistics, _strptime, time, heapq, re
_ALLOWED = frozenset(%(allowed)r)
_MAXOUT = %(maxout)d
_src = sys.stdin.buffer.read(%(maxcode)d).decode("utf-8", "replace")
_err = sys.stderr
_out = sys.stdout
def _fail(msg):
    _err.write(msg.replace("\n", " ")[:300] + "\n")
    _err.flush()
    sys.exit(1)
try:
    _code = compile(_src, "<calc>", "exec")
except SyntaxError as e:
    _fail("SyntaxError: %%s (line %%s)" %% (e.msg, e.lineno))
class _Cap:
    n = 0
    def write(self, t):
        _Cap.n += len(t)
        if _Cap.n > _MAXOUT:
            raise RuntimeError("output too long")
        return _out.write(t)
    def flush(self):
        _out.flush()
_real_import = builtins.__import__
# C code (datetime.strftime -> time, strptime -> _strptime) imports through the running code's __import__ too;
# those internals were loaded above and the static check keeps the script itself from importing them
_INTERNAL = frozenset(("time", "_strptime", "heapq"))     # heapq: Counter.most_common (statistics.mode)
def _imp(name, globals=None, locals=None, fromlist=(), level=0):
    if level or not (name in _ALLOWED or (name in _INTERNAL and name in sys.modules)):
        raise ImportError("module not allowed: " + str(name))
    m = _real_import(name, globals, locals, fromlist, level)
    for f in fromlist or ():
        if f == "*" or f.startswith("_") or type(getattr(m, f, None)) is type(sys):
            raise ImportError("name not allowed: " + str(f))
    return m
_DENY = %(deny)r
_b = {k: v for k, v in vars(builtins).items() if k not in _DENY}
_b["__import__"] = _imp
_g = {"__name__": "__main__", "__builtins__": _b}
_g.update(_re=re, _dt=datetime, _F=fractions.Fraction, math=math, date=datetime.date, timedelta=datetime.timedelta)  # helpers' modules;
# math / date / timedelta are there for scripts that forget the import
exec(compile(%(helpers)r, "<helpers>", "exec"), _g)     # convert(), days_until(), feet_and_inches(), ...
def _hook(ev, args, _c=_code):
    if ev == "exec" and args and args[0] is _c:
        return
    raise RuntimeError("blocked: " + ev)
sys.stdout = _Cap()
sys.addaudithook(_hook)
_rc = 0
try:
    exec(_code, _g)
except SystemExit as e:
    _rc = 0 if e.code in (None, 0) else 1
    if _rc:
        _err.write("SystemExit: %%s\n" %% (e.code,))
except BaseException as e:
    _rc = 1
    _err.write(("%%s: %%s" %% (type(e).__name__, e)).replace("\n", " ")[:300] + "\n")
try:
    _out.flush()
    _err.flush()
except BaseException:
    pass
sys.exit(_rc)
'''
_DENY_BUILTINS = ("open", "exec", "eval", "compile", "input", "globals", "locals", "vars", "getattr", "setattr",
                  "delattr", "breakpoint", "exit", "quit", "help", "memoryview", "__loader__", "__spec__")
# Trusted helpers defined in the script's namespace, so a small model doesn't have to remember conversion factors
# or the "this year or next" rule. Internals start with "_" so the script can't reach them.
HELPERS = r'''
def _unit_table():
    t = {}
    def add(dim, factor, *names):
        # exact: a decimal factor is taken as written (0.3048 is 381/1250), so 1 ft is exactly 12 in
        f = factor if isinstance(factor, _F) else _F(repr(float(factor)))
        for n in names:
            t[n] = (dim, f)
    # length (meters)
    add("length", 0.001, "mm", "millimeter", "millimetre")
    add("length", 0.01, "cm", "centimeter", "centimetre")
    add("length", 1.0, "m", "meter", "metre")
    add("length", 1000.0, "km", "kilometer", "kilometre", "kms")
    add("length", 1e-6, "um", "micrometer", "micron")
    add("length", 1e-9, "nm", "nanometer")
    add("length", 0.0254, "in", "inch", "inches", '"')
    add("length", 0.3048, "ft", "foot", "feet", "'")
    add("length", 0.9144, "yd", "yard")
    add("length", 1609.344, "mi", "mile", "statute mile")
    add("length", 1852.0, "nmi", "nautical mile")
    add("length", 9460730472580800.0, "ly", "light year", "lightyear")
    # mass (kilograms)
    add("mass", 1e-6, "mg", "milligram")
    add("mass", 0.001, "g", "gram", "gramme")
    add("mass", 1.0, "kg", "kilogram", "kilo")
    add("mass", 1000.0, "t", "tonne", "metric ton")
    add("mass", 0.028349523125, "oz", "ounce")
    add("mass", 0.45359237, "lb", "lbs", "pound")
    add("mass", 6.35029318, "st", "stone")
    add("mass", 907.18474, "ton", "short ton", "us ton")
    add("mass", 1016.0469088, "long ton", "imperial ton")
    # volume (liters; US customary unless "imperial")
    add("volume", 0.001, "ml", "milliliter", "millilitre", "cc", "cm3", "cubic centimeter", "cubic centimetre")
    add("volume", 0.01, "cl", "centiliter", "centilitre")
    add("volume", 0.1, "dl", "deciliter", "decilitre")
    add("volume", 1.0, "l", "liter", "litre")
    add("volume", 1000.0, "m3", "cubic meter", "cubic metre")
    add("volume", 0.00492892159375, "tsp", "teaspoon")
    add("volume", 0.01478676478125, "tbsp", "tablespoon")
    add("volume", 0.0295735295625, "fl oz", "floz", "fluid ounce")
    add("volume", 0.2365882365, "cup")
    add("volume", 0.473176473, "pt", "pint")
    add("volume", 0.946352946, "qt", "quart")
    add("volume", 3.785411784, "gal", "gallon", "us gallon")
    add("volume", 4.54609, "imperial gallon", "uk gallon")
    add("volume", 0.56826125, "imperial pint", "uk pint")
    add("volume", 28.316846592, "cubic foot", "cubic feet", "cu ft", "ft3")
    add("volume", 0.016387064, "cubic inch", "cubic inches", "cu in", "in3")
    # speed (m/s)
    add("speed", 1.0, "m/s", "meter per second", "meters per second", "metre per second", "metres per second", "mps")
    add("speed", _F(5, 18), "km/h", "kmh", "kph", "km/hr", "kilometer per hour", "kilometers per hour",
        "kilometre per hour", "kilometres per hour")
    add("speed", 0.44704, "mph", "mile per hour", "miles per hour", "mi/h")
    add("speed", _F(1852, 3600), "knot", "kn", "kt")
    add("speed", 0.3048, "ft/s", "fps", "foot per second", "feet per second")
    # area (square meters)
    add("area", 1e-6, "mm2", "square millimeter", "square millimetre", "sq mm")
    add("area", 1e-4, "cm2", "square centimeter", "square centimetre", "sq cm")
    add("area", 1.0, "m2", "square meter", "square metre", "sq m", "sqm")
    add("area", 1e6, "km2", "square kilometer", "square kilometre", "sq km")
    add("area", 0.00064516, "in2", "square inch", "square inches", "sq in")
    add("area", 0.09290304, "ft2", "square foot", "square feet", "sq ft", "sqft")
    add("area", 0.83612736, "yd2", "square yard", "sq yd")
    add("area", 2589988.110336, "mi2", "square mile", "sq mi")
    add("area", 4046.8564224, "acre", "ac")
    add("area", 10000.0, "ha", "hectare")
    # time (seconds); a year is 365 days, a month a twelfth of that
    add("time", 1e-9, "ns", "nanosecond")
    add("time", 1e-6, "us", "microsecond")
    add("time", 0.001, "ms", "millisecond")
    add("time", 1.0, "s", "sec", "second")
    add("time", 60.0, "min", "minute")
    add("time", 3600.0, "h", "hr", "hour")
    add("time", 86400.0, "d", "day")
    add("time", 604800.0, "wk", "week")
    add("time", 1209600.0, "fortnight")
    add("time", 31536000.0 / 12, "mo", "month")
    add("time", 31536000.0, "yr", "year")
    add("time", 315360000.0, "decade")
    add("time", 3153600000.0, "century", "centuries")
    # data (bytes)
    add("data", 0.125, "bit")
    add("data", 1.0, "b", "byte")
    add("data", 1e3, "kb", "kilobyte")
    add("data", 1e6, "mb", "megabyte")
    add("data", 1e9, "gb", "gigabyte")
    add("data", 1e12, "tb", "terabyte")
    add("data", 1e15, "pb", "petabyte")
    add("data", 1024.0, "kib", "kibibyte")
    add("data", 1024.0 ** 2, "mib", "mebibyte")
    add("data", 1024.0 ** 3, "gib", "gibibyte")
    add("data", 1024.0 ** 4, "tib", "tebibyte")
    # energy (joules)
    add("energy", 1.0, "j", "joule")
    add("energy", 1000.0, "kj", "kilojoule")
    add("energy", 4.184, "cal", "small calorie")
    add("energy", 4184.0, "kcal", "kilocalorie", "calorie", "food calorie")
    add("energy", 3600.0, "wh", "watt hour", "watt-hour")
    add("energy", 3.6e6, "kwh", "kilowatt hour", "kilowatt-hour")
    add("energy", 1055.05585262, "btu")
    # power (watts)
    add("power", 1.0, "w", "watt")
    add("power", 1000.0, "kw", "kilowatt")
    add("power", 1e6, "mw", "megawatt")
    add("power", 745.6998715822702, "hp", "horsepower")
    # pressure (pascals)
    add("pressure", 1.0, "pa", "pascal")
    add("pressure", 1000.0, "kpa", "kilopascal")
    add("pressure", 1e5, "bar")
    add("pressure", 6894.757293168, "psi")
    add("pressure", 101325.0, "atm", "atmosphere")
    add("pressure", 133.322387415, "mmhg")
    # force (newtons)
    add("force", 1.0, "n", "newton")
    add("force", 4.4482216152605, "lbf", "pound force", "pound-force")
    return t
_UNITS = _unit_table()
_TEMPS = {"c": "c", "celsius": "c", "centigrade": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}
def _unit(name):
    u = str(name).lower().replace("^2", "2").replace("²", "2").replace("^3", "3").replace("³", "3")
    u = " ".join(u.replace("_", " ").replace("°", " ").replace(".", "").split())
    for word, pre in ((" squared", "square "), (" cubed", "cubic ")):
        if u.endswith(word):
            u = pre + u[:-len(word)]
    for p in ("degrees ", "degree ", "deg "):
        if u.startswith(p):
            u = u[len(p):]
    cands = [u]
    if u.startswith("square "):
        cands.append("sq " + u[7:])
    if u.startswith("sq "):
        cands.append("square " + u[3:])
    for c in list(cands):
        if c.endswith("es"):
            cands.append(c[:-2])
        if c.endswith("s"):
            cands.append(c[:-1])
    for c in cands:
        if c in _TEMPS:
            return ("temp", _TEMPS[c])
        if c in _UNITS:
            return _UNITS[c]
    raise ValueError("unknown unit: " + repr(name))
def convert(value, from_unit, to_unit):
    """value in from_unit expressed in to_unit (exact factors)."""
    a, b = _unit(from_unit), _unit(to_unit)
    if a[0] != b[0]:
        # "ounces in a gallon": an ounce next to a volume is a fluid ounce
        if a == _UNITS["oz"] and b[0] == "volume":
            a = _UNITS["fl oz"]
        elif b == _UNITS["oz"] and a[0] == "volume":
            b = _UNITS["fl oz"]
        else:
            raise ValueError("can't convert %s to %s" % (from_unit, to_unit))
    if isinstance(value, float):
        value = _F(repr(value))
    elif not isinstance(value, (int, _F)):
        value = _F(str(value))
    value = _F(value)
    if a[0] == "temp":
        k = _F("273.15")
        c = {"c": value, "f": (value - 32) * 5 / 9, "k": value - k}[a[1]]
        return float({"c": c, "f": c * 9 / 5 + 32, "k": c + k}[b[1]])
    return float(value * a[1] / b[1])
def _nth_weekday(year, month, weekday, n):
    datetime = _dt
    if n > 0:
        d = datetime.date(year, month, 1)
        d += datetime.timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
    else:
        nxt = datetime.date(year + (month == 12), month % 12 + 1, 1)
        d = nxt - datetime.timedelta(days=1)
        d -= datetime.timedelta(days=(d.weekday() - weekday) % 7)
    return d
def _easter(year):
    datetime = _dt
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    month = (h + l - 7 * m + 90) // 25
    return datetime.date(year, month, (h + l - 7 * m + 33 * month + 19) % 32)
_HOLIDAYS = {
    "christmas": (12, 25), "christmas day": (12, 25), "xmas": (12, 25), "christmas eve": (12, 24),
    "new year": (1, 1), "new years": (1, 1), "new year's": (1, 1), "new year's day": (1, 1), "new years day": (1, 1),
    "new year's eve": (12, 31), "new years eve": (12, 31), "halloween": (10, 31), "valentine's day": (2, 14),
    "valentines day": (2, 14), "valentine's": (2, 14), "valentines": (2, 14), "independence day": (7, 4),
    "fourth of july": (7, 4), "july 4th": (7, 4), "4th of july": (7, 4), "st patrick's day": (3, 17),
    "st patricks day": (3, 17), "saint patrick's day": (3, 17), "april fools": (4, 1), "april fools day": (4, 1),
    "april fool's day": (4, 1), "juneteenth": (6, 19), "veterans day": (11, 11), "groundhog day": (2, 2),
    "thanksgiving": lambda y: _nth_weekday(y, 11, 3, 4), "easter": _easter,
    "mother's day": lambda y: _nth_weekday(y, 5, 6, 2), "mothers day": lambda y: _nth_weekday(y, 5, 6, 2),
    "father's day": lambda y: _nth_weekday(y, 6, 6, 3), "fathers day": lambda y: _nth_weekday(y, 6, 6, 3),
    "labor day": lambda y: _nth_weekday(y, 9, 0, 1), "memorial day": lambda y: _nth_weekday(y, 5, 0, -1),
}
_MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                "november", "december")
def _parse_day(text):
    """(month, day, year or None) from "June 3", "3 June", "march 1st, 2027", "2027-03-01", "6/3", "6/3/2027";
    None when the text is not a date."""
    re = _re
    s = " ".join(text.lower().replace(",", " ").replace(".", " ").split())
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return int(m.group(2)), int(m.group(3)), int(m.group(1))
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", s)
    if m:
        y = m.group(3)
        return int(m.group(1)), int(m.group(2)), (int(y) + 2000 if y and len(y) == 2 else int(y) if y else None)
    mon = r"(" + "|".join(n[:3] + r"[a-z]*" for n in _MONTH_NAMES) + r")"
    m = re.fullmatch(mon + r" (\d{1,2})(?:st|nd|rd|th)?(?: (\d{4}))?", s)
    if m:
        mo, d, y = m.group(1), m.group(2), m.group(3)
    else:
        m = re.fullmatch(r"(?:the )?(\d{1,2})(?:st|nd|rd|th)?(?: of)? " + mon + r"(?: (\d{4}))?", s)
        if not m:
            return None
        d, mo, y = m.group(1), m.group(2), m.group(3)
    month = next((i + 1 for i, n in enumerate(_MONTH_NAMES) if n.startswith(mo[:3])), None)
    return (month, int(d), int(y) if y else None) if month else None
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
def _on(year, month, day):
    datetime = _dt
    if isinstance(month, str):
        key = " ".join(month.lower().replace("’", "'").split())
        key = key[4:] if key.startswith("the ") else key
        h = _HOLIDAYS.get(key)
        if h is not None:
            return h(year) if callable(h) else datetime.date(year, *h)
        p = _parse_day(key)
        if p is None:
            raise ValueError("unknown date or holiday: " + repr(month) + " (use month and day numbers)")
        return datetime.date(p[2] or year, p[0], p[1])
    return datetime.date(year, int(month), int(day))
def next_date(month, day=None, year=None):
    """The next date that is month/day (or a holiday name like "christmas", or a date like "June 3"): this year,
    or next year if it has already passed (today counts). With a year (argument or in the text), that date."""
    datetime = _dt
    if isinstance(month, str) and day is not None:      # days_until("June", 3)
        month, day = "%s %s" % (month, day), None
    wd = month.strip().lower().rstrip("s") if isinstance(month, str) else ""
    if wd in _WEEKDAYS and year is None:        # "days until friday": the next one (today counts)
        today = datetime.date.today()
        return today + datetime.timedelta(days=(_WEEKDAYS.index(wd) - today.weekday()) % 7)
    if year is None and isinstance(month, str):
        p = _parse_day(month)
        if p and p[2]:
            year = p[2]
    if year is not None:
        return _on(int(year), month, day)
    today = datetime.date.today()
    d = _on(today.year, month, day)
    return d if d >= today else _on(today.year + 1, month, day)
def days_until(month, day=None, year=None):
    """Days from today until month/day (or a holiday name), the next time it comes; with ``year``, that date."""
    datetime = _dt
    return (next_date(month, day, year) - datetime.date.today()).days
def feet_and_inches(value, unit="in"):
    """A length as "5 ft 10.87 in"."""
    inches = round(convert(value, unit, "in"), 2)
    feet = int(inches // 12)
    rest = round(inches - feet * 12, 2)
    if rest >= 12:
        feet, rest = feet + 1, 0.0
    return "%d ft %s in" % (feet, ("%.2f" % rest).rstrip("0").rstrip("."))
'''
PRELUDE = _PRELUDE % {"allowed": ALLOWED_MODULES, "maxout": MAX_OUTPUT_CHARS, "maxcode": MAX_CODE_CHARS * 4 + 16,
                      "deny": _DENY_BUILTINS, "helpers": HELPERS}


def python_exe() -> Optional[str]:
    """A console python.exe for the child: the base interpreter behind the venv (a venv's python.exe is a launcher
    that starts it as a grandchild, which would escape the job object). None when none is found (calculations
    are then off)."""
    cands = []
    for p in (getattr(sys, "_base_executable", "") or "", sys.executable or ""):
        if not p:
            continue
        pp = Path(p)
        if pp.name.lower() == "pythonw.exe":
            pp = pp.with_name("python.exe")
        cands.append(pp)
    for pp in cands:
        if pp.name.lower() == "python.exe" and pp.is_file():
            return str(pp)
    return None


def available() -> bool:
    return python_exe() is not None


class _Job:
    """Windows job object: memory cap, one active process, no UI access, kill on close. Best effort: any
    failure leaves ``ok`` False and the run goes ahead with the other layers."""

    def __init__(self, mem_mb: int = MEMORY_LIMIT_MB):
        self.handle = None
        self.ok = False
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes as wt
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wt.HANDLE
            k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wt.LPCWSTR)
            k32.SetInformationJobObject.argtypes = (wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD)
            k32.AssignProcessToJobObject.argtypes = (wt.HANDLE, wt.HANDLE)
            k32.CloseHandle.argtypes = (wt.HANDLE,)

            class BASIC(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD),
                            ("SchedulingClass", wt.DWORD)]

            class IO(ctypes.Structure):
                _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

            class EXT(ctypes.Structure):
                _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            class UI(ctypes.Structure):
                _fields_ = [("UIRestrictionsClass", wt.DWORD)]

            self._k32 = k32
            h = k32.CreateJobObjectW(None, None)
            if not h:
                return
            self.handle = h
            ext = EXT()
            ext.Basic.LimitFlags = 0x100 | 0x200 | 0x2000 | 0x8 | 0x400    # process + job memory, kill on close,
            ext.Basic.ActiveProcessLimit = 1                                 # one process, die on crash
            ext.ProcessMemoryLimit = mem_mb * 1024 * 1024
            ext.JobMemoryLimit = mem_mb * 1024 * 1024
            if not k32.SetInformationJobObject(h, 9, ctypes.byref(ext), ctypes.sizeof(ext)):
                return
            ui = UI(0xFF)       # JOB_OBJECT_UILIMIT_ALL: no clipboard, desktops, global atoms, other windows
            k32.SetInformationJobObject(h, 4, ctypes.byref(ui), ctypes.sizeof(ui))
            self.ok = True
        except Exception as e:
            log.debug("job object unavailable: %s", e)

    def assign(self, proc: subprocess.Popen) -> bool:
        if not (self.ok and self.handle):
            return False
        try:
            return bool(self._k32.AssignProcessToJobObject(self.handle, int(proc._handle)))
        except Exception:
            return False

    def close(self) -> None:
        if self.handle:
            try:
                self._k32.CloseHandle(self.handle)      # KILL_ON_JOB_CLOSE ends anything still running
            except Exception:
                pass
            self.handle = None


_POINT_ZERO_RE = re.compile(r"(?<![\d.])(\d[\d,]*)\.0(?![\d.])")


def tidy_output(text: str) -> str:
    """The script's printed text as the answer: Windows line ends normalised, outer blank lines dropped and a float
    that is a whole number shown without ".0" (print(12.0) -> "12")."""
    lines = [ln.rstrip() for ln in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return _POINT_ZERO_RE.sub(r"\1", "\n".join(lines).strip("\n"))


HELPER_NAMES = frozenset({"convert", "feet_and_inches", "days_until", "next_date"})


def _redefines_helper(stmt: ast.stmt) -> bool:
    if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
        return stmt.name in HELPER_NAMES
    if isinstance(stmt, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id in HELPER_NAMES for t in stmt.targets)
    return False


def prepare_code(code: str) -> str:
    """Small fixes before a checked script runs: drop the script's own (placeholder) definitions of the ready-made
    helpers, and give a script that computes the answer but never prints it a print() of its last value (like the
    REPL): a trailing expression is printed, or the variable a trailing assignment set."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    changed = False
    if any(_redefines_helper(st) for st in tree.body):
        tree.body = [st for st in tree.body if not _redefines_helper(st)] or [ast.Pass()]
        changed = True
    if not tree.body or any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
                            for n in ast.walk(tree)):
        pass
    elif isinstance(tree.body[-1], ast.Expr):
        tree.body[-1] = ast.Expr(ast.Call(ast.Name("print", ast.Load()), [tree.body[-1].value], []))
        changed = True
    elif isinstance(tree.body[-1], (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        last = tree.body[-1]
        tgt = last.targets[0] if isinstance(last, ast.Assign) else last.target
        if isinstance(tgt, ast.Name):
            tree.body.append(ast.Expr(ast.Call(ast.Name("print", ast.Load()), [ast.Name(tgt.id, ast.Load())], [])))
            changed = True
    return ast.unparse(ast.fix_missing_locations(tree)) if changed else code


def run_code(code: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Run a checked script. (True, printed output) on success (exit 0 and something printed), else
    (False, error) - the error is the last line of stderr, "timed out", or why it could not start."""
    why = check_code(code)
    if why:
        return False, why
    code = prepare_code(code)
    exe = python_exe()
    if not exe:
        return False, "no Python interpreter found"
    tmp = tempfile.mkdtemp(prefix="lf-calc-")
    job = _Job()
    proc = None
    try:
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "SystemRoot", "WINDIR") if k in os.environ}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen([exe, "-I", "-X", "utf8", "-S", "-c", PRELUDE], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmp, env=env,
                                creationflags=flags)
        # the prelude blocks on stdin before running anything, so the job is in place before the script runs
        if job.ok and not job.assign(proc):
            log.warning("calc: could not put the Python process in a job object (no memory limit)")
        bufs = {"out": bytearray(), "err": bytearray()}
        over = []

        def pump(pipe, key, cap):
            try:
                while True:
                    b = pipe.read1(65536)
                    if not b:
                        break
                    if len(bufs[key]) + len(b) > cap:
                        over.append(key)
                        proc.kill()
                        break
                    bufs[key] += b
            except (OSError, ValueError):
                pass
        readers = [threading.Thread(target=pump, args=(proc.stdout, "out", MAX_OUTPUT_CHARS * 4 + 4096), daemon=True),
                   threading.Thread(target=pump, args=(proc.stderr, "err", 16384), daemon=True)]
        for t in readers:
            t.start()
        try:
            proc.stdin.write(code.encode("utf-8"))
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=max(0.5, float(timeout_s)))
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            return False, f"timed out after {timeout_s:g} s"
        finally:
            for t in readers:
                t.join(timeout=2)
        if over:
            return False, "output too long"
        out = tidy_output(bytes(bufs["out"]).decode("utf-8", "replace"))
        err_lines = [ln for ln in bytes(bufs["err"]).decode("utf-8", "replace").splitlines() if ln.strip()]
        if out.strip() == "None":       # print(x.sort()) / an auto-printed call that returns nothing
            out = ""
        if proc.returncode == 0 and out.strip():
            if "\n" not in out:
                out = out.strip()
            if len(out) > MAX_OUTPUT_CHARS:
                out = out[:MAX_OUTPUT_CHARS].rstrip()
            return True, out
        if proc.returncode == 0:
            return False, "the script printed nothing"
        if err_lines:
            return False, err_lines[-1].strip()[:300]
        return False, f"exit code {proc.returncode}"
    except OSError as e:
        return False, f"could not start Python: {e}"
    finally:
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if pipe:
                        pipe.close()
                except Exception:
                    pass
        job.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- routing pre-gate
# A request counts as a calculation only with a concrete signal: a quantity ("5 feet", "$84", "15%"), an arithmetic
# expression ("17 times 23", "square root of 2025"), a date calculation ("days until christmas") or a "how many X
# in Y" / "convert X to Y" between two units. A number and a unit word somewhere ("what year did World War 2 end")
# is not enough.
_NUMBER_WORDS = (r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
                 r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
                 r"hundred|thousand|million|billion|trillion|half|quarter|dozen")
_NUM = r"(?:\d[\d,]*(?:\.\d+)?|\.\d+|(?:" + _NUMBER_WORDS + r"|a|an))"
_NUM_STRICT = r"(?:\d[\d,]*(?:\.\d+)?|\.\d+|(?:" + _NUMBER_WORDS + r"))"
_UNITS = (
    r"inch(?:es)?|foot|feet|ft|yards?|yds?|miles?|meters?|metres?|kilometers?|kilometres?|kms?|"
    r"centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|micrometers?|nanometers?|nautical miles?|"
    r"light ?years?|grams?|kilograms?|kilos?|kg|milligrams?|mg|pounds?|lbs?|ounces?|oz|tons?|tonnes?|stones?|"
    r"liters?|litres?|milliliters?|millilitres?|ml|gallons?|gal|quarts?|qt|pints?|cups?|tablespoons?|tbsp|"
    r"teaspoons?|tsp|fluid ounces?|fl oz|(?:degrees? )?(?:fahrenheit|celsius|centigrade|kelvin)|degrees?|mph|kph|"
    r"km/h|kmh|knots?|acres?|hectares?|square \w+|sq ft|cubic \w+|bytes?|kilobytes?|megabytes?|gigabytes?|"
    r"terabytes?|bits?|kb|mb|gb|tb|kib|mib|gib|calories?|kcal|joules?|kilojoules?|watts?|kilowatts?|kwh|"
    r"horsepower|psi|pascals?|atmospheres?|newtons?|"
    r"(?:miles|kilometers|kilometres|meters|metres|feet) (?:per|an|a) (?:hour|second)")
_TIME_UNITS = (r"seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|decades?|centur(?:y|ies)|"
               r"milliseconds?|microseconds?|nanoseconds?|fortnights?")
_CURRENCY = r"dollars?|bucks|cents?|euros?|yen|rupees?|pesos?|francs?|usd|eur|gbp"
# a time span alone ("remind me in 5 minutes", "a 30 minute dinner") is not a quantity to compute with; it counts
# when converted ("3 weeks in days") or asked about ("how many minutes in 3.5 hours")
_QUANTITY_RE = re.compile(
    r"(?<![\w.])" + _NUM + r"\s*(?:-\s*)?(?:(?:" + _UNITS + r"|" + _CURRENCY + r")\b|%|percent\b|per cent\b)"
    r"|[$€£¥₹]\s*\d"
    r"|(?<![\w.])" + _NUM + r"\s*(?:" + _TIME_UNITS + r")\b.*\b(?:in|to|into|as)\s+(?:" + _TIME_UNITS + r")\b", re.I)
_UNITS = _UNITS + "|" + _TIME_UNITS
_ARITH_RE = re.compile(
    r"(?<![\w.])" + _NUM_STRICT + r"\s*(?:[-+*/^×÷x]|plus|minus|times|multiplied by|divided by|over|"
    r"to the (?:power|\w+ power)(?: of)?|raised to(?: the power of)?|mod|modulo)\s*" + _NUM_STRICT + r"(?!\w)"
    r"|\b(?:square root|cube root|sqrt|factorial|log(?:arithm)?|natural log|sine|cosine|tangent|sin|cos|tan|"
    r"average|mean|median|sum|product|total|half|double|triple|a third|a quarter)\s+(?:of\s+)?" + _NUM_STRICT +
    r"(?!\w)"
    r"|(?<![\w.])" + _NUM_STRICT + r"\s+(?:squared|cubed|factorial)\b"
    r"|\b(?:calculate|compute|work out)\s+" + _NUM_STRICT + r"(?!\w)"
    r"|\badd\s+" + _NUM_STRICT + r"\s+(?:and|to|plus)\s+" + _NUM_STRICT + r"(?!\w)"
    r"|(?<![\w.])" + _NUM_STRICT + r"\s+to the \w+ power\b", re.I)
_MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"
_DATE_CALC_RE = re.compile(
    r"\b(?:days?|weeks?|months?|years?|hours?|minutes?|seconds?|sleeps?)\s+(?:until|till|til|to go until|left until|"
    r"left till|before|since|after|between|from now|ago|away)\b"
    r"|\b(?:how long|how much time)\s+(?:is it\s+|is there\s+|left\s+)?(?:until|till|til|since|before)\b"
    r"|\b(?:until|till|til|since)\s+(?:christmas|new year'?s?|halloween|thanksgiving|easter|valentine'?s|"
    r"my birthday|the weekend|friday|monday|tuesday|wednesday|thursday|saturday|sunday|" + _MONTHS + r")\b"
    r"|\bday of the week\b|\bwhat day (?:is|was|will|does|did)\b.*\d|\bleap year\b.*\d|\bweek number\b"
    r"|\bdays? (?:in|are in|is in|are there in) (?:a |the )?(?:year|month|leap year|" + _MONTHS + r")\b"
    r"|\b\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\s+(?:from|after|before)\s+"
    r"(?:now|today|tomorrow|yesterday|\w+day\b|" + _MONTHS + r")"
    , re.I)
_UNIT_PAIR_RE = re.compile(
    r"\bhow (?:many|much)\s+(?:" + _UNITS + r")\b.*\b(?:in|are in|is in|make|per|equal|for)\b.*\b(?:" + _UNITS +
    r"|" + _NUM_STRICT + r")\b"
    r"|\b(?:convert|change|turn)\b.*\b(?:" + _UNITS + r")\b.*\b(?:to|into|in)\b\s+(?:" + _UNITS + r")\b", re.I)
# code-writing requests: the user wants the code itself, not its result
_CODE_RE = re.compile(
    r"\b(?:scripts?|functions?|code|coding|program(?:s|me)?|snippets?|regex(?:es)?|regular expressions?|"
    r"python|javascript|typescript|java|c\+\+|c#|rust|golang|sql|bash|powershell|shell|excel|spreadsheets?|"
    r"formulas?|macros?|class(?:es)?|methods?|algorithms?|implement(?:ation)?|pseudo ?code|json|yaml|html|css|"
    r"api|lambda|loop)\b", re.I)
# writing / editing requests (and number-to-words / other languages) keep the normal command path
_WRITE_RE = re.compile(
    r"^\W*(?:(?:hey|ok|okay|so|um|uh|please|just|can you|could you|would you)\W+)*"
    r"(?:write|draft|compose|reply|respond|send|rewrite|rephrase|reword|translate|summari[sz]e|list|bullet|email|"
    r"text|message|tell (?:him|her|them)|make (?:a|an|me a|me an) (?:list|note|table|email|message|text|poem|"
    r"story|joke)|explain|describe|define|tell me about|remind|set (?:a|an|the) )\b"
    r"|\b(?:poem|haiku|essay|story|paragraph|bullet points?|tweet|joke|limerick|song|email|letter|"
    r"note|memo|summary|table|chart|graph|spanish|french|german|italian|japanese|chinese|in words|"
    r"spell(?:ed)? out)\b", re.I)
# anything about selected text ("convert this to celsius") keeps the normal path (NO_SELECTION without one)
_REFERS_RE = re.compile(
    r"\b(?:this|these|those|that one|selected|selection|highlighted|above|below|the text|my text|the paragraph|"
    r"the sentence|the list|the table|the numbers|the measurements?|the values|the prices|the recipe|"
    r"the document|the file|the page|them)\b"
    r"|(?<!\bis )(?<!\bwas )(?<!\bwill )(?<!\bdoes )(?<!\bdid )\bit\b(?!'s|\s+(?:is|was|be|will))"
    r"|\bthat\b(?!'s| is| was| will)", re.I)
_EXCHANGE_RE = re.compile(r"\b(?:exchange|forex|stocks?|share price|bitcoin|btc|crypto|ethereum|"
                          r"market (?:price|value)|worth in)\b", re.I)
_CURRENCY_KINDS = (("dollar", "buck", "usd", "$"), ("euro", "eur", "€"), ("sterling", "gbp", "£", "quid"),
                   ("yen", "jpy", "¥"), ("rupee", "₹"), ("peso",), ("franc",))


def _currency_kinds(text: str) -> int:
    low = text.lower()
    return sum(any(w in low for w in kind) for kind in _CURRENCY_KINDS)


# A command whose result is TYPED into the document must also be shaped like a calculation: it starts with a
# calculation word or is a bare conversion / percentage / expression. "say the package weighs 3 pounds" or "a
# caption for my 10 mile hike" contain a quantity but ask for writing.
_LEAD = r"^\W*(?:(?:hey|ok|okay|so|um|uh|please|just|quickly|can you|could you|would you|will you)\W+)*"
_INSERT_SHAPE_RE = re.compile(
    _LEAD + r"(?:(?:the|give me|tell me|number of)\s+)?"
    r"(?:convert|conversion|calculate|compute|work out|figure out|what(?:'s| is| are| was|s)\b|"
    r"how (?:many|much|long|far)\b|what day\b|split\b|divide\b|multiply\b|add up\b|subtract\b|sum of\b|"
    r"total of\b|half (?:of|an?)\b|a third of\b|a quarter of\b|double(?!\s+check)\b|triple(?!\s+check)\b|"
    r"square root\b|cube root\b|average\b|mean of\b|median\b|factorial\b|tip on\b|"
    r"add\s+" + _NUM_STRICT + r"\s+(?:and|to|plus)\b|"
    r"(?:an?\s+)?" + _NUM_STRICT + r"\s*(?:%|percent\b|per cent\b)|"
    r"[$€£¥₹]?\s*" + _NUM + r"\s*(?:" + _UNITS + r"|" + _CURRENCY + r")\b.*\b(?:to|in|into|as)\s+(?:"
    + _UNITS + r")(?:\s+please)?\W*$|"
    r"" + _NUM_STRICT + r"\s*(?:[-+*/^×÷x]|plus|minus|times|divided by|multiplied by|over|to the)\s|"
    r"" + _NUM_STRICT + r"\s+(?:squared|cubed|factorial)\b|"
    r"(?:days?|weeks?|months?|hours?|minutes?|sleeps?)\s+(?:until|till|til|since|between|left)\b)", re.I)


def looks_like_calc(text: str, insert: bool = False) -> bool:
    """Cheap deterministic pre-gate: could this request be a calculation (conversion, percentage, arithmetic,
    date maths) whose answer a short script can compute? False for writing / code / selection requests. With
    ``insert`` (the result is typed at the cursor) it must also be shaped like a calculation. Only a gate: the
    calc prompt itself can still answer NO_CALC."""
    s = " ".join((text or "").replace("’", "'").split())
    if not s or len(s) > 300:
        return False
    if insert and not _INSERT_SHAPE_RE.search(s):
        return False
    if _CODE_RE.search(s) or _WRITE_RE.search(s) or _REFERS_RE.search(s) or _EXCHANGE_RE.search(s):
        return False
    if _currency_kinds(s) >= 2:         # currency exchange needs live rates a script doesn't have
        return False
    return bool(_QUANTITY_RE.search(s) or _ARITH_RE.search(s) or _DATE_CALC_RE.search(s) or _UNIT_PAIR_RE.search(s))


# numbers with 7+ decimals are float noise (0.30000000000000004, 10.866141732283467) unless precision was asked for
_NOISE_RE = re.compile(r"(?<![\w.])(-?\d+)\.(\d{7,})(?!\d)")
# "exactly" is not a request for float noise ("convert one foot to inches exactly" -> 12, not 12.000000000000002)
_PRECISION_RE = re.compile(r"\b(?:decimals?|digits?|significant)\b", re.I)


def round_noise(text: str, request: str = "") -> str:
    """Round float noise in the printed answer to 2 decimals (3 significant digits below 1) unless the request
    asks for precision."""
    if _PRECISION_RE.search(request or ""):
        return text

    def fix(m: re.Match) -> str:
        v = float(m.group(0))
        s = f"{v:.2f}" if abs(v) >= 1 else f"{v:.3g}"
        if "e" in s:
            return m.group(0)
        return s.rstrip("0").rstrip(".") if "." in s else s
    return _NOISE_RE.sub(fix, text or "")
