"""Offline tests for command mode calculations: the routing pre-gate, extracting / checking / running the model's
Python script (pycalc) and Cleaner.command()/answer() taking the calc path with a mocked LLM. Scripts really run
(a locked-down child python.exe); no LLM server is used.

   .venv\\Scripts\\python.exe tests\\test_unit_pycalc.py
"""
import datetime as dt
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import cleanup, pycalc  # noqa: E402
from localflow.config import Config  # noqa: E402

PY = "```python\n{}\n```".format

CALCS = ["convert one foot to inches", "5 feet 11 inches in cm", "3 pounds in ounces", "15% tip on 84 dollars",
         "how many days until christmas", "what's 2 to the power of 64", "how many seconds in a week",
         "100 km/h in mph", "72 fahrenheit in celsius", "convert 26.2 miles to kilometers",
         "how many meters is 6 feet 2 inches", "how many hours are in 3 weeks", "what day of the week is July 4th 2027",
         "what is the square root of 2025", "what is 30 percent off 79.99", "convert 1 acre to square feet",
         "how many weeks until March 1st 2027", "what's 17 times 23", "split 120 dollars between 4 people",
         "how many days between January 1 and March 15 2027", "convert 180 centimeters to feet and inches",
         "how many minutes in 3.5 hours", "convert a foot to inches", "how many inches in a foot", "3 weeks in days",
         "what is five plus seven", "half of 37", "how many days in february 2028"]
NOT_CALCS = ["what is the capital of France", "write a script that converts feet to inches", "make this more formal",
             "translate this to German", "who wrote hamlet", "what time is it", "write a haiku about the ocean",
             "reply saying I will be ten minutes late", "how many people live in France",
             "what year did World War 2 end", "how tall is Mount Everest",
             "draft a text saying the meeting moved from 3 to 4", "what does HTTP stand for",
             "write three bullet points about our 20 percent growth", "convert this to celsius",
             "convert these measurements to metric", "convert 100 dollars to euros",
             "how far is Paris from London in kilometers", "write a python function that adds two numbers",
             "list five fruits", "how many players are on a soccer team", "summarize the paragraph above",
             "how long is the Nile in miles", "remind me in 5 minutes", "set a timer for 10 minutes",
             "give me a 5 minute workout", "convert it to celsius", "what's 5 dollars in yen", ""]

# payloads that got past the naive AST check during planning, plus a few more escape routes
BYPASSES = {
    "statistics.sys -> os": "import statistics\nprint(statistics.sys.modules['os'].getcwd())",
    "from-import of a module": "from fractions import sys\nprint(sys.modules['os'].getlogin())",
    "calendar.sys builtins open": "import calendar\nb = calendar.sys.modules['builtins']\nprint(b.open('x', 'w'))",
    "fractions.operator attrgetter": "import fractions\ng = fractions.operator.attrgetter('real')\nprint(g(1))",
    "statistics.random": "import statistics\nprint(statistics.random.random())",
    "import os": "import os\nprint(os.getcwd())",
    "import dotted": "import datetime.os\nprint(1)",
    "relative import": "from . import x\nprint(x)",
    "star import": "from datetime import *\nprint(date.today())",
    "dunder attribute": "print(().__class__.__bases__[0].__subclasses__())",
    "dunder in format string": "print('{0.__class__}'.format(1))",
    "open builtin": "print(open('x', 'w'))",
    "eval": "print(eval('1+1'))",
    "getattr": "print(getattr(1, 'real'))",
    "generator frame": "g = (i for i in [1])\nprint(g.gi_frame.f_globals)",
    "type()": "print(type(1).mro())",
    "underscore name": "print(__builtins__)",
    "alias to banned": "import math as sys\nprint(sys.pi)",
    # class patterns read attributes by name (found by the verifier: reached the real builtins this way)
    "match class pattern": ("class A:\n    pass\na = A()\nmatch a:\n    case A(__reduce_ex__=r):\n        t = r(1)\n"
                            "print(t)"),
    "except handler name": "try:\n    1 / 0\nexcept Exception as __x:\n    print(1 + 1)",
}

# writing commands typed at the cursor that mention a quantity: not calculations (live, these got a copied
# example script inserted before the insert-path shape gate)
WRITING_WITH_QUANTITY = ["add a line saying the rent is 1200 dollars", "say the package weighs 3 pounds",
                         "insert a sentence about how the project is 50% complete",
                         "a subject line for the 10 percent discount announcement",
                         "give me a headline about gas hitting 5 dollars", "generate a title for my 5 km race report",
                         "thank him for the 50 dollars", "congratulate her on running 26 miles",
                         "name ideas for a 5 km charity run", "a caption for my 10 mile hike photo"]


class PreGate(unittest.TestCase):
    def test_calculations_pass(self):
        missed = [q for q in CALCS if not pycalc.looks_like_calc(q)]
        self.assertEqual(missed, [])

    def test_everything_else_does_not(self):
        hit = [q for q in NOT_CALCS if pycalc.looks_like_calc(q)]
        self.assertEqual(hit, [])

    def test_insert_path_needs_a_calculation_shape(self):
        self.assertEqual([q for q in WRITING_WITH_QUANTITY if pycalc.looks_like_calc(q, insert=True)], [])
        inserted = ["convert one foot to inches", "5 feet 11 inches in cm", "3 pounds in ounces", "15% tip on 84 dollars",
                    "100 km/h in mph", "72 fahrenheit in celsius", "split 120 dollars between 4 people",
                    "convert 180 centimeters to feet and inches", "an 18 percent tip on 62 dollars",
                    "60 mph in kilometers per hour", "3 weeks in days", "please convert 2 gallons to liters",
                    "calculate 15% of 240", "what's 17 times 23", "17 times 23", "half of 37", "a foot to inches",
                    "the square root of 144", "10 squared", "5 cubed", "a third of 90", "5 feet in meters please",
                    "give me 5 feet in meters", "half a cup in tablespoons", "number of days until christmas",
                    "add 12 and 30", "tip on 84 dollars at 15 percent", "average of 5 10 and 15",
                    "2 to the 10th power"]
        self.assertEqual([q for q in inserted if not pycalc.looks_like_calc(q, insert=True)], [])
        self.assertFalse(pycalc.looks_like_calc("double check the 15% figure", insert=True))

    def test_rates_still_route(self):
        self.assertTrue(pycalc.looks_like_calc("interest on 5000 dollars at a 5 percent rate for 3 years"))
        self.assertTrue(pycalc.looks_like_calc("40 hours at an hourly rate of 25 dollars"))
        self.assertFalse(pycalc.looks_like_calc("what's the exchange rate for 100 dollars"))

    def test_copied_example_detection(self):
        tip = pycalc.extract_code(cleanup.CALC_SHOTS[1][1])
        self.assertTrue(cleanup._copied_shot("double check the 15% figure", tip.replace('"', "'") + "\n# copy\n"))
        self.assertFalse(cleanup._copied_shot(cleanup.CALC_SHOTS[1][0], tip))
        self.assertFalse(cleanup._copied_shot("15% tip on 84 dollars", tip))     # reuse with the same numbers
        feet = pycalc.extract_code(cleanup.CALC_SHOTS[0][1])
        self.assertFalse(cleanup._copied_shot("5 feet 11 inches in cm", feet))
        self.assertTrue(cleanup._copied_shot("say the rent is 1200 dollars", feet))
        self.assertFalse(cleanup._copied_shot("15% tip on 90 dollars", tip.replace("84", "90")))
        # a script with the request's numbers rewritten (5 thousand -> 5000) is a real script
        self.assertFalse(cleanup._copied_shot("convert 5 thousand feet to miles",
                                              "print(round(convert(5000, 'ft', 'mi'), 2), 'miles')"))


class Extract(unittest.TestCase):
    def test_fenced(self):
        self.assertEqual(pycalc.extract_code("<think></think>" + PY("print(1 + 1)") + "\nnote"), "print(1 + 1)")
        self.assertEqual(pycalc.extract_code("```py\nx = 2\nprint(x)\n```"), "x = 2\nprint(x)")
        self.assertEqual(pycalc.extract_code("```\nprint(3 * 4)\n```"), "print(3 * 4)")

    def test_bare_needs_print_and_valid_python(self):
        self.assertEqual(pycalc.extract_code("print(2 ** 10)"), "print(2 ** 10)")
        self.assertIsNone(pycalc.extract_code("12"))              # a bare number is a mental-maths answer
        self.assertIsNone(pycalc.extract_code("12 inches"))
        self.assertIsNone(pycalc.extract_code("print(2 ** 10)", allow_bare=False))

    def test_closed_only_for_cut_off_replies(self):
        self.assertEqual(pycalc.extract_code(PY("print(1 + 2)") + "\nWait, let me rethink" * 20, closed_only=True),
                         "print(1 + 2)")
        self.assertIsNone(pycalc.extract_code("```python\nprint(1 +", closed_only=True))
        self.assertEqual(pycalc.extract_code("```python\nprint(1 + 2)"), "print(1 + 2)")

    def test_no_calc(self):
        for r in ("NO_CALC", " no_calc. ", "`NO_CALC`", "NO_CALC\n"):
            self.assertTrue(pycalc.is_no_calc(r), r)
        self.assertFalse(pycalc.is_no_calc(PY("print(1 + 1)")))
        self.assertFalse(pycalc.is_no_calc("12"))


class Check(unittest.TestCase):
    def test_allowed(self):
        for code in ("import math\nprint(math.sqrt(2025))", "from datetime import date, timedelta\nprint(date.today())",
                     "from fractions import Fraction\nprint(Fraction(3, 8) + 1)", "for _ in range(2):\n    pass\nprint(_)",
                     "import statistics, calendar, decimal, cmath\nprint(statistics.mean([1, 2]))",
                     "print(convert(1, 'ft', 'in'))", "x = 9\nprint(x ** 2)"):
            self.assertEqual(pycalc.check_code(code), "", code)

    def test_every_bypass_is_rejected(self):
        for label, code in BYPASSES.items():
            self.assertNotEqual(pycalc.check_code(code), "", label)

    def test_limits_and_fixed_answers(self):
        self.assertIn("too long", pycalc.check_code("print(1 + 1)\n" * 400))
        self.assertIn("SyntaxError", pycalc.check_code("print(("))
        self.assertIn("fixed value", pycalc.check_code("print(45)"))
        self.assertIn("fixed value", pycalc.check_code("print('12 inches')"))
        self.assertIn("fixed value", pycalc.check_code("print(round(14, 2))"))
        self.assertIn("fixed value", pycalc.check_code("x = 15.96\nprint(x)"))
        self.assertIn("fixed value", pycalc.check_code("x = 15.96\nprint(f'{x} days')"))
        self.assertEqual(pycalc.check_code("x = 15\nx = x * 2\nprint(x)"), "")
        self.assertEqual(pycalc.check_code("d = days_until('christmas')\nprint(d)"), "")
        self.assertEqual(pycalc.check_code("for i in range(3):\n    pass\nprint(i)"), "")
        self.assertEqual(pycalc.check_code(""), "empty script")


class Run(unittest.TestCase):
    def test_success_and_tidy(self):
        self.assertEqual(pycalc.run_code("print(6.0 * 2, 'inches')"), (True, "12 inches"))
        self.assertEqual(pycalc.run_code("print(2 ** 64)"), (True, "18446744073709551616"))

    def test_utf8_output(self):
        self.assertEqual(pycalc.run_code("print('≈ € π', 1 + 1)"), (True, "≈ € π 2"))

    def test_dates_and_lazy_imports(self):
        ok, out = pycalc.run_code("from datetime import datetime, date\n"
                                  "print(datetime.strptime('2027-07-04', '%Y-%m-%d').strftime('%A'), date.today().year)")
        self.assertEqual((ok, out), (True, f"Sunday {dt.date.today().year}"))
        self.assertEqual(pycalc.run_code("import statistics\nprint(statistics.mode([1, 1, 2]))"), (True, "1"))

    def test_errors(self):
        self.assertEqual(pycalc.run_code("print(1 / 0)"), (False, "ZeroDivisionError: division by zero"))
        self.assertEqual(pycalc.run_code("x = [1, 2]\nx.sort()"), (False, "the script printed nothing"))
        ok, err = pycalc.run_code("import os\nprint(os.getcwd())")
        self.assertFalse(ok)
        self.assertIn("not allowed", err)

    def test_timeout(self):
        ok, err = pycalc.run_code("while True:\n    x = 1 + 1", timeout_s=1.0)
        self.assertEqual((ok, err), (False, "timed out after 1 s"))

    def test_output_cap(self):
        ok, err = pycalc.run_code("while True:\n    print('a' * 100)", timeout_s=3)
        self.assertEqual((ok, err), (False, "RuntimeError: output too long"))

    @unittest.skipUnless(os.name == "nt", "job objects are Windows only")
    def test_memory_limit(self):
        self.assertTrue(pycalc._Job().ok)
        ok, err = pycalc.run_code("x = bytearray(400 * 1024 * 1024)\nprint(len(x))")
        self.assertFalse(ok)
        self.assertIn("MemoryError", err)

    def test_no_python_means_no_routing(self):
        orig = pycalc.python_exe
        try:
            pycalc.python_exe = lambda: None
            self.assertFalse(pycalc.available())
            self.assertEqual(pycalc.run_code("print(1 + 1)"), (False, "no Python interpreter found"))
        finally:
            pycalc.python_exe = orig


class RuntimeLayer(unittest.TestCase):
    """The child's own defences, with the static check skipped (as if a payload had slipped past it)."""

    def raw(self, code):
        p = subprocess.run([pycalc.python_exe(), "-I", "-X", "utf8", "-S", "-c", pycalc.PRELUDE],
                           input=code.encode("utf-8"), capture_output=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace").strip()

    def test_open_is_blocked(self):
        rc, _out, err = self.raw("import calendar\nb = calendar.sys.modules['builtins']\nprint(b.open('x', 'w'))")
        self.assertEqual((rc, err), (1, "RuntimeError: blocked: open"))

    def test_os_system_is_blocked(self):
        rc, _out, err = self.raw("import statistics\nstatistics.sys.modules['os'].system('echo hi')")
        self.assertEqual((rc, err), (1, "RuntimeError: blocked: os.system"))

    def test_new_imports_are_blocked(self):
        for code in ("import os", "import ctypes", "import socket", "from fractions import sys"):
            rc, _out, err = self.raw(code)
            self.assertEqual(rc, 1, code)
            self.assertIn("not allowed", err, code)
        # the real __import__, reached through sys.modules: the audit hook still stops a new module loading
        rc, _out, err = self.raw("import statistics\nstatistics.sys.modules['builtins'].__import__('socket')")
        self.assertEqual((rc, err), (1, "RuntimeError: blocked: import"))

    def test_parent_caps_output_written_past_print(self):
        orig = pycalc.check_code
        try:
            pycalc.check_code = lambda code: ""         # as if the payload had passed the static check
            ok, err = pycalc.run_code("import statistics\nw = statistics.sys.modules['os'].write\n"
                                      "while True:\n    w(1, b'a' * 65536)", timeout_s=5)
        finally:
            pycalc.check_code = orig
        self.assertEqual((ok, err), (False, "output too long"))

    def test_restricted_builtins(self):
        rc, _out, err = self.raw("print(eval('1'))")
        self.assertEqual((rc, err), (1, "NameError: name 'eval' is not defined"))


class Helpers(unittest.TestCase):
    def run_ok(self, code):
        ok, out = pycalc.run_code(code)
        self.assertTrue(ok, out)
        return out

    def test_convert(self):
        out = self.run_ok("print(round(convert(1, 'foot', 'inches'), 2), round(convert(5, 'km', 'miles'), 2), "
                          "round(convert(72, '°F', 'celsius'), 2), round(convert(1, 'gallon', 'ounces'), 2), "
                          "round(convert(12, 'm^2', 'sq ft'), 2), round(convert(100, 'km/h', 'mph'), 2), "
                          "round(convert(3, 'weeks', 'hours'), 2))")
        self.assertEqual(out, "12 3.11 22.22 128 129.17 62.14 504")
        # exact factors: no float noise, even without round()
        self.assertEqual(self.run_ok("print(convert(1, 'ft', 'in'), convert(1, 'mile', 'feet'), "
                                     "convert(36, 'km/h', 'm/s'), convert(0, 'c', 'k'))"), "12 5280 10 273.15")
        ok, err = pycalc.run_code("print(convert(1, 'kg', 'm'))")
        self.assertEqual((ok, err), (False, "ValueError: can't convert kg to m"))

    def test_feet_and_inches(self):
        self.assertEqual(self.run_ok("print(feet_and_inches(180, 'cm'))"), "5 ft 10.87 in")

    def test_days_until(self):
        today = dt.date.today()

        def nxt(m, d):
            t = dt.date(today.year, m, d)
            return (t if t >= today else dt.date(today.year + 1, m, d)) - today
        out = self.run_ok("print(days_until('christmas'), days_until(6, 3), days_until('June 3'), "
                          "days_until('June', 3), days_until(3, 1, 2027), days_until('2027-03-01'))")
        want = [nxt(12, 25).days, nxt(6, 3).days, nxt(6, 3).days, nxt(6, 3).days,
                (dt.date(2027, 3, 1) - today).days, (dt.date(2027, 3, 1) - today).days]
        self.assertEqual(out, " ".join(map(str, want)))
        self.assertEqual(self.run_ok("print(next_date('thanksgiving', year=2026), next_date('easter', year=2027))"),
                         "2026-11-26 2027-03-28")
        fri = (4 - today.weekday()) % 7
        self.assertEqual(self.run_ok("print(days_until('friday'), days_until('Fridays'))"), f"{fri} {fri}")

    def test_prepare_code(self):
        self.assertEqual(pycalc.prepare_code("x = 9 ** 2"), "x = 9 ** 2\nprint(x)")
        self.assertEqual(pycalc.prepare_code("round(2.5 * 2, 2)"), "print(round(2.5 * 2, 2))")
        self.assertEqual(pycalc.prepare_code("print(1 + 1)"), "print(1 + 1)")
        self.assertEqual(pycalc.run_code("def convert(v, a, b):\n    return v\nprint(convert(1, 'ft', 'in'))"),
                         (True, "12"))     # the script's placeholder convert() is dropped

    def test_round_noise(self):
        self.assertEqual(pycalc.round_noise("5 ft 10.866141732283467 in, 0.30000000000000004, 3.14159"),
                         "5 ft 10.87 in, 0.3, 3.14159")
        self.assertEqual(pycalc.round_noise("3.14159265358979", "pi to 10 decimal places"), "3.14159265358979")
        self.assertEqual(pycalc.round_noise("12.000000000000002", "convert one foot to inches exactly"), "12")


# ============================================================================ Cleaner with a mocked LLM
class FakeResp:
    def __init__(self, content, finish="stop"):
        self._j = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


class CleanerCalc(unittest.TestCase):
    def make(self, replies, **cfg_over):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "c.json")
        cfg.set("cleanup.model_path", "models/LFM3-1.7B-Q8_0.gguf", save=False)
        for k, v in cfg_over.items():
            cfg.set(k, v, save=False)
        c = cleanup.Cleaner(cfg)
        c.server.ready = True
        c.server.proc = None
        self.bodies = []
        replies = list(replies)

        def post(url, json=None, timeout=None):
            self.bodies.append(json)
            r = replies.pop(0)
            return r if isinstance(r, FakeResp) else FakeResp(r)
        c.http.post = post
        return c

    def test_command_runs_the_script(self):
        c = self.make([PY("print(round(convert(1, 'foot', 'inches'), 2), 'inches')")])
        out, info = c.command("convert one foot to inches", "", "notepad.exe")
        self.assertEqual(out, "12 inches")
        self.assertTrue(info["python"] and info["llm"])
        self.assertIn("convert(1, 'foot'", info["code"])
        self.assertEqual(len(self.bodies), 1)
        b = self.bodies[0]
        self.assertEqual(b["messages"][0], {"role": "system", "content": cleanup.CALC_SYSTEM_PROMPT})
        self.assertEqual(len(b["messages"]), 2 + 2 * len(cleanup.CALC_SHOTS))
        self.assertEqual(b["messages"][-1], {"role": "user", "content": "convert one foot to inches"})
        self.assertEqual(b["temperature"], 0.0)
        self.assertEqual(b["max_tokens"], 400)

    def test_answer_runs_the_script_and_joins_lines(self):
        c = self.make([PY("print(6 * 7)\nprint('days')")])
        out, info = c.answer("how many days until christmas", "", "notepad.exe")
        self.assertEqual((out, info["python"]), ("42 days", True))

    def test_no_calc_falls_back_to_the_normal_path(self):
        c = self.make(["NO_CALC", "Two hundred"])
        out, info = c.command("convert 5 feet to inches", "", "notepad.exe")
        self.assertEqual(out, "Two hundred")
        self.assertFalse(info.get("python"))
        self.assertEqual(len(self.bodies), 2)
        want, _t, _w = c.command_request("convert 5 feet to inches", "", "notepad.exe")
        self.assertEqual(self.bodies[1], want)

    def test_writing_command_with_a_quantity_never_reaches_the_calculator(self):
        c = self.make(["The package weighs 3 pounds."])
        out, info = c.command("say the package weighs 3 pounds", "", "x")
        self.assertEqual((out, len(self.bodies)), ("The package weighs 3 pounds.", 1))
        self.assertEqual(self.bodies[0]["messages"][0]["content"], cleanup.COMMAND_SYSTEM_PROMPT)

    def test_copied_example_script_takes_the_normal_path(self):
        copied = PY("cm = convert(5, 'ft', 'cm') + convert(11, 'in', 'cm')\nprint(f'{round(cm, 2)} cm')")
        c = self.make([copied, "about 152 cm"])
        out, info = c.answer("how many cm in 60 inches of rope", "", "x")
        self.assertEqual(out, "about 152 cm")
        self.assertFalse(info.get("python"))

    def test_prose_reply_takes_the_normal_path(self):
        c = self.make(["It is 12 inches.", "12 inches"])
        out, info = c.command("convert one foot to inches", "", "x")
        self.assertEqual((out, info.get("python")), ("12 inches", None))
        self.assertEqual(self.bodies[1], c.command_request("convert one foot to inches", "", "x")[0])

    def test_failed_script_gets_one_retry(self):
        c = self.make([PY("print(1 / 0)"), PY("print(12 * 1)")])
        out, info = c.command("convert one foot to inches", "", "x")
        self.assertEqual(out, "12")
        retry = self.bodies[1]["messages"]
        self.assertEqual(retry[-2], {"role": "assistant", "content": PY("print(1 / 0)")})
        self.assertIn("ZeroDivisionError", retry[-1]["content"])

    def test_two_failures_insert_nothing_and_never_fall_back(self):
        c = self.make([PY("print(1 / 0)"), "it is 12 inches"])
        out, info = c.command("convert one foot to inches", "", "x")
        self.assertEqual((out, info["reason"]), ("", "couldn't calculate"))
        self.assertEqual(len(self.bodies), 2)          # no third request: no mental-maths fallback
        self.assertFalse(info.get("python"))

    def test_cut_off_reply_uses_the_finished_block(self):
        c = self.make([FakeResp(PY("print(3 * 4)") + "\nWait, actually" * 30, finish="length")])
        out, info = c.command("what's 3 times 4", "", "x")
        self.assertEqual(out, "12")

    def test_selection_never_runs_code(self):
        c = self.make(["300.48 cm"])
        out, info = c.command("convert 5 feet to cm", "5 feet", "x")
        self.assertFalse(c.wants_calc("convert 5 feet to cm", "5 feet"))
        self.assertEqual(self.bodies[0], c.command_request("convert 5 feet to cm", "5 feet", "x")[0])
        c = self.make(["about 152 cm"])
        c.answer("how many cm is 5 feet", "some selected text", "x")
        self.assertEqual(self.bodies[0]["messages"][0]["content"], cleanup.ANSWER_SYSTEM_PROMPT)

    def test_run_python_off_is_the_old_path(self):
        c = self.make(["12 inches"], **{"command_mode.run_python": False})
        self.assertFalse(c.wants_calc("convert one foot to inches"))
        out, info = c.command("convert one foot to inches", "", "x")
        self.assertEqual(out, "12 inches")
        self.assertEqual(self.bodies[0], c.command_request("convert one foot to inches", "", "x")[0])
        self.assertEqual(len(self.bodies[0]["messages"]), 2)

    def test_not_a_calculation_skips_the_calc_request(self):
        c = self.make(["Paris"])
        out, _info = c.answer("what is the capital of France", "", "x")
        self.assertEqual(out, "Paris")
        self.assertEqual(self.bodies[0]["messages"][0]["content"], cleanup.ANSWER_SYSTEM_PROMPT)

    def test_existing_request_bodies_have_no_extra_turns(self):
        c = self.make([])
        for body in (c.command_request("make this formal", "hi", "x")[0], c.answer_request("who wrote hamlet")[0]):
            self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])


if __name__ == "__main__":
    unittest.main()
