"""Per-model prompt mode (cleanup.model_prompt), reasoning (cleanup.model_reasoning), the request builders an eval
harness reuses, and the model_defaults_v1 config migration. No llama-server is started."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localflow import assistant as assistant_mod  # noqa: E402
from localflow import cleanup, styles  # noqa: E402
from localflow.cleanup import (CONTEXT_FEWSHOT, FEWSHOT, SYSTEM_PROMPT, Cleaner, build_messages,  # noqa: E402
                               unspoken_stops)
from localflow.config import DEFAULTS, LFM3_STOPS, Config, _migrate_model_defaults  # noqa: E402

OV = cleanup.REASONING_OVERHEAD_TOKENS     # forced end-of-thinking message + tags reserved on top of a budget

QWEN = "models/Qwen3.5-4B-Q8_0.gguf"
LFM3 = "models/LFM3-1.7B-Q8_0.gguf"
LOCALFLOW = "models/LFM3-1.7B-LocalFlow-Q8_0.gguf"


def tmp_cfg():
    return Config(path=Path(tempfile.mkdtemp()) / "config.json")


def make_cleaner(model=QWEN, launched=True):
    cfg = tmp_cfg()
    cfg.set("cleanup.model_path", model, save=False)
    cl = Cleaner(cfg)
    if launched:
        cl.server.launched = cl.server.launch()[1]
    return cl


# ---- frozen copies of the request building as it was before compact mode / reasoning (golden reference)
def _legacy_style_header(sd, rules=True):
    text = " ".join((sd.get("prompt", "") or "").split())
    if sd.get("name", styles.NEUTRAL) == styles.NEUTRAL and not text:
        return ""
    h = f"[style: {sd.get('name', styles.NEUTRAL)}]\n"
    if rules and text:
        h += f"[style rules: {text}]\n"
    return h


def _legacy_wrap(text, header):
    return f"{header or ''}<transcript>{text}</transcript>"


def legacy_messages(raw, sd, vocab, context=""):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": _legacy_wrap(u, "")})
        msgs.append({"role": "assistant", "content": a})
    if context:
        prev, u, a = CONTEXT_FEWSHOT
        msgs.append({"role": "user", "content": _legacy_wrap(u, f"[previous text: {prev}]\n")})
        msgs.append({"role": "assistant", "content": a})
    head = _legacy_style_header(sd)
    short_head = _legacy_style_header(sd, rules=False)
    for u, a in sd.get("examples", []) or []:
        msgs.append({"role": "user", "content": _legacy_wrap(u, short_head)})
        msgs.append({"role": "assistant", "content": a})
    header = head
    if context:
        header += "[previous text: " + "\n".join(
            l for l in (" ".join(x.split()) for x in context.strip().splitlines()) if l) + "]\n"
    if vocab:
        header += f"[vocabulary: {', '.join(vocab[:40])}]\n"
    msgs.append({"role": "user", "content": _legacy_wrap(raw, header)})
    return msgs


def legacy_cleanup_body(cl, raw, sd, vocab, context=""):
    n_in = max(8, len(raw) // 3)
    body = {"messages": legacy_messages(raw, sd, vocab, context),
            "temperature": 0.0, "top_k": 1, "max_tokens": int(n_in * 1.8 + 40),
            "cache_prompt": True, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    body.update(cl._request_overrides())
    if "stop" in body:
        body["stop"] = unspoken_stops(body["stop"], raw)
        if not body["stop"]:
            del body["stop"]
    timeout = float(cl.cfg.get("cleanup.timeout_s", 5.0)) + len(raw) / 200.0
    return body, timeout


def legacy_command_body(cl, instruction, selection, app_name):
    cfg = cl.cfg
    sel = (selection or "").replace("\r\n", "\n")
    user = f"<app>{app_name or 'unknown'}</app>\n<instruction>{instruction.strip()}</instruction>\n" + \
        (f"<selection>\n{sel}\n</selection>" if sel else "<selection></selection>")
    ctx = int(cfg.get("cleanup.ctx", 8192) or 8192)
    prompt_tok = int((len(cleanup.COMMAND_SYSTEM_PROMPT) + len(user)) / 3.0) + 32
    want = max(256, int(2.5 * len(sel) / 3.5)) if sel else int(cfg.get("command_mode.max_tokens_generate", 768))
    max_tokens = min(want, ctx - prompt_tok - 16)
    if max_tokens < 64:
        return None
    body = {"messages": [{"role": "system", "content": cleanup.COMMAND_SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "temperature": 0.2, "max_tokens": max_tokens, "cache_prompt": True, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(cl._request_overrides())
    body["max_tokens"] = max_tokens
    stops = cl._command_stops()
    if stops:
        body["stop"] = stops
    else:
        body.pop("stop", None)
    body["reasoning_format"] = "none"   # intentional change: raw content keeps the first line's indentation
    return body, float(cfg.get("command_mode.timeout_s", 30.0))


CASES = [
    ("so um this is a test of the thing", "neutral", [], ""),
    ("hey can you send me the notes note that its late", "casual", ["Maria", "Kubernetes"], ""),
    ("and then we should uh book the room", "professional", [], "Earlier sentence.\n- item one\n  - item two"),
    ("dear team the release is friday actually monday", "email", ["LocalFlow"], "Previous paragraph here."),
    ("def foo open paren x close paren", "code", [], ""),
    ("let me think wait no the meeting is at four", "neutral", ["Qwen"], "context"),
]


class FullModeIsUnchanged(unittest.TestCase):
    def test_build_messages_full_matches_legacy(self):
        cfg = tmp_cfg()
        for raw, style, vocab, ctx in CASES:
            sd = styles.get_style(cfg, style)
            self.assertEqual(build_messages(raw, sd, vocab, ctx), legacy_messages(raw, sd, vocab, ctx), style)
            self.assertEqual(build_messages(raw, sd, vocab, ctx, mode="full"), legacy_messages(raw, sd, vocab, ctx))

    def test_custom_style_examples_in_full_mode(self):
        cfg = tmp_cfg()
        cfg.set("styles.pirate", {"prompt": "Talk like a pirate.", "examples": [["hello there", "Ahoy there."]]},
                save=False)
        sd = styles.get_style(cfg, "pirate")
        self.assertTrue(sd["custom_examples"])
        self.assertEqual(build_messages("hi", sd, []), legacy_messages("hi", sd, []))

    def test_cleanup_and_command_bodies_unchanged_for_shipped_models(self):
        for model in (QWEN, LFM3):
            cl = make_cleaner(model)
            self.assertEqual(cl._prompt_mode(), "full")
            self.assertEqual(cl.reasoning_budget(), 0)
            for raw, style, vocab, ctx in CASES:
                sd = styles.get_style(cl.cfg, style)
                self.assertEqual(cl.cleanup_request(raw, sd, vocab, ctx), legacy_cleanup_body(cl, raw, sd, vocab, ctx),
                                 (model, raw))
            for ins, sel, app in (("make this formal", "hey whats up\n", "notepad.exe"),
                                  ("write a haiku about rain", "", ""), ("fix it", "x" * 60000, "code.exe")):
                got = cl.command_request(ins, sel, app)
                want = legacy_command_body(cl, ins, sel, app)
                if want is None:
                    self.assertIsNone(got[0])
                    self.assertEqual(got[2], "selection too long for the AI model")
                else:
                    self.assertEqual(got, (want[0], want[1], ""), (model, ins))

    def test_lfm3_stops_still_sent(self):
        cl = make_cleaner(LFM3)
        body, _ = cl.cleanup_request("so um we should meet at four", styles.get_style(cl.cfg, "neutral"), [])
        self.assertIn("</think>", body["stop"])
        self.assertIn("\n\nLet me", body["stop"])
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        cbody, _, _ = cl.command_request("write a list", "", "")
        self.assertEqual(cbody["stop"], ["</think>", "<think>"])

    def test_launch_unchanged_when_reasoning_off(self):
        for model in (QWEN, LFM3):
            cl = make_cleaner(model, launched=False)
            cmd, sig = cl.server.launch()
            self.assertNotIn("--reasoning-budget", cmd)
            self.assertEqual(cmd[-1], "--no-mmproj")      # nothing appended after the base command line
            cl.cfg.set("cleanup.model_reasoning", {Path(model).name: {"enabled": False, "budget": 512}}, save=False)
            self.assertEqual(cl.server.launch(), (cmd, sig))


class CompactMode(unittest.TestCase):
    def setUp(self):
        cleanup._warned.clear()

    def test_compact_drops_fewshot_and_builtin_examples(self):
        cfg = tmp_cfg()
        sd = styles.get_style(cfg, "casual")
        self.assertFalse(sd["custom_examples"])
        self.assertTrue(sd["examples"])     # built-in examples were filled in ...
        msgs = build_messages("so um hey whats up", sd, ["Maria"], "Earlier text.", mode="compact")
        self.assertEqual(len(msgs), 2)       # ... but compact mode leaves them out, like the FEWSHOT pairs
        self.assertEqual(msgs[0], {"role": "system", "content": SYSTEM_PROMPT})
        last = msgs[-1]["content"]
        self.assertEqual(last, build_messages("so um hey whats up", sd, ["Maria"], "Earlier text.")[-1]["content"])
        self.assertIn("[style: casual]\n[style rules: ", last)
        self.assertIn("[previous text: Earlier text.]\n", last)
        self.assertIn("[vocabulary: Maria]\n", last)
        self.assertTrue(last.endswith("<transcript>so um hey whats up</transcript>"))

    def test_compact_neutral_is_system_plus_request(self):
        sd = styles.get_style(tmp_cfg(), "neutral")
        self.assertEqual(build_messages("hello there", sd, [], mode="compact"),
                         [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": "<transcript>hello there</transcript>"}])

    def test_compact_keeps_user_examples(self):
        cfg = tmp_cfg()
        cfg.set("styles.casual", dict(styles.DEFAULT_STYLES["casual"], examples=[["um hi there", "hi there"]]),
                save=False)
        sd = styles.get_style(cfg, "casual")
        self.assertTrue(sd["custom_examples"])
        msgs = build_messages("so um yes", sd, [], mode="compact")
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[1], {"role": "user", "content": "[style: casual]\n<transcript>um hi there</transcript>"})
        self.assertEqual(msgs[2], {"role": "assistant", "content": "hi there"})
        empty = dict(styles.DEFAULT_STYLES["casual"], examples=[])      # user deleted every example
        cfg.set("styles.casual", empty, save=False)
        self.assertEqual(len(build_messages("so um yes", styles.get_style(cfg, "casual"), [], mode="compact")), 2)

    def test_default_config_makes_localflow_compact(self):
        cl = make_cleaner(LOCALFLOW)
        self.assertEqual(cl._prompt_mode(), "compact")
        body, _ = cl.cleanup_request("so um this is a test", styles.get_style(cl.cfg, "neutral"), [])
        self.assertEqual(len(body["messages"]), 2)
        self.assertEqual(body["stop"], ["<think>", "</think>"])
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])

    def test_mode_follows_loaded_model_mid_switch(self):
        cl = make_cleaner(QWEN)                                 # Qwen is running ...
        cl.cfg.set("cleanup.model_path", LOCALFLOW, save=False)  # ... while the switch to the fine-tune is under way
        self.assertEqual(cl._prompt_mode(), "full")
        cl = make_cleaner(LOCALFLOW)
        cl.cfg.set("cleanup.model_path", QWEN, save=False)
        self.assertEqual(cl._prompt_mode(), "compact")
        cl = make_cleaner(LOCALFLOW, launched=False)            # nothing launched yet: configured model
        self.assertEqual(cl._prompt_mode(), "compact")

    def test_case_insensitive_and_bad_values(self):
        cl = make_cleaner(QWEN)
        cl.cfg.set("cleanup.model_prompt", {"qwen3.5-4b-q8_0.GGUF": " Compact "}, save=False)
        self.assertEqual(cl._prompt_mode(), "compact")
        for bad in ("short", 1, None, ["compact"], {"x": 1}):
            cl.cfg.set("cleanup.model_prompt", {"Qwen3.5-4B-Q8_0.gguf": bad}, save=False)
            with self.assertLogs("localflow.cleanup", "WARNING") if bad is not None else _nolog():
                self.assertEqual(cl._prompt_mode(), "full")
        cl.cfg.set("cleanup.model_prompt", "compact", save=False)       # not a map at all
        self.assertEqual(cl._prompt_mode(), "full")


class _nolog:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Reasoning(unittest.TestCase):
    def setUp(self):
        cleanup._warned.clear()

    def enable(self, cl, model, **entry):
        cl.cfg.set("cleanup.model_reasoning", {Path(model).name: dict({"enabled": True}, **entry)}, save=False)

    def test_launch_flags(self):
        cl = make_cleaner(QWEN, launched=False)
        base, _ = cl.server.launch()
        self.enable(cl, QWEN, budget=512)
        cmd, sig = cl.server.launch()
        self.assertEqual(cmd, base + ["--reasoning-budget", "512", "--reasoning-budget-message",
                                      cleanup.REASONING_BUDGET_MESSAGE])
        self.assertEqual(sig[2], tuple(cmd[3:]))
        cl.cfg.set("cleanup.model_args", {"Qwen3.5-4B-Q8_0.gguf": ["--temp", "0"]}, save=False)
        cmd, _ = cl.server.launch()
        self.assertEqual(cmd[len(base):len(base) + 2], ["--temp", "0"])    # after model_args

    def test_budget_parsing(self):
        cl = make_cleaner(QWEN, launched=False)
        model = Path(cl.server._abs(QWEN))
        for entry, want in (({"enabled": True}, 256), ({"enabled": True, "budget": "bad"}, 256),
                            ({"enabled": True, "budget": 99999}, 8192), ({"enabled": True, "budget": -5}, 0),
                            ({"enabled": True, "budget": "300"}, 300), ({"enabled": False, "budget": 300}, 0),
                            ({"enabled": "yes", "budget": 300}, 0), ("on", 0), ({"budget": 300}, 0),
                            ({"enabled": True, "budget": float("nan")}, 256),
                            ({"enabled": True, "budget": float("inf")}, 256), ({"enabled": True, "budget": "nan"}, 256),
                            ({"enabled": True, "budget": 300.7}, 300)):
            cl.cfg.set("cleanup.model_reasoning", {"QWEN3.5-4B-Q8_0.GGUF": entry}, save=False)
            self.assertEqual(cleanup.reasoning_budget_for(cl.cfg, model), want, entry)

    def test_reasoning_budget_comes_from_running_server(self):
        cl = make_cleaner(QWEN, launched=False)
        self.enable(cl, QWEN, budget=384)
        self.assertEqual(cl.reasoning_budget(), 384)        # not launched yet: config
        cl.server.launched = cl.server.launch()[1]
        self.assertEqual(cl.reasoning_budget(), 384)
        cl.cfg.set("cleanup.model_reasoning", {}, save=False)   # turned off, server not restarted yet
        self.assertEqual(cl.reasoning_budget(), 384)
        cl.server.launched = cl.server.launch()[1]
        self.assertEqual(cl.reasoning_budget(), 0)
        cl.server.launched = ("exe", "model")                 # odd signature (test fakes): off, no crash
        self.assertEqual(cl.reasoning_budget(), 0)

    def test_cleanup_request_with_reasoning(self):
        cl = make_cleaner(LFM3, launched=False)
        self.enable(cl, LFM3, budget=200)
        cl.server.launched = cl.server.launch()[1]
        raw = "so um we should meet at four"
        sd = styles.get_style(cl.cfg, "neutral")
        body, timeout = cl.cleanup_request(raw, sd, [])
        off = make_cleaner(LFM3)
        base, base_timeout = off.cleanup_request(raw, sd, [])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("stop", body)
        self.assertEqual(body["max_tokens"], base["max_tokens"] + 200 + OV)
        self.assertAlmostEqual(timeout, base_timeout + 200 / 40.0)
        self.assertEqual(body["messages"], base["messages"])

    def test_reasoning_wins_over_request_overrides(self):
        cl = make_cleaner(QWEN, launched=False)
        self.enable(cl, QWEN, budget=64)
        cl.cfg.set("cleanup.model_request", {"Qwen3.5-4B-Q8_0.gguf": {
            "chat_template_kwargs": {"enable_thinking": False, "foo": 1}, "max_tokens": 100, "stop": ["x"]}},
            save=False)
        cl.server.launched = cl.server.launch()[1]
        body, _ = cl.cleanup_request("so um this is a test", styles.get_style(cl.cfg, "neutral"), [])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True, "foo": 1})
        self.assertEqual(body["max_tokens"], 164 + OV)
        self.assertNotIn("stop", body)

    def test_command_request_with_reasoning(self):
        cl = make_cleaner(LFM3, launched=False)
        self.enable(cl, LFM3, budget=300)
        cl.server.launched = cl.server.launch()[1]
        body, timeout, why = cl.command_request("write a haiku", "", "")
        self.assertEqual(why, "")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("stop", body)
        self.assertEqual(body["max_tokens"], 768 + 300 + OV)
        self.assertAlmostEqual(timeout, 30.0 + 300 / 40.0)
        # ctx cap: the thinking budget is capped by the context; the answer part still needs 64 tokens
        cl.cfg.set("cleanup.ctx", 2048, save=False)
        body, _, why = cl.command_request("write a haiku", "", "")
        prompt_tok = int((len(cleanup.COMMAND_SYSTEM_PROMPT) + len(
            "<app>unknown</app>\n<instruction>write a haiku</instruction>\n<selection></selection>")) / 3.0) + 32
        self.assertEqual(body["max_tokens"], min(768 + 300 + OV, 2048 - prompt_tok - 16))
        # a selection that leaves ~200 tokens of room: enough without reasoning, too little once 300 are reserved
        tag_len = len("<app>unknown</app>\n<instruction>shorten</instruction>\n<selection>\n\n</selection>")
        n = 3 * (2048 - 16 - 32 - 200) - len(cleanup.COMMAND_SYSTEM_PROMPT) - tag_len
        sel = ("word " * n)[:n]
        body, _, why = cl.command_request("shorten", sel, "")
        self.assertIsNone(body)
        self.assertEqual(why, "selection too long for the AI model")
        off = make_cleaner(LFM3)
        off.cfg.set("cleanup.ctx", 2048, save=False)
        body, _, why = off.command_request("shorten", sel, "")
        self.assertEqual(why, "")
        self.assertTrue(64 <= body["max_tokens"] <= 300)

    def test_reasoning_content_ignored(self):
        cl = make_cleaner(QWEN, launched=False)
        self.enable(cl, QWEN, budget=64)
        cl.server.launched = cl.server.launch()[1]
        resp = mock.Mock()
        resp.json.return_value = {"choices": [{"message": {"content": "We meet at 4.",
                                                           "reasoning_content": "The user said four..."},
                                               "finish_reason": "stop"}]}
        cl.http = mock.Mock()
        cl.http.post.return_value = resp
        self.assertEqual(cl._llm("so we meet at four", styles.get_style(cl.cfg, "neutral"), []), "We meet at 4.")
        sent = cl.http.post.call_args.kwargs["json"]
        self.assertTrue(sent["chat_template_kwargs"]["enable_thinking"])
        # warm-up only primes the prompt cache: with reasoning it must not spend the thinking budget ...
        cl._llm("so um this is just a warm up test", styles.get_style(cl.cfg, "neutral"), [], warm=True)
        self.assertEqual(cl.http.post.call_args.kwargs["json"]["max_tokens"], 1)
        # ... without reasoning warm-up requests are unchanged
        off = make_cleaner(QWEN)
        off.http = cl.http
        sd = styles.get_style(off.cfg, "neutral")
        off._llm("so um this is just a warm up test", sd, [], warm=True)
        self.assertEqual(off.http.post.call_args.kwargs["json"],
                         off.cleanup_request("so um this is just a warm up test", sd, [])[0])


class AssistantReasoning(unittest.TestCase):
    def make(self, budget=0, extra=None):
        cfg = tmp_cfg()
        return assistant_mod.Assistant(cfg.get, mock.Mock(), mock.Mock(), lambda: "http://x", lambda: True,
                                       request_extra_fn=(lambda: extra or {}),
                                       reasoning_budget_fn=(lambda: budget))

    def test_body_unchanged_without_reasoning(self):
        a = self.make(0, {"stop": ["</think>"], "top_k": 1})
        b = a._body([{"role": "user", "content": "x"}], 500)
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(b["max_tokens"], 500)
        self.assertEqual(b["top_k"], 1)
        self.assertNotIn("stop", b)
        no_fn = assistant_mod.Assistant(tmp_cfg().get, mock.Mock(), mock.Mock(), lambda: "", lambda: True)
        self.assertEqual(no_fn._body([], 500)["max_tokens"], 500)

    def test_body_with_reasoning(self):
        b = self.make(256)._body([{"role": "user", "content": "x"}], 500)
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(b["max_tokens"], 756 + OV)

    def test_full_budget_reserved_and_errors_are_off(self):
        self.assertEqual(self.make(4096)._reasoning_budget(), 4096 + OV)
        self.assertEqual(self.make(-3)._reasoning_budget(), 0)
        a = self.make(0)
        a.reasoning_budget_fn = mock.Mock(side_effect=RuntimeError("x"))
        self.assertEqual(a._reasoning_budget(), 0)

    def test_budget_too_large_for_context(self):
        a = self.make(8192)
        a._complete = mock.Mock(return_value="x")
        with self.assertRaisesRegex(assistant_mod.AssistantError, "reasoning budget is too large"):
            a.run("word " * 100, "summarize")
        a._complete.assert_not_called()
        a = self.make(4096)      # fits next to the answer in an 8192 context
        a._complete = mock.Mock(return_value="x")
        self.assertEqual(a.run("word " * 100, "summarize"), "x")
        self.assertEqual(a._complete.call_args.kwargs["think_cap"], 4096 + OV)

    def test_budget_too_large_for_map_reduce(self):
        a = self.make(2400)
        a.cfg_get = lambda k, d=None: 4096 if k == "cleanup.ctx" else d
        a._complete = mock.Mock(return_value="x")
        with self.assertRaisesRegex(assistant_mod.AssistantError, "reasoning budget is too large"):
            a.run("This is a sentence of a long transcript. " * 400, "summarize")
        a._complete.assert_not_called()

    def test_budget_raised_mid_run_fails_instead_of_overflowing(self):
        budget = {"v": 0}
        a = self.make(0)
        a.reasoning_budget_fn = lambda: budget["v"]
        a._wait_ready = mock.Mock(side_effect=lambda *x: budget.update(v=4096))   # reloaded while waiting
        a.http = mock.Mock()
        with self.assertRaisesRegex(assistant_mod.AssistantError, "reasoning setting changed"):
            a._complete([{"role": "user", "content": "x"}], 100, mock.Mock(is_set=lambda: False), None, None,
                        think_cap=0)
        a.http.post.assert_not_called()
        # lowered (or turned off) mid-run: fine, the request just uses the running server's budget
        budget["v"] = 512
        a._wait_ready = mock.Mock(side_effect=lambda *x: budget.update(v=0))
        a.gate.busy.return_value = False
        resp = mock.MagicMock()
        resp.__enter__.return_value = resp
        resp.iter_lines.return_value = iter([b'data: {"choices":[{"delta":{"content":"ok"}}]}', b"data: [DONE]"])
        a.http.post.return_value = resp
        self.assertEqual(a._complete([{"role": "user", "content": "x"}], 100, mock.Mock(is_set=lambda: False),
                                     None, None, think_cap=512), "ok")
        sent = a.http.post.call_args.kwargs["json"]
        self.assertEqual(sent["max_tokens"], 100)
        self.assertFalse(sent["chat_template_kwargs"]["enable_thinking"])

    def test_context_budget_reserves_thinking_tokens(self):
        for think in (0, 1024):
            a = self.make(think)
            a._complete = mock.Mock(return_value="notes")
            with mock.patch.object(assistant_mod, "budget_chars", wraps=assistant_mod.budget_chars) as bc:
                a.run("word " * 100, "summarize")
            max_tokens = max(64, min(1024, 8192 // 3))
            self.assertEqual(bc.call_args_list[0].args[2], max_tokens + cleanup.thinking_tokens(think))

    def test_streaming_ignores_reasoning_deltas(self):
        a = self.make(256)
        a.gate.busy.return_value = False
        lines = [b'data: {"choices":[{"delta":{"reasoning_content":"thinking hard"}}]}',
                 b'data: {"choices":[{"delta":{"reasoning_content":" more"}}]}',
                 b'data: {"choices":[{"delta":{"content":"The answer."}}]}',
                 b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}', b"data: [DONE]"]
        resp = mock.MagicMock()
        resp.__enter__.return_value = resp
        resp.iter_lines.return_value = iter(lines)
        a.http = mock.Mock()
        a.http.post.return_value = resp
        partials = []
        out = a._complete([{"role": "user", "content": "x"}], 100, mock.Mock(is_set=lambda: False),
                          partials.append, None)
        self.assertEqual(out, "The answer.")
        self.assertTrue(all("thinking" not in p for p in partials))
        self.assertEqual(a.http.post.call_args.kwargs["json"]["max_tokens"], 356 + OV)


class ModelDefaultsMigration(unittest.TestCase):
    OLD_MARKERS = ["email_style_v1", "casual_format_v1", "casual_format_v2"]

    def old_config(self, mutate):
        """A config.json written by the previous version (fresh file, then rewound), changed by ``mutate``."""
        p = Path(tempfile.mkdtemp()) / "config.json"
        Config(path=p)
        data = json.loads(p.read_text(encoding="utf-8"))
        data["migrations"] = list(self.OLD_MARKERS)
        data["cleanup"]["model_request"] = {"LFM3-1.7B-Q8_0.gguf": {"stop": LFM3_STOPS}}
        data["cleanup"].pop("model_prompt", None)
        data["cleanup"].pop("model_reasoning", None)
        mutate(data["cleanup"])
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    def test_old_config_gets_localflow_entries(self):
        p = self.old_config(lambda c: None)
        cfg = Config(path=p)
        self.assertEqual(cfg.get("cleanup.model_request"), {
            "LFM3-1.7B-Q8_0.gguf": {"stop": LFM3_STOPS}, "LFM3-1.7B-LocalFlow-Q8_0.gguf": {"stop": ["<think>", "</think>"]}})
        self.assertEqual(cfg.get("cleanup.model_prompt"), {"LFM3-1.7B-LocalFlow-Q8_0.gguf": "compact"})
        self.assertEqual(cfg.get("cleanup.model_reasoning"), {})
        disk = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(disk["cleanup"]["model_request"], cfg.get("cleanup.model_request"))
        self.assertEqual(disk["cleanup"]["model_prompt"], {"LFM3-1.7B-LocalFlow-Q8_0.gguf": "compact"})
        self.assertIn("model_defaults_v1", disk["migrations"])
        self.assertEqual(disk["cleanup"]["model_path"], DEFAULTS["cleanup"]["model_path"])

    def test_empty_maps_get_defaults(self):
        def m(c):
            c["model_request"] = {}
            c["model_prompt"] = {}
        cfg = Config(path=self.old_config(m))
        self.assertEqual(cfg.get("cleanup.model_request"), DEFAULTS["cleanup"]["model_request"])
        self.assertEqual(cfg.get("cleanup.model_prompt"), DEFAULTS["cleanup"]["model_prompt"])

    def test_existing_entries_kept(self):
        def m(c):
            c["model_request"] = {"lfm3-1.7b-localflow-q8_0.GGUF": {"stop": ["X"]}, "Other.gguf": {"top_k": 5}}
            c["model_prompt"] = {"LFM3-1.7B-LOCALFLOW-Q8_0.gguf": "full"}
            c["model_args"] = {"Other.gguf": ["--temp", "0"]}
            c["model_reasoning"] = {"Other.gguf": {"enabled": True, "budget": 128}}
        cfg = Config(path=self.old_config(m))
        self.assertEqual(cfg.get("cleanup.model_request"), {
            "lfm3-1.7b-localflow-q8_0.GGUF": {"stop": ["X"]}, "Other.gguf": {"top_k": 5},
            "LFM3-1.7B-Q8_0.gguf": {"stop": LFM3_STOPS}})       # only the missing LFM3 default was added
        self.assertEqual(cfg.get("cleanup.model_prompt"), {"LFM3-1.7B-LOCALFLOW-Q8_0.gguf": "full"})
        self.assertEqual(cfg.get("cleanup.model_args"), {"Other.gguf": ["--temp", "0"]})
        self.assertEqual(cfg.get("cleanup.model_reasoning"), {"Other.gguf": {"enabled": True, "budget": 128}})

    def test_non_map_left_alone(self):
        def m(c):
            c["model_request"] = "oops"
        cfg = Config(path=self.old_config(m))
        self.assertEqual(cfg.get("cleanup.model_request"), "oops")
        self.assertEqual(cfg.get("cleanup.model_prompt"), DEFAULTS["cleanup"]["model_prompt"])

    def test_reload_is_noop(self):
        p = self.old_config(lambda c: None)
        Config(path=p)
        before = p.read_bytes()
        mtime = p.stat().st_mtime_ns
        cfg = Config(path=p)
        cfg.reload_if_changed()
        self.assertEqual(p.read_bytes(), before)
        self.assertEqual(p.stat().st_mtime_ns, mtime)

    def test_function_idempotent(self):
        data = {"cleanup": {"model_request": {"LFM3-1.7B-Q8_0.gguf": {"stop": ["a"]}}}}
        self.assertTrue(_migrate_model_defaults(data))
        snap = copy.deepcopy(data)
        self.assertFalse(_migrate_model_defaults(data))
        self.assertEqual(data, snap)
        self.assertEqual(data["cleanup"]["model_request"]["LFM3-1.7B-Q8_0.gguf"], {"stop": ["a"]})
        self.assertFalse(_migrate_model_defaults({"cleanup": "bad"}))
        self.assertFalse(_migrate_model_defaults({}))

    def test_defaults_do_not_touch_shipped_models(self):
        for key in ("model_prompt", "model_reasoning", "model_args"):
            names = {k.lower() for k in DEFAULTS["cleanup"][key]}
            self.assertFalse(names & {"qwen3.5-4b-q8_0.gguf", "lfm3-1.7b-q8_0.gguf"}, key)
        self.assertEqual(DEFAULTS["cleanup"]["model_request"]["LFM3-1.7B-Q8_0.gguf"], {"stop": LFM3_STOPS})


class SettingsReasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.qapp = QApplication.instance() or QApplication([])

    def make(self, model=LFM3, table=None):
        from localflow.settings_ui import SettingsWindow
        cfg = tmp_cfg()
        cfg.set("cleanup.model_path", model, save=False)
        if table is not None:
            cfg.set("cleanup.model_reasoning", table, save=False)
        w = SettingsWindow(cfg, lambda: [], lambda: False, lambda v: None, get_last_ctx=lambda: ("", ""),
                           get_default_mic=lambda: "")
        w.reload()
        return cfg, w

    def test_toggle_and_budget(self):
        cfg, w = self.make()
        self.assertFalse(w.reason_sw.isChecked())
        self.assertEqual(w.reason_spin.value(), 256)
        seen = []
        cfg.on_change(lambda old, new: seen.append(copy.deepcopy(new["cleanup"]["model_reasoning"])))
        w.reason_spin.setValue(512)
        w.reason_spin.editingFinished.emit()      # off: saved (the app only reloads when the budget in use changes)
        self.assertEqual(seen, [{"LFM3-1.7B-Q8_0.gguf": {"enabled": False, "budget": 512}}])
        w.reload()
        self.assertEqual(w.reason_spin.value(), 512)
        self.assertFalse(w.reason_sw.isChecked())
        w.reason_sw.setChecked(True)
        self.assertEqual(cfg.get("cleanup.model_reasoning"), {"LFM3-1.7B-Q8_0.gguf": {"enabled": True, "budget": 512}})
        w.reason_spin.setValue(1024)
        w.reason_spin.editingFinished.emit()
        self.assertEqual(cfg.get("cleanup.model_reasoning")["LFM3-1.7B-Q8_0.gguf"]["budget"], 1024)
        n = len(seen)
        w.reason_spin.editingFinished.emit()      # unchanged value: no second write / reload
        self.assertEqual(len(seen), n)
        w.reason_sw.setChecked(False)
        self.assertEqual(cfg.get("cleanup.model_reasoning"), {"LFM3-1.7B-Q8_0.gguf": {"enabled": False, "budget": 1024}})

    def test_bad_budget_shown_as_default(self):
        for bad in (float("inf"), float("nan"), "x", None):
            cfg, w = self.make(table={"LFM3-1.7B-Q8_0.gguf": {"enabled": True, "budget": bad}})
            self.assertEqual(w.reason_spin.value(), 256)
            self.assertTrue(w.reason_sw.isChecked())

    def test_existing_key_case_kept_and_shown(self):
        cfg, w = self.make(table={"lfm3-1.7b-q8_0.GGUF": {"enabled": True, "budget": 8192, "x": 1},
                                  "Other.gguf": {"enabled": True}})
        self.assertTrue(w.reason_sw.isChecked())
        self.assertEqual(w.reason_spin.value(), 4096)          # shown clamped ...
        w.reason_spin.editingFinished.emit()
        self.assertEqual(cfg.get("cleanup.model_reasoning")["lfm3-1.7b-q8_0.GGUF"]["budget"], 8192)   # ... not written
        w.reason_sw.setChecked(False)
        self.assertEqual(cfg.get("cleanup.model_reasoning"), {
            "lfm3-1.7b-q8_0.GGUF": {"enabled": False, "budget": 4096, "x": 1}, "Other.gguf": {"enabled": True}})

    def test_follows_model_and_save_never_writes_it(self):
        cfg, w = self.make(table={"Qwen3.5-4B-Q8_0.gguf": {"enabled": True, "budget": 640}})
        self.assertFalse(w.reason_sw.isChecked())
        cfg.set("cleanup.model_path", QWEN, save=False)         # tray switch while the window is open
        w.reason_sw.setChecked(True)                            # stale controls: refreshed, not written
        self.assertEqual(cfg.get("cleanup.model_reasoning"), {"Qwen3.5-4B-Q8_0.gguf": {"enabled": True, "budget": 640}})
        self.assertTrue(w.reason_sw.isChecked())
        self.assertEqual(w.reason_spin.value(), 640)
        cfg.set("cleanup.model_path", LFM3, save=False)
        w.refresh_cleanup_model()
        self.assertFalse(w.reason_sw.isChecked())
        cfg.set("cleanup.model_reasoning", {"LFM3-1.7B-Q8_0.gguf": {"enabled": True, "budget": 96}}, save=False)
        self.assertTrue(w.save())
        self.assertEqual(cfg.get("cleanup.model_reasoning"), {"LFM3-1.7B-Q8_0.gguf": {"enabled": True, "budget": 96}})


class AppRestartTrigger(unittest.TestCase):
    def test_reasoning_budget_in(self):
        from localflow.app import _reasoning_budget_in
        base = copy.deepcopy(DEFAULTS)
        base["cleanup"]["model_path"] = LFM3
        self.assertEqual(_reasoning_budget_in(base), 0)
        off = copy.deepcopy(base)
        off["cleanup"]["model_reasoning"] = {"LFM3-1.7B-Q8_0.gguf": {"enabled": False, "budget": 512}}
        self.assertEqual(_reasoning_budget_in(off), 0)           # budget edited while off: no reload
        other = copy.deepcopy(base)
        other["cleanup"]["model_reasoning"] = {"Qwen3.5-4B-Q8_0.gguf": {"enabled": True, "budget": 512}}
        self.assertEqual(_reasoning_budget_in(other), 0)         # another model's entry: no reload
        on = copy.deepcopy(base)
        on["cleanup"]["model_reasoning"] = {"lfm3-1.7b-q8_0.gguf": {"enabled": True, "budget": 512}}
        self.assertEqual(_reasoning_budget_in(on), 512)
        self.assertEqual(_reasoning_budget_in({}), 0)


if __name__ == "__main__":
    unittest.main()
