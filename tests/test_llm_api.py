"""The dashboard's LLM features through the public `agent_trajviz.llm` module.

`ask` is the AI sidebar and `suggest_fixes` the Issues "Suggest fixes"
button. They must send what the dashboard sends (its brief), go through the
redacted HTTP path, and turn a missing configuration into a message.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import requests

import agent_trajviz as tv
from agent_trajviz import llm

REPO = Path(__file__).resolve().parent.parent
CC_RUN = REPO / "tests" / "fixtures" / "native_claude_code" / "django__django-11477.jsonl"
CONFIG = llm.LLMConfig(base_url="https://llm.example.test/v1", api_key="KEY", model="m", provider="openai",
                       temperature=None, max_tokens=256, timeout=5, source="analyze")


def _fake_post(reply: str, sent: list):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": reply}}]}

    def post(url, json=None, headers=None, timeout=None, allow_redirects=True):
        assert allow_redirects is False
        sent.append(json)
        return _Resp()

    return post


class LlmApiTests(unittest.TestCase):
    def setUp(self):
        self.traj = tv.load(CC_RUN)

    def test_the_brief_is_the_dashboards(self):
        from agent_trajviz.insight.assistant import build_analysis_brief_from_session
        from agent_trajviz.insight.session import load_session

        self.assertEqual(llm.brief(self.traj), build_analysis_brief_from_session(load_session(str(CC_RUN))))

    def test_ask_sends_the_brief_and_appends_the_reply(self):
        sent: list = []
        with mock.patch.object(requests, "post", _fake_post("Step 12 waits on a slow test run.", sent)):
            history = llm.ask(self.traj, "Why is it slow?", config=CONFIG)
            history = llm.ask(self.traj, "And then?", history, config=CONFIG)
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "user", "assistant"])
        self.assertEqual(history[1]["content"], "Step 12 waits on a slow test run.")
        system = sent[0]["messages"][0]["content"]
        self.assertIn(llm.brief(self.traj)[:200], system)

    def test_an_unconfigured_endpoint_is_a_message_not_an_exception(self):
        missing = llm.LLMConfig(base_url="", api_key="", model="", provider="openai", temperature=None,
                                max_tokens=64, timeout=5, source="missing")
        history = llm.ask(self.traj, "Why?", config=missing)
        self.assertIn("not configured", history[-1]["content"])

    def test_suggest_fixes_attaches_a_judgment_to_each_issue(self):
        reply = json.dumps({"where": "tests/i18n", "fix": "Read files, not directories.", "confidence": "high"})
        sent: list = []
        with mock.patch.object(requests, "post", _fake_post(reply, sent)):
            issues, errors = llm.suggest_fixes(self.traj, config=CONFIG)
        self.assertEqual(errors, [])
        self.assertEqual(len(issues), len(self.traj.view("issues").data))
        self.assertEqual(issues[0]["judgment"]["fix"], "Read files, not directories.")
        json.dumps(issues)

    def test_any_client_can_be_plugged_in_and_still_sees_redacted_text(self):
        seen = []

        def my_client(system, messages):
            seen.append((system, messages))
            return "Use a narrower grep."

        unconfigured = llm.LLMConfig(base_url="", api_key="", model="", provider="openai", temperature=None,
                                     max_tokens=64, timeout=5, source="missing")
        secret = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
        history = llm.ask(self.traj, f"Is {secret} leaked?", config=unconfigured, chat=my_client)
        self.assertEqual(history[-1]["content"], "Use a narrower grep.")
        system, messages = seen[0]
        self.assertIn(llm.brief(self.traj)[:200], system)
        self.assertNotIn(secret, messages[-1]["content"])

    def test_language_reaches_the_prompt(self):
        seen = []
        llm.ask(self.traj, "Why?", chat=lambda system, messages: seen.append(system) or "ok", language="English")
        self.assertIn("English", seen[0])
        self.assertIn("English", llm.first_question("English") + " English")

    def test_suggest_fixes_with_a_plugged_in_client(self):
        reply = json.dumps({"where": "tests/", "fix": "Check the path is a file first.", "confidence": "low"})
        issues, errors = llm.suggest_fixes(self.traj, chat=lambda system, messages: reply, language="English")
        self.assertEqual(errors, [])
        self.assertEqual(issues[0]["judgment"]["confidence"], "low")

    def test_the_module_names_its_extra(self):
        code = textwrap.dedent(f"""
            import importlib.abc, sys
            class _Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] == "requests":
                        raise ModuleNotFoundError(name)
            sys.meta_path.insert(0, _Block())
            sys.path.insert(0, {str(REPO)!r})
            import agent_trajviz
            try:
                import agent_trajviz.llm
            except agent_trajviz.MissingExtraError as exc:
                print("MISSING", exc.extra)
        """)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.stdout.strip(), "MISSING llm", proc.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
