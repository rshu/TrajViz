"""Regression tests for the 2026-07-06 review fixes — LLM layer.

Pins:
  1.2  arbiter outcome-blinding (no 'failed attempt' phrasing in prompts)
  1.3  OpenRouter body-level errors raise LLMError instead of yielding content=""
  1.13 hierarchical panel vote (covered in test_grounding) + derived confidence
  1.14 fabricated evidence-step citations are filtered and lose the evidence gate
  1.15 arbiter cache is claim-keyed (no thrash), legacy unkeyed caches still read

Run:  python3 -m unittest tests.test_review_fixes_llm -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


import agent_trajviz.awe._engine.llm as llm
import agent_trajviz.awe._engine.arbiter as arbiter
from agent_trajviz.awe._engine.judge import parse_verdict
from agent_trajviz.awe._engine.arbiter import parse_verify, SYSTEM_PROMPT as ARBITER_PROMPT, _render_user
from agent_trajviz.awe._engine.scout import parse_modes


class TestBodyErrors(unittest.TestCase):
    """Review 1.3: HTTP-200 error bodies must fail loudly, never parse as 'none'."""

    def _chat_with(self, resp):
        orig_post, orig_key = llm._post, llm.config.OPENROUTER_API_KEY
        llm._post = lambda *a, **k: resp
        llm.config.OPENROUTER_API_KEY = "sk-test"
        try:
            return llm.chat([{"role": "user", "content": "x"}], retries=1)
        finally:
            llm._post, llm.config.OPENROUTER_API_KEY = orig_post, orig_key

    def test_error_body_raises(self):
        with self.assertRaises(llm.LLMError):
            self._chat_with({"error": {"code": 403, "message": "moderation"}})

    def test_missing_choices_raises(self):
        with self.assertRaises(llm.LLMError):
            self._chat_with({"usage": {}})

    def test_normal_response_unaffected(self):
        r = self._chat_with({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
        self.assertEqual(r["content"], "hi")


class TestEvidenceStepValidation(unittest.TestCase):
    """Review 1.14: citations must point at steps rendered into the prompt."""

    def test_judge_fabricated_steps_lose_the_gate(self):
        content = json.dumps(
            {
                "requirement_understanding": {
                    "error_type": "missed_constraint",
                    "confidence": "high",
                    "evidence_steps": [9999, -1],
                    "rationale": "x",
                },
                "task_planning": {"error_type": "none", "confidence": "high", "evidence_steps": [], "rationale": ""},
            }
        )
        v = parse_verdict(content, valid_steps={1, 2, 3})
        ru = v["requirement_understanding"]
        self.assertEqual(ru["evidence_steps"], [])
        self.assertEqual(ru["confidence"], "low")  # evidence gate applied

    def test_judge_valid_steps_survive(self):
        content = json.dumps(
            {
                "requirement_understanding": {
                    "error_type": "missed_constraint",
                    "confidence": "high",
                    "evidence_steps": [2, 9999],
                    "rationale": "x",
                },
                "task_planning": {"error_type": "none", "confidence": "high", "evidence_steps": [], "rationale": ""},
            }
        )
        ru = parse_verdict(content, valid_steps={1, 2, 3})["requirement_understanding"]
        self.assertEqual(ru["evidence_steps"], [2])
        self.assertEqual(ru["confidence"], "high")

    def test_arbiter_fabricated_steps_cannot_demote(self):
        content = json.dumps({"support": "refute", "confidence": "high", "evidence_steps": [777], "rationale": "x"})
        v = parse_verify(content, valid_steps={1, 2})
        self.assertEqual(v["evidence_steps"], [])
        self.assertEqual(v["confidence"], "low")  # cannot pass the demote gate

    def test_scout_fabricated_steps_filtered(self):
        content = json.dumps(
            {
                "modes": [
                    {"label": "x-loop", "description": "d", "covered_by": None, "novel": True, "evidence_steps": [50]}
                ]
            }
        )
        modes = parse_modes(content, valid_steps={1, 2})
        self.assertEqual(modes[0]["evidence_steps"], [])


class TestArbiterBlinding(unittest.TestCase):
    """Review 1.2: the arbiter must not be told the attempt failed."""

    def test_prompts_carry_no_failure_leak(self):
        self.assertNotIn("failed attempt", ARBITER_PROMPT.lower())
        self.assertNotIn("failed bug-fix", ARBITER_PROMPT.lower())
        evidence = {
            "problem_statement": "issue",
            "rows": [{"step": 0, "kind": "says", "text": "hello"}],
            "hidden_reasoning": False,
        }
        user = _render_user(evidence, "code_editing", "relevant_change_omitted").lower()
        self.assertNotIn("failed attempt", user)
        self.assertNotIn("failed", user.split("claimed primary cause")[-1])


class TestClaimKeyedCache(unittest.TestCase):
    """Review 1.15: two different claims for one instance must not thrash."""

    @patch.dict("os.environ", {"AWE_STRICT_VERDICT_PROVENANCE": "0"})
    def test_two_claims_coexist_and_legacy_read_works(self):
        with tempfile.TemporaryDirectory() as td:
            orig = arbiter.ARBITER_CACHE_DIR
            arbiter.ARBITER_CACHE_DIR = Path(td)
            try:
                for cap, et in (
                    ("code_editing", "relevant_change_omitted"),
                    ("code_verification", "relevant_test_not_run"),
                ):
                    p = arbiter.verdict_path("a", "i", "m", cap, et)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps({"claim": arbiter._claim_key(cap, et), "verdict": {"support": "confirm"}}))
                r1 = arbiter.verdict_cached("a", "i", "m", "code_editing", "relevant_change_omitted")
                r2 = arbiter.verdict_cached("a", "i", "m", "code_verification", "relevant_test_not_run")
                self.assertIsNotNone(r1)
                self.assertIsNotNone(r2)
                self.assertNotEqual(r1["claim"], r2["claim"])
                # Legacy unkeyed record is still readable for its own claim only.
                legacy = arbiter.verdict_path("a", "j", "m")
                legacy.write_text(
                    json.dumps(
                        {"claim": arbiter._claim_key("task_planning", "flawed_plan"), "verdict": {"support": "refute"}}
                    )
                )
                self.assertIsNotNone(arbiter.verdict_cached("a", "j", "m", "task_planning", "flawed_plan"))
                self.assertIsNone(arbiter.verdict_cached("a", "j", "m", "task_planning", "no_decomposition"))
            finally:
                arbiter.ARBITER_CACHE_DIR = orig


if __name__ == "__main__":
    unittest.main()
