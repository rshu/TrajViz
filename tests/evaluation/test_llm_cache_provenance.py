import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from agent_trajviz.awe._engine import config, judge, arbiter
from agent_trajviz.awe._engine.provenance import validate_verdict


class TestLLMCacheProvenance(unittest.TestCase):
    def test_legacy_is_explicit_and_strict_mode_abstains(self):
        with warnings.catch_warnings(), patch.dict("os.environ", {"AWE_STRICT_VERDICT_PROVENANCE": "0"}):
            warnings.simplefilter("ignore")
            self.assertEqual(
                validate_verdict({}, lambda: "new", "test")["cache_provenance"]["status"], "legacy_unverified"
            )
        with warnings.catch_warnings(), patch.dict("os.environ", {"AWE_STRICT_VERDICT_PROVENANCE": "1"}):
            warnings.simplefilter("ignore")
            self.assertIsNone(validate_verdict({}, lambda: "new", "test"))

    def test_new_judge_and_arbiter_entries_track_raw_prompt_and_claim(self):
        with tempfile.TemporaryDirectory() as td, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root = Path(td)
            with (
                patch.object(config, "TRAJECTORY_DIR", root / "trajectory"),
                patch.object(config, "REQUIREMENTS_DIR", root / "requirements"),
                patch.object(config, "JUDGE_CACHE_DIR", root / "judge"),
                patch.object(arbiter, "ARBITER_CACHE_DIR", root / "arbiter"),
            ):
                raw = root / "trajectory/a/i.json"
                raw.parent.mkdir(parents=True)
                raw.write_text("original")
                req = root / "requirements/i.json"
                req.parent.mkdir()
                req.write_text(json.dumps({"problem_statement": "issue", "gold_patch": "secret"}))
                key = judge._verdict_key("a", "i", "m")
                jp = judge.verdict_path("a", "i", "m")
                jp.parent.mkdir(parents=True)
                jp.write_text(json.dumps({"input_fingerprint": key, "verdict": {}}))
                self.assertEqual(judge.verdict_cached("a", "i", "m")["cache_provenance"]["status"], "verified")
                req.write_text(json.dumps({"problem_statement": "issue", "gold_patch": "different secret"}))
                self.assertEqual(judge._verdict_key("a", "i", "m"), key)
                with patch.object(judge, "SYSTEM_PROMPT", "new prompt"):
                    self.assertIsNone(judge.verdict_cached("a", "i", "m"))
                cap, et = "code_editing", "relevant_change_omitted"
                ak = arbiter._verdict_key("a", "i", "m", cap, et)
                ap = arbiter.verdict_path("a", "i", "m", cap, et)
                ap.parent.mkdir(parents=True)
                ap.write_text(json.dumps({"input_fingerprint": ak, "claim": arbiter._claim_key(cap, et)}))
                self.assertIsNotNone(arbiter.verdict_cached("a", "i", "m", cap, et))
                legacy_path = arbiter.verdict_path("a", "i", "m")
                legacy_path.write_text(ap.read_text())
                self.assertIsNotNone(arbiter.verdict_cached("a", "i", "m"))
                raw.write_text("changed")
                self.assertIsNone(judge.verdict_cached("a", "i", "m"))
                self.assertIsNone(arbiter.verdict_cached("a", "i", "m", cap, et))
                raw.write_text("original")
                req.write_text(json.dumps({"problem_statement": "changed issue"}))
                self.assertIsNone(judge.verdict_cached("a", "i", "m"))
