import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_trajviz.awe._engine import adapters, config, outcomes, trajsignals
from agent_trajviz.awe._engine.provenance import fingerprint
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory, RUN_TEST
from .test_batch8_signals import _gold


class TestCacheProvenance(unittest.TestCase):
    def test_tuple_keyed_line_multisets_are_stable(self):
        self.assertEqual(fingerprint({(0, "x"): 1, (4, "y"): 2}), fingerprint({(4, "y"): 2, (0, "x"): 1}))
        self.assertNotEqual(fingerprint({(0, "x"): 1}), fingerprint({(0, "x"): 2}))

    def test_normalized_cache_tracks_raw_content_and_deletion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "trajectory/a/i.json"
            raw.parent.mkdir(parents=True)

            def write(name):
                raw.write_text(
                    json.dumps(
                        {
                            "trajectory": [
                                {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "tool_call", "id": "r", "name": "Read", "input": {"file_path": name}}
                                    ],
                                }
                            ]
                        }
                    )
                )

            with (
                patch.object(config, "TRAJECTORY_DIR", root / "trajectory"),
                patch.object(adapters, "_NORMTRAJ_CACHE", root / "cache"),
            ):
                write("old.py")
                self.assertEqual(adapters.load_normalized("a", "i").steps[0].files, ["old.py"])
                write("new.py")
                self.assertEqual(adapters.load_normalized("a", "i").steps[0].files, ["new.py"])
                raw.unlink()
                self.assertIsNone(adapters.load_normalized("a", "i"))

    def test_signal_cache_tracks_gold_inputs(self):
        t = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    0,
                    0,
                    "agent",
                    "tool_call",
                    canonical_action=RUN_TEST,
                    test_run=True,
                    args={"command": "pytest tests/test_core.py"},
                )
            ],
        )
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(trajsignals, "_CACHE", Path(td)),
            patch.object(adapters, "load_normalized", return_value=t),
        ):
            gold = _gold()
            self.assertTrue(trajsignals.signals_cached("a", "i", gold).fail_to_pass_referenced)
            new = replace(
                gold, test_patch_files=["tests/test_other.py"], fail_to_pass=["tests/test_other.py::test_other"]
            )
            self.assertFalse(trajsignals.signals_cached("a", "i", new).fail_to_pass_referenced)

    def test_outcomes_refresh_within_one_process(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "eval_a.json"
            with patch.object(config, "DATA", root):
                p.write_text('{"i": false}')
                self.assertIs(outcomes.resolved("a", "i"), False)
                p.write_text('{"i": true}')
                self.assertIs(outcomes.resolved("a", "i"), True)
