"""Tests for the v0.38 documented-outcome-override layer (agent_trajviz.awe._engine/outcomes.py):
the §0ac/§0ao "documented re-label, never a silent edit" contract made
mechanism.

Run:  python3 -m unittest tests.test_resolved_overrides -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from agent_trajviz.awe._engine import config, outcomes


def _with_overrides(doc):
    """Point PROJECT_DATA at a temp tree holding the given override doc."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "overrides").mkdir()
    (tmp / "overrides" / "resolved_overrides.json").write_text(json.dumps(doc))
    return tmp


class TestOverrides(unittest.TestCase):
    def setUp(self):
        self._data = config.PROJECT_DATA
        outcomes.load_overrides.cache_clear()

    def tearDown(self):
        config.PROJECT_DATA = self._data
        outcomes.load_overrides.cache_clear()

    def _entry(self, **kw):
        e = {"agent": "a", "instance_id": "i", "resolved": True, "reason": "r", "evidence": "e"}
        e.update(kw)
        return e

    def test_true_false_and_exclusion(self):
        config.PROJECT_DATA = _with_overrides(
            {
                "overrides": [
                    self._entry(instance_id="i1", resolved=True),
                    self._entry(instance_id="i2", resolved=False),
                    self._entry(instance_id="i3", resolved=None),
                ]
            }
        )
        self.assertIs(outcomes.resolved("a", "i1"), True)
        self.assertIs(outcomes.resolved("a", "i2"), False)
        self.assertIsNone(outcomes.resolved("a", "i3"))  # excluded
        self.assertIsNotNone(outcomes.override_for("a", "i3"))
        self.assertIsNone(outcomes.override_for("a", "other"))

    def test_malformed_entry_fails_loudly(self):
        config.PROJECT_DATA = _with_overrides(
            {"overrides": [{"agent": "a", "instance_id": "i", "resolved": True}]}
        )  # no reason/evidence
        with self.assertRaises(ValueError):
            outcomes.load_overrides()

    def test_non_bool_resolved_fails_loudly(self):
        config.PROJECT_DATA = _with_overrides({"overrides": [self._entry(resolved="yes")]})
        with self.assertRaises(ValueError):
            outcomes.load_overrides()

    def test_absent_file_is_empty(self):
        config.PROJECT_DATA = Path(tempfile.mkdtemp())
        self.assertEqual(outcomes.load_overrides(), {})


if __name__ == "__main__":
    unittest.main()
