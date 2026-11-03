"""v0.42 (DESIGN §0az): `retry_without_change` is demoted from a blamed fault
to a cohort feature — its fact calibrates protective. The signal is still
computed, the detector never emits it, the catalog no longer covers it, and
the legacy auditor still re-derives persisted pre-v0.42 records.

Run:  python3 -m unittest tests.test_demote_retry -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine import countermeasures as cm, validate
from agent_trajviz.awe._engine.annotation import ERRORS_BY_CAPABILITY
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH
from agent_trajviz.awe._engine.cohort import features_of
from agent_trajviz.awe._engine.detect import (
    objective_candidates,
    produced_error_types,
    expected_evidence_signal,
    grounding_signals,
)
from agent_trajviz.awe._engine.digest import PROCESS_COLS
from agent_trajviz.awe._engine.trajsignals import compute

from .test_new_error_types_batch2 import TestRetryWithoutChange, _gold
from .test_pertest_tier import _gold as _gold_p, _patch, _tsig
from agent_trajviz.awe._engine.signals import localization_signals


class TestDemotion(unittest.TestCase):
    def test_not_produced_anywhere(self):
        self.assertNotIn("retry_without_change", produced_error_types())
        self.assertIsNone(expected_evidence_signal("retry_without_change"))
        self.assertNotIn("retry_without_change", grounding_signals())
        self.assertNotIn("retry_without_change", ERRORS_BY_CAPABILITY["self_repair_loop"])
        self.assertNotIn("retry_without_change", PROCESS_COLS)

    def test_signal_still_computed_and_a_cohort_feature(self):
        traj = TestRetryWithoutChange()._traj(edit_between=False)
        sig = compute(traj, _gold())
        self.assertTrue(sig.retry_without_change)
        self.assertEqual(features_of(sig)["retry_without_change"], 1.0)

    def test_detector_never_emits_it(self):
        g = _gold_p()
        loc = localization_signals(g, _patch(), False)
        t = _tsig(retry_without_change=True, encountered_failure=True, n_failure_events=2, repair_not_reverified=True)
        ets = {c.error_type for c in objective_candidates(g, loc, t, False)}
        self.assertNotIn("retry_without_change", ets)
        self.assertIn("repair_not_reverified", ets)  # the sibling still fires

    def test_catalog_no_longer_covers_it_and_stays_valid(self):
        entries = cm.load_catalog()
        self.assertNotIn("retry_without_change", {e.error_type for e in entries})
        self.assertNotIn("CM-REPAIR-BLINDRETRY-1", {e.id for e in entries})
        r = validate.countermeasure_catalog()
        self.assertTrue(r["checks"]["referential_integrity"], r)
        self.assertTrue(r["checks"]["coverage"], r)
        self.assertTrue(r["checks"]["bundles"], r)

    def test_legacy_records_still_audit(self):
        fault = {
            "capability": "self_repair_loop",
            "error_type": "retry_without_change",
            "confidence": "medium",
            "detector_evidence": {"retry_without_change": True},
        }
        honest = TestRetryWithoutChange()._traj(edit_between=False)
        forged = TestRetryWithoutChange()._traj(edit_between=True)
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=honest)
        self.assertEqual(a["verdict"], RE_DERIVED)
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=forged)
        self.assertEqual(a["verdict"], MISMATCH)


if __name__ == "__main__":
    unittest.main()
