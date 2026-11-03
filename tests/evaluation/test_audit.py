# tests/test_audit.py
import unittest
from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.trajectory import NormalizedTrajectory, NormalizedStep, RUN_TEST, RUN_COMMAND, EDIT


def _step(i, **kw):
    return NormalizedStep(step_id=i, turn_id=i, **kw)


GOLD = (
    "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n"
    "@@ -1,2 +1,2 @@\n-old_line\n+new_line\n"
    "diff --git a/pkg/util.py b/pkg/util.py\n--- a/pkg/util.py\n+++ b/pkg/util.py\n"
    "@@ -1 +1 @@\n-a\n+b\n"
)
# agent edits the right files but reproduces only the core.py change, not util.py
AGENT_HIT = (
    "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n"
    "@@ -1,2 +1,2 @@\n-old_line\n+new_line\n"
    "diff --git a/pkg/util.py b/pkg/util.py\n--- a/pkg/util.py\n+++ b/pkg/util.py\n"
    "@@ -1 +1 @@\n-a\n+totally_different\n"
)


def _fault(**ev):
    return {"capability": "code_editing", "error_type": "incorrect_patch", "detector_evidence": ev}


class TestDeductiveAudit(unittest.TestCase):
    def test_incorrect_patch_re_derives(self):
        # core.py fully reproduced (2/2); util.py only the removal matches (1/2) -> 3/4 = 0.75
        f = _fault(hit=["pkg/core.py", "pkg/util.py"], reproduced_fraction=0.75)
        r = audit.audit_fault(f, agent="a", inst="i", gold_text=GOLD, agent_text=AGENT_HIT, resolved=False, traj=None)
        self.assertEqual(r["tier"], "deductive")
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])

    def test_tampered_atom_is_caught(self):  # ANTI-VACUOUS centerpiece
        f = _fault(hit=["pkg/core.py", "pkg/util.py"], reproduced_fraction=0.99)  # lie
        r = audit.audit_fault(f, agent="a", inst="i", gold_text=GOLD, agent_text=AGENT_HIT, resolved=False, traj=None)
        self.assertEqual(r["verdict"], "mismatch")
        self.assertTrue(any(m["atom"] == "reproduced_fraction" for m in r["mismatches"]))

    def test_outcome_must_be_failed(self):
        f = _fault(hit=["pkg/core.py", "pkg/util.py"], reproduced_fraction=0.5)
        r = audit.audit_fault(
            f, agent="a", inst="i", gold_text=GOLD, agent_text=AGENT_HIT, resolved=True, traj=None
        )  # resolved!
        self.assertEqual(r["verdict"], "mismatch")
        self.assertTrue(any(m["atom"] == "outcome" for m in r["mismatches"]))

    def test_missing_trajectory_cannot_certify_search_failure(self):
        # Reference divergence is known; search reachability is unknown.
        agent_partial = (
            "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n"
            "+++ b/pkg/core.py\n@@ -1,2 +1,2 @@\n-old_line\n+new_line\n"
        )
        f = {
            "capability": "code_localization",
            "error_type": "missed_relevant_file",
            "detector_evidence": {"missed_gold": ["pkg/util.py"], "recall": 0.5},
        }
        r = audit.audit_fault(
            f, agent="a", inst="i", gold_text=GOLD, agent_text=agent_partial, resolved=False, traj=None
        )
        self.assertEqual(r["verdict"], "unverifiable", r["mismatches"])

    def test_tamper_by_relabel_within_deductive_family_is_not_clean(self):
        """v0.31: `recall` is capability-generic — relabeling a loc fault to a
        different deductive type must NOT audit re_derived on outcome+recall
        alone (the type-specific branch finds none of its keys)."""
        agent_partial = (
            "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n"
            "+++ b/pkg/core.py\n@@ -1,2 +1,2 @@\n-old_line\n+new_line\n"
        )
        for relabel in ("incorrect_patch", "wrong_file_localization"):
            f = {
                "capability": "code_localization",
                "error_type": relabel,
                "detector_evidence": {"missed_gold": ["pkg/util.py"], "recall": 0.5},
            }
            r = audit.audit_fault(
                f, agent="a", inst="i", gold_text=GOLD, agent_text=agent_partial, resolved=False, traj=None
            )
            self.assertEqual(r["verdict"], "unverifiable", relabel)


class TestAssociationalAudit(unittest.TestCase):
    def test_ran_no_test_re_derives(self):
        traj = NormalizedTrajectory("a", "i", [_step(0, actor="agent", type="message", text="done")])
        f = {
            "capability": "code_verification",
            "error_type": "relevant_test_not_run",
            "detector_evidence": {"ran_any_test": False},
        }
        r = audit.audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj)
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])

    def test_ran_test_contradicts_claim(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [_step(0, actor="agent", type="tool_call", canonical_action=RUN_TEST, native_tool="Bash", test_run=True)],
        )
        f = {
            "capability": "code_verification",
            "error_type": "relevant_test_not_run",
            "detector_evidence": {"ran_any_test": False},
        }  # lie: a test WAS run
        r = audit.audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj)
        self.assertEqual(r["verdict"], "mismatch")

    def test_no_trajectory_is_unverifiable(self):
        f = {
            "capability": "self_repair_loop",
            "error_type": "repair_not_reverified",
            "detector_evidence": {"edited_after_failure_without_retest": True},
        }
        r = audit.audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=None)
        self.assertEqual(r["verdict"], "unverifiable")

    # --- v0.31: defining atoms of the verification faults are re-derived ---

    def _traj_ran_gating(self):
        return NormalizedTrajectory(
            "a",
            "i",
            [
                _step(
                    0,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    native_tool="Bash",
                    test_run=True,
                    call_id="c1",
                    args={"command": "pytest tests/test_core.py::test_gating"},
                ),
                _step(1, actor="tool", type="tool_result", call_id="c1", output_excerpt="1 passed"),
            ],
        )

    def test_fail_to_pass_referenced_lie_is_caught(self):
        """Fault claims the gating test was never referenced; the trajectory
        names the gating test file — the defining atom must mismatch."""
        f = {
            "capability": "code_verification",
            "error_type": "relevant_test_not_run",
            "detector_evidence": {"ran_any_test": True, "fail_to_pass_referenced": False},
        }
        r = audit.audit_fault(
            f,
            agent="a",
            inst="i",
            gold_text="",
            agent_text="",
            resolved=False,
            traj=self._traj_ran_gating(),
            test_files=["tests/test_core.py"],
            fail_to_pass=["tests/test_core.py::test_gating"],
        )
        self.assertEqual(r["verdict"], "mismatch")
        self.assertTrue(any(m["atom"] == "fail_to_pass_referenced" for m in r["mismatches"]))

    def test_fail_to_pass_referenced_without_inputs_is_unverifiable(self):
        """No test_files/fail_to_pass inputs -> the DEFINING atom cannot be
        re-derived; honest downgrade instead of re_derived on generic atoms."""
        f = {
            "capability": "code_verification",
            "error_type": "relevant_test_not_run",
            "detector_evidence": {"ran_any_test": True, "fail_to_pass_referenced": False},
        }
        r = audit.audit_fault(
            f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj_ran_gating()
        )
        self.assertEqual(r["verdict"], "unverifiable")

    def test_last_test_failed_lie_is_caught(self):
        """test_result_misinterpreted asserts the last test result FAILED; a
        trajectory whose last test passed must mismatch on that atom."""
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _step(
                    0,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    native_tool="Bash",
                    test_run=True,
                    call_id="c1",
                    args={"command": "pytest"},
                ),
                _step(1, actor="tool", type="tool_result", call_id="c1", output_excerpt="5 passed in 0.2s"),
                _step(2, actor="agent", type="message", text="All tests pass — the fix is complete."),
            ],
        )
        f = {
            "capability": "code_verification",
            "error_type": "test_result_misinterpreted",
            "detector_evidence": {"claimed_success_final": True, "last_test_failed": True},
        }
        r = audit.audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj)
        self.assertEqual(r["verdict"], "mismatch")
        self.assertTrue(any(m["atom"] == "last_test_failed" for m in r["mismatches"]))

    def test_last_test_failed_true_re_derives(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _step(
                    0,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    native_tool="Bash",
                    test_run=True,
                    call_id="c1",
                    args={"command": "pytest"},
                ),
                _step(
                    1,
                    actor="tool",
                    type="tool_result",
                    call_id="c1",
                    output_excerpt="FAILED tests/test_core.py::test_x - AssertionError",
                ),
                _step(2, actor="agent", type="message", text="All tests pass — the fix is complete."),
            ],
        )
        f = {
            "capability": "code_verification",
            "error_type": "test_result_misinterpreted",
            "detector_evidence": {"claimed_success_final": True, "last_test_failed": True},
        }
        r = audit.audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj)
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])


class TestSelfRepairToolAudit(unittest.TestCase):
    """Self-repair + tool-mechanic faults must be RE-DERIVED from the trajectory's
    failure→edit→retest / tool-error→no-retry ordering — no longer blanket-unverifiable
    — and a fault asserted on a trajectory that doesn't exhibit the condition must be
    caught as a mismatch (the non-vacuity guarantee for this path)."""

    def _fail_then_edit(self, retest):
        steps = [
            _step(
                0,
                actor="agent",
                type="tool_call",
                canonical_action=RUN_TEST,
                native_tool="Bash",
                test_run=True,
                call_id="t1",
            ),
            _step(1, actor="tool", type="tool_result", tool_error=True, output_excerpt="1 failed", call_id="t1"),
            _step(
                2,
                actor="agent",
                type="tool_call",
                canonical_action=EDIT,
                native_tool="Edit",
                files=["pkg/a.py"],
                call_id="e1",
            ),
            # the edit COMPLETED (result arrived) — a run ending on a dangling
            # tool_call would be truncated (killed ≠ gave up) and not flagged
            _step(3, actor="tool", type="tool_result", output_excerpt="ok", call_id="e1"),
        ]
        if retest:
            steps += [
                _step(
                    4,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    native_tool="Bash",
                    test_run=True,
                    call_id="t2",
                ),
                _step(5, actor="tool", type="tool_result", output_excerpt="ok", call_id="t2"),
            ]
        return NormalizedTrajectory("a", "i", steps)

    def _tool_error(self, retry):
        steps = [
            _step(0, actor="agent", type="tool_call", canonical_action=RUN_COMMAND, native_tool="Bash", call_id="c1"),
            _step(
                1, actor="tool", type="tool_result", output_excerpt="cat: x.py: No such file or directory", call_id="c1"
            ),
        ]
        if retry:
            steps += [
                _step(
                    2, actor="agent", type="tool_call", canonical_action=RUN_COMMAND, native_tool="Bash", call_id="c2"
                ),
                _step(3, actor="tool", type="tool_result", output_excerpt="ok", call_id="c2"),
            ]
        return NormalizedTrajectory("a", "i", steps)

    def _audit(self, et, cap, traj, **ev):
        f = {"capability": cap, "error_type": et, "detector_evidence": ev}
        return audit.audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj)

    def test_repair_not_reverified_re_derives(self):
        r = self._audit(
            "repair_not_reverified",
            "self_repair_loop",
            self._fail_then_edit(retest=False),
            edited_after_failure_without_retest=True,
        )
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])

    def test_repair_reverified_is_caught_as_mismatch(self):
        # The agent DID re-run the test after editing -> the fault does not hold.
        r = self._audit(
            "repair_not_reverified",
            "self_repair_loop",
            self._fail_then_edit(retest=True),
            edited_after_failure_without_retest=True,
        )
        self.assertEqual(r["verdict"], "mismatch")

    def test_tool_failure_not_retried_re_derives_flagless(self):
        # Codex-shaped: error only in the result text, no retry -> re-derived.
        r = self._audit(
            "tool_failure_not_retried", "tool_use", self._tool_error(retry=False), tool_error_without_followup=True
        )
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])

    def test_tool_failure_with_retry_is_mismatch(self):
        # The agent issued a follow-up tool call -> not 'gave up' -> mismatch.
        r = self._audit(
            "tool_failure_not_retried", "tool_use", self._tool_error(retry=True), tool_error_without_followup=True
        )
        self.assertEqual(r["verdict"], "mismatch")

    def test_repeated_ineffective_n_failure_events_cross_check(self):
        # Same file edited twice after a failure -> condition holds; but the evidence
        # lies about the failure-event count (1, actual 1... tamper to 9) -> mismatch.
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _step(
                    0,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    native_tool="Bash",
                    test_run=True,
                    call_id="t1",
                ),
                _step(1, actor="tool", type="tool_result", tool_error=True, output_excerpt="1 failed", call_id="t1"),
                _step(
                    2,
                    actor="agent",
                    type="tool_call",
                    canonical_action=EDIT,
                    native_tool="Edit",
                    files=["pkg/a.py"],
                    call_id="e1",
                ),
                _step(
                    3,
                    actor="agent",
                    type="tool_call",
                    canonical_action=EDIT,
                    native_tool="Edit",
                    files=["pkg/a.py"],
                    call_id="e2",
                ),
            ],
        )
        ok = self._audit(
            "repeated_ineffective_attempt",
            "self_repair_loop",
            traj,
            same_file_re_edited_after_failure=True,
            n_failure_events=1,
        )
        self.assertEqual(ok["verdict"], "re_derived", ok["mismatches"])
        bad = self._audit(
            "repeated_ineffective_attempt",
            "self_repair_loop",
            traj,
            same_file_re_edited_after_failure=True,
            n_failure_events=9,
        )
        self.assertEqual(bad["verdict"], "mismatch")


class TestModelInferredAudit(unittest.TestCase):
    def _traj(self, n=5):
        return NormalizedTrajectory("a", "i", [_step(i, actor="agent", type="message") for i in range(n)])

    def test_valid_structural_ok(self):
        f = {
            "capability": "requirement_understanding",
            "error_type": "misread_requirement",
            "confidence": "high",
            "detector_evidence": {"evidence_steps": [2]},
        }
        r = audit.audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj())
        self.assertEqual(r["verdict"], "structural_ok")
        self.assertNotEqual(r["verdict"], "re_derived")  # never re_derived

    def test_out_of_range_step_mismatches(self):
        f = {
            "capability": "requirement_understanding",
            "error_type": "misread_requirement",
            "confidence": "high",
            "detector_evidence": {"evidence_steps": [99]},
        }
        r = audit.audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj())
        self.assertEqual(r["verdict"], "mismatch")


class TestAuditCaseCounts(unittest.TestCase):
    def test_verdict_counts_aggregate(self):
        # audit_case delegates to audit_fault; verify the counts roll-up given a
        # hand-built persisted record + injected raw texts via the _inputs hook.
        faults = [
            {
                "capability": "code_editing",
                "error_type": "incorrect_patch",
                "detector_evidence": {"hit": ["pkg/core.py"], "reproduced_fraction": 0.5},
            },
        ]
        rec = {"case_id": "a__i", "agent": "a", "instance_id": "i", "faults": faults}
        gold = "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1 +1 @@\n-x\n+y\n"
        agent_t = (
            "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n"
            "+++ b/pkg/core.py\n@@ -1 +1 @@\n-x\n+z\n"
        )  # removal shared, addition missed -> 0.5
        out = audit.audit_case("a", "i", persisted=rec, _inputs=(gold, agent_t, False, None))
        self.assertEqual(out["verdict_counts"]["re_derived"], 1)
        self.assertEqual(out["verdict_counts"]["mismatch"], 0)


class TestIndependence(unittest.TestCase):
    def test_audit_does_not_import_detector(self):
        # The auditor must re-derive independently — never re-run the detector.
        # Guards the feature's central non-vacuousness invariant against regression.
        import inspect
        from agent_trajviz.awe._engine import audit as _audit

        src = inspect.getsource(_audit)
        for banned in (
            "agent_trajviz.awe._engine.signals",
            "agent_trajviz.awe._engine.detect",
            "agent_trajviz.awe._engine.rootcause",
            "agent_trajviz.awe._engine.trajsignals",
            "from .signals",
            "from .detect",
            "from .rootcause",
            "from .trajsignals",
        ):
            self.assertNotIn(banned, src, f"audit.py must not depend on {banned}")


if __name__ == "__main__":
    unittest.main()


class TestGatingTokenSpecificity(unittest.TestCase):
    def test_short_stems_are_filtered_like_the_detector(self):
        """v0.32: the audit's token builder applies the detector's len>=5 bar —
        a gold test file named test.py must not contribute the stem 'test',
        which would substring-match every test command and mint false
        mismatches."""
        toks = audit._gating_test_tokens(["tests/test.py"], ["tests/test.py::test_x"])
        self.assertNotIn("test", toks)
        self.assertIn("tests/test.py", toks)  # full path is >= 5 chars
        toks2 = audit._gating_test_tokens(["tests/test_core.py"], [])
        self.assertIn("test_core", toks2)
