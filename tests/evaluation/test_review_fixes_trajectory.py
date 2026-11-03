"""Regression tests for the 2026-07-06 review fixes — trajectory layer.

Each test pins a specific defect from REVIEW-2026-07-06.md §1:
  1.1  '0 failed' / '5 passed, 0 failed' misread as test failures
  1.5  sub-agent steps appended after the main thread corrupting ordering signals
  1.6  substring test-command detection (pip install pytest, cat tox.ini, ...)
  1.7  redirect/heredoc write targets contaminating the read-set; sed -i as read
  1.8  environment_error / tool-error text scan running over READ+SEARCH output
  1.15 unversioned signal caches silently serving stale results

Run:  python3 -m unittest tests.test_review_fixes_trajectory -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from agent_trajviz.awe._engine.trajectory import (
    NormalizedStep,
    NormalizedTrajectory,
    command_read_targets,
    excerpt,
    is_test_command,
    looks_like_test_failure,
    repo_relative,
    EDIT,
    READ,
    RUN_COMMAND,
    RUN_TEST,
    SEARCH,
)
from agent_trajviz.awe._engine.trajsignals import compute, _repair_and_tool_signals, _run_failed
from agent_trajviz.awe._engine.gold import GoldReference


def _step(i, **kw):
    return NormalizedStep(step_id=i, turn_id=i, **kw)


def _call(i, cid, action=RUN_COMMAND, test=False, files=None, cmd="cmd", sub=None):
    return _step(
        i,
        actor="agent",
        type="tool_call",
        canonical_action=action,
        test_run=test,
        files=files or [],
        call_id=cid,
        args={"command": cmd},
        subagent_id=sub,
    )


def _result(i, cid, output="", error=False, sub=None, args=None):
    return _step(
        i,
        actor="tool",
        type="tool_result",
        output_excerpt=output,
        tool_error=error,
        call_id=cid,
        subagent_id=sub,
        args=args or {},
    )


def _gold():
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=["pkg/a.py"],
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=[],
        fail_to_pass=["pkg/tests/test_x.py::test_x"],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
    )


class TestFailureClassifier(unittest.TestCase):
    """Review 1.1: clean summaries must never double as failure markers."""

    def test_zero_failed_is_clean(self):
        self.assertFalse(looks_like_test_failure("5 passed, 0 failed"))
        self.assertFalse(looks_like_test_failure("Ran 12 tests. 0 failed"))
        self.assertFalse(looks_like_test_failure("0 errors"))
        self.assertFalse(looks_like_test_failure("no errors, all tests passed"))

    def test_real_failures_still_fire(self):
        self.assertTrue(looks_like_test_failure("2 failed, 3 passed"))
        self.assertTrue(looks_like_test_failure("FAILED pkg/tests/test_x.py::test_x"))
        self.assertTrue(looks_like_test_failure("Traceback (most recent call last):\n  ..."))
        self.assertTrue(looks_like_test_failure("the build failed"))

    def test_clean_summary_plus_real_failure_marker_fires(self):
        self.assertTrue(looks_like_test_failure("0 errors so far\nTraceback (most recent call last):"))


class TestTestCommandDetection(unittest.TestCase):
    """Review 1.6: runner must be the command head (or a -m target / test script),
    not a substring anywhere in the line."""

    def test_substring_false_positives_gone(self):
        for cmd in (
            "pip install pytest",
            "pip install tox",
            "cat tox.ini",
            "grep -rn unittest .",
            'git commit -m "fix pytest issue"',
            "echo unittest",
        ):
            self.assertFalse(is_test_command(cmd), cmd)

    def test_real_invocations_detected(self):
        for cmd in (
            "pytest tests/test_x.py",
            "python -m pytest -x",
            "python3 -m unittest discover",
            "tox -e py39",
            "cd repo && python -m pytest tests/",
            "python tests/runtests.py app.tests",
            "python test_foo.py",
            "make test",
            "./tests/runtests.py view_tests",
            "python setup.py test",
            "python manage.py test app",
        ):
            self.assertTrue(is_test_command(cmd), cmd)

    def test_plain_script_run_is_not_a_test(self):
        self.assertFalse(is_test_command("python repro.py"))
        self.assertFalse(is_test_command("python setup.py build"))


class TestReadTargets(unittest.TestCase):
    """Review 1.7: write targets / sibling-segment files are not reads."""

    def test_redirect_target_excluded(self):
        self.assertEqual(command_read_targets("sed -n 1,50p a.py > out.txt"), ["a.py"])

    def test_heredoc_write_excluded(self):
        self.assertEqual(command_read_targets("cat <<'EOF' > repro.py"), [])

    def test_sibling_segment_files_excluded(self):
        self.assertEqual(command_read_targets("cat a.py && python b.py"), ["a.py"])

    def test_sed_in_place_is_not_a_read(self):
        self.assertEqual(command_read_targets("sed -i 's/x/y/' file.py"), [])

    def test_pipeline_read_kept(self):
        self.assertEqual(command_read_targets("cat pkg/a.py | grep foo"), ["pkg/a.py"])

    def test_repo_relative_normalizes_dot_slash(self):
        self.assertEqual(repo_relative("./pkg/a.py"), "pkg/a.py")
        self.assertEqual(repo_relative("pkg/./a.py"), "pkg/a.py")


class TestScanScoping(unittest.TestCase):
    """Review 1.8: env/tool-error text scans must skip READ and SEARCH results."""

    def test_env_error_in_read_output_ignored(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _call(0, "c1", action=READ, files=["pkg/compat.py"]),
                _result(
                    1,
                    "c1",
                    output="try:\n    import x\nexcept ImportError:\n"
                    "    raise ModuleNotFoundError('no module named x')",
                ),
            ],
        )
        self.assertFalse(compute(traj, _gold()).environment_error)

    def test_env_error_in_command_output_detected(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _call(0, "c1", action=RUN_COMMAND, cmd="python repro.py"),
                _result(1, "c1", output="ModuleNotFoundError: No module named 'x'"),
            ],
        )
        self.assertTrue(compute(traj, _gold()).environment_error)

    def test_search_output_not_a_tool_error(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _call(0, "c1", action=SEARCH),
                _result(1, "c1", output="pkg/io.py:41: raise OSError('No such file or directory: %s' % path)"),
            ],
        )
        self.assertEqual(_repair_and_tool_signals(traj)["n_tool_call_errors"], 0)


class TestSubagentOrdering(unittest.TestCase):
    """Review 1.5: appended sub-agent steps must not corrupt ordering signals."""

    def test_subagent_edit_does_not_fake_unreverified_repair(self):
        # Main thread: test fails -> edit -> test passes (properly re-verified).
        # A sub-agent block appended AFTER contains an edit; with global ordering
        # it would look like an edit after the final test run.
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _call(0, "t1", action=RUN_TEST, test=True, cmd="pytest tests/test_x.py"),
                _result(1, "t1", output="1 failed"),
                _step(2, actor="agent", type="tool_call", canonical_action=EDIT, files=["pkg/a.py"], call_id="e1"),
                _call(3, "t2", action=RUN_TEST, test=True, cmd="pytest tests/test_x.py"),
                _result(4, "t2", output="1 passed"),
                _step(
                    5,
                    actor="agent",
                    type="tool_call",
                    canonical_action=EDIT,
                    files=["pkg/b.py"],
                    call_id="e2",
                    subagent_id="sub1",
                ),
            ],
        )
        sig = _repair_and_tool_signals(traj)
        self.assertTrue(sig["encountered_failure"])
        self.assertFalse(sig["repair_not_reverified"])

    def test_final_claim_reads_main_thread(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _step(0, actor="agent", type="message", text="All tests pass, the fix works.", subagent_id="sub1"),
                _step(1, actor="agent", type="message", text="I could not finish the change."),
            ],
        )
        self.assertFalse(compute(traj, _gold()).claimed_success_final)


class TestExitCodes(unittest.TestCase):
    """Exit codes, when recorded, are ground truth over text heuristics."""

    def test_exit_zero_vetoes_echoed_failure_text(self):
        r = _result(1, "c1", output="grep found: 'FAILED' in docs", args={"exit_code": 0})
        self.assertFalse(_run_failed(r))

    def test_nonzero_exit_is_failure_even_without_markers(self):
        r = _result(1, "c1", output="...long collection header only...", args={"exit_code": 2})
        self.assertTrue(_run_failed(r))


class TestExcerptAndCache(unittest.TestCase):
    def test_excerpt_keeps_tail(self):
        text = "collecting...\n" + ("x" * 2000) + "\n=== 1 failed in 0.2s ==="
        e = excerpt(text)
        self.assertIn("1 failed", e)
        self.assertLessEqual(len(e), 620)
        self.assertTrue(looks_like_test_failure(e))

    def test_stale_unversioned_signal_cache_ignored(self):
        """Review 1.15: a pre-version cache entry must be recomputed, not served."""
        import agent_trajviz.awe._engine.trajsignals as ts

        with tempfile.TemporaryDirectory() as td:
            old_cache = ts._CACHE
            ts._CACHE = Path(td)
            try:
                p = Path(td) / "a" / "i.json"
                p.parent.mkdir(parents=True)
                # Old flat format: no schema_version, stale field values.
                p.write_text(
                    json.dumps(
                        {
                            "n_steps": 999,
                            "read_set": [],
                            "edit_set_traj": [],
                            "ran_any_test": True,
                            "fail_to_pass_referenced": True,
                            "claimed_success_final": True,
                            "claimed_success_any": True,
                            "repeated_edit_files": [],
                            "edits_per_file": {},
                            "n_tool_errors": 0,
                            "n_test_runs": 9,
                            "hidden_reasoning": False,
                        }
                    )
                )
                # No raw trajectory exists for this fake agent/instance, so a
                # correctly-invalidated cache falls through to None instead of
                # serving the planted stale record.
                self.assertIsNone(ts.signals_cached("a", "i", _gold()))
            finally:
                ts._CACHE = old_cache


if __name__ == "__main__":
    unittest.main()
