# tests/test_step_repetition.py
from __future__ import annotations
import os
import unittest
from agent_trajviz.awe._engine.trajectory import (
    NormalizedStep,
    NormalizedTrajectory,
    EDIT,
    READ,
    RUN_COMMAND,
    RUN_TEST,
    SEARCH,
)
from agent_trajviz.awe._engine.trajsignals import compute, _step_repetition
from agent_trajviz.awe._engine.gold import GoldReference


def _step(i, **kw):
    return NormalizedStep(step_id=i, turn_id=i, **kw)


def _call(i, action=RUN_COMMAND, cmd="pytest", files=None, sub=None):
    return _step(
        i,
        actor="agent",
        type="tool_call",
        canonical_action=action,
        files=files or [],
        call_id=f"c{i}",
        args={"command": cmd},
        subagent_id=sub,
    )


def _read(i, path):
    return _step(i, actor="agent", type="tool_call", canonical_action=READ, files=[path], call_id=f"c{i}", args={})


def _edit(i, path):
    return _step(i, actor="agent", type="tool_call", canonical_action=EDIT, files=[path], call_id=f"c{i}", args={})


def _search(i, pattern, path=None, include=None):
    args = {"pattern": pattern}
    if path is not None:
        args["path"] = path
    if include is not None:
        args["include"] = include
    return _step(i, actor="agent", type="tool_call", canonical_action=SEARCH, files=[], call_id=f"c{i}", args=args)


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


class TestStepRepetition(unittest.TestCase):
    def test_blind_retest_is_unproductive(self):
        # same test command 3x, no edit between -> 2 unproductive repeats
        traj = NormalizedTrajectory(
            "a", "i", [_call(1, RUN_TEST, "pytest x"), _call(2, RUN_TEST, "pytest x"), _call(3, RUN_TEST, "pytest x")]
        )
        self.assertEqual(_step_repetition(traj), 2)
        sig = compute(traj, _gold())
        self.assertEqual(sig.n_unproductive_repeats, 2)
        self.assertTrue(sig.step_repetition)

    def test_healthy_iteration_is_silent(self):
        # edit -> test -> edit -> test (same cmd): each re-test follows an edit
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _edit(1, "pkg/a.py"),
                _call(2, RUN_TEST, "pytest x"),
                _edit(3, "pkg/a.py"),
                _call(4, RUN_TEST, "pytest x"),
            ],
        )
        self.assertEqual(_step_repetition(traj), 0)
        self.assertFalse(compute(traj, _gold()).step_repetition)

    def test_interleaved_thrash_counts(self):
        # read A, read B, read A, read B, no edits -> fragments miss this
        traj = NormalizedTrajectory("a", "i", [_read(1, "a.py"), _read(2, "b.py"), _read(3, "a.py"), _read(4, "b.py")])
        self.assertEqual(_step_repetition(traj), 2)

    def test_single_occurrence_is_zero(self):
        traj = NormalizedTrajectory(
            "a", "i", [_call(1, RUN_TEST, "pytest x"), _read(2, "a.py"), _call(3, RUN_COMMAND, "ls")]
        )
        self.assertEqual(_step_repetition(traj), 0)

    def test_env_repair_between_retests_is_not_unproductive(self):
        # pip install between two identical test runs = new information (parity with
        # retry_without_change's _ENV_REPAIR_RE guard) — must NOT count as a blind repeat.
        traj = NormalizedTrajectory(
            "a",
            "i",
            [_call(1, RUN_TEST, "pytest x"), _call(2, RUN_COMMAND, "pip install foo"), _call(3, RUN_TEST, "pytest x")],
        )
        self.assertEqual(_step_repetition(traj), 0)

    def test_subagent_steps_ignored(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _call(1, RUN_TEST, "pytest x"),
                _step(
                    2,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    files=[],
                    call_id="c2",
                    args={"command": "pytest x"},
                    subagent_id="s1",
                ),
            ],
        )
        self.assertEqual(_step_repetition(traj), 0)

    # --- native search tools (adapter shapes: claude_code {pattern,path},
    # --- opencode {pattern,include}; files=[] for both) ---------------------
    def test_repeated_identical_search_is_unproductive(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [_search(1, "def foo", path="src"), _search(2, "def foo", path="src"), _search(3, "def foo", path="src")],
        )
        self.assertEqual(_step_repetition(traj), 2)

    def test_different_search_patterns_are_not_repeats(self):
        traj = NormalizedTrajectory("a", "i", [_search(1, "def foo"), _search(2, "def bar"), _search(3, "class Baz")])
        self.assertEqual(_step_repetition(traj), 0)

    def test_same_pattern_different_scope_is_not_a_repeat(self):
        traj = NormalizedTrajectory("a", "i", [_search(1, "def foo", path="src"), _search(2, "def foo", path="tests")])
        self.assertEqual(_step_repetition(traj), 0)

    def test_opencode_search_shape_include_scope(self):
        traj = NormalizedTrajectory(
            "a", "i", [_search(1, "def foo", include="*.py"), _search(2, "def foo", include="*.py")]
        )
        self.assertEqual(_step_repetition(traj), 1)

    def test_edit_between_searches_resets(self):
        traj = NormalizedTrajectory("a", "i", [_search(1, "def foo"), _edit(2, "pkg/a.py"), _search(3, "def foo")])
        self.assertEqual(_step_repetition(traj), 0)

    # --- env-repair carve-out must not excuse blind re-TESTS ----------------
    def test_blind_make_test_retest_counts(self):
        # `make test` matches _ENV_REPAIR_RE (make\b) AND _TEST_PHRASES; a blind
        # re-run of the same test command must count like any other blind re-test.
        traj = NormalizedTrajectory(
            "a",
            "i",
            [_call(1, RUN_TEST, "make test"), _call(2, RUN_TEST, "make test"), _call(3, RUN_TEST, "make test")],
        )
        self.assertEqual(_step_repetition(traj), 2)

    # --- chunked reads of one large file are one logical read ---------------
    def test_chunked_read_distinct_offsets_not_unproductive(self):
        def rd(i, offset):
            return _step(
                i,
                actor="agent",
                type="tool_call",
                canonical_action=READ,
                files=["big.py"],
                call_id=f"c{i}",
                args={"path": "big.py", "offset": offset, "limit": 2000},
            )

        traj = NormalizedTrajectory("a", "i", [rd(1, 0), rd(2, 2000), rd(3, 4000)])
        self.assertEqual(_step_repetition(traj), 0)

    def test_reread_same_span_still_counts(self):
        def rd(i):
            return _step(
                i,
                actor="agent",
                type="tool_call",
                canonical_action=READ,
                files=["big.py"],
                call_id=f"c{i}",
                args={"path": "big.py", "offset": 0, "limit": 2000},
            )

        traj = NormalizedTrajectory("a", "i", [rd(1), rd(2)])
        self.assertEqual(_step_repetition(traj), 1)

    # --- signature hygiene ---------------------------------------------------
    def test_command_whitespace_collapsed_in_signature(self):
        # spec §3.1: strip + collapse whitespace
        traj = NormalizedTrajectory("a", "i", [_call(1, RUN_TEST, "pytest  x"), _call(2, RUN_TEST, "pytest x")])
        self.assertEqual(_step_repetition(traj), 1)

    def test_run_command_vs_run_test_same_string_distinct(self):
        traj = NormalizedTrajectory("a", "i", [_call(1, RUN_COMMAND, "tox -e py"), _call(2, RUN_TEST, "tox -e py")])
        self.assertEqual(_step_repetition(traj), 0)


class TestIsolation(unittest.TestCase):
    def test_fault_layers_never_reference_step_repetition(self):
        # spec §5/§6: step_repetition is a calibrate-first BEHAVIORAL signal; the
        # fault-attribution layers must not consume it until it is promoted.
        import agent_trajviz.awe._engine

        root = os.path.dirname(os.path.abspath(agent_trajviz.awe._engine.__file__))
        for mod in ("detect.py", "rootcause.py", "scoring.py"):
            with open(os.path.join(root, mod)) as fh:
                src = fh.read()
            self.assertNotIn("step_repetition", src, f"{mod} references step_repetition")
            self.assertNotIn("n_unproductive_repeats", src, f"{mod} references n_unproductive_repeats")


class TestCohortFeature(unittest.TestCase):
    def test_features_of_exposes_step_repetition(self):
        from agent_trajviz.awe._engine import cohort

        traj = NormalizedTrajectory("a", "i", [_call(1, RUN_TEST, "pytest x"), _call(2, RUN_TEST, "pytest x")])
        feats = cohort.features_of(compute(traj, _gold()))
        self.assertEqual(feats["step_repetition"], 1.0)
        self.assertEqual(feats["n_unproductive_repeats"], 1.0)

    def test_step_repetition_registered_binary(self):
        # binary presence flag must be in BINARY_FEATURES; the graded count must NOT
        from agent_trajviz.awe._engine.cohort import BINARY_FEATURES

        self.assertIn("step_repetition", BINARY_FEATURES)
        self.assertNotIn("n_unproductive_repeats", BINARY_FEATURES)
