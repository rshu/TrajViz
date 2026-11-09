"""Code changes without any validation: write tools touched code files but no
test/build/lint command ran in the whole session.

Only writes to code files count (docs/config have nothing a test run would
catch); writes through shell redirection are not counted, keeping the rule
explainable in one sentence.
"""

import unittest

from agent_trajviz.insight.patterns import detect_unvalidated_writes
from agent_trajviz.insight.issues import prepare_overview_issues
from agent_trajviz.insight.presenters.issues import build_overview_issues_html
from tests.overview_session_stub import overview_session as _session

def _step(index, calls=None):
    return {"index": index, "role": "assistant", "tool_calls": calls or [], "tokens": {}}

def _write(path="src/app.py", status="completed"):
    return {
        "tool_name": "edit",
        "status": status,
        "input": {"filePath": path, "old_string": "x", "new_string": "y"},
        "output": "ok",
    }

def _bash(command, status="completed"):
    return {
        "tool_name": "bash",
        "status": status,
        "input": {"command": command},
        "output": "ok",
    }

class DetectUnvalidatedWritesTests(unittest.TestCase):
    def test_code_writes_without_validation_are_flagged(self):
        steps = [
            _step(2, [_write("src/app.py")]),
            _step(5, [_write("src/lib/util.py")]),
        ]
        hits = detect_unvalidated_writes(steps)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["files"], 2)
        self.assertEqual(hits[0]["steps"], [2, 5])

    def test_a_single_validation_command_clears_the_flag(self):
        steps = [
            _step(2, [_write("src/app.py")]),
            _step(4, [_bash("pytest -q tests/")]),
        ]
        self.assertEqual(detect_unvalidated_writes(steps), [])

    def test_build_and_lint_commands_also_count_as_validation(self):
        for cmd in ("make -j8", "npm run build", "ruff check src/"):
            steps = [_step(1, [_write("src/app.py")]), _step(3, [_bash(cmd)])]
            self.assertEqual(detect_unvalidated_writes(steps), [], cmd)

    def test_doc_and_config_writes_are_ignored(self):
        steps = [
            _step(1, [_write("README.md")]),
            _step(2, [_write("config/settings.json")]),
        ]
        self.assertEqual(detect_unvalidated_writes(steps), [])

    def test_failed_writes_do_not_count(self):
        steps = [_step(1, [_write("src/app.py", status="error")])]
        self.assertEqual(detect_unvalidated_writes(steps), [])

    def test_running_the_written_program_clears_the_flag(self):
        # python3 app.py is a smoke run — it checks the result.
        steps = [
            _step(1, [_write("bisect_demo.py")]),
            _step(3, [_bash("python3 bisect_demo.py")]),
        ]
        self.assertEqual(detect_unvalidated_writes(steps), [])

    def test_version_probes_are_not_smoke_runs(self):
        steps = [
            _step(1, [_write("src/app.py")]),
            _step(3, [_bash("python3 --version")]),
        ]
        self.assertEqual(len(detect_unvalidated_writes(steps)), 1)

    def test_tooling_python_commands_are_not_smoke_runs(self):
        # pip / -c / bare interpreter never exercise the written code.
        for cmd in (
            "python3 -m pip install foo",
            "python3 -c 'print(1)'",
            "python3",
            "python -m venv .venv",
        ):
            steps = [
                _step(1, [_write("src/app.py")]),
                _step(3, [_bash(cmd)]),
            ]
            self.assertEqual(
                len(detect_unvalidated_writes(steps)), 1, cmd,
            )

    def test_python_module_entrypoint_clears_the_flag(self):
        steps = [
            _step(1, [_write("pkg/main.py")]),
            _step(3, [_bash("python -m pkg.main")]),
        ]
        self.assertEqual(detect_unvalidated_writes(steps), [])

    def test_read_only_sessions_are_not_flagged(self):
        steps = [_step(1, [{"tool_name": "read", "status": "completed",
                            "input": {"filePath": "src/app.py"}, "output": "..."}])]
        self.assertEqual(detect_unvalidated_writes(steps), [])

    def test_same_file_written_twice_counts_one_file(self):
        steps = [_step(1, [_write("src/app.py")]), _step(3, [_write("src/app.py")])]
        hits = detect_unvalidated_writes(steps)
        self.assertEqual(hits[0]["files"], 1)
        self.assertEqual(hits[0]["steps"], [1, 3])

class UnvalidatedWritesCardTests(unittest.TestCase):
    def _hits(self):
        return [{"files": 3, "steps": [2, 5, 9]}]

    def test_card_is_built_from_the_detector_output(self):
        issues = prepare_overview_issues(_session(unvalidated_writes=self._hits()))
        card = [i for i in issues if i.source_id == "antipattern:unvalidated_writes"]
        self.assertEqual(len(card), 1)
        self.assertEqual(card[0].kind, "antipattern")
        self.assertEqual(card[0].title, "Code changes were never validated")
        self.assertIn("3 code file(s) written in steps 2–9", card[0].detail)
        self.assertEqual(card[0].steps, (2, 5, 9))

    def test_no_card_when_nothing_was_flagged(self):
        issues = prepare_overview_issues(_session())
        self.assertFalse(
            [i for i in issues if "unvalidated_writes" in i.source_id]
        )

    def test_card_is_localized_to_chinese(self):
        html = build_overview_issues_html(_session(unvalidated_writes=self._hits()))
        self.assertIn("改了代码但全程未验证", html)
        self.assertIn("改了 3 个代码文件（第 2–9 步）", html)
        self.assertIn("整个运行没有任何测试/构建/检查命令", html)
        self.assertIn("正确性全凭智能体自信", html)
        self.assertIn("至少验证一次", html)

if __name__ == "__main__":
    unittest.main()
