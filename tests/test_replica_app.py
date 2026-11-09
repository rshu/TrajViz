"""End to end: the TrajViz dashboard rebuilt on the public API alone.

examples/trajviz_app reproduces the dashboard -- KPI strip, Overview
sections, Patterns, Attribution, Comparison, Workflow, Raw Data and the AI
sidebar -- with nothing but the public API. These tests check that by AST,
then launch it on a local port and drive every interaction over HTTP through
Gradio's own client, and check its charts are the original dashboard's.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import agent_trajviz as tv

REPO = Path(__file__).resolve().parent.parent
APP_DIR = REPO / "examples" / "trajviz_app"
REF = REPO / "tests" / "fixtures" / "native_claude_code"
DJANGO, ASTROPY = REF / "django__django-11477.jsonl", REF / "astropy__astropy-13033.jsonl"
PUBLIC = {"agent_trajviz", "agent_trajviz.figures", "agent_trajviz.llm", "agent_trajviz.attribution",
          "agent_trajviz.dashboard"}


class PublicApiOnlyTests(unittest.TestCase):
    def test_the_replica_imports_nothing_internal(self):
        for module in sorted(APP_DIR.glob("*.py")):
            tree = ast.parse(module.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                for name in names:
                    if name.startswith("agent_trajviz"):
                        with self.subTest(module=module.name, imports=name):
                            self.assertIn(name, PUBLIC)
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tv":
                    with self.subTest(module=module.name, attr=node.attr):
                        self.assertIn(node.attr, tv.__all__)


def _load_app():
    sys.path.insert(0, str(APP_DIR))
    try:
        spec = importlib.util.spec_from_file_location("trajviz_replica_app", APP_DIR / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(APP_DIR))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _labels(trajectory: Path, directory: Path) -> Path:
    run = tv.load(trajectory)
    phases = [("understand", "code_reading"), ("implement", "code_editing"), ("validate", "test_running")]
    records = [{"index": s["index"], "role": s["role"],
                **({"phase": "user", "action": "user_prompt"} if s["role"] == "user"
                   else dict(zip(("phase", "action"), phases[i % 3], strict=True))),
                "duration_s": s.get("duration") or 0} for i, s in enumerate(run.steps)]
    path = directory / f"{trajectory.stem}_labeled.json"
    path.write_text(json.dumps({"schema_version": "trajectory_labels.v2", "taxonomy_version": "v1",
                                "model": "m", "trajectory_file": str(trajectory), "steps": records}))
    return path


class LiveReplicaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gradio_client import Client, handle_file

        cls.file = staticmethod(handle_file)
        cls.module = _load_app()
        cls.app = cls.module.build_app()
        cls.app.launch(server_name="127.0.0.1", server_port=_free_port(), prevent_thread_lock=True, quiet=True)
        cls.client = Client(cls.app.local_url, verbose=False)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.app.close()
        cls.tmp.cleanup()

    def _load(self):
        return self.client.predict(self.file(str(DJANGO)), "", api_name="/load")

    def test_load_fills_every_panel_with_the_original_dashboards_numbers(self):
        out = self._load()
        text = json.dumps(out, default=str)
        for needle in ("Claude Code", "602,021", "217.2 gen tok/s", "95.8%", "EISDIR", "Showing 24 of 24 steps",
                       "Session details"):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_charts_are_the_original_dashboards(self):
        from agent_trajviz.insight.presenters.overview import build_chart_outputs
        from agent_trajviz.insight.session import load_session

        out = self._load()
        plots = [o for o in out if isinstance(o, dict) and o.get("type") == "plotly"]
        self.assertGreaterEqual(len(plots), 10)
        original = build_chart_outputs(load_session(str(DJANGO)))
        replica_token, replica_duration = json.loads(plots[0]["plot"]), json.loads(plots[1]["plot"])
        for replica, fig in ((replica_token, original["tok_fig"]), (replica_duration, original["dur_fig"])):
            expected = json.loads(fig.to_json())
            self.assertEqual(replica["data"], expected["data"])
            self.assertEqual(replica["layout"]["title"], expected["layout"]["title"])

    def test_labels_run_group_compare_and_step_detail(self):
        self._load()
        labels = self.client.predict(self.file(str(_labels(DJANGO, Path(self.tmp.name)))), api_name="/labels")
        self.assertIn("Labels loaded", json.dumps(labels))

        group = self.client.predict([self.file(str(ASTROPY))], "", api_name="/run_group")
        self.assertIn("astropy__astropy-13033", group[0])
        self.assertIn("django__django-11477", group[0])

        compared = self.client.predict(self.file(str(ASTROPY)), None, None, "", api_name="/compare")
        self.assertIn("Alignment", compared[0])

        detail = self.client.predict(1, api_name="/step")
        self.assertIn("Step #1", detail)
        self.assertIn("translate_url", detail)

    def test_export_writes_the_html_report(self):
        self._load()
        exported = self.client.predict(api_name="/export")
        path = exported["value"] if isinstance(exported, dict) else exported
        self.assertIn("<html", Path(path["path"] if isinstance(path, dict) else path).read_text()[:2000].lower())

    def test_native_capture_without_clocks_renders_unavailable_instead_of_zero(self):
        from tests.test_cursor_evidence import capture

        source = Path(self.tmp.name) / "untimed-cursor.json"
        source.write_text(json.dumps(capture()), encoding="utf-8")
        out = self.client.predict(self.file(str(source)), "", api_name="/load")
        rendered = json.dumps(out, default=str)
        self.assertIn("P95 n/a", rendered)
        self.assertNotIn("P95 0.00s", rendered)
        self.assertIn("limited evidence", rendered)
        self.assertNotIn("No diagnostic issues detected", rendered)
        self.assertNotIn("none found", rendered)

    def test_multi_session_capture_selects_explicitly_and_preserves_unknown_usage(self):
        from tests.test_otlp_e2e import _bundle

        with tempfile.TemporaryDirectory() as tmp:
            _, archive = _bundle(Path(tmp), partial=True)
            inspected = self.client.predict(self.file(str(archive)), api_name="/sessions")
            self.assertTrue(inspected["visible"])
            self.assertIsNone(inspected["value"])
            self.assertEqual({item[1] for item in inspected["choices"]}, {"demo-session", "second-session"})
            missing = self.client.predict(self.file(str(archive)), "", api_name="/load")
            self.assertIn("multiple sessions", json.dumps(missing).lower())
            selected = self.client.predict(self.file(str(archive)), "", "second-session",
                                           api_name="/select_session")
            rendered = json.dumps(selected, default=str)
            self.assertIn("n/a tokens", rendered)
            self.assertIn("limited evidence", rendered)
            exported = self.client.predict(api_name="/export")
            path = exported["value"] if isinstance(exported, dict) else exported
            html = Path(path["path"] if isinstance(path, dict) else path).read_text()
            self.assertIn("second-session", html)

    def test_new_failed_or_cleared_baseline_resets_labels_comparisons_and_export(self):
        invalid = Path(self.tmp.name) / "invalid.json"
        invalid.write_text('{"not_a_trajectory": true}', encoding="utf-8")
        for target in (None, str(invalid), str(ASTROPY)):
            with self.subTest(target=target):
                self._load()
                labels = self.client.predict(self.file(str(_labels(DJANGO, Path(self.tmp.name)))), api_name="/labels")
                self.assertIn("Labels loaded", json.dumps(labels))
                self.client.predict([self.file(str(ASTROPY))], "", api_name="/run_group")
                self.client.predict(self.file(str(ASTROPY)), None, None, "", api_name="/compare")
                self.client.predict(api_name="/export")
                output = self.client.predict(self.file(target) if target else None, "", api_name="/load")
                rendered = json.dumps(output, default=str)
                self.assertNotIn("Labels loaded", rendered)
                self.assertNotIn("Alignment", rendered)
                self.assertNotIn("trajviz-report-", rendered)
                self.assertNotIn("602,021", rendered)
                if target != str(ASTROPY):
                    self.assertEqual(output[1], "")
                    callback = next(fn.fn for fn in self.app.fns.values() if fn.api_name == "load")
                    packed = callback(target, "")
                    tabs = next(component for component in packed if component.get_block_name() == "tabs")
                    export = next(component for component in packed if component.get_block_name() == "button"
                                  and component.value == "Export HTML")
                    self.assertFalse(packed[tabs]["visible"])
                    self.assertFalse(packed[export]["interactive"])
                    with self.assertRaises(Exception) as caught:
                        self.client.predict(api_name="/export")
                    self.assertIn("Load a trajectory first", str(caught.exception))
                else:
                    callback = next(fn.fn for fn in self.app.fns.values() if fn.api_name == "load")
                    packed = callback(target, "")
                    tabs = next(component for component in packed if component.get_block_name() == "tabs")
                    self.assertTrue(packed[tabs]["visible"])

    def test_compressed_native_file_is_accepted_by_the_upload_component(self):
        from tests.test_dsh_native import events, write

        with tempfile.TemporaryDirectory() as tmp:
            source = write(Path(tmp) / "native", events())
            rendered = json.dumps(self.client.predict(self.file(str(source)), "", api_name="/load"), default=str)
            self.assertIn("DeepSeek Harness", rendered)
            self.assertIn("70 tokens", rendered)

    def test_llm_features_without_configuration_answer_with_a_message(self):
        # Hermetic: a developer's .env (loaded into this process by app
        # startup) must not leak in. load_env_files only fills unset
        # variables, so empty strings win and resolve sees them as missing.
        keys = [k for k in os.environ if k.startswith(("ANALYZE_", "LABEL_"))]
        saved = {k: os.environ[k] for k in keys}
        try:
            for k in keys:
                os.environ[k] = ""
            self._load()
            history, _ = self.client.predict("What is slow?", [], api_name="/ask")
        finally:
            os.environ.update(saved)
        content = history[-1]["content"]  # Gradio 6 sends message content as parts
        text = content if isinstance(content, str) else " ".join(part.get("text", "") for part in content)
        self.assertEqual(history[-1]["role"], "assistant")
        self.assertIn("not configured", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
