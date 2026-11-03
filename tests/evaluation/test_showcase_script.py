"""The offline showcase is import-safe and renders without browser JavaScript."""
from __future__ import annotations

import contextlib
import io
import json
from agent_trajviz.awe import _engine as _engine_package
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest

_ENGINE_ROOT = Path(_engine_package.__file__).resolve().parent


class ShowcaseScriptTests(unittest.TestCase):
    def test_saved_showcase_produces_inert_html_and_svg_without_javascript(self):
        source = _ENGINE_ROOT / "scripts" / "build_showcase.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "data").mkdir()
            data = {"dataset": {"benchmark": "Script fixture", "n_scored_runs": 2, "n_solved": 1, "n_failed": 1},
                    "insights": {"agent": {"scores": {"code_editing": 80.0}}},
                    "agents_meta": {"agent": {"name": "<img src=x onerror=alert(1)>"}}, "cases": [], "countermeasures": []}
            (root / "data" / "showcase_data.json").write_text(json.dumps(data), encoding="utf-8")
            script = root / "scripts" / source.name
            shutil.copy2(source, script)
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(script), run_name="__main__")
            report = (root / "data" / "showcase_results.html").read_text(encoding="utf-8")
            self.assertIn("<svg", report)
            self.assertNotIn("<script", report.lower())
            self.assertNotIn("<img src=x", report)
            self.assertIn("&lt;img", report)
            self.assertIn("80", report)


if __name__ == "__main__":
    unittest.main()
