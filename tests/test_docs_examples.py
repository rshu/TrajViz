"""The docs checker executes only opted-in, offline, isolated examples."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / ".github" / "scripts" / "check_docs_examples.py"
SPEC = importlib.util.spec_from_file_location("trajviz_docs_examples", CHECKER)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def _run(directory: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(CHECKER), "--package-root", str(ROOT), "--docs-root", str(directory), *arguments],
        text=True, capture_output=True, timeout=30,
    )


@pytest.mark.parametrize("info", ["python", "python title=example", "text literal"])
def test_unmarked_fences_and_markers_inside_them_are_not_executed(tmp_path, info):
    page = tmp_path / "example.md"
    page.write_text(
        f"```{info}\nraise AssertionError('unmarked')\n<!-- trajviz:run -->\n```\n\n"
        "<!-- trajviz:run -->\n```python\nassert 2 + 2 == 4\n```\n", encoding="utf-8",
    )
    examples = checker.read_examples(page)
    assert len(examples) == 1 and examples[0].line == 8
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["executed"] == 1


@pytest.mark.parametrize("text", [
    "<!-- trajviz:run -->\n```bash\necho bad\n```\n",
    "<!-- trajviz:run -->\nprose instead of code\n",
    "<!-- trajviz:run extras=imaginary -->\n```python\npass\n```\n",
    "<!-- trajviz:run extras=charts;llm -->\n```python\npass\n```\n",
    "<!-- trajviz:run -->\n```python\npass\n",
])
def test_invalid_or_incomplete_markers_fail_loudly(tmp_path, text):
    page = tmp_path / "bad.md"
    page.write_text(text, encoding="utf-8")
    with pytest.raises(checker.ExampleError):
        checker.read_examples(page)


def test_extra_examples_are_skipped_unless_requested(tmp_path):
    (tmp_path / "extras.md").write_text(
        "<!-- trajviz:run -->\n```python\nassert True\n```\n"
        "<!-- trajviz:run extras=charts -->\n```python\nraise AssertionError('extra selected')\n```\n",
        encoding="utf-8",
    )
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["skipped"] == 1
    selected = _run(tmp_path, "--extras", "charts")
    assert selected.returncode == 1 and "extra selected" in selected.stderr


def test_each_page_has_a_fresh_registry_and_its_own_synthetic_inputs(tmp_path):
    text = '''<!-- trajviz:run -->
```python
import agent_trajviz as tv
assert tv.load("capture.jsonl").format == "claude_code"
assert len(tv.list_sessions("sessions.zip")) == 2
assert tv.label_views("labels.json")["labels_summary"].data["classified"] > 0
tv.register_format(tv.Format(
    key="docs_checker_example", label="Example", parse=lambda doc: [{"role": "assistant"}],
))
```
'''
    for name in ("first.md", "second.md"):
        (tmp_path / name).write_text(text, encoding="utf-8")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["executed"] == 2


def test_explicit_readme_executes_quickstart_without_local_docs(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- trajviz:run -->\n```python\nimport agent_trajviz as tv\n"
                      "run = tv.load('trajectory.json')\nassert run.format == 'claude_code'\n"
                      "tv.export_json(run, 'out/')\n```\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "private.md").write_text("<!-- trajviz:run -->\n```python\nraise AssertionError('local notes')\n```\n")
    result = subprocess.run([sys.executable, "-I", str(CHECKER), "--package-root", str(ROOT),
                             "--readme", str(readme)], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["executed"] == 1 and report["pages"] == 1
    assert report["results"][0]["page"] == "README.md"


@pytest.mark.parametrize("code", [
    "import socket\nsocket.create_connection(('example.invalid', 443))\n",
    "import subprocess\nsubprocess.run(['echo', 'unexpected'], check=True)\n",
])
def test_executed_examples_cannot_call_networks_or_spawn_processes(tmp_path, code):
    (tmp_path / "offline.md").write_text(
        "<!-- trajviz:run -->\n```python\n" + code + "```\n", encoding="utf-8",
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "must not use network access or child processes" in result.stderr


def test_requested_package_version_is_checked_before_execution(tmp_path):
    (tmp_path / "version.md").write_text("<!-- trajviz:run -->\n```python\nassert True\n```\n", encoding="utf-8")
    result = _run(tmp_path, "--expected-version", "0.0.0-does-not-exist")
    assert result.returncode == 1 and "expected version" in result.stderr


def test_reference_covers_public_api_and_validates_guide_fragments(tmp_path):
    module = ModuleType("example_public_module")
    module.__all__ = ["load", "Trajectory"]
    reference, guide = tmp_path / "reference.html", tmp_path / "guide.md"
    reference.write_text('<div id="load"></div><div id="Trajectory"></div>', encoding="utf-8")
    guide.write_text('[load](api/agent_trajviz.html#load)', encoding="utf-8")
    assert checker.check_api_reference(reference, [guide], module) == 2
    guide.write_text('[typo](api/agent_trajviz.html#missing)', encoding="utf-8")
    with pytest.raises(checker.ExampleError, match="unknown public API link"):
        checker.check_api_reference(reference, [guide], module)
    reference.write_text('<div id="load"></div>', encoding="utf-8")
    with pytest.raises(checker.ExampleError, match="missing public API anchors: Trajectory"):
        checker.check_api_reference(reference, [], module)
