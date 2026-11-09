"""Readable task/code evidence keeps exact raw artifacts and remains inert."""

from __future__ import annotations

import copy
import json
from html.parser import HTMLParser

from agent_trajviz.awe import render


class _PreBlocks(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.current = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "pre" and self.current is not None:
            self.blocks.append("".join(self.current))
            self.current = None


def _case(task, code):
    return {
        "case_id": "agent__task",
        "agent": "agent",
        "instance_id": "task",
        "task": task,
        "faults": [
            {
                "capability": "code_editing",
                "error_type": "incorrect_patch",
                "is_primary": True,
                "blame_weight": 0.7,
                "confidence": "medium",
                "code_evidence": code,
                "evidence_chain": {"causal_support": {"status": "not_tested"}},
                "detector_evidence": {"source": {"artifact": "faults.json", "json_pointer": "/0/task"}},
            }
        ],
    }


def _json_blocks(output):
    result = []
    for block in _PreBlocks(output).blocks:
        try:
            result.append(json.loads(block))
        except ValueError:
            pass
    return result


def test_task_problem_is_readable_markdown_with_all_reference_and_raw_fields():
    task = {
        "problem_statement": "## Reproduce\r\n\r\nRun `sample()` first.\r\n- Verify the output\r\n\r\n```python\r\nprint('first')\r\nprint('second')\r\n```",
        "gold_files": ["src/a.py"],
        "future_task_field": {"recorded": "ORIGINAL-UNKNOWN"},
    }
    case = _case(task, {})
    original = copy.deepcopy(case)
    output = render.case_html(case)
    assert "<h5>Reproduce</h5>" in output
    assert "<code>sample()</code>" in output and "<li>Verify the output</li>" in output
    assert "print('first')\nprint('second')" in _PreBlocks(output).blocks
    assert "src/a.py" in output and "ORIGINAL-UNKNOWN" in output
    assert "Raw task artifact" in output and task in _json_blocks(output)
    assert "Recorded diagnostic attribution" in output and "causal effects remain untested" in output
    assert case == original


def test_flat_and_nested_code_strings_render_real_lines_and_keep_complete_raw_artifact():
    patch = "@@ -1 +1 @@\n-old()\n+new()\n"
    literal_escape = 'print("\\n")\nprint("literal escape retained")'
    code = {
        "agent_patch": patch,
        "reference_fix": {"src/a_file.py": [patch, literal_escape]},
        "source_code": {"file": {"text": "first line\r\nsecond line"}},
        "future_shape": [{"nested": [1, True, None], "kept": "UNKNOWN-CODE-METADATA"}],
    }
    case = _case({"problem_statement": "Compare the recorded changes."}, code)
    original = copy.deepcopy(case)
    output = render.case_html(case)
    blocks = _PreBlocks(output).blocks
    assert patch in blocks and literal_escape in blocks
    assert "first line\r\nsecond line" in blocks
    assert "src/a_file.py" in output and "UNKNOWN-CODE-METADATA" in output
    assert "Raw code evidence artifact" in output and code in _json_blocks(output)
    assert "json_pointer" in output and "/0/task" in output
    assert case == original


def test_task_markdown_code_and_imported_keys_are_escaped_without_executable_html():
    unsafe = "<script>alert('fixture')</script>\n<img src=x onerror=alert(1)>"
    task = {"problem_statement": "# Task\n" + unsafe + "\n[bad](javascript:alert(1))", "reference": unsafe}
    code = {"<img src=x onerror=alert(1)>": unsafe, "reference_fix": {"<script>file()</script>": [unsafe]}}
    output = render.case_html(_case(task, code))
    assert "<script>" not in output and "<img" not in output
    assert 'href="javascript:' not in output
    assert "&lt;script&gt;" in output and "&lt;img" in output
    assert unsafe in _PreBlocks(output).blocks
    assert task in _json_blocks(output) and code in _json_blocks(output)


def test_unknown_task_and_code_shapes_keep_json_fallbacks():
    task = {"problem_statement": {"future": ["structured", "statement"]}, "unknown": False}
    code = [{"new_code_shape": {"nested": [False, None, "original"]}}]
    output = render.case_html(_case(task, code))
    assert task in _json_blocks(output) and code in _json_blocks(output)
    assert "Raw task artifact" in output and "Raw code evidence artifact" in output
