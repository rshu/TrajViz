"""Native import → prompt text model → colored, literal HTML integration."""

from __future__ import annotations

import copy
import inspect
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest

import agent_trajviz as tv
from agent_trajviz.i18n import translate
from agent_trajviz.insight.ui import prompt_tab

_MODULE = '''"""A small repository module with literal markup in its data."""
from __future__ import annotations

class Checkpoint:
    label = "Settings <script>alert('literal')</script>"

    def ready(self) -> bool:
        return True
'''
_CHECKPOINT = "Checkpoint: keep Settings literal; inspect <script> as text before continuing."
_DIAGNOSTIC = "python -m pytest tests/test_checkpoint.py -q"


class _HTMLText(HTMLParser):
    """Read visible text and semantic classes without depending on HTML spacing."""

    def __init__(self, value: str):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, set[str]]] = []
        self.fragments: list[tuple[set[str], str]] = []
        self.pre_text: list[str] = []
        self.tags: list[str] = []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        classes = set(dict(attrs).get("class", "").split())
        self.stack.append((tag, classes))

    def handle_endtag(self, tag):
        for position in range(len(self.stack) - 1, -1, -1):
            if self.stack[position][0] == tag:
                del self.stack[position:]
                break

    def handle_data(self, data):
        classes = set().union(*(classes for _, classes in self.stack)) if self.stack else set()
        self.fragments.append((classes, data))
        if any(tag == "pre" for tag, _ in self.stack):
            self.pre_text.append(data)

    def marked(self, css_class):
        return "".join(value for classes, value in self.fragments if css_class in classes)


def _at_step(run, step_index):
    catalog = run.view("prompt_construction").data["requests"]
    selected = next(row for row in catalog if row.get("step_index") == step_index)
    return run.view("prompt_construction", request_id=selected["id"]).data["selected_request"]


def _at_message(run, message_id):
    index = next(step["index"] for step in run.steps
                 if step.get("id") == message_id or step.get("message_id") == message_id)
    return _at_step(run, index)


def _contains_text(value, text):
    if isinstance(value, str):
        return text in value
    if isinstance(value, dict):
        return any(_contains_text(item, text) for item in value.values())
    if isinstance(value, list):
        return any(_contains_text(item, text) for item in value)
    return False


@pytest.fixture
def repository_run(tmp_path):
    """A portable native OpenCode session covering the real-session workflows."""
    def message(eid, role, created, content, *, tool=None):
        parts = [{"type": "text", "text": content}] if content else []
        if tool:
            name, arguments, result = tool
            parts.append({"type": "tool", "tool": name, "callID": f"call-{eid}",
                          "state": {"status": "completed", "input": arguments, "output": result,
                                    "time": {"start": created + 20, "end": created + 80}}})
        return {"info": {"id": eid, "role": role, "sessionID": "synthetic-repository",
                         "modelID": "recorded-test-model", "providerID": "recorded-test-provider",
                         "time": {"created": created, "completed": created + 100}}, "parts": parts}

    native = {"info": {"id": "synthetic-repository", "directory": "/scratch/repository",
                       "time": {"created": 1000, "updated": 6100}},
              "messages": [message("u1", "user", 1000, "Fix the repository checkpoint tests."),
                           message("diagnostic-1", "assistant", 2000, "", tool=(
                               "shell", {"command": _DIAGNOSTIC}, "1 failed <assertion>")),
                           message("diagnostic-2", "assistant", 3000, "", tool=(
                               "shell", {"command": _DIAGNOSTIC}, "1 failed <assertion>")),
                           message("module-write", "assistant", 4000, "", tool=(
                               "write", {"path": "/scratch/repository/checkpoint.py", "content": _MODULE}, "Written")),
                           message("checkpoint", "user", 5000, _CHECKPOINT),
                           message("after-checkpoint", "assistant", 6000, "CURRENT RESPONSE MUST NOT ENTER INPUT")]}
    path = tmp_path / "repository-task.json"
    path.write_text(json.dumps(native), encoding="utf-8")
    run = tv.load(path)
    assert run.format == "opencode"
    return run


def test_repository_checkpoint_is_added_green_without_invented_rewrite(repository_run):
    from agent_trajviz.insight.ui.prompt_text import render_text_evolution

    detail = _at_message(repository_run, "after-checkpoint")
    evolution = detail["text_evolution"]
    assert evolution["comparison_available"] and evolution["evidence"] == "estimated"
    assert evolution["rewrites"] == []
    checkpoint = next(block for block in evolution["blocks"] if block["text"] == _CHECKPOINT)
    assert checkpoint["change"] == "added" and checkpoint["role"] == "user"
    assert not checkpoint["repeated"]
    rendered = render_text_evolution(detail)
    parsed = _HTMLText(rendered)
    assert _CHECKPOINT in parsed.marked("pc-text-added")
    assert "script" not in parsed.tags
    assert "&lt;script&gt;" in rendered
    assert "CURRENT RESPONSE MUST NOT ENTER INPUT" not in json.dumps(detail)


def test_repeated_diagnostic_and_module_write_remain_readable(repository_run):
    from agent_trajviz.insight.ui.prompt_text import render_component_text, render_text_evolution

    repeated_detail = _at_message(repository_run, "module-write")
    repeated = [block for block in repeated_detail["text_evolution"]["blocks"]
                if block["kind"] == "tool_arguments" and block["repeat_kind"] == "same_tool_arguments"]
    assert any(_DIAGNOSTIC in block["text"] and block["repeated"] and block["repeat_count"] >= 1
               for block in repeated)
    repeat_html = render_text_evolution(repeated_detail)
    assert "pc-text-repeat-badge" in repeat_html
    assert _DIAGNOSTIC in _HTMLText(repeat_html).marked("pc-text-repeated")

    detail = _at_message(repository_run, "after-checkpoint")
    component = next(part for part in detail["components"] if _contains_text(part["content"], _MODULE))
    original = copy.deepcopy(component)
    rendered = render_component_text(component, detail)
    parsed = _HTMLText(rendered)
    assert _MODULE in "".join(parsed.pre_text)
    assert "class Checkpoint:\n" in "".join(parsed.pre_text)
    assert "script" not in parsed.tags and "&lt;script&gt;" in rendered
    assert component == original


@pytest.fixture
def pi_rewrite_run(tmp_path):
    def message(eid, parent, role, content="", **extra):
        return {"type": "message", "id": eid, "parentId": parent,
                "message": {"role": role, "content": content, **extra}}

    before = "Run tests once.\nSettings <script>must remain literal</script>."
    after = "Run tests twice.\nSettings <script>must remain literal</script>."
    events = [{"type": "session", "id": "native-rewrite", "version": 3},
              message("system", None, "system", sections={"policy": "Keep changes focused."}, toolsAdded=[]),
              message("request", "system", "user", before), message("answer", "request", "assistant", "First reply"),
              {"type": "context_edit", "id": "rewrite", "parentId": "answer", "targetId": "request", "replacement": after},
              message("next", "rewrite", "assistant", "Second reply")]
    path = tmp_path / "native-rewrite.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    return tv.load(path), before, after


def test_native_rewrite_has_highlighted_before_after_and_literal_source(pi_rewrite_run):
    from agent_trajviz.insight.ui.prompt_text import render_text_evolution

    run, before, after = pi_rewrite_run
    detail = run.view("prompt_construction").data["selected_request"]
    rewrites = detail["text_evolution"]["rewrites"]
    assert detail["scope"] == "native_prompt_state" and detail["evidence"] == "reconstructed"
    assert any(row["before_text"] == before and row["after_text"] == after for row in rewrites)
    rendered = render_text_evolution(detail)
    parsed = _HTMLText(rendered)
    assert "pc-text-rewrite" in rendered and "ins" in parsed.tags and "del" in parsed.tags
    assert before in "".join(parsed.pre_text) and after in "".join(parsed.pre_text)
    assert "once" in parsed.marked("pc-text-removed") and "twice" in parsed.marked("pc-text-added")
    assert "script" not in parsed.tags and "&lt;script&gt;" in rendered


def test_language_switch_localizes_controls_without_translating_source(pi_rewrite_run):
    from agent_trajviz.insight.ui.prompt_text import legend, render_text_evolution

    run, before, after = pi_rewrite_run
    english = prompt_tab.outputs(run, language="en")
    chinese = prompt_tab.outputs(run, language="zh-CN")
    assert english[-1] == chinese[-1]
    assert english[0]["choices"][0][0] != chinese[0]["choices"][0][0]
    rendered = render_text_evolution(chinese[-1])
    assert before in "".join(_HTMLText(rendered).pre_text)
    assert after in "".join(_HTMLText(rendered).pre_text)
    for label in ("Added text", "Removed text", "Retained text", "Uncompared text", "Repeated content"):
        assert f'data-tv-i18n="{label}"' in legend()
        assert translate(label, "zh-CN") != label
    assert "Settings" in "".join(_HTMLText(rendered).pre_text)


def _run(session):
    return tv.Trajectory.from_steps([
        {"index": 0, "role": "user", "session_id": session,
         "parts": [{"type": "text", "text": f"Private input for {session}"}]},
        {"index": 1, "role": "assistant", "session_id": session,
         "parts": [{"type": "text", "text": f"Answer for {session}"}]}])


def test_stale_selection_recovers_current_run_choices_before_gradio_preprocessing():
    old, current = _run("old-session"), _run("current-session")
    stale_id = prompt_tab.outputs(old)[0]["value"]
    values = prompt_tab.outputs(current, stale_id)
    selector, detail = values[0], values[-1]
    current_ids = {row["id"] for row in current.view("prompt_construction").data["requests"]}
    assert selector["choices"] and selector["value"] in current_ids
    assert detail["id"] in current_ids and "old-session" not in json.dumps(detail)
    # Gradio validates dropdown membership before invoking our callback. The
    # recovered server-side choices must accept the value returned to clients.
    assert gr.Dropdown(choices=selector["choices"]).preprocess(selector["value"]) == selector["value"]


def test_prompt_callbacks_keep_selection_and_language_input_arity():
    from gradio.helpers import special_args

    from agent_trajviz.insight.ui.shared import record_latest_run

    run = _run("callback-session")
    with gr.Blocks() as app:
        state = gr.State(run)
        language = gr.State("en")
        refs = prompt_tab.layout()
        trigger = gr.Button("Load test trajectory")
        event = trigger.click(fn=lambda: run, outputs=[state])
        prompt_tab.bind(refs, SimpleNamespace(state_run=state, state_dark=gr.State(False)), [event], language=language)
    callbacks = [fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "").startswith("on_prompt_")]
    assert callbacks
    selected_id = run.view("prompt_construction").data["selected_request"]["id"]
    request = gr.Request(session_hash="prompt-text-e2e-callback-session")
    record_latest_run(request.session_hash, run)
    try:
        for callback in callbacks:
            parameters = inspect.signature(callback.fn).parameters
            assert len([name for name in parameters if name != "request"]) == len(callback.inputs)
            assert callback.inputs[0] is state and callback.inputs[-1] is language
            inputs = [run, "en"] if len(callback.inputs) == 2 else [run, selected_id, "en"]
            arguments = special_args(callback.fn, inputs=inputs, request=request)[0]
            assert arguments[-1] is request
            values = callback.fn(*arguments)
            assert len(values) == len(callback.outputs)
            assert values[-1]["id"] == selected_id
    finally:
        record_latest_run(request.session_hash, None)


def test_slow_old_prompt_render_does_not_replace_a_new_loaded_session(monkeypatch):
    from agent_trajviz.insight.ui.shared import record_latest_run

    old, current = _run("old-session"), _run("current-session")
    with gr.Blocks() as app:
        state = gr.State(old)
        language = gr.State("en")
        refs = prompt_tab.layout()
        trigger = gr.Button("Load test trajectory")
        event = trigger.click(fn=lambda: old, outputs=[state])
        prompt_tab.bind(refs, SimpleNamespace(state_run=state), [event], language=language)
    on_load = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "on_prompt_load")
    session_hash = "prompt-text-e2e-stale-session"
    request = SimpleNamespace(session_hash=session_hash)
    original_outputs = prompt_tab.outputs

    def interrupted_render(run, request_id=None, language="en"):
        assert run is old
        result = original_outputs(run, request_id, language)
        record_latest_run(session_hash, current)
        return result

    monkeypatch.setattr(prompt_tab, "outputs", interrupted_render)
    record_latest_run(session_hash, old)
    try:
        values = on_load(old, "en", request=request)
        assert len(values) == 6 and all(value == gr.skip() for value in values)
    finally:
        record_latest_run(session_hash, None)


@pytest.fixture(scope="module")
def real_opencode_run():
    fixture_path = os.environ.get("TRAJVIZ_OPENCODE_SESSION_FIXTURE")
    if not fixture_path:
        pytest.skip("Set TRAJVIZ_OPENCODE_SESSION_FIXTURE to the 204-step OpenCode walkthrough export")
    path = Path(fixture_path)
    assert path.is_file(), "The configured OpenCode session fixture is not a file"
    run = tv.load(path)
    assert run.format == "opencode" and len(run.steps) > 100
    return run


def test_real_opencode_checkpoint_before_native_183_is_added_text(real_opencode_run):
    from agent_trajviz.insight.ui.prompt_text import render_text_evolution

    run = real_opencode_run
    detail = _at_step(run, 183)
    checkpoint = next(step for step in run.steps if step["index"] == 182)
    content = "\n".join(part["text"] for part in checkpoint["parts"] if part.get("type") == "text")
    assert checkpoint["role"] == "user" and "checkpoint" in content.lower()
    evolution = detail["text_evolution"]
    assert evolution["evidence"] == "estimated" and evolution["rewrites"] == []
    assert any(block["change"] == "added" and block["role"] == "user" and content == block["text"]
               for block in evolution["blocks"])
    assert content in _HTMLText(render_text_evolution(detail)).marked("pc-text-added")


def test_real_opencode_repeated_diagnostics_and_source_22_module_are_readable(real_opencode_run):
    from agent_trajviz.insight.ui.prompt_text import render_component_text, render_text_evolution

    run = real_opencode_run
    assert run.steps[67]["tool_calls"][0]["input"] == run.steps[69]["tool_calls"][0]["input"]
    command = run.steps[69]["tool_calls"][0]["input"]["command"]
    repeated_detail = _at_step(run, 70)
    repeated = [block for block in repeated_detail["text_evolution"]["blocks"]
                if block["repeated"] and block["repeat_kind"] == "same_tool_arguments"]
    assert any(command in block["text"] and block["repeat_count"] >= 1 for block in repeated)
    assert "pc-text-repeat-badge" in render_text_evolution(repeated_detail)

    detail = _at_step(run, 23)
    code = run.steps[22]["tool_calls"][0]["input"]["content"]
    component = next(part for part in detail["components"] if _contains_text(part["content"], code[:200]))
    assert any(source.get("raw_index") == 22 or source.get("step_index") == 22
               for source in detail["sources"])
    assert len(code) > 10_000 and "class " in code
    rendered = render_component_text(component, detail)
    readable = "".join(_HTMLText(rendered).pre_text)
    assert code[:200] in readable and code.split("class ", 1)[1].splitlines()[0] in readable
    assert "\n" in readable and "\\n" not in readable[:200]
    assert "script" not in _HTMLText(rendered).tags
