"""Language changes preserve session data and relabel asynchronously updated UI."""

import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import gradio as gr
import pytest

from agent_trajviz.i18n import attrs, catalog, normalize_language, text, translate, validate_catalog
from agent_trajviz.i18n.ui import bind_language, language_selector
from agent_trajviz.insight.connection_test import ConnectionTestResult, _http_failure
from agent_trajviz.insight.llm_config import AnalysisLLMConfig, config_status_html
from agent_trajviz.insight.ui.settings import _test_status_html


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.data = []

    def handle_starttag(self, tag, attributes):
        self.tags.append((tag, dict(attributes)))

    def handle_data(self, data):
        self.data.append(data)


def test_catalog_placeholders_and_unknown_locale_are_safe():
    validate_catalog()
    assert normalize_language("zh-CN") == "zh-CN"
    assert normalize_language("en") == "en"
    assert normalize_language({"locale": "zh-CN"}) == "en"
    assert translate("Workflow", "zh-CN") == "工作流"
    assert translate("unregistered technical_identifier", "zh-CN") == "unregistered technical_identifier"
    # Repeated catalog reads must remain callable after importing its modules.
    assert catalog() == catalog()


def test_owned_text_escapes_data_and_keeps_original_arguments():
    value = '<img src=x onerror="run()"> & Reference'
    rendered = text("Input errors ({count})", count=value)
    parsed = _HTML()
    parsed.feed(rendered)
    assert [tag for tag, _ in parsed.tags] == ["span"]
    attributes = parsed.tags[0][1]
    assert attributes["data-tv-i18n"] == "Input errors ({count})"
    assert json.loads(attributes["data-tv-i18n-args"]) == {"count": value}
    assert "".join(parsed.data) == f"Input errors ({value})"
    assert "<img" not in rendered


def test_owned_attributes_are_restricted_and_escaped():
    rendered = '<button ' + attrs("Step {index}", attribute="aria-label", index='" onclick="run()') + '>x</button>'
    parsed = _HTML()
    parsed.feed(rendered)
    attributes = parsed.tags[0][1]
    assert attributes["aria-label"] == 'Step " onclick="run()'
    assert "onclick" not in attributes
    assert attributes["data-tv-i18n-attr"] == "aria-label"
    with pytest.raises(ValueError):
        attrs("Workflow", attribute="onclick")


def _language_fixture():
    with gr.Blocks(analytics_enabled=False) as app:
        language = language_selector()
        model = gr.Textbox(label="Model", value="Reference", placeholder="Model ID")
        key = gr.Textbox(label="API key", value="private-session-value", type="password")
        tokens = gr.Number(label="Max output tokens", value=8192)
        choices = gr.Radio(choices=[("Reference", "ref"), ("Compared", "cmp")], value="cmp", label="Format")
        mutable_choices = gr.Dropdown(choices=[("Reference", "recording-id")], value="recording-id", label="Agent")
        result = gr.HTML("<p>Reference</p>", label="Comparison Report")
        test_button = gr.Button("Test API key")
        # Output membership alone must not stop an otherwise fixed button
        # label from being translated. It only receives interactivity updates.
        test_button.click(lambda: gr.update(interactive=False), outputs=test_button)
        test_button.click(lambda: gr.update(choices=[("Compared", "recording-id")]), outputs=mutable_choices)
        test_button.click(lambda: "<p>Reference</p>", outputs=result)
        bind_language(app, language)
    change = next(fn for fn in app.fns.values() if fn.fn.__name__ == "change")
    restore = next(fn for fn in app.fns.values() if fn.fn.__name__ == "restore")
    return app, change, restore, dict(language=language, model=model, key=key, tokens=tokens,
                                      choices=choices, mutable_choices=mutable_choices,
                                      result=result, test_button=test_button)


def _updates(event, language):
    return {component._id: update for component, update in zip(event.outputs, event.fn(language), strict=True)}


def test_compact_language_control_preserves_two_fixed_labels_and_canonical_values():
    _, change, restore, refs = _language_fixture()
    selector = refs["language"]
    assert isinstance(selector, gr.Radio)
    assert selector.choices == [("EN", "en"), ("中", "zh-CN")]
    assert selector.value == "en"
    assert selector.show_label is False
    assert selector.container is False
    assert selector.interactive is True
    assert selector.label  # Remains available to the native accessible control.
    assert selector._id not in _updates(change, "zh-CN")
    assert selector._id not in _updates(change, "en")
    restored = _updates(restore, "zh-CN")[selector._id]
    assert restored["value"] == "zh-CN" and "choices" not in restored
    assert selector.choices == [("EN", "en"), ("中", "zh-CN")]
    assert selector.value == "en"


def test_language_updates_preserve_credentials_selections_results_and_numbers():
    app, change, restore, refs = _language_fixture()
    updates = _updates(change, "zh-CN")
    for name in ("model", "key", "tokens", "choices", "mutable_choices", "result"):
        update = updates.get(refs[name]._id, {})
        assert "value" not in update
    assert updates[refs["model"]._id]["label"] == "模型"
    assert updates[refs["model"]._id]["placeholder"] == "模型 ID"
    assert refs["key"]._id not in updates
    assert "tv-secret-field" in refs["key"].elem_classes
    assert updates[refs["test_button"]._id]["value"] == "测试 API 密钥"
    assert "interactive" not in updates[refs["test_button"]._id]
    assert updates[refs["choices"]._id]["choices"] == [("参考", "ref"), ("对比", "cmp")]
    assert "choices" not in updates[refs["mutable_choices"]._id]
    assert refs["model"].value == "Reference"
    assert refs["key"].value == "private-session-value"
    assert refs["tokens"].value == 8192
    assert refs["choices"].value == "cmp"
    assert refs["mutable_choices"].value == "recording-id"
    preference = next(block for block in app.blocks.values() if isinstance(block, gr.BrowserState))
    assert preference.storage_key == "trajviz-ui-language-v1"
    assert preference.secret == "trajviz-ui-language-v1"
    restored = _updates(restore, "zh-CN")
    assert restored[refs["language"]._id]["value"] == "zh-CN"


def test_two_language_sessions_do_not_change_global_component_defaults():
    _, change, _, refs = _language_fixture()
    chinese = _updates(change, "zh-CN")
    english = _updates(change, "en")
    assert chinese[refs["model"]._id]["label"] == "模型"
    assert english[refs["model"]._id]["label"] == "Model"
    assert refs["model"].label == "Model"
    assert refs["key"].value == "private-session-value"


@pytest.mark.parametrize("status", [301, 400, 401, 402, 403, 404, 422, 429, 500, 503, 599, 418])
def test_connection_results_localize_only_safe_application_explanations(status):
    result = _http_failure(status)
    rendered = _test_status_html(result)
    parsed = _HTML()
    parsed.feed(rendered)
    marker = next(attributes for _, attributes in parsed.tags if "data-tv-i18n" in attributes)
    args = json.loads(marker["data-tv-i18n-args"])
    translated = translate(marker["data-tv-i18n"], "zh-CN", **args)
    assert translated != result.message
    if str(status) in result.message:
        assert str(status) in translated
    assert "".join(parsed.data) == result.message


def test_unknown_connection_evidence_stays_escaped_and_unmarked():
    original = "Unexpected message <script>secret()</script> & Reference"
    rendered = _test_status_html(ConnectionTestResult(False, original))
    parsed = _HTML()
    parsed.feed(rendered)
    assert "".join(parsed.data) == original
    assert "<script>" not in rendered
    assert not any("data-tv-i18n" in attributes for _, attributes in parsed.tags)


def test_analysis_connection_status_localizes_prose_and_preserves_safe_identifiers():
    rendered = config_status_html(loaded_steps=3, config=AnalysisLLMConfig(
        base_url="https://viewer:private-password@example.test:8443/private?key=private-query",
        model="Reference <model>", api_key="private-api-key", provider="openai", source="session",
        temperature=0.2, max_tokens=8192, timeout=120,
    ))
    parsed = _HTML()
    parsed.feed(rendered)
    keys = [attributes["data-tv-i18n"] for _, attributes in parsed.tags if "data-tv-i18n" in attributes]
    assert {"Ready", "sends to", "UI settings", "{count} steps in context"} <= set(keys)
    assert "Reference <model>" not in keys
    assert "example.test:8443" not in keys
    assert "<code>Reference &lt;model&gt;</code>" in rendered
    assert "<code>example.test:8443</code>" in rendered
    assert "private" not in rendered


_PLOT_TEST = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
    constructor(graph = false) { this.graph = graph; this.children = []; this.attributes = {}; }
    matches(selector) { return selector === '.js-plotly-plot' && this.graph; }
    querySelectorAll(selector) { return selector === '.js-plotly-plot' ? this.children : []; }
    closest(selector) { return this.matches(selector) ? this : null; }
    getAttribute(name) { return this.attributes[name] ?? null; }
    setAttribute(name, value) { this.attributes[name] = value; }
}
const graph = new Element(true);
graph.layout = {title: {text: 'Chart'}, xaxis: {title: {text: 'Step'}}, meta: {tv_i18n: {
    layout: {'title.text': 'Chart', 'xaxis.title.text': 'Step'}, traces: {'0': 'Reference'}
}}};
graph.data = [{name: 'Reference', x: [1, 2]}];
const events = {};
graph.on = (name, callback) => { events[name] = callback; };
const body = new Element(); body.children = [graph];
const document = {body, readyState: 'complete', documentElement: {lang: 'en'},
    querySelector: () => null, dispatchEvent: () => {}};
let calls = 0, holdNext = false, release = null, throwNext = false;
function write(object, path, value) {
    const parts = path.replace(/\[(\d+)\]/g, '.$1').split('.');
    for (const key of parts.slice(0, -1)) object = object[key] ??= {};
    object[parts.at(-1)] = value;
}
const window = {__tvTranslations: {Chart: '图表', Step: '步骤', Reference: '参考', Recall: '召回率'}};
window.Plotly = {
    relayout(node, updates) {
        calls++;
        if (throwNext) { throwNext = false; throw new Error('synthetic relayout failure'); }
        const apply = () => {
            for (const [key, value] of Object.entries(updates)) write(node.layout, key, value);
            events.plotly_afterplot?.();
        };
        if (holdNext) { holdNext = false; return new Promise(resolve => {
            release = () => { apply(); resolve(); };
        }); }
        apply(); return Promise.resolve();
    },
    restyle(node, updates, indices) {
        for (const [position, index] of indices.entries()) {
            for (const [key, value] of Object.entries(updates)) {
                node.data[index][key] = Array.isArray(value) ? value[position] : value;
            }
        }
        events.plotly_afterplot?.(); return Promise.resolve();
    }
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
    window, document, Element, WeakMap, Promise, JSON, console,
    MutationObserver: class { observe() {} },
    CustomEvent: class { constructor(name, options) { this.name = name; this.detail = options.detail; } },
});
const flush = async () => { for (let index = 0; index < 8; index++) await new Promise(setImmediate); };
(async () => {
    await flush();
    window.tvSetLanguage('zh-CN'); await flush();
    assert.equal(graph.layout.title.text, '图表');
    assert.equal(graph.data[0].name, '参考');
    // A new Gradio Plotly.react render can retain identical localization
    // metadata while restoring English chrome and changing only plot data.
    graph.layout.title.text = 'Chart'; graph.layout.xaxis.title.text = 'Step';
    graph.data[0] = {name: 'Reference', x: [9, 10]};
    events.plotly_afterplot(); await flush();
    assert.equal(graph.layout.title.text, '图表');
    assert.equal(graph.layout.xaxis.title.text, '步骤');
    assert.deepEqual(graph.data[0].x, [9, 10]);
    const stableCalls = calls;
    for (let index = 0; index < 10; index++) events.plotly_afterplot();
    await flush(); assert.equal(calls, stableCalls, 'translated charts must not loop');
    // A newer language choice wins even while an older relayout is pending.
    window.tvSetLanguage('en'); await flush();
    holdNext = true; window.tvSetLanguage('zh-CN'); await flush();
    assert.equal(typeof release, 'function');
    window.tvSetLanguage('en'); release(); await flush();
    assert.equal(graph.layout.title.text, 'Chart');
    assert.equal(graph.data[0].name, 'Reference');
    // Newly introduced figure metadata must be reconciled after a pending
    // relayout, even when every field in the old metadata already matches.
    holdNext = true; window.tvSetLanguage('zh-CN'); await flush();
    graph.layout.yaxis = {title: {text: 'Recall'}};
    graph.layout.meta = {tv_i18n: {layout: {'title.text': 'Chart',
        'xaxis.title.text': 'Step', 'yaxis.title.text': 'Recall'}, traces: {'0': 'Reference'}}};
    events.plotly_afterplot(); release(); await flush();
    assert.equal(graph.layout.yaxis.title.text, '召回率');
    // Synchronous Plotly failures must not leave the translator stuck busy.
    throwNext = true; window.tvSetLanguage('en'); await flush();
    window.tvSetLanguage('en'); await flush();
    assert.equal(graph.layout.title.text, 'Chart');
    assert.equal(graph.data[0].name, 'Reference');
    assert.deepEqual(graph.data[0].x, [9, 10]);
    // Parameterized titles localize their authored prefix while preserving
    // the escaped agent argument and every original plotted identity/value.
    window.__tvTranslations['Context Window Pressure — {agent}'] = '上下文窗口压力 — {agent}';
    graph.data[0] = {name: 'Input/<img>', x: [9, 10], customdata: ['Input/<img>']};
    graph.layout.meta = {tv_i18n: {layout: {'title.text': {
        template: 'Context Window Pressure — {agent}', params: {agent: 'Input/&lt;img&gt;'}
    }}}};
    const originalData = JSON.stringify(graph.data);
    window.tvSetLanguage('zh-CN'); await flush();
    assert.equal(graph.layout.title.text, '上下文窗口压力 — Input/&lt;img&gt;');
    assert.equal(JSON.stringify(graph.data), originalData);
    window.tvSetLanguage('en'); await flush();
    assert.equal(graph.layout.title.text, 'Context Window Pressure — Input/&lt;img&gt;');
    assert.equal(JSON.stringify(graph.data), originalData);
    // Authored list labels and hover prose translate independently of literal
    // category IDs, even when those IDs happen to match a translation key.
    Object.assign(window.__tvTranslations, {
        Total: '总计', 'First Edit': '首次编辑',
        'Compared − reference: {delta} steps': '对比 − 参考：{delta} 步',
        '%{x}<br>Recall: %{y}%<extra>Reference</extra>': '%{x}<br>召回率：%{y}%<extra>参考</extra>',
    });
    const labelMetadata = {layout: {}, layout_labels: {'xaxis.ticktext': [
        {literal: 'Reference'}, 'Total', {literal: 'Total'}
    ]}, trace_labels: {'0': {text: ['First Edit', {literal: 'Reference'}],
        hovertext: [{template: 'Compared − reference: {delta} steps', params: {delta: '+5'}},
            {literal: 'Compared − reference: +5 steps'}],
        hovertemplate: '%{x}<br>Recall: %{y}%<extra>Reference</extra>'}}};
    const resetLabels = () => {
        graph.layout.xaxis.tickvals = ['Reference', 'Total', 'recorded-Total'];
        graph.layout.xaxis.ticktext = ['Reference', 'Total', 'Total'];
        graph.data[0] = {name: 'Reference', x: ['Reference', 'Total'], y: [2, 4],
            customdata: ['First Edit', 'Recall'], text: ['First Edit', 'Reference'],
            hovertext: ['Compared − reference: +5 steps', 'Compared − reference: +5 steps'],
            hovertemplate: '%{x}<br>Recall: %{y}%<extra>Reference</extra>'};
    };
    graph.layout.meta = {tv_i18n: labelMetadata}; resetLabels();
    const originalCoordinates = JSON.stringify({x: graph.data[0].x, y: graph.data[0].y,
        customdata: graph.data[0].customdata, ticks: graph.layout.xaxis.tickvals});
    window.tvSetLanguage('zh-CN'); await flush();
    const verifyChinese = () => {
        assert.equal(JSON.stringify(graph.layout.xaxis.ticktext), '["Reference","总计","Total"]');
        assert.equal(JSON.stringify(graph.data[0].text), '["首次编辑","Reference"]');
        assert.equal(JSON.stringify(graph.data[0].hovertext),
            '["对比 − 参考：+5 步","Compared − reference: +5 steps"]');
        assert.equal(graph.data[0].hovertemplate, '%{x}<br>召回率：%{y}%<extra>参考</extra>');
        assert.equal(graph.data[0].name, 'Reference', 'raw trace names are not implicitly owned');
        assert.equal(JSON.stringify({x: graph.data[0].x, y: graph.data[0].y,
            customdata: graph.data[0].customdata, ticks: graph.layout.xaxis.tickvals}), originalCoordinates);
    };
    verifyChinese();
    const arrayCalls = calls;
    for (let index = 0; index < 10; index++) events.plotly_afterplot();
    await flush(); assert.equal(calls, arrayCalls, 'equal label arrays must not loop');
    resetLabels(); events.plotly_afterplot(); await flush(); verifyChinese();
    window.tvSetLanguage('en'); await flush();
    assert.equal(JSON.stringify(graph.layout.xaxis.ticktext), '["Reference","Total","Total"]');
    assert.equal(JSON.stringify(graph.data[0].text), '["First Edit","Reference"]');
    assert.equal(graph.data[0].hovertemplate, '%{x}<br>Recall: %{y}%<extra>Reference</extra>');
    holdNext = true; window.tvSetLanguage('zh-CN'); await flush();
    window.tvSetLanguage('en'); release(); await flush();
    assert.equal(JSON.stringify(graph.data[0].text), '["First Edit","Reference"]');
    window.tvSetLanguage('zh-CN'); await flush(); verifyChinese();
    console.log('plot localization state and race checks passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def _run_frontend_test(source, marker):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the isolated frontend unit test")
    script = Path(__file__).resolve().parents[1] / "agent_trajviz" / "i18n" / "ui.js"
    result = subprocess.run([node, "-e", source, str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker in result.stdout


def test_plot_updates_and_latest_language_win_without_recomputing_data():
    _run_frontend_test(_PLOT_TEST, "plot localization state and race checks passed")


_DOM_HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Text {
    constructor(value) { this.nodeType = 3; this.textContent = value; this.parentElement = null; }
    replaceWith(node) {
        const nodes = this.parentElement.childNodes;
        nodes[nodes.indexOf(this)] = node; node.parentElement = this.parentElement;
    }
}
class Element {
    constructor(tag, attributes = {}, children = []) {
        this.nodeType = 1; this.tag = tag; this.attributes = attributes;
        this.childNodes = children; for (const node of children) node.parentElement = this;
    }
    get textContent() { return this.childNodes.map(node => node.textContent).join(''); }
    set textContent(value) {
        this.writes = (this.writes || 0) + 1;
        this.childNodes = [new Text(value)]; this.childNodes[0].parentElement = this;
    }
    getAttribute(name) { return this.attributes[name] ?? null; }
    setAttribute(name, value) { this.attributes[name] = value; }
    matches(selector) {
        if (selector === '[data-tv-i18n]') return 'data-tv-i18n' in this.attributes;
        if (selector === 'pre, code, textarea, .analysis-chatbot, .wf-preview')
            return ['pre', 'code', 'textarea'].includes(this.tag)
                || (this.attributes.class || '').split(' ').some(name => ['analysis-chatbot', 'wf-preview'].includes(name));
        if (selector === 'button[aria-dropeffect="copy"]')
            return this.tag === 'button' && this.attributes['aria-dropeffect'] === 'copy';
        if (selector === 'input[type="file"][data-testid="file-upload"]')
            return this.tag === 'input' && this.attributes.type === 'file'
                && this.attributes['data-testid'] === 'file-upload';
        if (selector === '.wrap, .wrap .or') return (this.attributes.class || '').split(' ').includes('wrap')
            || (this.attributes.class || '').split(' ').includes('or') && this.parentElement?.attributes.class === 'wrap';
        return false;
    }
    closest(selector) { for (let node = this; node; node = node.parentElement)
        if (node.matches(selector)) return node; return null; }
    querySelectorAll(selector) {
        const result = [];
        for (const node of this.childNodes) if (node instanceof Element) {
            if (node.matches(selector)) result.push(node);
            result.push(...node.querySelectorAll(selector));
        }
        return result;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
"""


_UPLOAD_TEST = _DOM_HARNESS + r"""
const input = new Element('input', {type: 'file', 'data-testid': 'file-upload', 'aria-label': 'File upload'});
input.value = 'Drop File Here.json';
const wrapper = new Element('div', {class: 'wrap'}, [new Text('将文件拖放到此处 '),
    new Element('span', {class: 'or'}, [new Text('- 或 -')]), new Text(' 点击上传')]);
// Filenames can equal catalog phrases. Even one inside the upload button
// must remain data because it is not fixed dropzone prose in .wrap.
const filename = new Element('span', {class: 'stem'}, [new Text('Click to Upload')]);
const uploader = new Element('button', {'aria-dropeffect': 'copy'}, [wrapper, filename, input]);
const arbitrary = new Element('div', {}, [new Text('将文件拖放到此处 Click to Upload - or -')]);
const unrelatedButton = new Element('button', {'aria-dropeffect': 'copy'}, [
    new Element('div', {class: 'wrap'}, [new Text('Drop File Here')])]);
const body = new Element('body', {}, [uploader, arbitrary, unrelatedButton]);
const document = {body, readyState: 'complete', documentElement: {lang: 'zh-CN'},
    createElement: tag => new Element(tag), querySelector: () => null, dispatchEvent: () => {}};
const window = {__tvTranslations: {'Drop File Here': '将文件拖放到此处',
    'Click to Upload': '点击上传', '- or -': '- 或 -', 'File upload': '上传文件',
    'Click to upload or drop files': '点击上传或拖放文件'}};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
    window, document, Element, WeakMap, Promise, JSON, console,
    MutationObserver: class { observe() {} },
    CustomEvent: class { constructor(name, options) { this.name = name; this.detail = options.detail; } },
});
window.tvSetLanguage('en');
assert.equal(wrapper.textContent, 'Drop File Here- or -Click to Upload');
assert.equal(uploader.getAttribute('aria-label'), 'Click to upload or drop files');
window.tvSetLanguage('zh-CN');
assert.equal(wrapper.textContent, '将文件拖放到此处- 或 -点击上传');
assert.equal(uploader.getAttribute('aria-label'), '点击上传或拖放文件');
assert.equal(input.getAttribute('aria-label'), '上传文件');
assert.equal(input.value, 'Drop File Here.json');
assert.equal(filename.textContent, 'Click to Upload');
assert.equal(filename.getAttribute('data-tv-i18n'), null);
assert.equal(arbitrary.textContent, '将文件拖放到此处 Click to Upload - or -');
assert.equal(unrelatedButton.textContent, 'Drop File Here');
assert.equal(unrelatedButton.getAttribute('data-tv-i18n'), null);
window.tvSetLanguage('en');
assert.equal(wrapper.textContent, 'Drop File Here- or -Click to Upload');
assert.equal(filename.textContent, 'Click to Upload');
console.log('scoped upload prose preserves filenames and evidence');
"""


def test_native_upload_prose_aliases_preserve_filenames_and_unmarked_content():
    _run_frontend_test(_UPLOAD_TEST, "scoped upload prose preserves filenames and evidence")


_MUTATION_TEST = _DOM_HARNESS + r"""
const key = 'Load complete: {count} steps';
const owned = new Element('span', {'data-tv-i18n': key, 'data-tv-i18n-args': '{"count":3}'},
    [new Text('Load complete: 3 steps')]);
const log = new Element('div', {}, [new Text('Load complete: 3 steps')]);
const protectedMarkers = ['pre', 'code', 'textarea', 'analysis-chatbot', 'wf-preview'].map(kind => {
    const marker = new Element('span', {'data-tv-i18n': key, 'data-tv-i18n-args': '{"count":3}'},
        [new Text('Load complete: 3 steps')]);
    const wrapper = ['pre', 'code', 'textarea'].includes(kind)
        ? new Element(kind, {}, [marker]) : new Element('div', {class: kind}, [marker]);
    return {marker, wrapper};
});
const body = new Element('body', {}, [owned, log, ...protectedMarkers.map(item => item.wrapper)]);
const document = {body, readyState: 'complete', documentElement: {lang: 'en'},
    createElement: tag => new Element(tag), querySelector: () => null, dispatchEvent: () => {}};
const window = {__tvTranslations: {[key]: '加载完成：{count} 个步骤'}};
let observerCallback = null, observerOptions = null;
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
    window, document, Element, WeakMap, Promise, JSON, console,
    MutationObserver: class {
        constructor(callback) { observerCallback = callback; }
        observe(target, options) { assert.equal(target, body); observerOptions = options; }
    },
    CustomEvent: class { constructor(name, options) { this.name = name; this.detail = options.detail; } },
});
assert.equal(observerOptions.characterData, true);
assert.equal(observerOptions.childList, true);
assert.equal(observerOptions.subtree, true);
window.tvSetLanguage('zh-CN');
assert.equal(owned.textContent, '加载完成：3 个步骤');
assert.equal(log.textContent, 'Load complete: 3 steps');
// Gradio can change an existing text node without adding any elements.
owned.setAttribute('data-tv-i18n-args', '{"count":5}');
owned.childNodes[0].textContent = 'Load complete: 5 steps';
observerCallback([{type: 'characterData', target: owned.childNodes[0], addedNodes: []}]);
assert.equal(owned.textContent, '加载完成：5 个步骤');
// It can instead replace only the text child of an existing marked span.
owned.setAttribute('data-tv-i18n-args', '{"count":7}');
const fresh = new Text('Load complete: 7 steps'); fresh.parentElement = owned;
owned.childNodes = [fresh];
observerCallback([{type: 'childList', target: owned, addedNodes: [fresh]}]);
assert.equal(owned.textContent, '加载完成：7 个步骤');
// The observer's own translated write must settle without rewriting again.
const stableWrites = owned.writes;
for (let index = 0; index < 10; index++)
    observerCallback([{type: 'characterData', target: owned.childNodes[0], addedNodes: []}]);
assert.equal(owned.writes, stableWrites);
// Similar text in logs stays literal, including when it changes in place.
log.childNodes[0].textContent = 'Load complete: 9 steps';
observerCallback([{type: 'characterData', target: log.childNodes[0], addedNodes: []}]);
assert.equal(log.textContent, 'Load complete: 9 steps');
// Even an explicit-looking marker must never rewrite source code, a raw
// workflow preview, editable text, or a model response.
for (const {marker} of protectedMarkers) {
    observerCallback([{type: 'characterData', target: marker.childNodes[0], addedNodes: []}]);
    assert.equal(marker.textContent, 'Load complete: 3 steps');
    assert.equal(marker.writes, undefined);
}
window.tvSetLanguage('en');
assert.equal(owned.textContent, 'Load complete: 7 steps');
assert.equal(log.textContent, 'Load complete: 9 steps');
console.log('in-place owned updates translate without touching logs or looping');
"""


def test_in_place_html_text_updates_localize_without_rewriting_logs_or_code():
    _run_frontend_test(_MUTATION_TEST, "in-place owned updates translate without touching logs or looping")


_ATTRIBUTE_MUTATION_TEST = _DOM_HARNESS + r"""
const key = 'Load complete: {count} steps';
const attributes = (field, count) => ({'data-tv-i18n': key, 'data-tv-i18n-args': JSON.stringify({count}),
    'data-tv-i18n-en': `Load complete: ${count} steps`, 'data-tv-i18n-attr': field,
    [field]: `Load complete: ${count} steps`});
const child = new Element('span', {'data-tv-i18n': key, 'data-tv-i18n-args': '{"count":3}'},
    [new Text('Load complete: 3 steps')]);
const title = new Element('span', attributes('title', 3), [child]);
const accessible = new Element('button', attributes('aria-label', 3), [new Text('raw button text')]);
const help = new Element('span', attributes('data-help', 3), [new Text('raw help text')]);
const raw = new Element('span', {title: 'Reference', 'aria-label': 'Reference', 'data-help': 'Reference'},
    [new Text('Reference')]);
const protectedMarkers = ['pre', 'code', 'textarea', 'analysis-chatbot', 'wf-preview'].map(kind => {
    const marker = new Element('span', attributes('title', 3), [new Text('Load complete: 3 steps')]);
    const wrapper = ['pre', 'code', 'textarea'].includes(kind)
        ? new Element(kind, {}, [marker]) : new Element('div', {class: kind}, [marker]);
    return {marker, wrapper};
});
const body = new Element('body', {}, [title, accessible, help, raw, ...protectedMarkers.map(item => item.wrapper)]);
const document = {body, readyState: 'complete', documentElement: {lang: 'en'},
    createElement: tag => new Element(tag), querySelector: () => null, dispatchEvent: () => {}};
const window = {__tvTranslations: {[key]: '加载完成：{count} 个步骤', Reference: '参考'}};
let observerCallback = null, observerOptions = null, observing = false, writes = 0;
const pending = [];
const originalSet = Element.prototype.setAttribute;
Element.prototype.setAttribute = function(name, value) {
    writes++; originalSet.call(this, name, value);
    if (observing && observerOptions.attributeFilter.includes(name)) {
        pending.push({type: 'attributes', target: this, attributeName: name, addedNodes: []});
    }
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
    window, document, Element, WeakMap, Promise, JSON, console,
    MutationObserver: class {
        constructor(callback) { observerCallback = callback; }
        observe(target, options) {
            assert.equal(target, body); observerOptions = options; observing = true;
        }
    },
    CustomEvent: class { constructor(name, options) { this.name = name; this.detail = options.detail; } },
});
assert.equal(observerOptions.attributes, true);
assert.equal(JSON.stringify(observerOptions.attributeFilter), JSON.stringify([
    'data-tv-i18n', 'data-tv-i18n-args', 'data-tv-i18n-en', 'data-tv-i18n-attr',
    'title', 'aria-label', 'data-help']));
const settle = () => {
    for (let batch = 0; pending.length && batch < 20; batch++) observerCallback(pending.splice(0));
    assert.equal(pending.length, 0, 'translated attributes must settle without an observer loop');
};
window.tvSetLanguage('zh-CN'); settle();
assert.equal(title.getAttribute('title'), '加载完成：3 个步骤');
assert.equal(child.textContent, '加载完成：3 个步骤');
assert.equal(accessible.getAttribute('aria-label'), '加载完成：3 个步骤');
assert.equal(help.getAttribute('data-help'), '加载完成：3 个步骤');
assert.equal(accessible.textContent, 'raw button text');
assert.equal(help.textContent, 'raw help text');
// An outer tooltip marker and its inner text marker refresh independently.
title.setAttribute('data-tv-i18n-args', '{"count":7}');
title.setAttribute('data-tv-i18n-en', 'Load complete: 7 steps');
title.setAttribute('title', 'Load complete: 7 steps');
child.setAttribute('data-tv-i18n-args', '{"count":5}');
child.setAttribute('data-tv-i18n-en', 'Load complete: 5 steps');
settle();
assert.equal(title.getAttribute('title'), '加载完成：7 个步骤');
assert.equal(child.textContent, '加载完成：5 个步骤');
// A renderer can restore English attributes without replacing any elements.
accessible.setAttribute('aria-label', 'Load complete: 3 steps');
help.setAttribute('data-help', 'Load complete: 3 steps'); settle();
assert.equal(accessible.getAttribute('aria-label'), '加载完成：3 个步骤');
assert.equal(help.getAttribute('data-help'), '加载完成：3 个步骤');
child.setAttribute('data-tv-i18n', 'Reference');
child.setAttribute('data-tv-i18n-args', '{}');
child.setAttribute('data-tv-i18n-en', 'Reference'); settle();
assert.equal(child.textContent, '参考');
assert.equal(title.getAttribute('title'), '加载完成：7 个步骤');
const stableWrites = writes;
for (let index = 0; index < 10; index++) observerCallback([
    {type: 'attributes', target: title, attributeName: 'title', addedNodes: []},
    {type: 'attributes', target: help, attributeName: 'data-help', addedNodes: []}]);
settle(); assert.equal(writes, stableWrites, 'already translated attributes must not be written again');
// Raw attributes and explicit-looking markers inside protected regions stay literal.
raw.setAttribute('title', 'Load complete: 9 steps');
raw.setAttribute('aria-label', 'Reference'); raw.setAttribute('data-help', 'Reference');
for (const {marker} of protectedMarkers) {
    marker.setAttribute('data-tv-i18n-args', '{"count":9}');
    marker.setAttribute('title', 'Load complete: 9 steps');
}
const rawWrites = writes; settle(); assert.equal(writes, rawWrites);
assert.equal(raw.getAttribute('title'), 'Load complete: 9 steps');
assert.equal(raw.getAttribute('aria-label'), 'Reference');
assert.equal(raw.getAttribute('data-help'), 'Reference');
assert.equal(raw.textContent, 'Reference');
for (const {marker} of protectedMarkers) {
    assert.equal(marker.getAttribute('title'), 'Load complete: 9 steps');
    assert.equal(marker.textContent, 'Load complete: 3 steps');
}
window.tvSetLanguage('en'); settle();
assert.equal(title.getAttribute('title'), 'Load complete: 7 steps');
assert.equal(child.textContent, 'Reference');
assert.equal(raw.getAttribute('title'), 'Load complete: 9 steps');
console.log('owned attribute refresh preserves nested labels, raw content and protected regions');
"""


def test_in_place_owned_attribute_and_argument_updates_settle_without_touching_raw_content():
    _run_frontend_test(_ATTRIBUTE_MUTATION_TEST,
                       "owned attribute refresh preserves nested labels, raw content and protected regions")
