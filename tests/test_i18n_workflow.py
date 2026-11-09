"""Workflow filter summaries localize without changing their canonical chips."""

import json
import shutil
import subprocess

import pytest

from agent_trajviz.i18n import catalog
from agent_trajviz.insight.ui.workflow_tab import WORKFLOW_JS


_HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = SOURCE;
const translations = TRANSLATIONS;
let language = 'zh-CN';
const chips = [
    {dataset: {filterGroup: 'role', filter: 'Assistant'}, textContent: 'Assistant', active: true},
    {dataset: {filterGroup: 'role', filter: 'User'}, textContent: 'User', active: true},
    {dataset: {filterGroup: 'feature', filter: 'Errors'}, textContent: 'Errors', active: true},
    {dataset: {filterGroup: 'feature', filter: 'Tool Calls'}, textContent: 'Tool Calls', active: true},
    {dataset: {filterGroup: 'feature', filter: 'Reasoning'}, textContent: 'Reasoning', active: false},
    {dataset: {filterGroup: 'feature', filter: 'All'}, textContent: 'All', active: false},
    {dataset: {filterGroup: 'agent', filter: 'agent:literal-one'}, textContent: 'Input', active: true},
    {dataset: {filterGroup: 'agent', filter: 'agent:literal-two'}, textContent: 'Raw <img>', active: true},
    {dataset: {filterGroup: 'agent', filter: 'agent:All'}, textContent: 'All agents', active: false},
];
const query = {textContent: ''};
const wrapper = {
    querySelectorAll(selector) {
        const group = selector.match(/data-filter-group="([^"]+)"/)[1];
        return chips.filter(chip => chip.dataset.filterGroup === group
            && (!selector.includes('.chip-active') || chip.active));
    },
    querySelector(selector) {
        if (selector === '#wf-filter-query') return query;
        assert.equal(selector, '[data-filter-group="agent"]');
        return chips.find(chip => chip.dataset.filterGroup === 'agent');
    }
};
// The summary is a sibling of the filter bar. Looking inside the bar is a
// regression even if querying the document would happen to find another one.
const bar = {parentElement: wrapper,
    querySelector() { throw new Error('query summary is outside the filter bar'); },
    querySelectorAll() { throw new Error('initial refresh used filter bar instead of its parent'); }};
const listeners = [];
const document = {
    getElementById(id) { assert.equal(id, 'wf-filter-bar'); return bar; },
    querySelectorAll(selector) { assert.equal(selector, '#wf-filter-bar'); return [bar]; },
    addEventListener(name, callback) { assert.equal(name, 'trajviz:language'); listeners.push(callback); }
};
const window = {tvTranslate: key => language === 'zh-CN' ? translations[key] || key : key};
const context = vm.createContext({window, document});
const originalChips = JSON.stringify(chips);
vm.runInContext(source, context);
// Initial mounting after a saved Chinese preference already took effect
// must produce a Chinese summary without waiting for another language event.
assert.equal(query.textContent, translations.Role + ': ' + translations.Assistant + translations[' or ']
    + translations.User + ' · ' + translations['Step feature'] + ': ' + translations.Errors
    + translations[' or '] + translations['Tool Calls'] + ' · ' + translations.Agent + ': Input'
    + translations[' or '] + 'Raw <img>');
assert.equal(JSON.stringify(chips), originalChips);
assert.equal(listeners.length, 1);
language = 'en'; listeners[0]();
assert.equal(query.textContent, 'Role: Assistant or User · Step feature: Errors or Tool Calls · Agent: Input or Raw <img>');
assert.equal(JSON.stringify(chips), originalChips);
language = 'zh-CN'; listeners[0]();
assert.ok(query.textContent.includes(translations.Assistant));
assert.ok(query.textContent.includes('Input') && !query.textContent.includes(translations.Input));
assert.equal(JSON.stringify(chips), originalChips);
// An All sentinel is translated as interface text, while retaining its exact
// machine token. Specific agent names remain literal in the preceding case.
for (const chip of chips) {
    if (chip.dataset.filterGroup === 'feature') chip.active = chip.dataset.filter === 'All';
    if (chip.dataset.filterGroup === 'agent') chip.active = chip.dataset.filter === 'agent:All';
}
const sentinelState = JSON.stringify(chips);
listeners[0]();
assert.ok(query.textContent.endsWith(translations['Step feature'] + ': ' + translations.All
    + ' · ' + translations.Agent + ': ' + translations.All));
assert.equal(JSON.stringify(chips), sentinelState);
language = 'en'; listeners[0]();
assert.ok(query.textContent.endsWith('Step feature: All · Agent: All'));
assert.equal(JSON.stringify(chips), sentinelState);
// The workflow initialization can run again after a re-render; the language
// listener remains single-bound and the initial sibling lookup still works.
vm.runInContext(source, context);
assert.equal(listeners.length, 1);
assert.ok(query.textContent.endsWith('Step feature: All · Agent: All'));
console.log('initial and refreshed workflow summaries preserve canonical chip data');
"""


def test_initial_chinese_workflow_summary_uses_sibling_scope_and_preserves_filters():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the isolated frontend unit test")
    start = WORKFLOW_JS.index("window.__updateWorkflowFilterQuery =")
    end = WORKFLOW_JS.index("/* Pure chip state machine", start)
    source = WORKFLOW_JS[start:end]
    labels = ("Role", "Assistant", "User", "Step feature", "Errors", "Tool Calls", "Agent", "All", " or ", "Input")
    translations = {key: catalog()[key] for key in labels}
    harness = _HARNESS.replace("SOURCE", json.dumps(source)).replace("TRANSLATIONS", json.dumps(translations))
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "initial and refreshed workflow summaries preserve canonical chip data" in result.stdout
