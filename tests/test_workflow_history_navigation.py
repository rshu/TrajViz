"""Execute the dashboard's shipped navigation JS against browser history semantics."""

from __future__ import annotations

import shutil
import subprocess
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def navigation_js():
    from agent_trajviz.insight import insight

    with patch.object(insight, "load_env_files"):
        app = insight.build_ui()
    try:
        scripts = [fn.js for fn in app.fns.values() if fn.js and "window.tvGotoWorkflowStep = function" in fn.js]
        assert len(scripts) == 1
        source = scripts[0]
        return source[source.index("window.tvMainTabButtons = function") : source.index("/* Test contract")]
    finally:
        app.close()


_BROWSER = r"""
const assert = require('node:assert/strict');
let activeTab = 'Patterns';
let entries = [];
let position = 0;
let selected = [];
const captureClicks = [];
const popstates = [];
const window = {
  location: new URL('https://fixture.invalid/proxy/dashboard?capture=public'),
  addEventListener: (name, callback) => { if (name === 'popstate') popstates.push(callback); },
  tvSelectWorkflowCard: (card, options) => {
    assert.equal(options.pushHistory, false);
    selected.push(card.index);
  }
};
const wrapper = {querySelectorAll: selector => {
  assert.ok(selector.includes('.overflow-dropdown button'));
  return tabs.concat(measurement, nestedTab);
}};
const mainTabsRoot = {querySelector: selector => {
  assert.equal(selector, ':scope > .tab-wrapper');
  return wrapper;
}};
const tabs = ['Overview', 'Patterns', 'Comparison', 'Workflow'].map(label => ({
  textContent: label,
  overflow: false,
  getAttribute(name) { return name === 'aria-selected' && !this.overflow ? String(activeTab === label) : null; },
  classList: {contains: name => name === 'selected' && activeTab === label},
  closest(selector) {
    if (selector === '.tab-wrapper') return wrapper;
    if (selector === '[aria-hidden="true"]') return null;
    return this;
  },
  click() {
    // Browser capture listeners run BEFORE Gradio changes aria-selected.
    captureClicks.forEach(callback => callback({target: this}));
    if (this.fail) throw new Error('fixture tab click failed');
    activeTab = label;
  }
}));
const measurement = {
  textContent: 'Workflow', classList: {contains: () => true},
  getAttribute: () => 'true',
  closest: selector => selector === '[aria-hidden="true"]' ? {} : wrapper,
  click() { throw new Error('measurement clone clicked'); }
};
const nestedTab = {
  textContent: 'Workflow', classList: {contains: () => true},
  getAttribute: () => 'true',
  closest(selector) { return selector === 'button' ? this : {}; },
  click() { captureClicks.forEach(callback => callback({target: this})); }
};
const document = {
  querySelectorAll: () => [],
  querySelector: () => null,
  getElementById: id => {
    if (id === 'main-tabs') return mainTabsRoot;
    // Workflow cards mount only after the real Gradio tab changes.
    if (activeTab !== 'Workflow') return null;
    return {index: Number(id.replace('wf-card-', '')),
      classList: {add() {}, remove() {}}, scrollIntoView() {}};
  },
  addEventListener: (name, callback, capture) => {
    if (name === 'click') {
      assert.equal(capture, true);
      captureClicks.push(callback);
    }
  }
};
const history = {
  get state() { return entries[position].state; },
  pushState(state, unused, url) {
    entries = entries.slice(0, position + 1);
    entries.push({state, url: new URL(url, window.location).href});
    position += 1;
    window.location = new URL(entries[position].url);
  },
  replaceState(state, unused, url) {
    entries[position] = {state, url: new URL(url, window.location).href};
    window.location = new URL(entries[position].url);
  },
  go(delta) {
    position += delta;
    assert.ok(position >= 0 && position < entries.length);
    window.location = new URL(entries[position].url);
    popstates.forEach(callback => callback({state: this.state}));
  }
};
function reset(tab, hash = '', state = null) {
  activeTab = tab;
  window.location = new URL('https://fixture.invalid/proxy/dashboard?capture=public' + hash);
  entries = [{state, url: window.location.href}];
  position = 0;
  selected = [];
}
function setTimeout(callback) { callback(); return 1; }
function clearTimeout() {}
reset('Patterns');
"""

_SCENARIOS = {
    "pattern_back_forward": r"""
window.tvGotoWorkflowStep(2);
assert.equal(activeTab, 'Workflow');
assert.equal(history.state.tvTab, 'Workflow');
assert.deepEqual(entries.slice(-2).map(entry => entry.state.tvTab), ['Patterns', 'Workflow']);
assert.equal(new URL(entries[position - 1].url).hash, '');
const length = entries.length;
history.go(-1);
assert.equal(activeTab, 'Patterns');
assert.equal(window.location.hash, '');
history.go(1);
assert.equal(activeTab, 'Workflow');
assert.equal(window.location.hash, '#step-2');
assert.equal(entries.length, length);
assert.equal(selected.at(-1), 2);
assert.equal(window.location.pathname, '/proxy/dashboard');
assert.equal(window.location.search, '?capture=public');
""",
    "stale_step_hash_is_not_a_non_workflow_return_point": r"""
reset('Comparison', '#step-19', {tvTab: 'Workflow'});
window.tvGotoWorkflowStep(5);
assert.equal(new URL(entries[position - 1].url).hash, '');
assert.equal(entries[position - 1].state.tvTab, 'Comparison');
history.go(-1);
assert.equal(activeTab, 'Comparison');
assert.equal(window.location.hash, '');
history.go(1);
assert.equal(activeTab, 'Workflow');
assert.equal(window.location.hash, '#step-5');
""",
    "step_navigation_and_repeat_do_not_insert_extra_history": r"""
reset('Workflow', '#step-2', {tvTab: 'Workflow'});
window.tvGotoWorkflowStep(7);
assert.equal(entries.length, 2);
window.tvGotoWorkflowStep(7);
assert.equal(entries.length, 2);
history.go(-1);
assert.equal(window.location.hash, '#step-2');
assert.equal(selected.at(-1), 2);
history.go(1);
assert.equal(window.location.hash, '#step-7');
assert.equal(selected.at(-1), 7);
""",
    "manual_workflow_click_keeps_return_and_restore_never_pushes": r"""
reset('Overview', '#step-8', {tvTab: 'Workflow'});
tabs.find(tab => tab.textContent === 'Workflow').click();
assert.equal(activeTab, 'Workflow');
assert.equal(history.state.tvTab, 'Overview');
assert.equal(window.location.hash, '');
// Workflow card selection normally records the selected step after mounting.
history.pushState({tvTab: 'Workflow'}, '', window.tvWorkflowStepUrl(0));
const length = entries.length;
history.go(-1);
assert.equal(activeTab, 'Overview');
history.go(1);
assert.equal(activeTab, 'Workflow');
assert.equal(entries.length, length);
activeTab = 'Patterns';
window.tvGotoWorkflowStep(0, {restore: true});
assert.equal(activeTab, 'Workflow');
assert.equal(entries.length, length);
assert.equal(history.state.tvTab, 'Workflow');
""",
    "mobile_overflow_jump_and_history_use_real_main_tab": r"""
const workflow = tabs.find(tab => tab.textContent === 'Workflow');
workflow.overflow = true;
const comparison = tabs.find(tab => tab.textContent === 'Comparison');
comparison.overflow = true;
assert.deepEqual(window.tvMainTabButtons(), tabs);
window.tvGotoWorkflowStep(7);
assert.equal(activeTab, 'Workflow');
assert.equal(window.tvActiveMainTab(), 'Workflow');
assert.equal(selected.at(-1), 7);
const length = entries.length;
history.go(-1);
assert.equal(activeTab, 'Patterns');
history.go(1);
assert.equal(activeTab, 'Workflow');
assert.equal(selected.at(-1), 7);
assert.equal(entries.length, length);
reset('Comparison');
workflow.click();
assert.equal(history.state.tvTab, 'Comparison');
history.pushState({tvTab: 'Workflow'}, '', window.tvWorkflowStepUrl(3));
history.go(-1);
assert.equal(activeTab, 'Comparison');
assert.equal(window.tvActiveMainTab(), 'Comparison');
history.go(1);
assert.equal(activeTab, 'Workflow');
assert.equal(selected.at(-1), 3);
""",
    "nested_tabs_do_not_change_main_history": r"""
nestedTab.click();
assert.equal(entries.length, 1);
assert.equal(history.state, null);
assert.equal(activeTab, 'Patterns');
""",
    "programmatic_suppression_resets_when_click_throws": r"""
const workflow = tabs.find(tab => tab.textContent === 'Workflow');
workflow.fail = true;
assert.throws(() => window.tvGotoWorkflowStep(4), /fixture tab click failed/);
assert.equal(window.__tvProgrammaticWorkflowTab, undefined);
workflow.fail = false;
reset('Comparison');
workflow.click();
assert.equal(history.state.tvTab, 'Comparison');
assert.equal(activeTab, 'Workflow');
""",
}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required to execute browser navigation behavior")
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_browser_history_navigation(navigation_js, scenario):
    result = subprocess.run(
        ["node", "-e", _BROWSER + navigation_js + _SCENARIOS[scenario]],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
