"""Charts rendered inside hidden sections must fit when those sections open."""

from __future__ import annotations

import shutil
import subprocess

import pytest


def test_revealed_charts_resize_to_their_cards_without_changing_height():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to verify the chart visibility handler")
    from agent_trajviz.insight.insight import build_ui

    app = build_ui()
    try:
        scripts = [event["js"] for event in app.get_config_file()["dependencies"]
                   if "tvResizeVisibleCharts" in (event.get("js") or "")]
        assert len(scripts) == 1
    finally:
        app.close()

    source = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const timers = new Map();
const listeners = {};
const calls = [];
let timerId = 0, mutated;
function chart(width, height = 380, visible = true) {
    return {
        layout: {autosize: true, height},
        _fullLayout: {width: 700, height},
        clientWidth: width,
        visible,
        style: {height: height + 'px'},
        getClientRects() { return this.visible ? [{}] : []; }
    };
}
const left = chart(509), right = chart(509);
const timeline = chart(1036, 1300), hidden = chart(509, 380, false);
const fixed = chart(450); fixed.layout.autosize = false;
const plots = [left, right, timeline, hidden, fixed];
const window = {
    addEventListener(type, fn) { listeners[type] = fn; },
    Plotly: {Plots: {resize(plot) {
        calls.push(plot);
        return Promise.resolve().then(() => { plot._fullLayout.width = plot.clientWidth; });
    }}}
};
const document = {
    documentElement: {style: {}, classList: {remove() {}}},
    body: {classList: {remove() {}}},
    querySelector() { return null; },
    querySelectorAll(selector) { return selector === '.js-plotly-plot' ? plots : []; },
    getElementById() { return null; },
    addEventListener() {}
};
const context = {
    window, document,
    MutationObserver: class {
        constructor(callback) { mutated = callback; }
        observe() {}
    },
    setTimeout(fn, delay) { timers.set(++timerId, {fn, delay}); return timerId; },
    clearTimeout(id) { timers.delete(id); }
};
vm.runInNewContext('(' + process.argv[1] + ')()', context);
async function flush() { for (let i = 0; i < 5; i++) await Promise.resolve(); }
function runScheduled() {
    const job = [...timers].find(([, timer]) => timer.delay === 60);
    assert.ok(job, 'visibility and viewport changes schedule chart reconciliation');
    timers.delete(job[0]); job[1].fn();
}
(async () => {
    // A section reveal arrives through the same mutation scheduler as Gradio.
    mutated();
    const firstPass = [...timers].find(([, timer]) => timer.delay === 60)[0];
    for (let update = 0; update < 20; update++) mutated();
    assert.ok(timers.has(firstPass), 'continuous loading updates must not postpone chart binding forever');
    runScheduled();
    assert.deepEqual(calls, [left, right, timeline]);
    // Plotly resize is asynchronous: mutations caused by its own redraw must
    // not start duplicate work for the same graph.
    window.tvResizeVisibleCharts();
    assert.equal(calls.length, 3);
    await flush();
    assert.deepEqual(plots.slice(0, 3).map(p => p._fullLayout.width), [509, 509, 1036]);
    assert.equal(timeline.layout.height, 1300);
    assert.equal(timeline.style.height, '1300px');
    window.tvResizeVisibleCharts();
    assert.equal(calls.length, 3, 'settled charts do not cause a resize/mutation loop');
    // A viewport change reconciles new card widths; hidden charts wait until
    // opened rather than inheriting a zero or stale hidden-container width.
    left.clientWidth = right.clientWidth = 420;
    listeners.resize(); runScheduled(); await flush();
    assert.deepEqual(calls.slice(3), [left, right]);
    hidden.visible = true;
    mutated(); runScheduled(); await flush();
    assert.equal(calls.at(-1), hidden);
    assert.equal(hidden._fullLayout.width, 509);
    assert.equal(calls.includes(fixed), false, 'explicitly fixed figures remain fixed');
    console.log('visible charts fit their cards and preserve timeline height');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run([node, "-e", source, scripts[0]], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "visible charts fit their cards" in result.stdout
