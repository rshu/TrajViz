"""Serve the composed dashboard and exercise its shipped browser handlers.

Node executes the JavaScript returned by Gradio's actual /config response, using
native-value/event and Plotly doubles. The HTTP load test uses real uploads and
Gradio sessions; browser-only private callbacks then run through process_api
against those same sessions. These checks do not claim browser layout coverage.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import pytest
from gradio_client import Client, handle_file

from agent_trajviz.insight import insight

_FIXTURES = Path(__file__).parent / "fixtures"


def _function(app, name):
    functions = [fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == name]
    assert functions, f"Missing composed app callback: {name}"
    return functions[0]


@pytest.fixture(scope="module")
def served_prompt_app():
    with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}), patch.object(insight, "load_env_files"):
        app = insight.build_ui()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        app.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                   quiet=True, inbrowser=False, share=False)
        try:
            with urllib.request.urlopen(app.local_url.rstrip("/") + "/config", timeout=30) as response:
                config = json.load(response)
            yield app, config
        finally:
            app.close()


def _load_http(app, client, path):
    fn = _function(app, "_do_load")
    values = {"Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(path)),
              "Format": "", "Captured session": None}
    inputs = [values.get(component.label, component.value) for component in fn.inputs
              if not isinstance(component, (gr.State, gr.BrowserState))]
    return client.predict(*inputs, api_name="/" + fn.api_name)


def _private_callback(app, client, name, *, selection=None, language="en"):
    fn = _function(app, name)
    inputs = []
    for component in fn.inputs:
        if isinstance(component, gr.State):
            inputs.append(None)
        elif isinstance(component, gr.Radio):
            inputs.append(language)
        else:
            inputs.append(selection)
    values = asyncio.run(app.process_api(
        fn, inputs, state=app.state_holder[client.session_hash],
        request=gr.Request(session_hash=client.session_hash), session_hash=client.session_hash))["data"]
    return [value.model_dump() if hasattr(value, "model_dump") else value for value in values]


def test_served_prompt_click_bridge_rebinds_without_loading_indicator_starvation(served_prompt_app):
    _app, config = served_prompt_app
    scripts = [event.get("js") for event in config["dependencies"]
               if "window.tvBindPromptEvolution = function" in (event.get("js") or "")]
    assert len(scripts) == 1
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the served Plotly bridge")
    script = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = JSON.parse(process.argv[1]);
let now = 0, serial = 0, mutations, styleWrites = 0;
const timers = new Map(), microtasks = [], clicked = [], dispatched = [];
const setTimeout = (fn, delay) => { const id=++serial; timers.set(id,{fn,at:now+delay}); return id; };
const clearTimeout = id => timers.delete(id);
function advance(ms) {
  const end=now+ms;
  for (let guard=0; guard<1000; guard++) {
    const due=[...timers].filter(([,t])=>t.at<=end).sort((a,b)=>a[1].at-b[1].at)[0];
    if (!due) { now=end; return; }
    now=due[1].at; timers.delete(due[0]); due[1].fn();
  }
  throw new Error('browser scheduling did not settle');
}
class NativeInput {
  constructor(){ this.tagName='TEXTAREA'; this.stored=''; }
  dispatchEvent(event){ dispatched.push({type:event.type,value:this.value,bubbles:event.bubbles}); }
}
Object.defineProperty(NativeInput.prototype,'value',{get(){return this.stored;},set(value){this.stored=value;}});
const input=new NativeInput();
const button={click(){clicked.push(input.value);}};
class DOMEvent { constructor(type,options){this.type=type;Object.assign(this,options);} }
function plot(){
  let listeners=[];
  return {style:new Proxy({}, {set(target,key,value){styleWrites++;target[key]=value;return true;}}),
    on(event,fn){if(event==='plotly_click')listeners.push(fn);},
    removeListener(event,fn){listeners=listeners.filter(item=>item!==fn);},
    fire(data){listeners.slice().forEach(fn=>fn(data));},wipe(){listeners=[];},count(){return listeners.length;}};
}
let current=plot(), bridgePresent=true, buttonPresent=true;
const document={documentElement:{style:{},classList:{remove(){}}},body:{classList:{remove(){}}},
  getElementById(id){return id==='prompt-evolution-chart'?{querySelector(){return current;}}:null;},
  querySelector(selector){
    if(selector.startsWith('#pc-request-bridge '))return bridgePresent?input:null;
    if(selector.startsWith('#pc-select-request button'))return buttonPresent?button:null;
    return null;
  },querySelectorAll(){return [];},addEventListener(){}};
const window={HTMLTextAreaElement:NativeInput,HTMLInputElement:NativeInput,addEventListener(){}};
const context={window,document,InputEvent:DOMEvent,Event:DOMEvent,setTimeout,clearTimeout,
  queueMicrotask:fn=>microtasks.push(fn),MutationObserver:class{constructor(fn){mutations=fn;}observe(){}}};
vm.runInNewContext('('+source+')()',context);
advance(0);
// A loading indicator emits mutations every 10 ms. The first binding must run
// at 60 ms instead of being endlessly delayed until the indicator disappears.
for(let i=0;i<7;i++){mutations();advance(10);}
assert.equal(current.count(),1);
window.tvBindPromptEvolution();window.tvBindPromptEvolution();
assert.equal(current.count(),1);
assert.equal(styleWrites,1,'rebinding must not keep mutating cursor style');
const first='claude:main:request-1', literal='claude:child:<script>literal</script>"';
current.fire({points:[{customdata:[first,'conversation']}]});
assert.equal(clicked.length,0,'the click follows native input/change events');
while(microtasks.length)microtasks.shift()();
assert.deepEqual(clicked,[first]);
assert.deepEqual(dispatched.map(event=>event.type),['input','change']);
assert.ok(dispatched.every(event=>event.bubbles && event.value===first));
current.wipe();mutations();advance(60);
assert.equal(current.count(),1,'Plotly.react dropped listeners and must be rebound');
current.fire({points:[{customdata:[literal,'tools']}]});
while(microtasks.length)microtasks.shift()();
assert.deepEqual(clicked,[first,literal]);
for(const invalid of [undefined,[],{points:[]},{points:[{}]},{points:[{customdata:[0]}]},
  {points:[{customdata:['']}]},{points:[{customdata:{id:first}}]}])current.fire(invalid);
while(microtasks.length)microtasks.shift()();
assert.equal(clicked.length,2);
bridgePresent=false;current.fire({points:[{customdata:[first]}]});
bridgePresent=true;buttonPresent=false;current.fire({points:[{customdata:[first]}]});
while(microtasks.length)microtasks.shift()();
assert.equal(clicked.length,2,'unmounted Gradio controls cannot trigger stale selection');
buttonPresent=true;current=plot();mutations();advance(60);
assert.equal(current.count(),1,'a replacement graph receives the same binding');
console.log('served prompt bridge, native events, bounded scheduling, and Plotly rebinding passed');
"""
    result = subprocess.run([node, "-e", script, json.dumps(scripts[0])], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "bounded scheduling" in result.stdout


class _RenderedHTML(HTMLParser):
    def __init__(self, value):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.pre = False
        self.literal = []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "pre":
            self.pre = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.pre = False

    def handle_data(self, data):
        if self.pre:
            self.literal.append(data)


def test_served_upload_family_selection_language_and_session_switch_share_real_state(served_prompt_app, tmp_path):
    app, config = served_prompt_app
    components = {component["id"]: component for component in config["components"]}
    prompt_functions = [event for event in config["dependencies"] if event.get("backend_fn") and any(
        components.get(identity, {}).get("props", {}).get("elem_id") == "prompt-request-json"
        for identity in event.get("outputs", []))]
    assert prompt_functions and all(event["trigger_mode"] == "always_last" for event in prompt_functions)
    archive = tmp_path / "native-family.zip"
    family = _FIXTURES / "claude_current"
    with zipfile.ZipFile(archive, "w") as handle:
        for source in sorted(family.rglob("*.jsonl")):
            handle.write(source, source.relative_to(family))
    first, second = Client(app.local_url, verbose=False), Client(app.local_url, verbose=False)
    _load_http(app, first, archive)
    loaded = _private_callback(app, first, "on_prompt_load")
    detail = loaded[-1]
    assert detail and detail["scope"] == "recorded_request"
    choices = loaded[0]["choices"]
    requested = choices[0][1]
    selected = _private_callback(app, first, "on_prompt_select", selection=requested)
    assert selected[-1]["id"] == requested
    chinese = _private_callback(app, first, "on_prompt_select", selection=requested, language="zh-CN")
    assert chinese[-1] == selected[-1]
    assert any("已记录的请求" in label for label, _identity in chinese[0]["choices"])
    assert _private_callback(app, second, "on_prompt_load")[-1] is None
    original = 'Rules <script>alert("literal")</script> and <img> remain source text.'
    records = [
        {"type": "session", "version": 3, "id": "browser-pi"},
        {"type": "message", "id": "s", "parentId": None,
         "message": {"role": "system", "sections": {"rules": original}, "toolsAdded": []}},
        {"type": "message", "id": "u", "parentId": "s", "message": {"role": "user", "content": "Neutral task"}},
        {"type": "message", "id": "a", "parentId": "u", "message": {"role": "assistant", "content": "Reply"}},
    ]
    pi = tmp_path / "native-<img>.jsonl"
    pi.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    _load_http(app, first, pi)
    switched = _private_callback(app, first, "on_prompt_select", selection=requested)
    assert switched[-1]["id"].startswith("pi:browser-pi:")
    assert switched[0]["value"] != requested
    parsed = _RenderedHTML(switched[3])
    assert original in "".join(parsed.literal)
    assert "script" not in parsed.tags and "img" not in parsed.tags
    assert "&lt;script&gt;" in switched[3] and "&lt;img&gt;" in switched[3]
    broken = tmp_path / "invalid.json"
    broken.write_text('{"unknown": "not a trajectory"}', encoding="utf-8")
    _load_http(app, first, broken)
    cleared = _private_callback(app, first, "on_prompt_load")
    assert cleared[-1] is None and not cleared[3] and not cleared[4]
