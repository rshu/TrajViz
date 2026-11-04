"""State-preserving dashboard language controls, using public Gradio updates."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

from . import LANGUAGES, catalog, normalize_language, translate

_STORAGE_KEY = "trajviz-ui-language-v1"


class _OwnedHTML(HTMLParser):
    """Mark only initial, app-authored HTML; never called on imported results."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.skip = 0
        self.stack: list[tuple[str, bool]] = []
        self.translations = catalog()

    def handle_starttag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())
        if tag not in ("area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"):
            self.stack.append((tag, tag in ("script", "style", "pre", "code")
                               or any(key == "data-tv-i18n" for key, _ in attrs)))
            self.skip = sum(blocked for _, blocked in self.stack)

    def handle_endtag(self, tag):
        self.parts.append(f"</{tag}>")
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        self.skip = sum(blocked for _, blocked in self.stack)

    def handle_startendtag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())

    def handle_data(self, data):
        from . import text
        key = data.strip()
        if not self.skip and key in self.translations:
            left = data[:len(data) - len(data.lstrip())]
            right = data[len(data.rstrip()):]
            self.parts.extend((left, text(key), right))
        else:
            self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")

    def handle_comment(self, data):
        self.parts.append(f"<!--{data}-->")


def language_selector():
    """Build a visible, bilingual selector without changing input data."""
    import gradio as gr
    return gr.Dropdown(choices=list(LANGUAGES), value="en", label="Language / 语言",
                       interactive=True, filterable=False, allow_custom_value=False,
                       scale=0, min_width=160, elem_id="tv-language-selector")


def bind_language(app, selector):
    """Register presentation props once, then update them per browser session.

    Input values, selections, keys, plot data and result bodies are never
    overwritten. Mutable choice lists are intentionally excluded.
    """
    import gradio as gr
    translations = catalog()
    targets = []
    specs = []
    outputs = {getattr(output, "_id", None) for fn in app.fns.values() for output in fn.outputs}
    for component in list(app.blocks.values()):
        if component is selector:
            continue
        if isinstance(component, gr.Textbox) and component.type == "password":
            # A presentation-only Gradio update can still replace a password
            # input's unsaved DOM value. Mark its authored captions locally and
            # never include the secret component in the language event outputs.
            component.elem_classes = [*(component.elem_classes or []), "tv-secret-field"]
            continue
        props = {}
        for name in ("label", "info", "placeholder"):
            value = getattr(component, name, None)
            if isinstance(value, str) and value in translations:
                props[name] = value
        if isinstance(component, gr.Button) or isinstance(component, gr.Markdown) and component._id not in outputs:
            value = component.value
            if isinstance(value, str) and value in translations:
                props["value"] = value
        if isinstance(component, gr.Dataframe) and component.headers:
            props["headers"] = tuple(component.headers)
        if isinstance(component, (gr.Dropdown, gr.Radio, gr.CheckboxGroup)) and component._id not in outputs:
            props["choices"] = tuple(component.choices)
        if isinstance(component, gr.HTML) and isinstance(component.value, str) and component.value:
            parser = _OwnedHTML()
            parser.feed(component.value)
            component.value = "".join(parser.parts)
        if props:
            targets.append(component)
            specs.append(props)

    def updates(language):
        language = normalize_language(language)
        result = []
        for spec in specs:
            props = {}
            for name, value in spec.items():
                if name == "choices":
                    props[name] = [(translate(label, language), canonical) for label, canonical in value]
                elif name == "headers":
                    props[name] = [translate(label, language) for label in value]
                else:
                    props[name] = translate(value, language)
            result.append(gr.update(**props))
        return result

    # The preference contains one nonsecret language code. A stable secret
    # keeps Gradio BrowserState readable across app restarts and dashboards.
    preference = gr.BrowserState("en", storage_key=_STORAGE_KEY, secret=_STORAGE_KEY)
    locale = gr.State("en")

    def change(language):
        language = normalize_language(language)
        return language, language, *updates(language)

    def restore(language):
        language = normalize_language(language)
        return gr.update(value=language), language, *updates(language)

    selector.change(change, inputs=selector, outputs=[locale, preference, *targets],
                    js="(language) => { window.tvSetLanguage?.(language); return language; }",
                    queue=False, api_name=False, show_progress="hidden")
    app.load(restore, inputs=preference, outputs=[selector, locale, *targets],
             queue=False, api_name=False, show_progress="hidden")
    return locale


def presentation(options: dict) -> dict:
    """Add a self-contained local translator to existing launch presentation."""
    translations = json.dumps(catalog(), ensure_ascii=False).replace("<", "\\u003c")
    script = Path(__file__).with_name("ui.js").read_text(encoding="utf-8")
    head = options.get("head", "") + f"<script>window.__tvTranslations={translations};\n{script}</script>"
    css = options.get("css", "") + """
    #tv-app-header {align-items:center; gap:20px; padding-right:180px; box-sizing:border-box;}
    #tv-app-header .tv-header {padding-right:0;}
    #tv-converge-header {align-items:center;gap:20px;}
    #tv-app-header .app-title {flex:1; min-width:220px;}
    #tv-language-selector {max-width:200px; margin-left:auto; margin-bottom:8px;}
    #tv-language-selector input {cursor:pointer;}
    html[lang="zh-CN"] .gradio-container {font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}
    #analysis-sidebar.right:not(.open) .toggle-button[aria-label="AI 轨迹分析"]::after {content:"AI 轨迹分析";}
    @media(max-width:600px){
        #tv-app-header{padding-right:0;flex-direction:column!important;align-items:stretch;gap:6px;}
        #tv-app-header .app-title{min-width:0;}
        #tv-app-header>.form{width:100%;}
        #tv-language-selector{max-width:100%;width:100%;margin-bottom:6px;}
    }
    """
    return {**options, "head": head, "css": css}
