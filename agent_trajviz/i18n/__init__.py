"""Presentation-only English and Simplified Chinese localization.

Recorded content and API data never pass through this module. HTML renderers
mark their own prose explicitly; the dashboard changes only those markers.
"""

from __future__ import annotations

import copy
import html
import importlib
import json
import re
from string import Formatter
from typing import Any

LANGUAGES = (("EN", "en"), ("中", "zh-CN"))


def normalize_language(value: object) -> str:
    return "zh-CN" if value == "zh-CN" else "en"


def catalog() -> dict[str, str]:
    translations: dict[str, str] = {}
    for name in ("catalog_common", "catalog_insight", "catalog_awe", "catalog_converge", "catalog_settings",
                 "catalog_comparison", "catalog_comparison_ui", "catalog_comparison_charts",
                 "catalog_comparison_outcomes"):
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise
            continue
        for key, value in module.ZH_CN.items():
            # Shared terminology wins across panels; section catalogues add
            # context-specific phrases rather than overriding shared labels.
            translations.setdefault(key, value)
    return translations


def translate(template: str, language: str = "en", **params: Any) -> str:
    selected = catalog().get(template, template) if normalize_language(language) == "zh-CN" else template
    return selected.format(**params) if params else selected


def _marker(template: str, params: dict[str, Any]) -> str:
    args = json.dumps(params, ensure_ascii=False, allow_nan=False, default=str)
    visible = template.format(**params) if params else template
    return (f'data-tv-i18n="{html.escape(template, quote=True)}" '
            f'data-tv-i18n-args="{html.escape(args, quote=True)}" '
            f'data-tv-i18n-en="{html.escape(visible, quote=True)}"')


def text(template: str, **params: Any) -> str:
    """Mark app-authored HTML text, escaping every interpolated data value."""
    visible = template.format(**params) if params else template
    # Gradio's prose reset explicitly colors every descendant. The wrapper
    # must inherit its parent's semantic badge/status color in either locale.
    return f'<span {_marker(template, params)} style="color:inherit">{html.escape(visible)}</span>'


def attrs(template: str, *, attribute: str = "title", **params: Any) -> str:
    """Mark a title or accessible label without adding invalid nested markup."""
    if attribute not in ("title", "aria-label", "data-help"):
        raise ValueError("Only presentation labels and help can be localized.")
    visible = template.format(**params) if params else template
    return (f'{attribute}="{html.escape(visible, quote=True)}" '
            f'{_marker(template, params)} data-tv-i18n-attr="{attribute}"')


_CHART_LAYOUT_LABEL = re.compile(
    r"(?:title\.text|legend\.title\.text|[xy]axis\d*\.(?:title\.text|ticktext)|annotations\[\d+\]\.text)\Z"
)
_CHART_TRACE_LABELS = frozenset(("name", "text", "hovertext", "hovertemplate"))


def _chart_label_spec(value: Any) -> Any:
    """Copy explicit presentation expressions without treating literal data as prose."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_chart_label_spec(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"literal"} and isinstance(value["literal"], (str, int, float, bool, type(None))):
            return copy.deepcopy(value)
        if (set(value) <= {"template", "params"} and isinstance(value.get("template"), str)
                and isinstance(value.get("params", {}), dict)):
            # Plotly labels accept HTML, so interpolated data stays escaped.
            return {"template": value["template"], "params": {
                key: html.escape(item) if isinstance(item, str) else item
                for key, item in value.get("params", {}).items()
            }}
    raise ValueError("Chart labels must be authored strings, lists, templates, or literal scalars.")


def _chart_label_source(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_chart_label_source(item) for item in value]
    if "literal" in value:
        return value["literal"]
    return value["template"].format(**value["params"]) if value["params"] else value["template"]


def _set_chart_layout_label(fig: Any, path: str, value: Any) -> None:
    if not _CHART_LAYOUT_LABEL.fullmatch(path):
        raise ValueError("Only chart titles, tick labels, and annotation text can be localized.")
    parts = re.sub(r"\[(\d+)\]", r".\1", path).split(".")
    target = fig.layout
    for part in parts[:-1]:
        target = target[int(part) if part.isdigit() else part]
    target[parts[-1]] = value


def localize_figure(fig: Any, *, trace_names: bool = False, annotations: bool = False,
                    layout_text: dict[str, tuple[str, dict[str, Any]]] | None = None,
                    layout_labels: dict[str, Any] | None = None,
                    trace_labels: dict[int, dict[str, Any]] | None = None) -> Any:
    """Record authored chart chrome while retaining plotted identities and values.

    Extra display labels are opt-in: strings are prose keys, lists combine
    display labels, ``{\"literal\": value}`` preserves a recorded category, and
    ``{\"template\": key, \"params\": args}`` formats escaped data arguments.
    Only presentation fields can be registered; x/y/customdata stay untouched.
    """
    layout = fig.layout.to_plotly_json()
    prior = layout.get("meta", {})
    prior = prior.get("tv_i18n", {}) if isinstance(prior, dict) else {}
    owned: dict[str, Any] = {key: value for key, value in prior.get("layout", {}).items()
                             if key.startswith("annotations[")}
    for key, value in layout.items():
        if key == "title" or key == "legend" or key.startswith(("xaxis", "yaxis")):
            title = value if key == "title" else value.get("title", {})
            if isinstance(title, dict) and isinstance(title.get("text"), str):
                path = "title.text" if key == "title" else f"{key}.title.text"
                previous = prior.get("layout", {}).get(path)
                if (isinstance(previous, dict) and "template" in previous
                        and previous["template"].format(**previous.get("params", {})) == title["text"]):
                    owned[path] = previous
                else:
                    owned[path] = title["text"]
    if annotations:
        for index, annotation in enumerate(layout.get("annotations", [])):
            if isinstance(annotation.get("text"), str):
                owned[f"annotations[{index}].text"] = annotation["text"]
    for path, (template, params) in (layout_text or {}).items():
        owned[path] = _chart_label_spec({"template": template, "params": params})
        _set_chart_layout_label(fig, path, _chart_label_source(owned[path]))
    meta = dict(layout.get("meta") or {}) if isinstance(layout.get("meta"), (dict, type(None))) else {}
    meta["tv_i18n"] = {"layout": owned}
    if prior.get("traces"):
        meta["tv_i18n"]["traces"] = dict(prior["traces"])
    if trace_names:
        meta["tv_i18n"]["traces"] = {
            str(index): trace.name for index, trace in enumerate(fig.data) if isinstance(trace.name, str)
        }
    for section in ("layout_labels", "trace_labels"):
        if prior.get(section):
            meta["tv_i18n"][section] = copy.deepcopy(prior[section])
    for path, expression in (layout_labels or {}).items():
        spec = _chart_label_spec(expression)
        _set_chart_layout_label(fig, path, _chart_label_source(spec))
        meta["tv_i18n"].setdefault("layout_labels", {})[path] = spec
    for index, labels in (trace_labels or {}).items():
        if not isinstance(index, int) or not 0 <= index < len(fig.data):
            raise ValueError("Chart trace label index is out of range.")
        for field, expression in labels.items():
            if field not in _CHART_TRACE_LABELS:
                raise ValueError("Only chart trace names, text, and hover labels can be localized.")
            spec = _chart_label_spec(expression)
            fig.data[index][field] = _chart_label_source(spec)
            meta["tv_i18n"].setdefault("trace_labels", {}).setdefault(str(index), {})[field] = spec
    fig.update_layout(meta=meta)
    return fig


def validate_catalog() -> None:
    """Fail clearly on a malformed translation before shipping a dashboard."""
    formatter = Formatter()
    for source, target in catalog().items():
        source_fields = {field for _, field, _, _ in formatter.parse(source) if field is not None}
        target_fields = {field for _, field, _, _ in formatter.parse(target) if field is not None}
        if source_fields != target_fields:
            raise ValueError(f"Translation placeholders differ: {source!r}")
