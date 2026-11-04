"""Presentation-only English and Simplified Chinese localization.

Recorded content and API data never pass through this module. HTML renderers
mark their own prose explicitly; the dashboard changes only those markers.
"""

from __future__ import annotations

import html
import importlib
import json
from string import Formatter
from typing import Any

LANGUAGES = (("English", "en"), ("简体中文", "zh-CN"))


def normalize_language(value: object) -> str:
    return "zh-CN" if value == "zh-CN" else "en"


def catalog() -> dict[str, str]:
    translations: dict[str, str] = {}
    for name in ("catalog_common", "catalog_insight", "catalog_awe", "catalog_converge", "catalog_settings"):
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
    return f'<span {_marker(template, params)}>{html.escape(visible)}</span>'


def attrs(template: str, *, attribute: str = "title", **params: Any) -> str:
    """Mark a title or accessible label without adding invalid nested markup."""
    if attribute not in ("title", "aria-label", "data-help"):
        raise ValueError("Only presentation labels and help can be localized.")
    visible = template.format(**params) if params else template
    return (f'{attribute}="{html.escape(visible, quote=True)}" '
            f'{_marker(template, params)} data-tv-i18n-attr="{attribute}"')


def localize_figure(fig: Any, *, trace_names: bool = False, annotations: bool = False,
                    layout_text: dict[str, tuple[str, dict[str, Any]]] | None = None) -> Any:
    """Record authored chart chrome; plotted data and category labels stay intact."""
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
        # Plotly titles accept HTML; data arguments must remain literal text.
        safe_params = {
            key: html.escape(str(value)) if isinstance(value, str) else value
            for key, value in params.items()
        }
        owned[path] = {"template": template, "params": safe_params}
        target = fig.layout
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = template.format(**safe_params)
    meta = dict(layout.get("meta") or {}) if isinstance(layout.get("meta"), (dict, type(None))) else {}
    meta["tv_i18n"] = {"layout": owned}
    if prior.get("traces"):
        meta["tv_i18n"]["traces"] = dict(prior["traces"])
    if trace_names:
        meta["tv_i18n"]["traces"] = {
            str(index): trace.name for index, trace in enumerate(fig.data) if isinstance(trace.name, str)
        }
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
