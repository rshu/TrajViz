"""Localize only known taxonomy display labels, never recorded identities."""

from __future__ import annotations

import html
from typing import Any

from . import attrs, text
from .catalog_labels import ACTION_LABELS, PHASE_LABELS, RESERVED_ACTION_LABELS, RESERVED_PHASE_LABELS


def label_key(kind: str, value: Any) -> str | None:
    """Return an authored key for a built-in ID, or None for a custom label."""
    if kind == "phase":
        known = PHASE_LABELS.keys() | RESERVED_PHASE_LABELS.keys()
    elif kind == "action":
        known = ACTION_LABELS.keys() | RESERVED_ACTION_LABELS.keys()
    else:
        raise ValueError("Label kind must be phase or action.")
    return value if isinstance(value, str) and value in known else None


def label_spec(kind: str, value: Any) -> str | dict[str, str]:
    """A Plotly presentation expression with explicit literal fallback."""
    return label_key(kind, value) or {"literal": html.escape(str(value))}


def label_text(kind: str, value: Any) -> str:
    """Safe HTML for a display name, leaving custom labels unmarked."""
    key = label_key(kind, value)
    return text(key) if key is not None else html.escape(str(value))


def label_attrs(kind: str, value: Any, *, attribute: str = "title", count: Any = None) -> str:
    """Owned label/count tooltip, or a literal escaped custom tooltip."""
    if attribute not in {"title", "aria-label", "data-help"}:
        raise ValueError("Only presentation labels and help can be localized.")
    key = label_key(kind, value)
    if key is not None:
        return attrs(key, attribute=attribute) if count is None else attrs(
            key + ": {count}", attribute=attribute, count=count,
        )
    shown = str(value) if count is None else f"{value}: {count}"
    return f'{attribute}="{html.escape(shown, quote=True)}"'
