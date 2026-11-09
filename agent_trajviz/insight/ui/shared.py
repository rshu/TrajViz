"""Shared Gradio state passed into tab layout/bind helpers."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import gradio as gr

if TYPE_CHECKING:
    from ...api import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class SharedState:
    state_steps: gr.State
    state_raw: gr.State
    # Write-once: `insight.build_ui` pins the dashboard to light (its app.load
    # handler is the only writer, and APP_CSS declares `color-scheme: light`),
    # so every `dark=` this package threads through is False. The parameter
    # itself is live for `python -m agent_trajviz.insight --report --dark`; this State
    # is the single switch point should the dashboard ever offer a toggle.
    state_dark: gr.State
    state_analysis_brief: gr.State
    # The loaded agent_trajviz.Trajectory (every step), or None before a load.
    # Handlers that need the run itself read it here rather than re-parsing
    # `state_raw`, which holds nothing for a format registered by a plugin.
    state_run: gr.State
    # Ranked Overview issues as JSON-friendly dicts (optional judgments).
    # Cards pass a 1-based index; the sidebar rebuilds the AI question here.
    state_overview_issues: gr.State
    # Session-only model overrides; credentials never enter browser storage.
    state_settings: gr.State | None = None


def format_choices() -> list[tuple[str, str]]:
    """Format dropdown choices: auto-detect first, then every format, registered ones included."""
    from ...api import FORMATS

    return [("Auto-detect", ""), *((label, key) for key, label in FORMATS.items())]


# Gradio owns each loaded run. Weak references let its state eviction free
# both the trajectory and this index, including clients that never unload.
_latest_run_by_session: WeakValueDictionary[str, Trajectory] = WeakValueDictionary()
_latest_run_lock = Lock()


def record_latest_run(session_hash: str | None, run: Trajectory | None) -> None:
    """Track the run the session displays, or invalidate it after a clear."""
    if session_hash is None:
        return
    with _latest_run_lock:
        if run is None:
            _latest_run_by_session.pop(session_hash, None)
        else:
            _latest_run_by_session[session_hash] = run


def is_latest_run(session_hash: str | None, run: Trajectory) -> bool:
    """True while *run* is still what *session_hash* last loaded.

    Only direct callers without a session bypass this check. A cleared,
    evicted, or closed browser session supersedes its pending work.
    """
    if session_hash is None:
        return True
    with _latest_run_lock:
        return _latest_run_by_session.get(session_hash) is run


def drop_latest_run(session_hash: str) -> None:
    """Forget *session_hash* on browser unload."""
    with _latest_run_lock:
        _latest_run_by_session.pop(session_hash, None)


def safe_callback(label: str) -> Callable:
    """Turn an unexpected callback failure into a visible toast, not a silent 500.

    Gradio's ``show_error`` defaults to False (and the dashboard does not set
    it), so a raising callback answers HTTP 500 with ``{"error": null}``: the
    browser shows nothing at all and every output keeps its previous value, so
    the user reads stale numbers as if they were the new ones. Raising
    ``gr.Error`` renders a toast even with ``show_error`` off.

    Only the exception *type* is reported. Enabling ``show_error`` — or putting
    ``str(exc)`` in the message — would ship absolute server paths and instance
    ids to the browser, which is the same reason
    ``assistant._public_http_error`` exists.

    Apply this only to callbacks with no containment at any layer. ``do_load``,
    ``prepare_html_export``, ``do_load_labels``, the sidebar chat path and
    ``run_comparison`` all render their own error surface; wrapping them would
    replace it with a toast.
    """

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except gr.Error:
                # Already a user-facing message (e.g. a nested safe_callback).
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s failed", label)
                raise gr.Error(f"{label} failed: {type(exc).__name__}. See the server log.") from exc

        wrapper.__tv_safe__ = True
        return wrapper

    return decorate
