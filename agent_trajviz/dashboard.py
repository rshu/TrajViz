"""Gradio support: the TrajViz dashboards, and components that show views.

Needs the ``app`` extra (``pip install "agent-trajviz[app]"``).

Run a dashboard::

    >>> from agent_trajviz import dashboard
    >>> dashboard.launch_insight()                  # http://127.0.0.1:7860

Or build your own app from the analysis -- every view becomes a component::

    >>> import gradio as gr
    >>> with gr.Blocks() as demo:
    ...     for view in run.views(only=["summary", "token_usage", "issues"]).values():
    ...         dashboard.view_component(view)

Both dashboards bind to 127.0.0.1 by default. Exposing one (``share=True`` or
another ``host``) warns unless a login is set (``auth=`` or the
``TRAJVIZ_AUTH=user:password`` environment variable) and removes the inputs
that name paths on the serving machine.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._extras import require

require("app", "agent_trajviz.dashboard")

import gradio as gr  # noqa: E402

from . import _serve  # noqa: E402
from .figures import FIGURE_VIEWS, figure  # noqa: E402

if TYPE_CHECKING:
    from .views import View

__all__ = [
    "converge_app",
    "converge_presentation",
    "insight_app",
    "insight_presentation",
    "launch_converge",
    "launch_insight",
    "render_views",
    "view_component",
]


# -- the dashboards -----------------------------------------------------------

def insight_app(*, allow_server_paths: bool = True) -> gr.Blocks:
    """The Insight dashboard as an unlaunched ``gr.Blocks``.

    Gradio applies CSS at launch, so pass :func:`insight_presentation` to
    ``launch()`` (or use :func:`launch_insight`). ``allow_server_paths=False``
    removes inputs that name paths on the serving machine; use it whenever
    the app is reachable from other machines.
    """
    from .insight.insight import build_ui

    return build_ui(allow_server_paths=allow_server_paths)


def insight_presentation() -> dict[str, Any]:
    """``launch()`` keyword arguments that style the Insight dashboard."""
    from .insight.insight import LAUNCH_PRESENTATION

    return dict(LAUNCH_PRESENTATION)


def converge_app(*, data_root: str | os.PathLike[str] | None = None) -> gr.Blocks:
    """The Converge comparison dashboard as an unlaunched ``gr.Blocks``.

    Batch-manifest paths resolve against *data_root* (default: the current
    directory) and must stay inside it.
    """
    from .converge.app import build_ui

    return build_ui(data_root=data_root)


def converge_presentation() -> dict[str, Any]:
    """``launch()`` keyword arguments that style the Converge dashboard."""
    from .converge.app import LAUNCH_PRESENTATION

    return dict(LAUNCH_PRESENTATION)


def _resolve_auth(auth: tuple[str, str] | None) -> tuple[str, str] | None:
    return auth if auth is not None else _serve.auth_from_env()


def launch_insight(
    *,
    host: str = "127.0.0.1",
    port: int | None = 7860,
    share: bool = False,
    auth: tuple[str, str] | None = None,
    max_file_size: str | int | None = _serve.DEFAULT_MAX_FILE_SIZE,
    **launch_kwargs: Any,
) -> Any:
    """Build and launch the Insight dashboard; returns what ``Blocks.launch`` returns.

    *auth* defaults to ``TRAJVIZ_AUTH``. Other keyword arguments go to
    ``Blocks.launch`` (for example ``prevent_thread_lock=True``).
    """
    login = _resolve_auth(auth)
    server = _serve.server_kwargs(host=host, port=port, share=share, max_file_size=max_file_size, auth=login)
    exposed = share or not _serve.is_loopback(host)
    app = insight_app(allow_server_paths=not exposed)
    return app.launch(**server, **insight_presentation(), **launch_kwargs)


def launch_converge(
    *,
    host: str = "127.0.0.1",
    port: int | None = 7861,
    share: bool = False,
    auth: tuple[str, str] | None = None,
    max_file_size: str | int | None = _serve.DEFAULT_MAX_FILE_SIZE,
    data_root: str | os.PathLike[str] | None = None,
    **launch_kwargs: Any,
) -> Any:
    """Build and launch the Converge dashboard (see :func:`launch_insight`)."""
    login = _resolve_auth(auth)
    server = _serve.server_kwargs(host=host, port=port, share=share, max_file_size=max_file_size, auth=login)
    return converge_app(data_root=data_root).launch(**server, **converge_presentation(), **launch_kwargs)


# -- views as components ------------------------------------------------------

def _is_record_table(data: Any) -> bool:
    return isinstance(data, list) and bool(data) and all(isinstance(row, Mapping) for row in data)


def _cell(value: Any) -> Any:
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def view_component(view: View, *, dark: bool = False, **kwargs: Any) -> gr.components.Component:
    """A Gradio component showing *view*: a plot, a table, or JSON.

    Views with a renderer in :data:`agent_trajviz.figures.FIGURE_VIEWS` become
    ``gr.Plot``. Otherwise, nonempty lists of records become ``gr.Dataframe``
    (nested values as JSON text), and other data becomes ``gr.JSON``. Call it
    inside a ``gr.Blocks`` context to place the component; keyword arguments
    go to the component.
    """
    kwargs.setdefault("label", view.title)
    if view.id in FIGURE_VIEWS:
        return gr.Plot(value=figure(view, dark=dark), **kwargs)
    if _is_record_table(view.data):
        headers: list[str] = []
        for row in view.data:
            headers.extend(k for k in row if k not in headers)
        rows = [[_cell(row.get(h)) for h in headers] for row in view.data]
        return gr.Dataframe(value=rows, headers=headers, wrap=True, **kwargs)
    return gr.JSON(value=view.data, **kwargs)


def render_views(views: Mapping[str, View], *, dark: bool = False, open_first: int = 3) -> None:
    """Place one accordion per view -- title, description, component -- in the current Blocks."""
    for position, view in enumerate(views.values()):
        with gr.Accordion(view.title, open=position < open_first):
            gr.Markdown(view.description)
            view_component(view, dark=dark, show_label=False)
