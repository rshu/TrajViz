"""Expanded Workflow step body: full text, collapsed reasoning and tool I/O.

The card header stays in the Workflow renderer. This module only builds the
content under it. Tool output is not truncated (the detail panel caps at
200,000 characters). Every dynamic string goes through ``html.escape``.
"""

from __future__ import annotations

import html
import json

from ..metrics import tool_call_duration_ms

# Same hint keys as the Workflow detail panel.
_TITLE_ARG_KEYS = ("command", "file_path", "pattern", "description", "prompt")

_EMPTY = "(no content)"

TRANSCRIPT_CSS = """<style>
.tv-meta {
    font-size: 12px; color: var(--ov-muted); margin-top: 4px;
    display: flex; gap: 8px; flex-wrap: wrap;
}
.tv-text {
    white-space: pre-wrap; word-break: break-word; font-size: 13px;
    line-height: 1.55; margin-top: 8px; color: var(--ov-text);
}
.tv-details { margin-top: 8px; }
.tv-details pre, .dp-details-body pre { white-space: pre-wrap; word-break: break-word; }
.tv-io-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--ov-muted); margin: 8px 0 2px;
}
.tv-empty { color: var(--ov-muted); font-style: italic; font-size: 12px; margin-top: 6px; }
</style>"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _plain(value: object) -> str:
    """Readable text for a tool input, output, or error value."""
    if isinstance(value, str):
        return value
    if (
        isinstance(value, list)
        and value
        and all(isinstance(b, dict) and b.get("type") == "text" for b in value)
    ):
        return "\n".join(str(b.get("text", "")) for b in value)
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _details(summary: str, body: str) -> str:
    return (
        "<details class='dp-details tv-details'>"
        f"<summary>{summary}</summary>"
        f"<div class='dp-details-body'>{body}</div>"
        "</details>"
    )


def reasoning_block_label(
    text: str,
    tokens: int | None,
    part_no: int = 1,
    part_count: int = 1,
) -> str:
    """Reasoning summary: the step's token total, else a character count.

    A step-level total cannot be split across parts, so every part of a
    multi-part step is labelled with that total plus its ordinal.
    """
    if tokens:
        if part_count > 1:
            return f"Reasoning ({tokens:,} tok · part {part_no}/{part_count})"
        return f"Reasoning ({tokens:,} tok)"
    return f"Reasoning ({len(text):,} chars)"


def _tool_call_html(part: dict) -> str:
    tool_name = str(part.get("tool_name") or "?")
    title = str(part.get("title") or "")
    if not title:
        inp = part.get("input")
        if isinstance(inp, dict):
            for key in _TITLE_ARG_KEYS:
                if inp.get(key):
                    title = str(inp[key])[:80]
                    break
    dur_ms = tool_call_duration_ms(part)
    status = str(part.get("status") or "?")
    is_error = bool(part.get("error")) or status == "error"
    bits = "".join((
        f" &middot; {_esc(title)}" if title else "",
        f" &mdash; {dur_ms / 1000.0:.2f}s" if dur_ms is not None else "",
        " &middot; error" if is_error else "",
    ))
    body = [
        f"<div class='tv-meta'><span>status: <code>{_esc(status)}</code></span></div>",
        f"<div class='tv-io-label'>Input</div><pre>{_esc(_plain(part.get('input', {})))}</pre>",
    ]
    output_text = _plain(part.get("output", ""))
    if output_text:
        body.append(f"<div class='tv-io-label'>Output</div><pre>{_esc(output_text)}</pre>")
    if part.get("error"):
        body.append(f"<div class='tv-io-label'>Error</div><pre>{_esc(_plain(part['error']))}</pre>")
    return _details(f"<code>{_esc(tool_name)}</code>{bits}", "".join(body))


def _parts_html(step: dict) -> str:
    parts = [p for p in (step.get("parts") or []) if isinstance(p, dict)]
    reasoning_total = (step.get("tokens") or {}).get("reasoning") or None
    reasoning_count = sum(1 for p in parts if p.get("type") == "reasoning" and p.get("text"))
    reasoning_no = 0
    chunks: list[str] = []
    for part in parts:
        ptype = part.get("type")
        if ptype == "text":
            text = part.get("text") or ""
            chunk = (
                f"<div class='tv-io-label'>Text</div><div class='tv-text'>{_esc(text)}</div>"
                if text.strip() else ""
            )
        elif ptype == "reasoning" and part.get("text"):
            reasoning_no += 1
            text = part["text"]
            chunk = _details(
                reasoning_block_label(text, reasoning_total, reasoning_no, reasoning_count or 1),
                f"<pre>{_esc(text)}</pre>",
            )
        elif ptype == "tool_call":
            chunk = _tool_call_html(part)
        elif ptype == "patch" and part.get("diff_content"):
            patch_hash = str(part.get("hash") or "")
            label = f"Patch {_esc(patch_hash[:12])}" if patch_hash else "Patch"
            chunk = _details(label, f"<pre>{_esc(part['diff_content'])}</pre>")
        elif ptype == "compaction" and part.get("summary"):
            chunk = _details("Compaction summary", f"<pre>{_esc(part['summary'])}</pre>")
        else:
            chunk = ""
        if chunk:
            chunks.append(chunk)
    return "".join(chunks) if chunks else f"<div class='tv-empty'>{_EMPTY}</div>"


def build_step_parts_html(step: dict) -> str:
    """Conversation content of one step, embedded under an expanded Workflow card."""
    return _parts_html(step)
