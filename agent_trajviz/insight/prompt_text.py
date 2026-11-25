"""Readable selected-pair prompt text, with conservative repetition evidence.

Content hashes and native identity decide changes elsewhere. This module only
extracts literal text and compares complete earlier blocks, never previews.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

_OPAQUE_TYPES = {"image", "image_url", "input_image", "audio", "input_audio", "video", "document", "base64",
                 "thinking", "redacted_thinking", "reasoning", "reasoning_content"}
_OPAQUE_KEYS = {"encrypted_content", "signature", "base64", "image_url", "input_audio"}
_MAX_REFS = 8
_MAX_BLOCKS = 80
_MAX_CHARS = 6000
_MAX_TOTAL_CHARS = 120_000


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pointer(path: str, key: Any) -> str:
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _opaque_string(value: str) -> bool:
    return value.lstrip().lower().startswith("data:") and ";base64," in value[:200].lower()


def _kind(value: dict) -> str:
    kind = value.get("type")
    return kind if isinstance(kind, str) else ""


def _visible(value: Any, depth: int = 0) -> Any:
    """Drop binary/encrypted fields from fallback argument/definition rendering."""
    if depth > 64:
        return None
    if isinstance(value, dict):
        if _kind(value) in _OPAQUE_TYPES or value.get("encoding") == "base64":
            return None
        return {key: item for key, raw in value.items() if key not in _OPAQUE_KEYS
                and (item := _visible(raw, depth + 1)) is not None}
    if isinstance(value, list):
        return [item for raw in value if (item := _visible(raw, depth + 1)) is not None]
    return None if isinstance(value, str) and _opaque_string(value) else value


def _texts(value: Any, path: str, *, depth: int = 0) -> Iterator[tuple[str, str]]:
    """Known text-bearing shapes only; arbitrary metadata is not prose."""
    if depth > 64:
        return
    if isinstance(value, str):
        if value and not _opaque_string(value):
            yield path, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _texts(item, _pointer(path, index), depth=depth + 1)
    elif isinstance(value, dict) and _kind(value) not in _OPAQUE_TYPES:
        for key in ("text", "content", "output"):
            if key in value:
                yield from _texts(value[key], _pointer(path, key), depth=depth + 1)
                return


def _literal_blocks(component: dict) -> Iterator[dict]:
    value, role = component.get("content"), str(component.get("role") or "")
    base = {"component_id": str(component.get("id") or ""), "role": role,
            "source_refs": [dict(ref) for ref in component.get("source_refs", [])[:_MAX_REFS]
                            if isinstance(ref, dict)]}

    def emit(path: str, text: str, kind: str = "text", *, tool_name: str = "", payload: Any = None) -> dict:
        # Include the whole argument/result payload: equal prefixes or a single
        # equal command inside otherwise different arguments never prove repeats.
        signature = _json([kind, tool_name, text, payload])
        return {**base, "path": path, "kind": kind, "tool_name": tool_name, "text": text,
                "_signature": hashlib.sha256(signature.encode()).hexdigest()}

    def walk(item: Any, path: str, depth: int = 0) -> Iterator[dict]:
        if depth > 64:
            return
        if isinstance(item, str):
            if item and not _opaque_string(item):
                yield emit(path, item)
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                yield from walk(child, _pointer(path, index), depth + 1)
            return
        if not isinstance(item, dict) or _kind(item) in _OPAQUE_TYPES:
            return
        kind = _kind(item)
        item_role = item.get("role") or role
        item_role = item_role if isinstance(item_role, str) else role
        name = str(item.get("name") or item.get("toolName") or item.get("tool_name") or "")
        if kind in {"tool_use", "toolCall", "function_call"}:
            key = "input" if "input" in item else "arguments"
            arguments = item.get(key, {})
            if kind == "function_call" and isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    pass
            found = []
            if isinstance(arguments, dict):
                for text_key in ("command", "content"):
                    found.extend(_texts(arguments.get(text_key), _pointer(_pointer(path, key), text_key)))
            if not found:
                visible = _visible(arguments)
                if visible not in (None, {}, [], ""):
                    found = [(_pointer(path, key), visible if isinstance(visible, str)
                              else json.dumps(visible, ensure_ascii=False, sort_keys=True, indent=2))]
            for location, text in found:
                yield emit(location, text, "tool_arguments", tool_name=name, payload=arguments)
            return
        if kind in {"tool_result", "function_call_output"} or item_role in {"tool", "toolResult"}:
            key = "output" if "output" in item else "content"
            payload = [item.get(key), item.get("isError", item.get("is_error"))]
            for location, text in _texts(item.get(key), _pointer(path, key)):
                yield emit(location, text, "tool_result", tool_name=name, payload=payload)
            return
        if isinstance(item.get("text"), str):
            if item["text"] and not _opaque_string(item["text"]):
                yield emit(_pointer(path, "text"), item["text"])
        elif "content" in item:
            yield from walk(item["content"], _pointer(path, "content"), depth + 1)

    if component.get("kind") == "tool":
        visible = _visible(value)
        if visible not in (None, {}, [], ""):
            name = str(value.get("name") or "") if isinstance(value, dict) else ""
            yield emit("", json.dumps(visible, ensure_ascii=False, sort_keys=True, indent=2), tool_name=name)
    else:
        yield from walk(value, "")


def _display(block: dict, max_chars: int) -> dict:
    text = block["text"]
    return {key: value for key, value in block.items() if key not in {"text", "_signature"}} | {
        "text": text[:max_chars], "text_truncated": len(text) > max_chars}


def readable_component_blocks(component: dict, *, max_chars: int = _MAX_CHARS,
                              max_blocks: int = _MAX_BLOCKS) -> list[dict]:
    """Extract bounded literal text for a snapshot; opaque payloads stay in JSON.

    Paths are JSON pointers into ``component.content``. No escaping/translation
    occurs here; HTML renderers must escape recorded text at their boundary.
    """
    max_chars, max_blocks = max(0, min(max_chars, _MAX_CHARS)), max(0, min(max_blocks, _MAX_BLOCKS))
    output = []
    for block in _literal_blocks(component):
        if len(output) >= max_blocks:
            break
        output.append(_display(block, max_chars))
    return output


def _joined(component: dict) -> tuple[str, bool]:
    text, shortened = "", False
    for index, block in enumerate(_literal_blocks(component)):
        if index >= _MAX_BLOCKS:
            shortened = True
            break
        addition = ("\n\n" if text else "") + block["text"]
        room = _MAX_CHARS - len(text)
        text += addition[:room]
        if len(addition) > room:
            shortened = True
            break
    return text, shortened


def build_text_evolution(selected: dict, previous_components: list[dict]) -> dict:
    """Readable changes for an already validated comparison, selected pair only."""
    changes = selected.get("changes", {})
    available = changes.get("available") is True
    result = {"comparison_available": available, "evidence": changes.get("evidence", selected["evidence"]),
              "scope": selected["scope"], "comparison_basis": changes.get("basis", "recorded_request_state"),
              "blocks": [], "rewrites": [], "limitations": [], "blocks_truncated": False}
    if not available:
        return result
    before = {part["id"]: part for part in previous_components}
    after = {part["id"]: part for part in selected.get("components", [])}
    rewritten_ids = set()
    for rewrite in changes.get("rewritten_components", []):
        old, new = before.get(rewrite["before_id"]), after.get(rewrite["after_id"])
        if old is None or new is None:
            continue
        old_text, old_short = _joined(old)
        new_text, new_short = _joined(new)
        rewritten_ids.update((old["id"], new["id"]))
        if old_text or new_text:
            result["rewrites"].append({"before_id": old["id"], "after_id": new["id"], "label": rewrite["label"],
                                       "before_text": old_text, "after_text": new_text,
                                       "before_truncated": old_short, "after_truncated": new_short,
                                       "source_refs": (old["source_refs"] + new["source_refs"])[:_MAX_REFS]})
        if len(result["rewrites"]) >= _MAX_BLOCKS:
            result["blocks_truncated"] = True
            break
    # Only the comparable previous snapshot is indexed; no cross-branch/global
    # search or current/future matching is allowed. Store counts and bounded IDs.
    earlier: dict[str, dict] = {}
    for component in previous_components:
        for block in _literal_blocks(component):
            match = earlier.setdefault(block["_signature"], {"count": 0, "ids": [], "truncated": False})
            match["count"] += 1
            if component["id"] not in match["ids"]:
                if len(match["ids"]) < _MAX_REFS:
                    match["ids"].append(component["id"])
                else:
                    match["truncated"] = True
    chars = 0
    for change, ids, components in (("added", changes.get("added", []), after),
                                    ("removed", changes.get("removed", []), before)):
        for identifier in ids:
            if identifier in rewritten_ids or identifier not in components:
                continue
            for block in _literal_blocks(components[identifier]):
                if len(result["blocks"]) >= _MAX_BLOCKS or chars >= _MAX_TOTAL_CHARS:
                    result["blocks_truncated"] = True
                    break
                displayed = _display(block, min(_MAX_CHARS, _MAX_TOTAL_CHARS - chars))
                chars += len(displayed["text"])
                match = earlier.get(block["_signature"]) if change == "added" else None
                count = match["count"] if match else 0
                displayed.update({"change": change, "repeated": bool(count), "repeat_count": count,
                                  "repeat_kind": ({"text": "same_text", "tool_arguments": "same_tool_arguments",
                                                   "tool_result": "same_tool_result"}[block["kind"]] if count else None),
                                  "repeat_source_component_ids": match["ids"][:] if match else [],
                                  "repeat_sources_truncated": bool(match and match["truncated"])})
                result["blocks"].append(displayed)
    if result["blocks_truncated"]:
        result["limitations"].append("Readable changes are bounded previews; full recorded components remain in evidence JSON.")
    return result
