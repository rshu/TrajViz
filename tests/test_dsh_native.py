"""Released DSH v4 native storage, including compressed sibling children."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import agent_trajviz as tv
from agent_trajviz.errors import LoadError
from agent_trajviz.insight import loaders
from agent_trajviz.insight.formats import dsh_native as native

T0 = 1_790_000_000_000


def events(sid="root", *, parent=None):
    header = {"type": "session", "version": 4, "id": sid, "createdAt": T0,
              "cwd": "/workspace", "isSeeded": False, "delegationDepth": int(bool(parent))}
    if parent:
        header.update(origin="subagent", parentSession=parent)
    return [header,
            {"type": "step/start", "seq": 0, "time": T0, "data": {"turn": 1, "step": 1}},
            {"type": "user/message", "seq": 1, "time": T0 + 1,
             "data": {"source": {"kind": "user"}, "id": "u", "content": [{"type": "text", "text": "Inspect fixture"}]}},
            {"type": "assistant/message", "seq": 2, "time": T0 + 2000,
             "data": {"turn": 1, "step": 1, "message": {"id": sid + "-a", "role": "assistant",
                      "source": {"kind": "model", "provider": "deepseek-official", "model": "deepseek-flash"},
                      "content": [{"type": "tool-call", "id": sid + "-call", "name": "bash", "arguments": "{\"command\":\"true\"}"}]},
                      "usage": {"inputTokens": 10, "outputTokens": 20, "cacheReadTokens": 40,
                                "reasoningTokens": 5, "totalTokens": 70}, "stream": []}},
            {"type": "tool/call", "seq": 3, "time": T0 + 2001,
             "data": {"callId": sid + "-call", "name": "bash", "arguments": "{\"command\":\"true\"}"}},
            {"type": "tool/result", "seq": 4, "time": T0 + 7001,
             "data": {"message": {"source": {"kind": "tool", "callId": sid + "-call"},
                      "content": [{"type": "tool-result", "toolCallId": sid + "-call",
                                   "content": [{"type": "text", "text": "ok"}], "isError": False}]}}},
            {"type": "turn/end", "seq": 5, "time": T0 + 7002, "data": {"turn": 1, "reason": {"kind": "stop"}}}]


def encode(rows):
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def compress(data):
    try:
        return importlib.import_module("compression.zstd").compress(data)
    except ImportError:
        try:
            return importlib.import_module("zstandard").ZstdCompressor(write_checksum=True).compress(data)
        except ImportError:
            raise unittest.SkipTest("compression support is optional") from None


def write(directory, rows, *, compressed=True, generation=4):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"session.v{generation}.jsonl" if generation else "session.jsonl")
    data = encode(rows)
    if compressed:
        path = path.with_suffix(path.suffix + ".zstd")
        # Released DSH writes a header frame and append-batch frames.
        data = compress(encode(rows[:1])) + compress(encode(rows[1:]))
    path.write_bytes(data)
    return path


class DshNativeTests(unittest.TestCase):
    def test_sanitized_latest_real_capture_excerpts(self):
        fixture = Path(__file__).parent / "fixtures" / "dsh_native_v4"
        with tempfile.TemporaryDirectory() as tmp:
            for source in fixture.glob("*/session.v4.jsonl"):
                rows = [json.loads(line) for line in source.read_text().splitlines()]
                write(Path(tmp) / source.parent.name, rows)
            captured = tv.load(Path(tmp) / "root")
            self.assertEqual(captured.warnings, ())
            self.assertEqual(len(captured.agents), 3)
            self.assertEqual(captured.metrics["tokens"]["total"], 20_549)
            self.assertEqual(captured.metrics["tool_call_count"], 6)
            for include_raw in (False, True):
                restored = tv.Trajectory.from_dict(captured.to_dict(include_raw=include_raw))
                self.assertEqual(restored.steps, captured.steps)
                self.assertEqual(restored.metrics["tokens"], captured.metrics["tokens"])
                if include_raw:
                    self.assertEqual(restored.metrics, captured.metrics)

    def test_select_highest_generation_and_refuse_future_or_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for version in (0, 2, 4):
                rows = events()
                rows[0]["version"] = version
                newest = write(root, rows, compressed=False, generation=version)
            self.assertEqual(native.resolve_native_dsh_path(tmp), str(newest))
            self.assertEqual(tv.load(root).format, "dsh")
            future = root / "session.v5.jsonl"
            future.write_text("{}")
            with self.assertRaisesRegex(LoadError, "newer"):
                tv.load(root)
            future.unlink()
            (root / "session.v4.jsonl.zstd").write_bytes(b"not-decoded")
            with self.assertRaisesRegex(LoadError, "ambiguous"):
                tv.load(root)

    def test_versioned_session_accepts_bom_marked_utf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = events()
            path = write(Path(tmp), rows, compressed=False)
            path.write_bytes(path.read_text(encoding="utf-8").encode("utf-16"))
            captured = tv.load(path.parent)
        self.assertEqual(captured.raw["_dsh_events"]["root"], rows)
        self.assertEqual(captured.metrics["tokens"]["total"], 70)

    def test_compressed_tree_uses_catalog_lineage_and_exact_source_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            rows = events()
            rows.append({"type": "subagent/catalog", "time": T0 + 3,
                         "data": {"childId": "child", "mode": "continuable", "label": "Reviewer"}})
            parent = write(project / "root", rows)
            child = write(project / "child", events("child", parent="root"))
            write(project / "unreferenced", events("private", parent="root"))
            captured = tv.load(parent.parent)
            self.assertEqual(captured.warnings, ())
            self.assertEqual(len(captured.agents), 2)
            self.assertEqual(captured.metrics["tokens"]["total"], 140)
            self.assertEqual(captured.metrics["tool_call_count"], 2)
            self.assertEqual(captured.steps[0]["role"], "user")
            model = next(step for step in captured.steps if step["role"] == "assistant")
            self.assertEqual(model["duration"], 2)
            self.assertEqual(model["timing_source"], "model_request")
            self.assertEqual(model["tool_calls"][0]["duration_ms"], 5000)
            self.assertEqual(captured.raw["_dsh_events"]["root"], rows)
            parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
            child_sha = hashlib.sha256(child.read_bytes()).hexdigest()
            self.assertEqual(captured.sha256, loaders._dsh_tree_sha256(parent_sha, [("child", str(child))], {"child": child_sha}))
            direct = tv.load(child)
            self.assertEqual(direct.sha256, child_sha)
            self.assertEqual(len(direct.agents), 1)

    def test_child_hash_is_the_snapshot_used_for_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            rows = events()
            rows.append({"type": "subagent/catalog", "data": {"childId": "child"}})
            parent = write(project / "root", rows)
            child = write(project / "child", events("child", parent="root"))
            expected = tv.load(parent).sha256
            original = loaders._apply_format

            def mutate_after_read(*args, **kwargs):
                result = original(*args, **kwargs)
                child.write_bytes(compress(encode(events("changed", parent="root"))))
                return result

            with patch.object(loaders, "_apply_format", side_effect=mutate_after_read):
                self.assertEqual(tv.load(parent).sha256, expected)
            self.assertNotEqual(tv.load(parent).sha256, expected)

    def test_wrong_parent_and_missing_child_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            rows = events()
            rows.extend({"type": "subagent/catalog", "data": {"childId": child}} for child in ("wrong", "missing"))
            parent = write(project / "root", rows, compressed=False)
            write(project / "wrong", events("wrong", parent="unrelated"), compressed=False)
            captured = tv.load(parent)
            self.assertEqual(len(captured.agents), 1)
            self.assertEqual(len(captured.warnings), 2)

    def test_missing_compression_extra_is_a_public_load_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.v4.jsonl.zstd"
            path.write_bytes(b"\x28\xb5\x2f\xfd")
            with (patch.object(native.importlib, "import_module", side_effect=ImportError),
                  self.assertRaisesRegex(LoadError, r"pip install 'agent-trajviz\[zstd\]'") as caught):
                tv.load(path)
            self.assertEqual(caught.exception.code, "missing_extra")

    def test_corrupt_and_expanded_size_limits(self):
        with self.assertRaisesRegex(LoadError, "frame header"):
            native.decode_dsh_native_bytes(b"garbage", "session.v4.jsonl.zstd")
        compressed = compress(b"x" * 1000)
        with patch.object(native, "MAX_NATIVE_BYTES", 100), self.assertRaises(LoadError) as caught:
            native.decode_dsh_native_bytes(compressed, "session.v4.jsonl.zstd")
        self.assertEqual(caught.exception.code, "size_limit")

    def test_seeded_v4_uses_last_inherited_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = events("child", parent="root")
            rows[0]["isSeeded"] = True
            own = events("child", parent="root")[1:]
            rows += [{"type": "session/end-seed", "data": {"inherited": True}}, *own]
            captured = tv.load(write(Path(tmp), rows, compressed=False))
            self.assertEqual(captured.metrics["tokens"]["total"], 70)
            self.assertEqual(captured.metrics["tool_call_count"], 1)

    def test_v4_flat_tool_result_preserves_outcome_and_direct_call_id(self):
        # Released v4 stores the result fields on message, with plain text
        # content, rather than nesting them in a tool-result content block.
        for source_has_id in (False, True):
            for is_error in (False, True):
                with self.subTest(source_has_id=source_has_id, is_error=is_error), tempfile.TemporaryDirectory() as tmp:
                    rows = events()
                    source = {"kind": "tool"}
                    if source_has_id:
                        source["callId"] = "root-call"
                    rows[-2]["data"]["message"] = {
                        "id": "result", "role": "tool", "source": source,
                        "toolCallId": "root-call", "isError": is_error,
                        "content": [{"type": "text", "text": "command failed" if is_error else "ok"}],
                    }
                    captured = tv.load(write(Path(tmp), rows, compressed=False))
                    model = next(step for step in captured.steps if step["role"] == "assistant")
                    self.assertEqual(captured.metrics["tool_call_count"], 1)
                    self.assertEqual(len(captured.steps), 2)
                    self.assertEqual(model["error_count"], int(is_error))
                    tool = model["tool_calls"][0]
                    self.assertEqual(tool["tool_id"], "root-call")
                    self.assertEqual(tool["status"], "error" if is_error else "success")
                    self.assertEqual(tool["output"], "command failed" if is_error else "ok")
                    self.assertEqual(tool["duration_ms"], 5000)

    def test_missing_reasoning_breakdown_is_not_reported_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = events()
            usage = rows[3]["data"]["usage"]
            del usage["reasoningTokens"]
            path = write(Path(tmp), rows, compressed=False)
            self.assertFalse(tv.load(path).metrics["reasoning_tokens_reported"])
            usage["reasoningTokens"] = 0
            path = write(Path(tmp), rows, compressed=False)
            self.assertTrue(tv.load(path).metrics["reasoning_tokens_reported"])


if __name__ == "__main__":
    unittest.main()
