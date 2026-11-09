"""Tests for the compaction-aftermath hint detector and its Overview card.

The rule under test: when a compaction sits between two consecutive
writes and the file reads feeding the second write shrank by more than
20% compared to the reads before the compaction, emit one advisory card.
"""

from __future__ import annotations

import unittest

from agent_trajviz.insight.context_usage import detect_compaction_aftermath
from agent_trajviz.insight.issues import _from_compaction_aftermath
from agent_trajviz.insight.presenters.issues import _localize_issue_text


def _tokens(*, inp, out=0, cache_read=0):
    return {
        "total": inp + out + cache_read,
        "input": inp,
        "output": out,
        "reasoning": 0,
        "cache_read": cache_read,
    }


def _step(
    idx,
    *,
    role="assistant",
    session_id="s1",
    tokens=None,
    parts=None,
    tool_calls=None,
    is_compaction_checkpoint=False,
    summary=False,
):
    return {
        "index": idx,
        "role": role,
        "agent": "",
        "is_sub_agent": False,
        "session_id": session_id,
        "tokens": tokens or _tokens(inp=0),
        "parts": parts or [],
        "tool_calls": tool_calls or [],
        "model_id": "m",
        "session_title": "",
        "summary": summary,
        "is_compaction_checkpoint": is_compaction_checkpoint,
        "message_type": "compaction" if is_compaction_checkpoint else "",
    }


def _read(path, output="x" * 400, status="completed"):
    return {"tool_name": "read", "input": {"filePath": path}, "output": output, "status": status}


def _edit(path, status="completed"):
    return {
        "tool_name": "edit",
        "input": {"filePath": path, "old_string": "a", "new_string": "b"},
        "output": "",
        "status": status,
    }


def _write(path, status="completed", content="regenerated from the summary"):
    return {
        "tool_name": "write",
        "input": {"filePath": path, "content": content},
        "output": "",
        "status": status,
    }


def _compaction(idx=2, session_id="s1", *, inp_after=3000):
    return _step(
        idx,
        role="user",
        session_id=session_id,
        tokens=_tokens(inp=inp_after, cache_read=0),
        is_compaction_checkpoint=True,
        parts=[{"type": "compaction", "summary": "earlier work summarized"}],
    )


#: ~1000 estimated tokens ("other" chars at 2.5 per token).
_BIG = "x" * 2500
#: ~100 estimated tokens.
_SMALL = "x" * 250


def _run(
    *,
    before_calls=None,
    after_calls=None,
    compaction_idx=2,
    session_id="s1",
    other_session_steps=(),
):
    steps = [
        _step(0, role="user", session_id=session_id, parts=[{"type": "text", "text": "do it"}]),
        _step(
            1,
            session_id=session_id,
            tokens=_tokens(inp=15000),
            tool_calls=before_calls or [],
        ),
        _compaction(compaction_idx, session_id),
        _step(
            3,
            session_id=session_id,
            tokens=_tokens(inp=3000),
            tool_calls=after_calls or [],
        ),
    ]
    steps.extend(other_session_steps)
    return steps


class DetectCompactionAftermathTests(unittest.TestCase):
    def test_fires_when_reads_shrink_across_compaction(self):
        steps = _run(
            before_calls=[_edit("a.py"), _read("big.py", _BIG)],
            after_calls=[_read("small.py", _SMALL), _edit("b.py")],
        )
        events = detect_compaction_aftermath(steps)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["step"], 2)
        self.assertEqual(ev["prev_write_step"], 1)
        self.assertEqual(ev["next_write_step"], 3)
        self.assertGreater(ev["reads_before_tokens"], 900)
        self.assertLess(ev["reads_after_tokens"], 150)
        self.assertGreaterEqual(ev["shrink_pct"], 80)

    def test_fires_when_second_write_reads_nothing(self):
        # Pure memory write: nothing re-read between compaction and write.
        steps = _run(
            before_calls=[_edit("a.py"), _read("big.py", _BIG)],
            after_calls=[_write("b.py")],
        )
        events = detect_compaction_aftermath(steps)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reads_after_tokens"], 0)
        self.assertEqual(events[0]["shrink_pct"], 100)

    def test_no_fire_when_post_compaction_reads_keep_pace(self):
        steps = _run(
            before_calls=[_edit("a.py"), _read("big.py", _BIG)],
            after_calls=[_read("big2.py", _BIG), _edit("b.py")],
        )
        self.assertEqual(detect_compaction_aftermath(steps), [])

    def test_no_fire_without_write_after_compaction(self):
        steps = _run(
            before_calls=[_edit("a.py"), _read("big.py", _BIG)],
            after_calls=[_read("small.py", _SMALL)],
        )
        self.assertEqual(detect_compaction_aftermath(steps), [])

    def test_no_fire_without_write_before_compaction(self):
        steps = _run(
            before_calls=[_read("big.py", _BIG)],
            after_calls=[_read("small.py", _SMALL), _edit("b.py")],
        )
        self.assertEqual(detect_compaction_aftermath(steps), [])

    def test_no_fire_when_all_reads_happened_after_compaction(self):
        steps = _run(
            before_calls=[_edit("a.py")],
            after_calls=[_read("big.py", _BIG), _edit("b.py")],
        )
        self.assertEqual(detect_compaction_aftermath(steps), [])

    def test_failed_writes_do_not_count_as_writes(self):
        # The errored edit must not act as the second write: the small read
        # before the successful write still counts as post-compaction reads.
        steps = _run(
            before_calls=[_edit("a.py"), _read("big.py", _BIG)],
            after_calls=[
                _edit("x.py", status="error"),
                _read("small.py", _SMALL),
                _write("y.py"),
            ],
        )
        events = detect_compaction_aftermath(steps)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["next_write_step"], 3)
        self.assertLess(events[0]["reads_after_tokens"], 150)

    def test_read_before_the_write_in_the_same_step_counts_as_after(self):
        # big read sits in the same step as the second write, before it.
        steps = _run(
            before_calls=[_edit("a.py"), _read("tiny.py", _SMALL)],
            after_calls=[_read("big.py", _BIG), _edit("b.py")],
        )
        self.assertEqual(detect_compaction_aftermath(steps), [])

    def test_other_sessions_do_not_mix_in(self):
        # A huge read in another session must not dilute this session's Y.
        steps = _run(
            before_calls=[_edit("a.py"), _read("big.py", _BIG)],
            after_calls=[_write("b.py")],
            other_session_steps=[
                _step(4, session_id="s2", tool_calls=[_read("sub.py", _BIG * 3)]),
            ],
        )
        events = detect_compaction_aftermath(steps)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reads_after_tokens"], 0)

    def test_multiple_compactions_judged_independently(self):
        steps = [
            _step(0, role="user", parts=[{"type": "text", "text": "go"}]),
            _step(1, tool_calls=[_edit("a.py"), _read("one.py", _BIG)]),
            _compaction(2),
            _step(3, tool_calls=[_read("two.py", _SMALL), _edit("b.py"), _read("three.py", _BIG)]),
            _compaction(4),
            _step(5, tool_calls=[_write("c.py")]),
        ]
        events = detect_compaction_aftermath(steps)
        self.assertEqual([e["step"] for e in events], [2, 4])
        self.assertEqual(events[0]["prev_write_step"], 1)
        self.assertEqual(events[0]["next_write_step"], 3)
        self.assertEqual(events[1]["prev_write_step"], 3)
        self.assertEqual(events[1]["next_write_step"], 5)
        self.assertEqual(events[1]["reads_after_tokens"], 0)

    def test_no_compaction_no_events(self):
        steps = [
            _step(0, role="user", parts=[{"type": "text", "text": "go"}]),
            _step(1, tool_calls=[_edit("a.py"), _read("big.py", _BIG)]),
            _step(2, tool_calls=[_write("b.py")]),
        ]
        self.assertEqual(detect_compaction_aftermath(steps), [])

    def test_opencode_dual_signal_compaction_reported_once(self):
        # OpenCode leaves two signals for one compaction: the compaction
        # part on the user turn and the summary turn right after it.
        # They must yield one event, not two identical ones.
        steps = [
            _step(0, role="user", parts=[{"type": "text", "text": "go"}]),
            _step(1, tokens=_tokens(inp=15000), tool_calls=[_edit("a.py"), _read("big.py", _BIG)]),
            _step(2, role="user", parts=[{"type": "compaction", "summary": "prior work"}]),
            _step(3, tokens=_tokens(inp=3000), summary=True),
            _step(4, tokens=_tokens(inp=3000), tool_calls=[_edit("b.py")]),
        ]
        events = detect_compaction_aftermath(steps)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["step"], 2)
        self.assertEqual(events[0]["prev_write_step"], 1)
        self.assertEqual(events[0]["next_write_step"], 4)


class _Session:
    def __init__(self, events):
        self.compaction_aftermath = events


class AftermathIssueCardTests(unittest.TestCase):
    def _cards(self, **overrides):
        steps = _run(
            before_calls=[_edit("a.py"), _read("big.py", _BIG)],
            after_calls=[_read("small.py", _SMALL), _edit("b.py")],
            **overrides,
        )
        events = detect_compaction_aftermath(steps)
        return _from_compaction_aftermath(_Session(events)), events

    def test_card_fields(self):
        cards, events = self._cards()
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card.kind, "risk")
        self.assertIn("#2", card.title)
        self.assertIn("consecutive writes at steps 1 and 3", card.detail)
        self.assertIn("% less", card.detail)
        self.assertEqual(card.steps, (2, 3))
        self.assertTrue(card.source_id.startswith("antipattern:compaction_thin_read:"))
        self.assertTrue(card.why)

    def test_card_includes_dropped_tokens(self):
        cards, _ = self._cards()
        self.assertIn("compaction dropped", cards[0].detail)

    def test_card_localizes_to_chinese(self):
        cards, _ = self._cards()
        title, detail, why = _localize_issue_text(cards[0])
        self.assertIn("压缩后凭摘要改代码（第 2 步压缩之后）", title)
        self.assertIn("第 2 步压缩之后", title)
        self.assertIn("第 1、3 步两次连续写之间发生压缩", detail)
        self.assertIn("压缩前读取 ≈", detail)
        self.assertIn("少 ", detail)
        self.assertIn("%）", detail)
        self.assertIn("可能偏离文件真实状态", why)

    def test_card_skips_non_dict_events(self):
        issues = _from_compaction_aftermath(_Session([{"step": 2}, "junk", None]))
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
