"""Redundant re-reads: the same file read again with the same parameters while
no successful write (write tool or shell rewrite like ``sed -i``) touched it
and no compaction dropped it from context.

Those events are exactly the cases where re-reading is justified, which keeps
this the honest opposite of the post-compaction thin-read signal.
"""

import unittest

from agent_trajviz.insight.patterns import detect_redundant_re_reads
from agent_trajviz.insight.issues import prepare_overview_issues
from agent_trajviz.insight.presenters.issues import build_overview_issues_html
from tests.overview_session_stub import overview_session as _session

def _step(index, calls=None, **extra):
    step = {"index": index, "role": "assistant", "tool_calls": calls or [], "tokens": {}}
    step.update(extra)
    return step

def _read(path="src/app.py", offset=None, limit=None, status="completed"):
    inp = {"filePath": path}
    if offset is not None:
        inp["offset"] = offset
    if limit is not None:
        inp["limit"] = limit
    return {"tool_name": "read", "status": status, "input": inp, "output": "..."}

def _write(path="src/app.py", status="completed"):
    return {
        "tool_name": "edit",
        "status": status,
        "input": {"filePath": path, "old_string": "x", "new_string": "y"},
        "output": "ok",
    }

class DetectRedundantReReadsTests(unittest.TestCase):
    def test_same_file_same_range_read_twice_is_flagged(self):
        steps = [
            _step(1, [_read()]),
            _step(4, [_read()]),
            _step(7, [_read()]),
        ]
        hits = detect_redundant_re_reads(steps)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["path"], "src/app.py")
        self.assertEqual(hits[0]["count"], 2)
        self.assertEqual(hits[0]["steps"], [4, 7])

    def test_different_ranges_are_different_content(self):
        steps = [
            _step(1, [_read(offset=1, limit=50)]),
            _step(4, [_read(offset=51, limit=50)]),
        ]
        self.assertEqual(detect_redundant_re_reads(steps), [])

    def test_write_in_between_justifies_the_re_read(self):
        steps = [
            _step(1, [_read()]),
            _step(3, [_write()]),
            _step(5, [_read()]),
        ]
        self.assertEqual(detect_redundant_re_reads(steps), [])

    def test_shell_sed_in_place_justifies_the_re_read(self):
        steps = [
            _step(1, [_read("/tmp/app.py")]),
            _step(3, [{
                "tool_name": "bash",
                "status": "completed",
                "input": {"command": "sed -i 's/a/b/' /tmp/app.py"},
                "output": "",
            }]),
            _step(5, [_read("/tmp/app.py")]),
        ]
        self.assertEqual(detect_redundant_re_reads(steps), [])

    def test_shell_redirect_write_justifies_the_re_read(self):
        steps = [
            _step(1, [_read("/tmp/out.txt")]),
            _step(3, [{
                "tool_name": "bash",
                "status": "completed",
                "input": {"command": "echo hello > /tmp/out.txt"},
                "output": "",
            }]),
            _step(5, [_read("/tmp/out.txt")]),
        ]
        self.assertEqual(detect_redundant_re_reads(steps), [])

    def test_failed_write_does_not_justify_the_re_read(self):
        steps = [
            _step(1, [_read()]),
            _step(3, [_write(status="error")]),
            _step(5, [_read()]),
        ]
        hits = detect_redundant_re_reads(steps)
        self.assertEqual(len(hits), 1)

    def test_compaction_part_step_justifies_the_re_read(self):
        # OpenCode compaction arrives as a user message with a compaction part
        # (message_type stays "user") — it must count as a checkpoint too.
        steps = [
            _step(1, [_read()]),
            {
                "index": 3,
                "role": "user",
                "tool_calls": [],
                "tokens": {},
                "parts": [{"type": "compaction", "summary": "...", "reason": "auto"}],
            },
            _step(5, [_read()]),
        ]
        self.assertEqual(detect_redundant_re_reads(steps), [])

    def test_compaction_in_between_justifies_the_re_read(self):
        steps = [
            _step(1, [_read()]),
            _step(3, [], is_compaction_checkpoint=True),
            _step(5, [_read()]),
        ]
        self.assertEqual(detect_redundant_re_reads(steps), [])

    def test_write_to_a_different_file_does_not_justify_it(self):
        steps = [
            _step(1, [_read("src/app.py")]),
            _step(3, [_write("src/other.py")]),
            _step(5, [_read("src/app.py")]),
        ]
        hits = detect_redundant_re_reads(steps)
        self.assertEqual(len(hits), 1)

    def test_subagent_reads_are_judged_independently(self):
        # A sub-agent holds its own context: reading a file the parent already
        # read is not redundant for the sub-agent's conversation.
        steps = [
            _step(1, [_read()], session_id="root"),
            _step(7, [_read()], session_id="sub"),
        ]
        self.assertEqual(detect_redundant_re_reads(steps), [])

    def test_single_read_is_never_flagged(self):
        steps = [_step(1, [_read()])]
        self.assertEqual(detect_redundant_re_reads(steps), [])

    def test_paths_are_reported_separately_most_redundant_first(self):
        steps = [
            _step(1, [_read("a.py")]),
            _step(2, [_read("b.py")]),
            _step(3, [_read("a.py")]),
            _step(4, [_read("b.py")]),
            _step(5, [_read("b.py")]),
        ]
        hits = detect_redundant_re_reads(steps)
        self.assertEqual([h["path"] for h in hits], ["b.py", "a.py"])
        self.assertEqual([h["count"] for h in hits], [2, 1])

class RedundantReReadCardTests(unittest.TestCase):
    def _hits(self):
        return [{"path": "src/app.py", "count": 2, "steps": [4, 7]}]

    def test_card_is_built_from_the_detector_output(self):
        issues = prepare_overview_issues(_session(redundant_re_reads=self._hits()))
        card = [i for i in issues if i.source_id == "antipattern:redundant_reread:0"]
        self.assertEqual(len(card), 1)
        self.assertEqual(card[0].kind, "antipattern")
        self.assertEqual(card[0].title, "Re-read an unchanged file (2×)")
        self.assertIn(
            "src/app.py re-read 2× with no write or compaction in between "
            "(steps 4, 7)",
            card[0].detail,
        )
        self.assertEqual(card[0].steps, (4, 7))

    def test_no_card_when_nothing_was_flagged(self):
        issues = prepare_overview_issues(_session())
        self.assertFalse([i for i in issues if "redundant_reread" in i.source_id])

    def test_card_is_localized_to_chinese(self):
        html = build_overview_issues_html(_session(redundant_re_reads=self._hits()))
        self.assertIn("重复读取未变更的文件 (2×)", html)
        self.assertIn("src/app.py 重复读取 2 次", html)
        self.assertIn("（第 4、7 步）", html)
        self.assertIn("内容本来就在上下文里", html)
        self.assertIn("让智能体依赖已读内容", html)

if __name__ == "__main__":
    unittest.main()
