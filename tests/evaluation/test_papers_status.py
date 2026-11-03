"""Tests for the paper-processing manifest helper.

Run:  python3 -m unittest tests.test_papers_status -v
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agent_trajviz.awe._engine.scripts import papers_status as ps


class TestPapersStatus(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.manifest = self.dir / "processed.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _pdf(self, name, content=b"%PDF-1.4 hello"):
        p = self.dir / name
        p.write_bytes(content)
        return p

    def test_unprocessed_lists_new_pdfs(self):
        self._pdf("a.pdf")
        self._pdf("b.pdf")
        self.assertEqual(set(ps.unprocessed(self.dir, self.manifest)), {"a.pdf", "b.pdf"})

    def test_processed_pdf_omitted(self):
        p = self._pdf("a.pdf")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        ps.record_processed(self.manifest, "a.pdf", sha, "2026-06-20", 2, 1)
        self.assertEqual(ps.unprocessed(self.dir, self.manifest), [])

    def test_changed_pdf_is_relisted(self):
        p = self._pdf("a.pdf")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        ps.record_processed(self.manifest, "a.pdf", sha, "2026-06-20", 2, 1)
        p.write_bytes(b"%PDF-1.4 CHANGED")
        self.assertEqual(ps.unprocessed(self.dir, self.manifest), ["a.pdf"])

    def test_record_processed_upsert_roundtrip(self):
        ps.record_processed(self.manifest, "a.pdf", "sha1", "2026-06-20", 1, 0)
        ps.record_processed(self.manifest, "a.pdf", "sha2", "2026-06-21", 3, 2)
        m = json.loads(self.manifest.read_text())
        self.assertEqual(m["a.pdf"]["sha256"], "sha2")
        self.assertEqual(m["a.pdf"]["refs_added"], 3)
        self.assertEqual(m["a.pdf"]["candidates"], 2)

    def test_unprocessed_empty_dir(self):
        self.assertEqual(ps.unprocessed(self.dir, self.manifest), [])


if __name__ == "__main__":
    unittest.main()
