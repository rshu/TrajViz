#!/usr/bin/env python3
"""Track which paper PDFs in papers/ have been processed into the countermeasure catalog.

Stateless helper for the /extract-countermeasures workflow: lists PDFs not yet recorded
(by content sha256) in papers/processed.json and upserts manifest entries. It NEVER reads
the clock — the caller passes processed_at (the Claude session knows the date)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "papers"
MANIFEST = PAPERS_DIR / "processed.json"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def unprocessed(papers_dir: Path = PAPERS_DIR, manifest_path: Path = MANIFEST) -> list:
    """PDF filenames in papers_dir whose content sha256 is not recorded (new or changed)."""
    m = _load_manifest(manifest_path)
    out = []
    for pdf in sorted(papers_dir.glob("*.pdf")):
        rec = m.get(pdf.name)
        if not rec or rec.get("sha256") != _sha256(pdf):
            out.append(pdf.name)
    return out


def record_processed(
    manifest_path: Path, filename: str, sha256: str, processed_at: str, refs_added: int, candidates: int
) -> None:
    """Upsert a manifest entry for a processed paper."""
    m = _load_manifest(manifest_path)
    m[filename] = {"sha256": sha256, "processed_at": processed_at, "refs_added": refs_added, "candidates": candidates}
    manifest_path.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_of(filename: str, papers_dir: Path = PAPERS_DIR) -> str:
    """Content sha256 of a paper file (for the caller to pass to record_processed)."""
    return _sha256(papers_dir / filename)


def main() -> None:
    for name in unprocessed():
        print(name)


if __name__ == "__main__":
    main()
