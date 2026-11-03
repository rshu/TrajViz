#!/usr/bin/env python3
"""Emit BibTeX for the paper/preprint references in the countermeasure catalog."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_trajviz.awe._engine.countermeasures import load_catalog


def _key(r):
    a = re.sub(r"[^A-Za-z]", "", (r.get("authors") or "ref").split()[0])
    return f"{a}{r.get('year', '')}"


def main():
    seen, out = set(), []
    for e in load_catalog():
        for r in e.references:
            if r.get("source_type") not in ("paper", "preprint"):
                continue
            k = _key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(
                "@misc{%s,\n  title={%s},\n  author={%s},\n  year={%s},\n  note={%s}\n}"
                % (k, r.get("title", ""), r.get("authors", ""), r.get("year", ""), r.get("id", ""))
            )
    Path("data/countermeasures.bib").write_text("\n\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote data/countermeasures.bib ({len(out)} entries)")


if __name__ == "__main__":
    main()
