#!/usr/bin/env python3
"""Render explicitly saved evaluation evidence as safe offline HTML/SVG."""

import json
from pathlib import Path

from agent_trajviz.awe.showcase import render_showcase

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    source = ROOT / "data" / "showcase_data.json"
    output = ROOT / "data" / "showcase_results.html"
    document = render_showcase(json.loads(source.read_text(encoding="utf-8")))
    output.write_text(document, encoding="utf-8")
    print(f"Saved results showcase: {output}")


if __name__ == "__main__":
    main()
