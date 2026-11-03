#!/usr/bin/env python3
"""Countermeasures grouped by SCAFFOLD LAYER — the developer-facing "where do I plug
this in?" view of the catalog (system_prompt / tool_definition / control_flow /
retry_policy / response_synthesis). Orthogonal to the artifact form
(prompt/guardrail/skill/plugin); every remedy keeps its grounding.

Run:  python3 scripts/scaffold_view.py [--write]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_trajviz.awe._engine.countermeasures import scaffold_layer_view_md

OUT = Path(__file__).resolve().parent.parent / "data" / "discovery"


def main():
    md = scaffold_layer_view_md()
    print(md)
    if "--write" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "countermeasures_by_layer.md").write_text(md, encoding="utf-8")
        print(f"wrote {OUT / 'countermeasures_by_layer.md'}")


if __name__ == "__main__":
    main()
