"""Presentation of retained historical catalog notes, distinct from run evidence."""
from __future__ import annotations

import json
import re
from typing import Any

EVALUATION_GUIDE = "https://rshu.github.io/TrajViz/dev/awe.html"
HISTORICAL_CALIBRATION_PROVENANCE = {
    "origin": "bundled_historical_catalog",
    "computed_for_current_run": False,
    "raw_metadata_field": "calibration",
    "interpretation": "Stored historical calibration metadata, not measurements or efficacy evidence computed for this run.",
}


def historical_calibration_text(value: Any) -> str:
    """Qualify display text while leaving its complete original metadata intact.

    Numerical measurements and named studies are retained verbatim. Historical
    first-person claims and unavailable internal-document pointers are rendered
    as historical context, rather than active references for the current run.
    """
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = re.sub(r"\bour cohort(?: data)?\b", "the historical catalog cohort", text, flags=re.IGNORECASE)
    text = re.sub(r"\bour calibration\b", "the historical catalog calibration", text, flags=re.IGNORECASE)
    text = re.sub(r"DESIGN\.md(?:\s*§[\w.-]+)?", "the historical catalog design note", text)
    text = text.replace("data/discovery/pertest_pilot.md", "the historical catalog per-test pilot note")
    return (
        "Historical catalog note; not recomputed for this run and not evidence of current-run remedy efficacy: "
        + text
    )
