#!/usr/bin/env python3
"""Refresh recommendation fields of an existing dossier without re-judging."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.dossier import CSV_COLUMNS, to_csv_rows, to_markdown
from agent_trajviz.awe._engine.remedies import refresh_case
from agent_trajviz.awe._engine.countermeasures import load_catalog
from agent_trajviz.awe._engine.report import remedy_lines


def refresh_report_advice(text):
    """Replace recognized current-catalog blocks, retaining scores and examples."""
    entries = {e.summary: e for e in load_catalog()}
    by_id = {e.id: e for e in load_catalog()}
    lines = text.splitlines()
    output = []
    i = 0
    while i < len(lines):
        match = re.match(r"^- \*\*(.*?)\*\* _\(.*?diagnostic match in (\d+) cases,", lines[i])
        recipe_id = re.search(r"; recipe ([A-Z0-9-]+);", lines[i])
        entry = None
        if match:
            entry = by_id.get(recipe_id[1]) if recipe_id else entries.get(match[1])
        if match and entry is not None:
            output.extend(remedy_lines(entry, int(match[2])))
            i += 1
            while i < len(lines) and lines[i].startswith("  "):
                i += 1
        else:
            output.append(lines[i])
            i += 1
    return "\n".join(output) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, default=config.OUTPUT_DATA / "dossier")
    parser.add_argument(
        "--reports", type=Path, help="Also refresh advice blocks in existing reports; preserve their scores"
    )
    args = parser.parse_args()
    path = args.dossier / "faults.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        refresh_case(case)
    path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    with (args.dossier / "faults.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(to_csv_rows(cases))
    for agent in sorted({case["agent"] for case in cases}):
        (args.dossier / (agent + ".md")).write_text(to_markdown([case for case in cases if case["agent"] == agent]), encoding="utf-8")
        if args.reports is not None:
            report = args.reports / (agent + ".md")
            if report.exists():
                report.write_text(refresh_report_advice(report.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Refreshed remedy decisions for {len(cases)} existing cases; no attribution rerun.")


if __name__ == "__main__":
    main()
