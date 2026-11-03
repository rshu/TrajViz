#!/usr/bin/env python3
"""Emit a stratified human-calibration set: blank annotation tasks + the
detector's own predictions (kept separate so annotators aren't anchored).

  python scripts/build_calibration_set.py [n]
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine.sampling import stratified_sample
from agent_trajviz.awe._engine.annotation import build_task_record, detector_label

OUT = Path(__file__).resolve().parent.parent / "data" / "calibration"

INSTRUCTIONS = """# Calibration annotation — instructions

You are diagnosing WHERE each coding-agent run failed, using the closed taxonomy
(see `label_options` in each task). This calibrates all 7 workflow capabilities
(Requirement Understanding, Task Planning, Code Localization, Code Editing, Code
Verification, Self-Repair Loop, Tool Use). Note `code_editing` includes
`relevant_change_omitted` — the agent had a relevant file open but did not edit it.

For each case in `annotation_tasks.json`, read `problem_statement`, `gold_files`,
and `trajectory_summary`, then fill the `annotation` block:
  - primary_capability    : one of label_options.capabilities
  - primary_error_type    : one of label_options.errors_by_capability[primary_capability]
  - secondary_errors      : list of other error_types present (may be empty)
  - onset_step            : step_id where the run first went wrong
  - evidence_steps        : step_ids that justify your diagnosis
  - confidence            : low | medium | high
  - rationale             : 1-2 sentences

RULES
  - Do NOT consult the detector's output. Annotate independently.
  - The final test failure is NOT itself a root cause; find the earlier workflow error.
  - If nothing in the taxonomy fits, use primary_capability='other_capability' and explain.

OUTPUT
  Save your labels as `annotations/<your_name>.json`, a JSON object mapping
  case_id -> the filled annotation block. At least TWO annotators per case so we
  can measure human-human agreement (the ceiling). Then run:
      python scripts/score_agreement.py
"""


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    failed, solved = stratified_sample(n)
    cases = failed + solved
    tasks = [build_task_record(a, i) for a, i in cases]
    preds = {f"{a}__{i}": detector_label(a, i) for a, i in cases}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "annotations").mkdir(exist_ok=True)
    (OUT / "annotation_tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    (OUT / "detector_predictions.json").write_text(json.dumps(preds, indent=2), encoding="utf-8")
    (OUT / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")

    print(f"wrote {len(tasks)} tasks -> {OUT}")
    print(f"  failed={len(failed)} solved-control={len(solved)}")
    print("  strata (agent, detector_primary_capability):")
    strata = Counter((a, preds[f"{a}__{i}"]["primary_capability"]) for a, i in failed)
    for (a, cap), c in sorted(strata.items()):
        print(f"    {a:<15} {str(cap):<22} {c}")


if __name__ == "__main__":
    main()
