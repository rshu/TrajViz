#!/usr/bin/env python3
"""Score inter-rater agreement for the calibration set.

python scripts/score_agreement.py            # real: reads data/calibration/annotations/*.json
python scripts/score_agreement.py --mock     # validate the scorer with synthetic annotators
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine.agreement import human_human, detector_vs_consensus, onset_metrics
from agent_trajviz.awe._engine.annotation import CAPABILITIES, taxonomy_options

CAL = Path(__file__).resolve().parent.parent / "data" / "calibration"


def _load_real():
    preds = json.loads((CAL / "detector_predictions.json").read_text(encoding="utf-8"))
    annotators = {}
    for f in sorted((CAL / "annotations").glob("*.json")):
        annotators[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return preds, annotators


def _make_mock():
    preds = json.loads((CAL / "detector_predictions.json").read_text(encoding="utf-8"))
    caps = [c for c in CAPABILITIES if c != "other_capability"]
    errs = [e for v in taxonomy_options().values() for e in v]
    rng = random.Random(7)
    annotators = {}
    for name, p in (("ann_A", 0.85), ("ann_B", 0.80), ("ann_C", 0.75)):
        labels = {}
        for cid, pred in preds.items():
            tcap = pred["primary_capability"] or "code_editing"
            terr = pred["primary_error_type"] or "incorrect_patch"
            labels[cid] = {
                "primary_capability": tcap if rng.random() < p else rng.choice(caps),
                "primary_error_type": terr if rng.random() < p else rng.choice(errs),
                "onset_step": rng.randint(1, 12),
            }
        annotators[name] = labels
    return preds, annotators


def main() -> None:
    mock = "--mock" in sys.argv
    preds, annotators = _make_mock() if mock else _load_real()
    if not annotators:
        print(
            "No annotations found in data/calibration/annotations/. "
            "Run build_calibration_set.py, have annotators fill them, or use --mock."
        )
        return
    case_ids = list(preds)
    print(
        f"\n=== Agreement ({'MOCK synthetic annotators' if mock else 'real'}) — "
        f"{len(annotators)} annotators, {len(case_ids)} cases ===\n"
    )

    for field in ("primary_capability", "primary_error_type"):
        hh = human_human(annotators, case_ids, field)
        dv = detector_vs_consensus(preds, annotators, case_ids, field)
        print(f"{field}:")
        print(
            f"  human-human   : Fleiss kappa={_f(hh['fleiss_kappa'])}  "
            f"pairwise mean Cohen={_f(hh['pairwise_mean_kappa'])}  (the ceiling)"
        )
        print(f"  detector<->consensus: accuracy={_f(dv['accuracy'])}  kappa={_f(dv['kappa'])}  (n={dv['n']})")

    # onset (pairwise of first two annotators)
    names = list(annotators)
    if len(names) >= 2:
        a = [annotators[names[0]].get(c, {}).get("onset_step") for c in case_ids]
        b = [annotators[names[1]].get(c, {}).get("onset_step") for c in case_ids]
        om = onset_metrics(a, b)
        print(
            f"\nonset_step ({names[0]} vs {names[1]}): exact={_f(om['exact_match'])}  "
            f"mean_abs_dist={_f(om['mean_abs_dist'])}  (n={om['n']})"
        )
    if mock:
        print("\n(MOCK validates the scorer + flow. Real numbers require human labels.)")


def _f(x):
    return "-" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    main()
