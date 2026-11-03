"""Controlled evidence-retention checks, not a production accuracy benchmark."""

from __future__ import annotations

import copy

from .evidenceview import build_view
from .provenance import fingerprint
from .symptoms import analyze
from .trajectory import NormalizedStep as S, NormalizedTrajectory as T, edit_reversals, read_observations, excerpt
from .verification import verification_observations


def project(traj, mode):
    """Deterministic telemetry ablations; source artifacts are never modified."""
    clone = T.from_dict(copy.deepcopy(traj.to_dict()))
    if mode == "canonical":
        return clone
    if mode == "drop_results":
        clone.steps = [s for s in clone.steps if s.type != "tool_result"]
    elif mode == "drop_result_text":
        for s in clone.steps:
            if s.type == "tool_result":
                s.output_excerpt = ""
    elif mode == "drop_state":
        for s in clone.steps:
            for key in ("verification_state", "verification_state_conflicts"):
                s.args.pop(key, None)
    elif mode == "legacy_lossy_proxy":
        # Reproduce the former two main storage losses, not all old adapters.
        for s in clone.steps:
            s.output_excerpt = excerpt(s.output_excerpt)
            s.native_input, s.source_ref = {}, {}
            for key in ("old_content", "new_content"):
                s.args.pop(key, None)
            for key in ("old_lines", "new_lines"):
                if isinstance(s.args.get(key), list):
                    s.args[key] = s.args[key][:60]
    else:
        raise ValueError("unknown projection: " + mode)
    return clone


def describe(traj, task="Controlled task", focus_steps=None):
    reads, attempted, unanswered = read_observations(traj)
    verification = verification_observations(traj)
    analysis = analyze(traj)
    view = build_view(task, traj, focus_steps=focus_steps)
    return {
        "source": dict(traj.source),
        "n_steps": len(traj.steps),
        "trajectory_sha256": fingerprint(traj.to_dict()),
        "tool_output_chars": sum(len(s.output_excerpt) for s in traj.steps),
        "successful_reads": sorted(reads),
        "attempted_reads": sorted(attempted),
        "unanswered_reads": sorted(unanswered),
        "edit_reversals": edit_reversals(traj),
        "verification_summary_status": [r["summary"]["status"] for r in verification["runs"]],
        "verification_summary_reasons": [r["summary"].get("reason") for r in verification["runs"]],
        "symptom_kinds": sorted({s["kind"] for s in analysis["symptoms"]}),
        "view": view["omissions"],
    }


def _edit_pair(old, new, inverse):
    steps = []
    for i, (a, b) in enumerate([(old, new), (new, old) if inverse else (old, new)]):
        steps.append(
            S(
                len(steps),
                i,
                "agent",
                "tool_call",
                canonical_action="edit",
                files=["a.py"],
                call_id="e%d" % i,
                args={"old_content": a, "new_content": b},
            )
        )
        steps.append(S(len(steps), i, "tool", "tool_result", call_id="e%d" % i, output_excerpt="ok"))
    return T("controlled", "i", steps)


def controlled_validation():
    """Independent, explicit observational contracts plus masking controls."""
    cases = {}
    cases["observed_inverse"] = _edit_pair("x=1", "x=2", True)
    cases["repeated_blank_removal"] = _edit_pair("\n\nimport copy", "\nimport copy", False)
    cases["middle_runner_summary"] = T(
        "controlled",
        "i",
        [
            S(
                0,
                0,
                "agent",
                "tool_call",
                canonical_action="run_test",
                test_run=True,
                call_id="t",
                args={"command": "pytest tests/a.py"},
            ),
            S(
                1,
                0,
                "tool",
                "tool_result",
                call_id="t",
                output_excerpt="x" * 1000 + "\n1 passed in 0.1s\n" + "y" * 1000,
            ),
        ],
    )
    cases["observed_read"] = T(
        "controlled",
        "i",
        [
            S(0, 0, "agent", "tool_call", canonical_action="read", call_id="r", files=["a.py"]),
            S(1, 0, "tool", "tool_result", call_id="r", output_excerpt="content"),
        ],
    )
    cases["pending_read"] = T(
        "controlled",
        "i",
        [S(0, 0, "agent", "tool_call", canonical_action="read", call_id="r", files=["a.py"], status="pending")],
    )
    cases["revision_conflict"] = T(
        "controlled",
        "i",
        [
            S(
                0,
                0,
                "agent",
                "tool_call",
                canonical_action="run_test",
                test_run=True,
                call_id="t",
                args={"command": "go test ./...", "verification_state": {"revision": "a"}},
            ),
            S(
                1,
                0,
                "tool",
                "tool_result",
                call_id="t",
                args={
                    "verification_state": {"revision": "b"},
                    "verification_tests": {"complete": True, "results": [{"test_id": "TestClamp", "status": "passed"}]},
                },
            ),
        ],
    )
    modes = ("canonical", "legacy_lossy_proxy", "drop_results", "drop_result_text", "drop_state")
    reports = {name: {mode: describe(project(traj, mode)) for mode in modes} for name, traj in cases.items()}
    checks = {
        "observed_inverse_retained": bool(reports["observed_inverse"]["canonical"]["edit_reversals"]),
        "missing_edit_result_abstains": not reports["observed_inverse"]["drop_results"]["edit_reversals"],
        "lossy_edit_content_abstains": not reports["observed_inverse"]["legacy_lossy_proxy"]["edit_reversals"],
        "same_direction_whitespace_is_not_inverse": not reports["repeated_blank_removal"]["canonical"][
            "edit_reversals"
        ],
        "middle_summary_retained": reports["middle_runner_summary"]["canonical"]["verification_summary_status"]
        == ["reported"],
        "legacy_middle_summary_loss_visible": reports["middle_runner_summary"]["legacy_lossy_proxy"][
            "verification_summary_status"
        ]
        == ["unknown"],
        "masked_summary_unknown": reports["middle_runner_summary"]["drop_result_text"]["verification_summary_status"]
        == ["unknown"],
        "observed_read_retained": reports["observed_read"]["canonical"]["successful_reads"] == ["a.py"],
        "missing_read_result_unknown": reports["observed_read"]["drop_results"]["unanswered_reads"] == ["a.py"],
        "pending_read_not_success": not reports["pending_read"]["canonical"]["successful_reads"],
        "recorded_conflict_visible": "verification_state_conflict"
        in reports["revision_conflict"]["canonical"]["symptom_kinds"],
        "missing_state_does_not_invent_conflict": "verification_state_conflict"
        not in reports["revision_conflict"]["drop_state"]["symptom_kinds"],
    }
    # Stable focused middle call/result across a deliberately long trace.
    long = T("controlled", "long", [S(i, i, "agent", "message", text="routine") for i in range(80)])
    long.steps[40:42] = [
        S(40, 40, "agent", "tool_call", canonical_action="read", call_id="middle"),
        S(41, 40, "tool", "tool_result", call_id="middle", output_excerpt="MIDDLE FEEDBACK"),
    ]
    view = build_view("x" * 2500 + "\nKEEP FINAL CONSTRAINT", long, focus_steps=[40])
    checks["focused_middle_feedback_kept"] = {40, 41} <= {r["step"] for r in view["rows"]}
    checks["task_constraint_kept"] = view["problem_statement"].endswith("KEEP FINAL CONSTRAINT")
    checks["view_accounting_and_budget"] = (
        len(view["rows"]) <= 28 and view["omissions"]["retained_rows"] + view["omissions"]["omitted_rows"] == 80
    )
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "fixtures_sha256": fingerprint({name: traj.to_dict() for name, traj in cases.items()}),
        "cases": reports,
        "focused_view": view["omissions"],
        "scope": "controlled structural evidence retention and unknown-state behavior; no model accuracy or causal precision estimate",
    }
