"""Bounded contradictions between an observed result and an explicit agent claim.

Coverage is deliberately narrow: an entire message must be one supported claim.
No semantic inference from prose, quoted output, or ambiguous parallel calls.
"""

from .trajectory import OBSERVED, SEARCH
from .verification import tool_observations, ref


def _claim(text):
    # Entire-message matching also excludes questions, code, quotations, and
    # hypothetical examples rather than treating them as endorsed assertions.
    return {
        "The last command succeeded.": ("command", True),
        "The last command failed.": ("command", False),
        "The last test run passed.": ("tests", True),
        "The last search found no matches.": ("search", False),
        "The last search found matches.": ("search", True),
    }.get((text or "").strip())


def contradictions(traj, verification):
    if traj is None:
        return []
    test_runs = {r["call"]["position"]: r for r in verification.get("runs", [])}
    rows = []
    for cp, call, rp, result, linkage in tool_observations(traj):
        if result is None:
            continue
        between = [s for s in traj.steps[cp + 1 : rp] if s.subagent_id == call.subagent_id]
        if any(s.type in ("tool_call", "tool_result") for s in between):
            continue  # "last" would be ambiguous with concurrent tool work
        for pos in range(rp + 1, len(traj.steps)):
            step = traj.steps[pos]
            if step.subagent_id != call.subagent_id:
                continue
            if step.type in ("tool_call", "tool_result"):
                break
            if step.actor != "agent" or step.type not in ("message", "final_answer"):
                continue
            # Only the first agent message following this result is eligible.
            claim = _claim(step.text) if step.observability == OBSERVED else None
            if claim is None:
                break
            category, positive = claim
            fact, conflicts = None, False
            code = (result.args or {}).get("exit_code")
            if category == "command" and (call.args or {}).get("command") and type(code) is int:
                fact = {"exit_code": code}
                # An explicit tool-error flag conflicts with a successful exit;
                # that combination cannot establish success for either claim.
                conflicts = positive and code != 0 or (not positive and code == 0 and not result.tool_error)
            elif category == "tests":
                summary = test_runs.get(cp, {}).get("summary", {})
                counts = summary.get("counts") or {}
                failures = counts.get("failed")
                if summary.get("status") == "reported" and type(failures) is int:
                    fact = {"runner": summary["runner"], "counts": dict(counts)}
                    conflicts = failures > 0
            elif category == "search" and call.canonical_action == SEARCH:
                import re

                output = result.output_excerpt.strip()
                m = re.fullmatch(r"Found (\d+) matches\.?", output)
                count = int(m[1]) if m else 0 if output == "No matches found." else None
                if count is not None:
                    fact = {"matches": count}
                    conflicts = positive != (count > 0)
            if conflicts:
                rows.append(
                    {
                        "symptom_id": f"tool_output_contradiction:{cp}:{pos}",
                        "kind": "tool_output_contradiction",
                        "observation_strength": "explicit_claim_against_bounded_result",
                        "call": ref(cp, call),
                        "result": ref(rp, result),
                        "claim": ref(pos, step),
                        "evidence": {"category": category, "claim": step.text.strip(), "fact": fact},
                        "lifecycle": "claim_observed",
                        "terminal_relevance": "unknown",
                        "causal_validation": "not_tested",
                        "scoring": "not_scored",
                    }
                )
            break
    return rows
