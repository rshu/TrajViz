"""Conservative Python assertion weakening candidates, never scored faults.

Only complete, parseable diff hunks containing actual test definitions qualify.
Hunk headings are hints, not executable context. Expected-value substitutions
are not inherently weaker and are deliberately excluded.
"""

import ast
import textwrap
from .diffutil import parse_unified_diff, is_test_file


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _functions(lines):
    try:
        tree = ast.parse(textwrap.dedent("\n".join(lines)))
    except (SyntaxError, ValueError):
        return {}
    result = {}

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                name = prefix + node.name
                if name in result:
                    result[name] = None  # duplicate definitions are ambiguous
                else:
                    result[name] = [s for s in node.body if isinstance(s, ast.Assert)]

    visit(tree.body)
    return result


def _number(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number(node.operand)
        return -value if value is not None else None
    return None


def _relaxed(old, new):
    if isinstance(new, ast.Constant) and new.value is True:
        return "assertion_replaced_by_true" if _dump(old) != _dump(new) else None
    if not (
        isinstance(old, ast.Compare)
        and isinstance(new, ast.Compare)
        and len(old.ops) == len(new.ops) == 1
        and _dump(old.left) == _dump(new.left)
    ):
        return None
    a, b = _number(old.comparators[0]), _number(new.comparators[0])
    if a is None or b is None:
        return None
    less, greater = (ast.Lt, ast.LtE), (ast.Gt, ast.GtE)
    for group, direction in ((less, 1), (greater, -1)):
        if isinstance(old.ops[0], group) and isinstance(new.ops[0], group):
            if direction * (b - a) > 0 or (a == b and type(old.ops[0]) is group[0] and type(new.ops[0]) is group[1]):
                return "numeric_bound_relaxed"
    return None


def oracle_candidates(patch_text):
    rows, eligible, unknown = [], 0, 0
    for fd in parse_unified_diff(patch_text or ""):
        if not is_test_file(fd.path) or not fd.path.endswith(".py") or fd.is_new or fd.is_deleted:
            continue
        parsed = []
        for h in fd.hunk_contents:
            old, new = _functions(h["old"]), _functions(h["new"])
            if not h["complete"] or not old or not new:
                unknown += 1
                continue
            parsed.append((h, old, new))
        # Suppress assertion relocation when the same AST appears in another
        # parseable hunk in this file. No cross-function replacement pairing.
        retained = {
            _dump(a) for _, _, new in parsed for asserts in new.values() if asserts is not None for a in asserts
        }
        for h, old, new in parsed:
            for name in old.keys() & new.keys():
                before, after = old[name], new[name]
                if before is None or after is None:
                    unknown += 1
                    continue
                eligible += 1
                removed = [a for a in before if _dump(a) not in retained]
                added = [a for a in after if _dump(a) not in {_dump(b) for b in before}]
                # A single changed assertion is pairable. Multiple changes
                # could belong to unrelated checks within the same test.
                kind = None
                if len(removed) == 1 and not added:
                    kind = "assertion_removed_in_hunk"
                elif len(removed) == len(added) == 1:
                    kind = _relaxed(removed[0].test, added[0].test)
                if kind:
                    rows.append(
                        {
                            "symptom_id": f"oracle_weakening_candidate:{fd.path}:{h['old_start']}:{name}",
                            "kind": "oracle_weakening_candidate",
                            "observation_strength": "structural_patch_candidate",
                            "evidence": {
                                "file": fd.path,
                                "old_hunk_start": h["old_start"],
                                "function": name,
                                "change": kind,
                                "old_assertion_ast": _dump(removed[0]),
                                "new_assertion_ast": _dump(added[0]) if added else None,
                            },
                            "assumptions": ["numeric bounds assume ordinary numeric comparison"]
                            if kind == "numeric_bound_relaxed"
                            else [],
                            "lifecycle": "final_patch_candidate",
                            "terminal_relevance": "unknown",
                            "causal_validation": "not_tested",
                            "scoring": "not_scored",
                        }
                    )
    return {
        "symptoms": rows,
        "coverage": {
            "eligible_test_fragments": eligible,
            "unknown_hunks_or_definitions": unknown,
            "scope": "complete Python hunks with parsed test definitions",
        },
    }
