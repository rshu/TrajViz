"""DEPRECATED shim. The grounded countermeasure catalog (agent_trajviz.awe._engine.countermeasures, backed
by configs/countermeasures.json) is the source of truth. This module re-exports the
back-compat surface so existing callers keep working; new code should import
agent_trajviz.awe._engine.countermeasures directly."""

from __future__ import annotations

from .countermeasures import recommend, load_catalog


def _build_recommendations() -> dict:
    """Reconstruct the legacy {error_type: (capability, [strategies])} map from the catalog."""
    out: dict[str, tuple[str, list[str]]] = {}
    for e in load_catalog():
        cap, strat = out.get(e.error_type, (e.capability, []))
        out[e.error_type] = (cap, strat + e.strategies)
    return out


RECOMMENDATIONS = _build_recommendations()

__all__ = ["recommend", "RECOMMENDATIONS"]
