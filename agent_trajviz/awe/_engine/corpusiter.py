"""The one corpus scan loop (v0.38, closing a §0an cleanup debt).

Every calibration/replay script iterates (agent × instance), skips cases
without an eval outcome, and builds the gold reference — previously five
copied loops that could drift. This is the shared form; per-script needs
(patch, trajectory signals) stay in the scripts.
"""

from __future__ import annotations

from . import config
from .gold import build_gold_reference
from .outcomes import resolved


def iter_cases(agents=None, instances=None):
    """Yield ``(agent, instance_id, resolved, gold)`` for every case with a
    known outcome (overrides applied — an excluded case is skipped like a
    missing one) and a loadable gold reference. Fails loudly on an empty
    instance list (``config.require_instances``)."""
    agents = list(agents) if agents is not None else config.AGENTS
    instances = config.require_instances(list(instances) if instances is not None else config.list_instances())
    for agent in agents:
        for inst in instances:
            res = resolved(agent, inst)
            if res is None:
                continue
            try:
                gold = build_gold_reference(inst)
            except Exception:
                continue
            yield agent, inst, res, gold
