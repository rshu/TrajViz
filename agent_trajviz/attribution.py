"""Built-in failure attribution: workflow capabilities and their evidence.

Reference patches and outcomes ground each verdict. Without matching inputs,
``diagnose`` returns an unavailable result and its reason; it never guesses.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .views import _jsonable

if TYPE_CHECKING:
    from .api import Trajectory

__all__ = ["available", "diagnose"]


def available() -> tuple[bool, str | None]:
    """``(True, None)`` when the bundled attribution engine can be imported, else ``(False, reason)``."""
    from .insight import attribution as _attr

    return bool(_attr.ATTRIBUTION_AVAILABLE), (None if _attr.ATTRIBUTION_AVAILABLE else str(_attr._IMPORT_ERROR))


def diagnose(
    run: Trajectory,
    *,
    agent: str | None = None,
    instance_id: str | None = None,
    reference_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Attribute *run*'s failure to workflow capabilities, as plain data.

    *agent* and *instance_id* default to the run's file layout
    (``.../<agent>/<instance_id>.json``). The displayed content is checked
    against the reference set by hash, so a file that changed after loading
    is refused rather than diagnosed. Returns attribution's result as a dict:
    ``available``, ``reason``, ``mode``, the primary failure, the capability
    scorecard and the faults.
    """
    from .insight import attribution as _attr

    src = run.source or ""
    if src:
        instance_id = instance_id or os.path.splitext(os.path.basename(src))[0]
        agent = agent or os.path.basename(os.path.dirname(src))
    result = _attr.diagnose(
        agent=agent or None,
        instance_id=instance_id or None,
        source_path=src or None,
        fmt=run.format,
        expected_sha=run.sha256,
        merged_sources=int(run.raw.get("_source_merged_count") or 0),
        reference_root=reference_root,
    )
    return _jsonable(asdict(result))
