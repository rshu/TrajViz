"""Bridge module: orchestrates Converge's comparison pipeline from Insight's UI.

Delegates to converge.alignment.build_comparison_report_from_steps — the same
core the file-path entry point (build_comparison_report) uses — so the two
pipelines cannot drift. This module only adapts already-loaded trajectory
dicts, parses the optional anchor patch via the shared helper, and renders the
HTML report.
"""

import html
import logging

from ..i18n import text as ui_text

from agent_trajviz.insight.parser import parse_steps

from agent_trajviz.converge.alignment import (
    DEFAULT_TOKEN_RATE,
    TranscriptAlignmentUnavailable,
    _parse_anchor_files,
    build_comparison_report_from_steps,
    empty_steps_reason,
)
from agent_trajviz.converge.rendering import build_comparison_report_html

logger = logging.getLogger(__name__)


def run_comparison(
    ref_raw: dict,
    cmp_raw: dict,
    anchor_path: str | None = None,
    token_rate: float = DEFAULT_TOKEN_RATE,
    fuzzy: bool = False,
    dark: bool = False,
    ref_labels: dict[int, dict[str, str]] | None = None,
    cmp_labels: dict[int, dict[str, str]] | None = None,
    *,
    ref_steps: list[dict] | None = None,
    cmp_steps: list[dict] | None = None,
    ref_path: str | None = None,
    cmp_path: str | None = None,
) -> dict:
    """Run Converge's full comparison pipeline.

    Parameters
    ----------
    ref_raw : dict
        Raw trajectory dict for the **reference / baseline** trajectory —
        the one the user uploaded on the Comparison tab (left of the anchor
        patch slot). Already loaded via ``load_trajectory``.
    cmp_raw : dict
        Raw trajectory dict for the **compared** trajectory — the one the
        user loaded on the Overview tab. Already loaded via ``load_trajectory``.
    anchor_path : str or None
        Optional path to a .patch/.diff file for anchor-grounded comparison.
    token_rate : float
        Token rate for cost computation (default 50.0).
    fuzzy : bool
        Enable fuzzy command matching.
    dark : bool
        Unused (the HTML report is theme-agnostic); retained for API
        stability with the Insight caller.
    ref_labels, cmp_labels : dict or None
        Optional step-index → {'phase', 'action'} maps from the step labeler.
        When supplied, CanonicalActions carry phase/action labels and
        divergence confidence scoring becomes phase-aware. Opt-in on
        purpose: the batch path must stay label-free so the
        ``patterns[].confidence`` values it already reported are not
        retroactively rescored.
    ref_steps, cmp_steps : list or None
        The runs' steps when already parsed (a loaded ``Trajectory``, of any
        format); otherwise they are parsed from the raw dicts.
    ref_path, cmp_path : str or None
        The runs' file paths (default: the raw dicts' ``_source_path``).

    Returns
    -------
    dict with keys:
        report_html : str
            The rendered comparison report (or an error banner on failure).
        ok : bool
            True when the comparison ran to completion; False when a
            trajectory failed to load, parsed to no steps, or the pipeline
            raised (B15's producer side — the caller branches its status
            line on this).
    """
    empty = {"report_html": "", "ok": False}

    try:
        # Escaped: a loader error quotes the uploaded file's path, and the file
        # name is attacker-chosen as far as this renderer is concerned.
        if "_error" in ref_raw:
            empty["report_html"] = (
                f"<div style='color:var(--ov-bad);padding:1em;'>"
                f"{ui_text('Error loading reference trajectory:')} {html.escape(str(ref_raw['_error']))}</div>"
            )
            return empty
        if "_error" in cmp_raw:
            empty["report_html"] = (
                f"<div style='color:var(--ov-bad);padding:1em;'>"
                f"{ui_text('Error loading compared trajectory:')} {html.escape(str(cmp_raw['_error']))}</div>"
            )
            return empty

        if ref_steps is None:
            ref_steps = parse_steps(ref_raw)
        if cmp_steps is None:
            cmp_steps = parse_steps(cmp_raw)
        if ref_path is None:
            ref_path = ref_raw.get("_source_path", "")
        if cmp_path is None:
            cmp_path = cmp_raw.get("_source_path", "")

        # The `_error` checks above are not enough: `load_trajectory`
        # returns an unrecognised JSON *object* with no `_error` at all, and it
        # then scores 0.0 on every metric. Same shared guard the file-path
        # entry point raises on, so the two cannot drift again.
        for label, raw, steps in (
            ("reference", ref_raw, ref_steps),
            ("compared", cmp_raw, cmp_steps),
        ):
            reason = empty_steps_reason(
                label, raw, ref_path if label == "reference" else cmp_path, steps)
            if reason:
                # Escaped: the reason quotes the uploaded file's path.
                empty["report_html"] = (
                    f"<div style='color:var(--ov-bad);padding:1em;'>"
                    f"{html.escape(reason)}</div>"
                )
                return empty

        report = build_comparison_report_from_steps(
            ref_raw, cmp_raw, ref_steps, cmp_steps,
            token_rate=token_rate,
            fuzzy_commands=fuzzy,
            anchor_files=_parse_anchor_files(anchor_path),
            ref_path=ref_path,
            cmp_path=cmp_path,
            ref_labels=ref_labels,
            cmp_labels=cmp_labels,
        )

        # Render HTML report. The Insight UI's Comparison tab suppresses the
        # divergence-patterns section (heading, glossary, table) — the patterns
        # are still computed for downstream consumers but are hidden from the
        # rendered report. The standalone Converge app (converge/app.py) goes
        # through build_comparison_report directly and is unaffected.
        report_html = build_comparison_report_html({**report, "patterns": []})

        return {"report_html": report_html, "ok": True}

    except TranscriptAlignmentUnavailable:
        # A known evidence limitation is actionable; never echo exception
        # text, which may be changed by a caller or contain private paths.
        return {
            "ok": False,
            "unavailable": "transcript_alignment",
            "report_html": (
                "<div style='color:var(--ov-warn);padding:1em;'>"
                f"<strong>{ui_text('Transcript alignment unavailable')}</strong>"
                f"<p>{ui_text('This capture records operations rather than a native conversation transcript. Transcript alignment and divergence scores have not been computed.')}</p>"
                f"<p>{ui_text('Inspect observed token usage, timing and tool metrics in Overview, or compare those measurements in Run group. Use native session logs for transcript alignment.')}</p></div>"
            ),
        }
    except Exception as e:
        # The traceback goes to the log, not into the page: it
        # carries absolute filesystem paths, and the exception text itself is
        # derived from trajectory content. Same split `do_load` uses — the
        # viewer gets the exception TYPE, the person running the server gets
        # everything needed to debug it.
        logger.exception("comparison failed")
        empty["report_html"] = (
            f"<div style='color:var(--ov-bad);padding:1em;'>"
            f"<strong>{ui_text('Comparison failed:')}</strong> {html.escape(type(e).__name__)}"
            f"<div style='font-size:12px;margin-top:6px;color:var(--ov-muted);'>"
            f"{ui_text('Details are in the server log.')}</div></div>"
        )
        return empty
