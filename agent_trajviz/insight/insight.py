"""Gradio UI for TrajViz — Blocks composer."""

from __future__ import annotations

import gradio as gr

from ..i18n.ui import bind_language, presentation

from .llm_config import load_env_files
from .charts.activity import (
    FILE_INTERACTION_CHROME_PX,
    FILE_INTERACTION_MIN_HEIGHT,
    FILE_INTERACTION_ROW_PX,
)
from .styles import APP_CSS
from .ui import (
    attribution_tab,
    awe_tab,
    comparison_tab,
    overview_tab,
    patterns_tab,
    raw_tab,
    sidebar,
    settings,
    trajectory_inspection,
    upload,
    views_tab,
    workflow_tab,
)
from .ui.load import bind_load, merge_load_slots
from .ui.shared import SharedState

# Gradio 6 moved `css`/`head` from the Blocks constructor to `launch()` (passing
# them to gr.Blocks is warned about and silently dropped), so `build_ui()` is
# deliberately unstyled and every embedder must spread this at launch — without
# it the page is served bare, including the light-mode pinning below.
LAUNCH_PRESENTATION = presentation({
    "css": APP_CSS,
    "head": '<meta name="color-scheme" content="light">',
    # The application has its own Settings panel; avoid a second settings menu.
    "footer_links": ["gradio"],
})


def build_ui(*, allow_server_paths: bool = True) -> gr.Blocks:
    """Build the full Gradio Blocks UI.

    ``allow_server_paths=False`` removes every input that names a path on the
    serving machine (the AWE attribution view's reference-data root), and the
    callbacks ignore such a value even if a client sends one. Launch with it
    whenever the app is reachable from other machines.
    """
    load_env_files()

    # analytics_enabled=False: Gradio otherwise fetches a version check and
    # reports launches to its servers; nothing leaves the machine unasked.
    with gr.Blocks(title="TrajViz", elem_classes=["trajectory-viz"], analytics_enabled=False) as app:
        shared = SharedState(
            state_steps=gr.State([]),
            # Light-only by design: the app.load handler below is the single
            # writer of this State and pins it False, and APP_CSS declares
            # `color-scheme: light`. `dark=` stays threaded through so the
            # `--report --dark` CLI keeps working, and this State is where a
            # future toggle would write (every chart would have to re-render).
            state_dark=gr.State(False),
            state_raw=gr.State({}),
            state_analysis_brief=gr.State(""),
            state_run=gr.State(None),
            state_settings=gr.State(None),
        )

        sidebar_refs = sidebar.layout()
        settings_refs = sidebar_refs.settings
        upload_refs = upload.layout()
        state_awe_mode = gr.State("inspection")
        # Global health strip stays above the main navigation.
        overview_kpi_html = gr.HTML(
            "",
            visible=False,
            elem_classes=["overview-kpi-strip"],
        )
        with gr.Tabs(visible=True, elem_id="main-tabs") as main_tabs:
            overview = overview_tab.layout(overview_kpi_html)
            patterns = patterns_tab.layout()
            with gr.Tab("AWE", elem_id="awe-workspace"):
                gr.Markdown("Inspect the loaded trajectory or explore saved evaluation results.")
                with gr.Tabs(selected="inspection", elem_id="awe-workspace-tabs") as awe_source_tabs:
                    with gr.Tab("Trajectory", id="inspection",
                                elem_id="awe-trajectory-workspace") as trajectory_workspace:
                        inspection = trajectory_inspection.layout()
                        with (
                            inspection.optional_controls,
                            gr.Accordion("Reference-based attribution (optional)", open=False,
                                         elem_id="awe-reference-attribution"),
                        ):
                            attribution = attribution_tab.layout(allow_server_paths=allow_server_paths, embedded=True)
                    with gr.Tab("Saved runs & evaluation", id="saved",
                                elem_id="awe-saved-workspace") as saved_workspace:
                        awe = awe_tab.layout(allow_server_paths=allow_server_paths, embedded=True)
            comparison = comparison_tab.layout()
            workflow = workflow_tab.layout()
            raw = raw_tab.layout()
            # Only when views were registered (by the application or a plugin).
            views = views_tab.layout()

        def _workspace_visibility(run, mode="inspection", is_awe=False):
            native = not (is_awe and mode == "saved")
            loaded = bool(run is not None and run.steps)
            return (gr.update(visible=native), gr.update(visible=native),
                    gr.update(visible=native and loaded), gr.update(visible=native and loaded))

        def on_main_tab_select(run, awe_mode, *, tab_label):
            mode = "saved" if awe_mode == "saved" else "inspection"
            visibility = _workspace_visibility(run, mode, is_awe=tab_label == "AWE")
            selection = gr.update(selected=mode) if tab_label == "AWE" else gr.skip()
            return (*visibility, selection)

        def on_awe_workspace_select(run, *, mode):
            return (mode, *_workspace_visibility(run, mode, is_awe=True))

        # The two AWE views retain their own data. Only the saved-run view hides
        # native-session controls; returning to AWE restores the selected view.
        from functools import partial
        workspace_outputs = [upload_refs.upload_accordion, sidebar_refs.analysis_sidebar,
                             overview_kpi_html, upload_refs.summary_area]
        for tab in main_tabs.children:
            if isinstance(tab, gr.Tab):
                tab.select(fn=partial(on_main_tab_select, tab_label=tab.label),
                           inputs=[shared.state_run, state_awe_mode],
                           outputs=[*workspace_outputs, awe_source_tabs], api_name=False)
        for workspace, mode in ((trajectory_workspace, "inspection"), (saved_workspace, "saved")):
            workspace.select(fn=partial(on_awe_workspace_select, mode=mode),
                             inputs=shared.state_run,
                             outputs=[state_awe_mode, *workspace_outputs], api_name=False)

        slots = merge_load_slots(
            main_tabs=main_tabs,
            shared=shared,
            refs={
                upload: upload_refs,
                overview_tab: overview,
                patterns_tab: patterns,
                workflow_tab: workflow,
                raw_tab: raw,
            },
        )
        # One Python load path; picking a file re-clicks Load in the browser.
        # Everything chained off these events (export writer, sidebar hook,
        # AWE trajectory inspection) therefore runs once per gesture.
        load_events = bind_load(
            file_upload=upload_refs.file_upload,
            load_btn=upload_refs.load_btn,
            format_selector=upload_refs.format_selector,
            session_selector=upload_refs.session_selector,
            state_dark=shared.state_dark,
            slots=slots,
        )
        upload.bind_export(upload_refs, shared, load_events)

        sidebar_events = sidebar.bind(sidebar_refs, shared, load_events)
        judge_events = overview_tab.bind(overview, shared, upload_refs, load_events)
        settings.bind(settings_refs, shared, sidebar_refs, overview, app,
                      model_events=sidebar_events + judge_events)
        attribution_tab.bind(attribution, shared, upload_refs, load_events,
                             allow_server_paths=allow_server_paths, auto_diagnose=False)
        trajectory_inspection.bind(inspection, shared, load_events)
        awe_tab.bind(awe, app, allow_server_paths=allow_server_paths, language=upload_refs.language)
        comparison_tab.bind(comparison, shared, upload_refs)
        workflow_tab.bind(workflow, shared)
        views_tab.bind(views, shared, load_events)
        bind_language(app, upload_refs.language)

        app.load(
            fn=lambda _dark=False: False,
            outputs=[shared.state_dark],
            js="""() => {
                document.documentElement.style.colorScheme = 'light';
                document.documentElement.classList.remove('dark');
                if (document.body) document.body.classList.remove('dark');
                const btn = document.querySelector('#analysis-sidebar .toggle-button');
                if (btn) {
                    btn.setAttribute('aria-label', 'AI Trajectory Analysis');
                    btn.setAttribute('title', 'AI Trajectory Analysis');
                }
                if (!window.__tvChartUiBound) {
                    window.__tvChartUiBound = true;
                    const ROW = __TV_ROW__, CHROME = __TV_CHROME__, MIN = __TV_MIN__;
                    window.tvExpandFileTimeline = function () {
                        document.querySelectorAll('.resizable-chart').forEach((root) => {
                            const gd = root.querySelector('.js-plotly-plot')
                                || root.querySelector('[data-testid="plotly"]');
                            if (!gd || !gd.layout) return;
                            const meta = gd.layout.meta || {};
                            const cats = (gd.layout.yaxis && gd.layout.yaxis.categoryarray)
                                || (gd._fullLayout && gd._fullLayout.yaxis
                                    && gd._fullLayout.yaxis._categories)
                                || [];
                            const n = Array.isArray(cats) ? cats.length : 0;
                            const h = Number(meta.tv_chart_height)
                                || (n ? Math.max(MIN, ROW * n + CHROME) : 0);
                            if (!h) return;
                            const plot = gd.classList.contains('js-plotly-plot')
                                ? gd
                                : (gd.querySelector('.js-plotly-plot') || gd);
                            if (plot.style.height === h + 'px'
                                && Math.abs(plot.clientHeight - h) < 4) return;
                            plot.style.height = h + 'px';
                            plot.style.minHeight = h + 'px';
                            root.style.height = 'auto';
                            root.style.maxHeight = 'none';
                            if (window.Plotly && window.Plotly.Plots) {
                                window.Plotly.Plots.resize(plot);
                            }
                        });
                    };
                    window.tvMainTabButtons = function () {
                        const root = document.getElementById('main-tabs');
                        const wrapper = root && root.querySelector(':scope > .tab-wrapper');
                        if (!wrapper) return [];
                        /* Gradio moves narrow-screen tabs into real dropdown
                           buttons. Measurement clones and nested tab groups
                           must never receive navigation or history clicks. */
                        return Array.from(wrapper.querySelectorAll(
                            '.tab-container[role="tablist"] button[role="tab"], '
                                + '.overflow-dropdown button'
                        )).filter((tab) => tab.closest('.tab-wrapper') === wrapper
                            && !tab.closest('[aria-hidden="true"]'));
                    };
                    window.tvClickMainTab = function (label) {
                        const tabs = window.tvMainTabButtons();
                        for (let ti = 0; ti < tabs.length; ti++) {
                            const canonical = window.tvCanonicalText?.(tabs[ti].textContent.trim())
                                || tabs[ti].textContent.trim();
                            if (canonical === label) {
                                tabs[ti].click();
                                return true;
                            }
                        }
                        return false;
                    };
                    window.tvActiveMainTab = function () {
                        const tabs = window.tvMainTabButtons();
                        for (let ti = 0; ti < tabs.length; ti++) {
                            if (tabs[ti].getAttribute('aria-selected') === 'true'
                                    || tabs[ti].classList.contains('selected')) {
                                return window.tvCanonicalText?.(tabs[ti].textContent.trim())
                                    || tabs[ti].textContent.trim();
                            }
                        }
                        return '';
                    };
                    window.tvWorkflowStepUrl = function (idx) {
                        return window.location.pathname
                            + window.location.search
                            + '#step-' + idx;
                    };
                    window.tvPushTabReturnPoint = function (tabLabel) {
                        if (!tabLabel || tabLabel === 'Workflow') return;
                        const cur = history.state || {};
                        if (cur.tvTab === tabLabel && !window.location.hash.startsWith('#step-')) {
                            return;
                        }
                        history.pushState(
                            { tvTab: tabLabel },
                            '',
                            window.location.pathname + window.location.search
                                + (window.location.hash.startsWith('#step-') ? '' : window.location.hash)
                        );
                    };
                    window.tvFocusWorkflowCard = function (card, opts) {
                        if (!card) return;
                        opts = opts || {};
                        const flash = opts.flash !== false;
                        document.querySelectorAll('.wf-card.wf-flash').forEach((el) => {
                            el.classList.remove('wf-flash');
                        });
                        card.scrollIntoView({behavior: 'smooth', block: 'center'});
                        if (typeof window.tvSelectWorkflowCard === 'function') {
                            window.tvSelectWorkflowCard(card, {
                                pushHistory: opts.pushHistory !== false,
                            });
                        } else {
                            card.click();
                        }
                        if (!flash) return;
                        card.classList.add('wf-flash');
                        if (card.__tvFlashTimer) clearTimeout(card.__tvFlashTimer);
                        card.__tvFlashTimer = setTimeout(() => {
                            card.classList.remove('wf-flash');
                            card.__tvFlashTimer = null;
                        }, 2000);
                    };
                    window.tvGotoWorkflowStep = function (idx, opts) {
                        opts = opts || {};
                        const restore = !!opts.restore;
                        const priorTab = window.tvActiveMainTab();
                        if (!restore && priorTab !== 'Workflow') {
                            window.tvPushTabReturnPoint(priorTab);
                        }
                        const targetHash = '#step-' + idx;
                        /* Set the hash first so Workflow js_on_load auto-select
                           does not steal the target, and so a late card mount
                           can still deep-link. Use pathname+search so a <base>
                           href or reverse-proxy prefix cannot drop the app path. */
                        if (!restore) {
                            const url = window.tvWorkflowStepUrl(idx);
                            const histState = { tvTab: 'Workflow' };
                            if (window.location.hash !== targetHash) {
                                history.pushState(histState, '', url);
                            } else {
                                history.replaceState(histState, '', url);
                            }
                        }
                        /* The prior tab precedes the step entry. Suppress the
                           manual-click capture hook during this synchronous tab
                           switch, including Back/Forward restoration. */
                        if (priorTab !== 'Workflow') {
                            const previous = window.__tvProgrammaticWorkflowTab;
                            window.__tvProgrammaticWorkflowTab = true;
                            try {
                                window.tvClickMainTab('Workflow');
                            } finally {
                                window.__tvProgrammaticWorkflowTab = previous;
                            }
                        }
                        window.__tvGotoGen = (window.__tvGotoGen || 0) + 1;
                        const gen = window.__tvGotoGen;
                        const started = Date.now();
                        let askedReset = false;
                        const tryFocus = function () {
                            if (gen !== window.__tvGotoGen) return;
                            const c = document.getElementById('wf-card-' + idx);
                            if (c) {
                                window.tvFocusWorkflowCard(c, {
                                    flash: !restore,
                                    pushHistory: false,
                                });
                                return;
                            }
                            const elapsed = Date.now() - started;
                            /* Cards are on screen but this step is not: filters
                               (or search) hid it. Reset chips once, then keep
                               waiting for the Gradio re-render. */
                            if (!askedReset && !restore
                                    && document.querySelector('.wf-card')
                                    && elapsed > 400) {
                                askedReset = true;
                                const reset = document.querySelector(
                                    '[data-wf-action="reset-filters"]'
                                );
                                if (reset) reset.click();
                            }
                            if (elapsed < 8000) {
                                setTimeout(tryFocus, 50);
                                return;
                            }
                            if (restore && document.querySelector('.wf-card')) {
                                /* Stale hash from an earlier session or another
                                   trajectory: drop it rather than guess. */
                                history.replaceState(
                                    null, '',
                                    window.location.pathname + window.location.search
                                );
                            }
                        };
                        setTimeout(tryFocus, 0);
                    };
                    if (!window.__tvHistoryBound) {
                        window.__tvHistoryBound = true;
                        window.addEventListener('popstate', function () {
                            const state = history.state || {};
                            if (state.tvTab && state.tvTab !== 'Workflow') {
                                window.tvClickMainTab(state.tvTab);
                                return;
                            }
                            const m = window.location.hash.match(/^#step-(\\d+)$/);
                            if (m) {
                                window.tvGotoWorkflowStep(Number(m[1]), { restore: true });
                            }
                        });
                        /* Capture phase: read the prior tab before Gradio switches. */
                        document.addEventListener('click', function (e) {
                            const tab = e.target && e.target.closest && e.target.closest('button');
                            const canonical = tab && (window.tvCanonicalText?.(tab.textContent.trim())
                                || tab.textContent.trim());
                            if (!tab || canonical !== 'Workflow'
                                    || !window.tvMainTabButtons().includes(tab)
                                    || window.__tvProgrammaticWorkflowTab) return;
                            window.tvPushTabReturnPoint(window.tvActiveMainTab());
                        }, true);
                    };
                    /* Test contract, not just a note: tests/test_workflow_detail_ui.py
                       slices inspect.getsource(build_ui) between the __TV_BIND_JUMPS_*
                       markers and runs the slice under Node (and tests/test_ui_resets.py
                       checks the slice without Node). Keep both markers, keep exactly
                       one top-level function between them, and let nothing follow its
                       closing `};` — the slice is taken up to its LAST `};`. */
                    /* __TV_BIND_JUMPS_BEGIN__ */
                    window.tvBindChartWorkflowJumps = function () {
                        /* Rebind on every schedule: Gradio Plotly.react / newPlot
                           reuses the graph div and drops .on() listeners, which
                           a one-shot bind flag would miss. */
                        ['duration-chart', 'tool-outcome-chart', 'tool-duration-chart', 'agent-swimlane-chart'].forEach((id) => {
                            const root = document.getElementById(id);
                            if (!root) return;
                            const gd = root.querySelector('.js-plotly-plot')
                                || root.querySelector('[data-testid="plotly"]');
                            if (!gd || typeof gd.on !== 'function') return;
                            if (gd.__tvJumpHandler && typeof gd.removeListener === 'function') {
                                gd.removeListener('plotly_click', gd.__tvJumpHandler);
                            }
                            const handler = function (data) {
                                const pt = data && data.points && data.points[0];
                                if (!pt) return;
                                let idx = pt.customdata;
                                if (Array.isArray(idx)) idx = idx[0];
                                const n = Number(idx);
                                if (!Number.isFinite(n)) return;
                                window.tvGotoWorkflowStep(Math.trunc(n));
                            };
                            gd.__tvJumpHandler = handler;
                            gd.on('plotly_click', handler);
                            gd.style.cursor = 'pointer';
                        });
                    };
                    /* __TV_BIND_JUMPS_END__ */
                    let timer = null;
                    const schedule = () => {
                        if (timer) clearTimeout(timer);
                        timer = setTimeout(() => {
                            window.tvExpandFileTimeline();
                            window.tvBindChartWorkflowJumps();
                        }, 60);
                    };
                    new MutationObserver(schedule).observe(document.documentElement, {
                        childList: true,
                        subtree: true,
                        attributes: true,
                        attributeFilter: ['style', 'class', 'hidden'],
                    });
                    window.addEventListener('resize', schedule);
                    [0, 80, 250, 600].forEach((ms) => setTimeout(schedule, ms));
                }
                return [false];
            }"""
            # Single-sourced from charts.activity so the JS expander and the
            # figure it resizes can never disagree. `.replace` rather than an
            # f-string or .format(): this literal is full of JS braces.
            .replace("__TV_ROW__", str(FILE_INTERACTION_ROW_PX))
            .replace("__TV_CHROME__", str(FILE_INTERACTION_CHROME_PX))
            .replace("__TV_MIN__", str(FILE_INTERACTION_MIN_HEIGHT)),
        )

        app.unload(upload.cleanup_session_export)

    return app
