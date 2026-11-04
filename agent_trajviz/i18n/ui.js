/* Localize explicitly authored presentation; recorded content is never scanned. */
(() => {
    if (window.__tvLanguageBound) return;
    window.__tvLanguageBound = true;
    const translations = window.__tvTranslations || {};
    let language = "en";
    const figures = new WeakMap();
    const normalize = value => value === "zh-CN" ? "zh-CN" : "en";
    const format = (template, args) => template.replace(/\{([A-Za-z_]\w*)(?::([^}]*))?\}/g,
        (match, key, spec) => {
            if (!Object.prototype.hasOwnProperty.call(args, key)) return match;
            const value = args[key];
            if (typeof value !== "number" || !spec) return String(value);
            const precision = spec.match(/\.(\d+)f/);
            if (precision || spec.includes(",")) {
                const decimals = precision ? Number(precision[1]) : Number.isInteger(value) ? 0 : 3;
                return value.toLocaleString("en-US", {useGrouping: spec.includes(","),
                    minimumFractionDigits: decimals, maximumFractionDigits: decimals});
            }
            return String(value);
        });
    window.tvTranslate = (key, args = {}) => format(language === "zh-CN" ? translations[key] || key : key, args);
    window.tvCanonicalText = value => {
        if (language !== "zh-CN") return value;
        return Object.keys(translations).find(key => translations[key] === value) || value;
    };
    function mark(node) {
        if (node.closest("pre, code, textarea, .analysis-chatbot, .wf-preview")) return;
        const key = node.getAttribute("data-tv-i18n");
        if (key === null) return;
        let args = {};
        try { args = JSON.parse(node.getAttribute("data-tv-i18n-args") || "{}"); } catch (_) { return; }
        const translated = language === "en" ? node.getAttribute("data-tv-i18n-en") || format(key, args)
            : window.tvTranslate(key, args);
        const attribute = node.getAttribute("data-tv-i18n-attr");
        if (attribute === "title" || attribute === "aria-label" || attribute === "data-help") {
            if (node.getAttribute(attribute) !== translated) node.setAttribute(attribute, translated);
        } else if (!attribute && node.textContent !== translated) {
            node.textContent = translated;
        }
    }
    function plot(node) {
        const metadata = node.layout?.meta?.tv_i18n;
        if (!metadata || !window.Plotly?.relayout) return;
        const fingerprint = JSON.stringify(metadata);
        const chrome = value => {
            if (typeof value === "string") return window.tvTranslate(value);
            if (Array.isArray(value)) return value.map(chrome);
            if (Object.prototype.hasOwnProperty.call(value, "literal")) return value.literal;
            const params = {...value.params};
            for (const [key, expression] of Object.entries(value.labels || {})) params[key] = chrome(expression);
            return window.tvTranslate(value.template, params);
        };
        const same = (actual, expected) => Array.isArray(expected)
            ? Array.isArray(actual) && actual.length === expected.length
                && expected.every((item, index) => same(actual[index], item))
            : actual === expected;
        // Explicit display fields never include plotted coordinates, category
        // IDs or customdata. Literal expressions keep recorded labels intact.
        const layoutLabels = {...metadata.layout};
        for (const [path, expression] of Object.entries(metadata.layout_labels || {})) {
            if (/^(?:title\.text|legend\.title\.text|[xy]axis\d*\.(?:title\.text|ticktext)|annotations\[\d+\]\.text)$/.test(path)) {
                layoutLabels[path] = expression;
            }
        }
        const traceLabels = {};
        for (const [index, original] of Object.entries(metadata.traces || {})) {
            traceLabels[index] = {name: original};
        }
        for (const [index, fields] of Object.entries(metadata.trace_labels || {})) {
            for (const [field, expression] of Object.entries(fields)) {
                if (["name", "text", "hovertext", "hovertemplate"].includes(field)) {
                    (traceLabels[index] ||= {})[field] = expression;
                }
            }
        }
        let state = figures.get(node);
        if (!state) {
            state = {language: null, fingerprint: null, pending: false};
            figures.set(node, state);
            node.on?.("plotly_afterplot", () => plot(node));
        }
        const read = (object, path) => path.replace(/\[(\d+)\]/g, ".$1").split(".")
            .reduce((value, key) => value == null ? undefined : value[key], object);
        const current = () => Object.entries(layoutLabels).every(
            ([key, original]) => same(read(node.layout, key), chrome(original)))
            && Object.entries(traceLabels).every(([index, fields]) => Object.entries(fields).every(
                ([field, original]) => same(node.data?.[Number(index)]?.[field], chrome(original))));
        if (state.pending || state.language === language && state.fingerprint === fingerprint && current()) return;
        state.pending = true;
        state.language = language;
        state.fingerprint = fingerprint;
        const requested = language;
        const update = {};
        for (const [key, original] of Object.entries(layoutLabels)) {
            update[key] = chrome(original);
        }
        let succeeded = false;
        Promise.resolve().then(() => window.Plotly.relayout(node, update)).then(async () => {
            for (const [index, fields] of Object.entries(traceLabels)) {
                const updates = {};
                for (const [field, original] of Object.entries(fields)) {
                    const value = chrome(original);
                    // Plotly treats an outer array as values for trace indices.
                    // One trace's list of labels therefore needs another array.
                    updates[field] = Array.isArray(value) ? [value] : value;
                }
                await window.Plotly.restyle(node, updates, [Number(index)]);
            }
            succeeded = true;
        }).catch(() => { state.fingerprint = null; }).finally(() => {
            state.pending = false;
            if (requested !== language || succeeded && (!current()
                    || JSON.stringify(node.layout?.meta?.tv_i18n) !== fingerprint)) plot(node);
        });
    }
    function apply(root) {
        if (!(root instanceof Element)) return;
        const alias = {
            "Drop File Here": "Drop File Here", "将文件拖放到此处": "Drop File Here",
            "Click to Upload": "Click to Upload", "点击上传": "Click to Upload",
            "- or -": "- or -", "- 或 -": "- or -"
        };
        const uploaders = new Set(root.querySelectorAll('button[aria-dropeffect="copy"]'));
        const parentUploader = root.closest('button[aria-dropeffect="copy"]');
        if (parentUploader) uploaders.add(parentUploader);
        for (const button of uploaders) {
            const input = button.querySelector('input[type="file"][data-testid="file-upload"]');
            if (!input) continue;
            button.setAttribute("data-tv-i18n", "Click to upload or drop files");
            button.setAttribute("data-tv-i18n-attr", "aria-label");
            input.setAttribute("data-tv-i18n", "File upload");
            input.setAttribute("data-tv-i18n-attr", "aria-label");
            for (const wrapper of button.querySelectorAll(".wrap, .wrap .or")) {
                for (const child of Array.from(wrapper.childNodes)) {
                    if (child.nodeType !== 3) continue;
                    const key = alias[child.textContent.trim()];
                    if (!key) continue;
                    const span = document.createElement("span");
                    span.setAttribute("data-tv-i18n", key);
                    span.setAttribute("style", "color:inherit");
                    span.textContent = key;
                    child.replaceWith(span);
                }
            }
        }
        // These opt-in regions contain generated Markdown tables, never saved
        // reports. Their headings/column names are owned; cells remain data.
        const ownedSelector = '.diagnostics-data-table h2, .diagnostics-data-table h3, '
            + '.diagnostics-data-table th, .tv-owned-markdown h2, .tv-owned-markdown h3, .tv-owned-markdown th, '
            + '.tv-secret-field [data-testid="block-info"], .tv-secret-field .info-text';
        const headings = Array.from(root.querySelectorAll(ownedSelector));
        if (root.matches(ownedSelector)) headings.push(root);
        for (const heading of headings) {
            for (const child of Array.from(heading.childNodes)) {
                if (child.nodeType !== 3) continue;
                const key = child.textContent.trim();
                if (!Object.prototype.hasOwnProperty.call(translations, key)) continue;
                const span = document.createElement("span");
                span.setAttribute("data-tv-i18n", key);
                span.setAttribute("style", "color:inherit");
                span.textContent = key;
                child.replaceWith(span);
            }
        }
        if (root.matches("[data-tv-i18n]")) mark(root);
        root.querySelectorAll("[data-tv-i18n]").forEach(mark);
        if (root.matches(".js-plotly-plot")) plot(root);
        root.querySelectorAll(".js-plotly-plot").forEach(plot);
    }
    window.tvSetLanguage = value => {
        language = normalize(value);
        document.documentElement.lang = language;
        apply(document.body);
        const button = document.querySelector("#analysis-sidebar .toggle-button");
        if (button) {
            button.setAttribute("aria-label", window.tvTranslate("AI Trajectory Analysis"));
            button.setAttribute("title", window.tvTranslate("AI Trajectory Analysis"));
        }
        document.dispatchEvent(new CustomEvent("trajviz:language", {detail: language}));
    };
    function start() {
        apply(document.body);
        new MutationObserver(records => {
            for (const record of records) {
                if (record.type === "attributes") {
                    // Renderers can update a marker's arguments or restore an
                    // English tooltip without replacing its DOM node. Only
                    // the explicitly marked target owns those attributes.
                    if (record.target instanceof Element && record.target.matches("[data-tv-i18n]")) {
                        mark(record.target);
                    }
                    continue;
                }
                // Gradio can update an existing HTML text node in place. A
                // child-list observer alone misses refreshed status/findings.
                const parent = record.target instanceof Element ? record.target : record.target.parentElement;
                const marked = parent?.closest("[data-tv-i18n]");
                if (marked) mark(marked);
                for (const added of record.addedNodes) apply(added);
                // Plotly fills a graph div after mounting its descendants.
                const graph = record.target instanceof Element && record.target.closest(".js-plotly-plot");
                if (graph) plot(graph);
            }
        }).observe(document.body, {childList: true, characterData: true, subtree: true,
            attributes: true, attributeFilter: ["data-tv-i18n", "data-tv-i18n-args", "data-tv-i18n-en",
                "data-tv-i18n-attr", "title", "aria-label", "data-help"]});
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, {once: true});
    else start();
})();
