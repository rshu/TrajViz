"""Build public guides, API reference and schemas without importing private modules recursively."""
from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from urllib.parse import unquote, urlsplit, urlunsplit

PUBLIC_MODULES = ("agent_trajviz", "agent_trajviz.figures", "agent_trajviz.dashboard",
                  "agent_trajviz.llm", "agent_trajviz.attribution", "agent_trajviz.awe")
GUIDES = ("index", "quickstart", "guides", "data-reference", "awe")
TEMPLATES = Path(__file__).resolve().parents[1] / "docs"
EXCLUDED_TERM = re.compile("de" + "caf", re.IGNORECASE)


class DocsError(ValueError):
    """An actionable documentation build or link failure."""


def guard_public_text(text: str, origin: str) -> None:
    if EXCLUDED_TERM.search(text):
        raise DocsError(f"Excluded legacy backend name in generated documentation: {origin}")


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _root_prefix(page: Path) -> str:
    return "../" * len(page.parent.parts)


def _versions_root(page: Path, base_url: str) -> str:
    if urlsplit(base_url).scheme or base_url.startswith("/"):
        return base_url.rstrip("/") + "/"
    return _root_prefix(page) + base_url.rstrip("/") + "/"


def _version_target(page: Path, base_url: str, channel: str) -> str:
    return _versions_root(page, base_url) + channel + "/"


def header(page: Path, version: str, label: str, base_url: str) -> str:
    prefix = _root_prefix(page)
    links = (("Home", "index.html"), ("Quickstart", "quickstart.html"), ("Guides", "guides.html"),
             ("Data reference", "data-reference.html"), ("API", "api/agent_trajviz.html"))
    nav = "".join(f'<a href="{prefix + target}">{title}</a>' for title, target in links)
    channels = "".join(f'<a data-version-link="true" href="{html.escape(_version_target(page, base_url, channel), quote=True)}">'
                       f'{title}</a>' for channel, title in (("stable", "Stable"), ("dev", "Development")))
    current = "stable" if label == "stable" else "dev" if label in {"dev", "development"} else version
    manifest = html.escape(_versions_root(page, base_url) + "versions.json", quote=True)
    picker = (f'<label class="tv-version-picker" hidden data-versions-url="{manifest}" '
              f'data-current-version="{html.escape(current, quote=True)}">Version '
              '<select aria-label="Documentation version"></select></label>')
    return (f'<a class="tv-skip" href="#tv-main">Skip to content</a><header class="tv-header">'
            f'<div class="tv-brand"><a href="{prefix}index.html">TrajViz</a>'
            f'<span class="tv-version">{html.escape(label)} · {html.escape(version)}</span></div>'
            f'<nav class="tv-nav" aria-label="Documentation">{nav}</nav>'
            f'<nav class="tv-channels" aria-label="Documentation version">{channels}</nav>{picker}</header>')


def page_html(page: Path, title: str, body: str, version: str, label: str, base_url: str) -> str:
    prefix = _root_prefix(page)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<meta name="generator" content="TrajViz docs / pdoc 16.0.0">'
            f'<title>{html.escape(title)} · TrajViz {html.escape(version)}</title>'
            f'<link rel="stylesheet" href="{prefix}assets/docs.css">'
            f'<script defer src="{prefix}assets/versions.js"></script></head><body class="tv-guide">'
            + header(page, version, label, base_url)
            + f'<main id="tv-main" class="tv-main">{body}</main>'
            + '<footer class="tv-footer">TrajViz documentation · '
              '<a href="https://github.com/rshu/TrajViz">Source repository</a></footer></body></html>')


def rewrite_markdown_links(rendered: str) -> str:
    def replace(match):
        target = html.unescape(match.group(2))
        parts = urlsplit(target)
        if not parts.scheme and not parts.netloc and parts.path.endswith(".md"):
            target = urlunsplit(("", "", parts.path[:-3] + ".html", parts.query, parts.fragment))
        return match.group(1) + html.escape(target, quote=True) + match.group(3)
    return re.sub(r'(href=["\'])([^"\']+)(["\'])', replace, rendered)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.classes: set[str] = set()
        self.links: list[tuple[str, bool]] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.classes.update(attrs.get("class", "").split())
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        target = attrs.get("href") if tag in {"a", "link"} else attrs.get("src") if tag in {"img", "script"} else None
        if target:
            self.links.append((target, attrs.get("data-version-link") == "true"))
        if attrs.get("data-versions-url"):
            self.links.append((attrs["data-versions-url"], True))


def check_site(root: Path | str, *, allow_version_links: bool = False) -> dict[str, int]:
    """Check generated text, local HTML links/anchors and referenced assets."""
    root = Path(root).resolve()
    parsed: dict[Path, _Links] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".html", ".js", ".json", ".css"}:
            text = path.read_text(encoding="utf-8")
            guard_public_text(text, str(path.relative_to(root)))
            if path.suffix == ".html":
                parser = _Links()
                parser.feed(text)
                parsed[path] = parser
    if not parsed:
        raise DocsError("No HTML pages found in documentation output.")
    errors = []
    count = 0
    for source, parser in parsed.items():
        for target, version_link in parser.links:
            if version_link and allow_version_links:
                continue
            split = urlsplit(target)
            if split.scheme or split.netloc:
                continue
            count += 1
            if split.path.startswith("/"):
                # A --base-url rooted at the hosting path is checked by the
                # version assembler, rather than against a filesystem root.
                if version_link:
                    continue
                destination = root / unquote(split.path).lstrip("/")
            else:
                destination = source.parent / unquote(split.path) if split.path else source
            destination = destination.resolve()
            if destination.is_dir():
                destination /= "index.html"
            if not destination.is_relative_to(root) or not destination.is_file():
                errors.append(f"{source.relative_to(root)}: missing local target {target}")
            elif split.fragment and destination.suffix == ".html":
                fragment = unquote(split.fragment)
                if destination not in parsed or fragment not in parsed[destination].ids:
                    errors.append(f"{source.relative_to(root)}: missing anchor {target}")
    if errors:
        raise DocsError("Broken documentation links:\n" + "\n".join(errors[:30]))
    return {"html_pages": len(parsed), "local_links": count}


def _public_docs():
    # Imports are deliberately lazy: basic checks and test collection need
    # only the standard library. Pdoc is a docs-job tool, not a runtime extra.
    try:
        import pdoc
        from pdoc import doc, render
    except ImportError as error:
        raise DocsError("Building the API reference requires pdoc==16.0.0.") from error
    if importlib.metadata.version("pdoc") != "16.0.0":
        raise DocsError("The documentation build requires pdoc==16.0.0.")
    del pdoc
    render.configure(docformat="restructuredtext", include_undocumented=True, show_source=False, search=True)
    modules = {}
    package_version = importlib.import_module("agent_trajviz").__version__
    legacy = tuple(int(part) for part in re.match(r"(\d+)\.(\d+)", package_version).groups()) < (0, 4)
    for name in PUBLIC_MODULES:
        if legacy and name == "agent_trajviz.awe":
            continue
        imported = importlib.import_module(name)
        names = getattr(imported, "__all__", None)
        if not isinstance(names, (list, tuple)) or not all(isinstance(item, str) for item in names):
            raise DocsError(f"Public module must declare __all__: {name}")
        documented = doc.Module.from_name(name)
        members = {key: value for key, value in documented.members.items() if key in names}
        # Pdoc otherwise treats a renamed class alias as a Variable, exposing
        # its internal implementation name instead of the public constructor.
        for key in names:
            value = getattr(imported, key)
            if isinstance(value, type) and not isinstance(members.get(key), doc.Class):
                members[key] = doc.Class(name, key, value, (name, key))
        if set(members) != set(names):
            raise DocsError(f"Missing exported API members in {name}")
        documented.members = members
        modules[name] = documented
    # The immutable published release has historical backend-specific prose.
    # Normalize only this prose in the documentation model, never package code
    # or API identifiers. The output guard below catches all other occurrences.
    attribution = modules["agent_trajviz.attribution"]
    if "agent_trajviz.awe" in modules:
        attribution.docstring = ("Failure attribution through TrajViz's bundled evaluator. "
                                 "Diagnosis requires matching task reference data and returns "
                                 "an unavailable result when the evidence cannot be verified.")
        attribution.members["available"].docstring = "Return (True, None) when the bundled evaluator is ready, otherwise (False, reason)."
    else:
        attribution.docstring = ("Failure attribution through an optional reference backend. "
                                 "Without matching reference data, diagnosis returns an unavailable result.")
        attribution.members["available"].docstring = "Return whether this package's optional attribution integration is ready."
    attribution.members["diagnose"].docstring = EXCLUDED_TERM.sub("the attribution engine", attribution.members["diagnose"].docstring)
    catalog = modules["agent_trajviz"].members["VIEW_CATALOG"]
    if isinstance(catalog, doc.Variable):
        catalog.default_value = doc.empty
    catalog.docstring = ("The read-only catalog of built-in views: titles, kinds, descriptions and options. "
                         "Browse the [view catalog and JSON schemas](../data/views.html).")
    return modules, render


def _inject_api(page: Path, rendered: str, version: str, label: str, base_url: str) -> str:
    prefix = _root_prefix(page)
    rendered = rendered.replace("</head>", f'<link rel="stylesheet" href="{prefix}assets/docs.css">'
                                f'<script defer src="{prefix}assets/versions.js"></script></head>', 1)
    rendered = re.sub(r"<body[^>]*>", '<body class="tv-api">' + header(page, version, label, base_url), rendered, count=1)
    rendered = rendered.replace('<main class="pdoc">', '<main class="pdoc" id="tv-main">', 1)
    if 'id="tv-main"' not in rendered:
        # The API module index has no main element in pdoc's default template.
        rendered = rendered.replace('<body class="tv-api">', '<body class="tv-api" id="tv-main">', 1)
    return rendered


def _write(output: Path, relative: Path | str, text: str) -> None:
    relative = Path(relative)
    guard_public_text(text, relative.as_posix())
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(output: Path, docs_root: Path, *, package_root: Path | None = None,
          version_label: str = "development", expected_version: str | None = None,
          base_url: str = "../", check_examples: bool = False) -> dict:
    output, docs_root = output.resolve(), docs_root.resolve()
    if output == docs_root or output.is_relative_to(docs_root) or docs_root.is_relative_to(output):
        raise DocsError("Keep generated output separate from Markdown sources.")
    if package_root is not None:
        package_root = package_root.resolve()
        sys.path.insert(0, str(package_root))
    package = importlib.import_module("agent_trajviz")
    package_file = Path(package.__file__).resolve()
    if package_root is not None and not package_file.is_relative_to(package_root):
        raise DocsError("Imported package does not come from --package-root.")
    if output.is_relative_to(package_file.parent):
        raise DocsError("Do not generate documentation inside the installed package.")
    version = package.__version__
    if expected_version and version != expected_version:
        raise DocsError(f"Expected package version {expected_version}, found {version}.")
    modules, render = _public_docs()
    import markdown2
    output.mkdir(parents=True, exist_ok=True)
    _write(output, "assets/docs.css", (TEMPLATES / "site.css").read_text())
    _write(output, "assets/versions.js", (TEMPLATES / "versions.js").read_text())
    for name in GUIDES:
        source = docs_root / (name + ".md")
        text = source.read_text(encoding="utf-8")
        if name == "awe" and "agent_trajviz.awe" not in modules:
            text = ("# Workflow evaluation\n\nSaved evaluation runs and the bundled evaluator were added in TrajViz 0.4.0. "
                    "This documentation describes an earlier package. See the development or latest stable documentation.")
        guard_public_text(text, source.name)
        body = rewrite_markdown_links(str(markdown2.markdown(text, extras=["fenced-code-blocks", "tables", "header-ids", "code-friendly"])))
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")), name)
        page = Path(name + ".html")
        _write(output, page, page_html(page, title, body, version, version_label, base_url))
    if (docs_root / "assets").is_dir():
        for asset in (docs_root / "assets").rglob("*"):
            if asset.is_file():
                if not asset.resolve().is_relative_to(docs_root):
                    raise DocsError("Documentation assets must stay inside the docs source directory.")
                destination = output / asset.relative_to(docs_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(asset, destination)
    for name, documented in modules.items():
        page = Path("api") / (name.replace(".", "/") + ".html")
        _write(output, page, _inject_api(page, render.html_module(documented, modules), version, version_label, base_url))
    if "agent_trajviz.awe" not in modules:
        page = Path("api/agent_trajviz/awe.html")
        _write(output, page, page_html(page, "Workflow evaluation", "<h1>Workflow evaluation</h1>"
               "<p>This API was added in TrajViz 0.4.0 and is unavailable in this package version.</p>",
               version, version_label, base_url))
    api_index = render.html_index(modules)
    if api_index:
        page = Path("api/index.html")
        _write(output, page, _inject_api(page, api_index, version, version_label, base_url))
    _write(output, "api/search.js", render.search_index(modules))
    schemas = package.view_schemas()
    for name, schema in schemas.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise DocsError("Invalid schema filename in public API.")
        _write(output, f"schema/{name}.json", json.dumps(schema, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    rows = []
    for name, entry in package.VIEW_CATALOG.items():
        entry = _plain(entry)
        options = entry.get("options") or {}
        detail = (f'<details><summary>Options</summary><pre>{html.escape(json.dumps(options, indent=2))}</pre></details>'
                  if options else "")
        rows.append(f'<tr id="{html.escape(name, quote=True)}"><th scope="row"><code>{html.escape(name)}</code></th>'
                    f'<td>{html.escape(entry["title"])}</td><td>{html.escape(entry["kind"])}</td>'
                    f'<td>{html.escape(entry["description"])}{detail}</td>'
                    f'<td><a href="../schema/{name}.json">JSON schema</a></td></tr>')
    additional = "".join(f'<li><a href="../schema/{name}.json"><code>{html.escape(name)}</code></a></li>'
                         for name in sorted(set(schemas) - set(package.VIEW_CATALOG)))
    body = (f'<h1 id="view-catalog">Views and JSON schemas</h1><p>{len(package.VIEW_CATALOG)} built-in views and '
            f'{len(schemas)} schemas from the installed public API, version {html.escape(version)}.</p>'
            '<p><a href="../data-reference.html">Read the data reference</a></p><div class="tv-table">'
            '<table><caption>Built-in trajectory views</caption><thead><tr><th>View ID</th><th>Title</th><th>Kind</th>'
            '<th>Description and options</th><th>Schema</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
            '<h2 id="additional-schemas">Comparison, group and label schemas</h2><ul>' + additional + '</ul>')
    page = Path("data/views.html")
    _write(output, page, page_html(page, "Views and schemas", body, version, version_label, base_url))
    report = {"version": version, "version_label": version_label, "package": str(package_file),
              "api_modules": list(modules), "catalog_views": len(package.VIEW_CATALOG), "schemas": len(schemas),
              **check_site(output, allow_version_links=True)}
    _write(output, "build-info.json", json.dumps({key: value for key, value in report.items() if key != "package"}, indent=2) + "\n")
    if check_examples:
        command = [sys.executable, str(Path(__file__).with_name("check_docs_examples.py")), "--docs-root", str(docs_root),
                   "--expected-version", version, "--extras", "charts,llm,app", "--api-html", str(output / "api/agent_trajviz.html")]
        if package_root is not None:
            command += ["--package-root", str(package_root)]
        subprocess.run(command, check=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("site"))
    parser.add_argument("--docs-root", "--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--version-label", default="development")
    parser.add_argument("--expected-version", "--version")
    parser.add_argument("--base-url", default="../", help="Versions root URL or path, relative to the build root by default.")
    parser.add_argument("--check", type=Path, help="Only check a complete assembled site; no pdoc dependency needed.")
    parser.add_argument("--check-examples", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check:
            report = check_site(args.check)
        else:
            report = build(args.output, args.docs_root, package_root=args.package_root,
                           version_label=args.version_label, expected_version=args.expected_version,
                           base_url=args.base_url, check_examples=args.check_examples)
        print(json.dumps(report))
    except (DocsError, OSError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
