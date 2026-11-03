"""Documentation checks remain stdlib-only; full rendering runs in docs CI."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trajviz_build_docs", ROOT / ".github/scripts/build_docs.py")
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_markdown_links_become_html_without_rewriting_external_urls():
    source = ('<a href="quickstart.md#load-a-run">start</a>'
              '<a href="guides.md?source=home&amp;lang=en#figures">guide</a>'
              '<a href="https://example.test/guide.md">external</a>')
    output = builder.rewrite_markdown_links(source)
    assert 'href="quickstart.html#load-a-run"' in output
    assert 'href="guides.html?source=home&amp;lang=en#figures"' in output
    assert 'href="https://example.test/guide.md"' in output


def test_nested_api_navigation_stays_relative_to_version_root():
    output = builder.header(Path("api/agent_trajviz/figures.html"), "0.3.1", "stable", "../")
    assert 'href="../../quickstart.html"' in output
    assert 'href="../../../stable/"' in output
    assert 'href="../../../dev/"' in output
    assert "stable · 0.3.1" in output
    assert 'data-versions-url="../../../versions.json"' in output
    assert 'data-current-version="stable"' in output
    hosted = builder.header(Path("index.html"), "0.3.1", "stable", "https://docs.example.test/project/")
    assert 'href="https://docs.example.test/project/stable/"' in hosted


def test_header_escapes_version_labels():
    output = builder.header(Path("index.html"), "0.3.1", '<script>"bad"</script>', "../")
    assert "<script>" not in output
    assert "&lt;script&gt;" in output


def test_link_check_covers_assets_queries_and_encoded_anchors(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/docs.css").write_text("body { color: black; }")
    (tmp_path / "api.html").write_text('<h1 id="a.b">API</h1><div id="a b"></div>')
    (tmp_path / "index.html").write_text('<link href="assets/docs.css" rel="stylesheet">'
                                          '<a href="api.html?x=1#a.b">API</a><a href="api.html#a%20b">encoded</a>')
    result = builder.check_site(tmp_path)
    assert result == {"html_pages": 2, "local_links": 3}


@pytest.mark.parametrize("target", ["missing.html", "api.html#missing", "../outside.html"])
def test_bad_local_links_fail_the_build(tmp_path, target):
    (tmp_path / "index.html").write_text(f'<a href="{target}">bad</a>')
    (tmp_path / "api.html").write_text('<div id="exists"></div>')
    with pytest.raises(builder.DocsError, match="Broken documentation links"):
        builder.check_site(tmp_path)


def test_version_links_are_checked_when_the_complete_site_is_assembled(tmp_path):
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/index.html").write_text('<a data-version-link="true" href="../stable/">stable</a>'
                                               '<label data-versions-url="../versions.json"></label>')
    assert builder.check_site(tmp_path / "dev", allow_version_links=True)["html_pages"] == 1
    with pytest.raises(builder.DocsError, match="missing local target"):
        builder.check_site(tmp_path)
    (tmp_path / "stable").mkdir()
    (tmp_path / "stable/index.html").write_text("<h1>Stable</h1>")
    (tmp_path / "versions.json").write_text('{"stable":"0.3.1","versions":["0.3.1"],"dev":"dev"}')
    assert builder.check_site(tmp_path)["html_pages"] == 2


@pytest.mark.parametrize("suffix", ["html", "js", "json"])
def test_generated_output_guard_checks_pages_search_and_schema_text(tmp_path, suffix):
    (tmp_path / "index.html").write_text("<h1>Documentation</h1>")
    (tmp_path / ("unsafe." + suffix)).write_text(("DE" + "CAF").lower())
    with pytest.raises(builder.DocsError, match="Excluded legacy backend name"):
        builder.check_site(tmp_path)


def test_generator_refuses_to_mix_output_and_markdown_source(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    with pytest.raises(builder.DocsError, match="separate from Markdown"):
        builder.build(docs / "generated", docs)


@pytest.mark.skipif(importlib.util.find_spec("pdoc") is None, reason="pdoc is a docs-job tool")
def test_full_public_site_build_has_all_exports_schemas_and_no_source(tmp_path):
    import agent_trajviz as tv

    output = tmp_path / "site"
    report = builder.build(output, ROOT / "docs", package_root=ROOT,
                           version_label="test", expected_version=tv.__version__)
    assert (output / "assets/versions.js").is_file()
    assert '<script defer src="assets/versions.js">' in (output / "index.html").read_text()
    assert report["api_modules"] == list(builder.PUBLIC_MODULES)
    assert report["catalog_views"] == len(tv.VIEW_CATALOG)
    assert report["schemas"] == len(tv.view_schemas())
    assert sorted(path.relative_to(output / "api").as_posix() for path in (output / "api").rglob("*.html")) == [
        "agent_trajviz.html", "agent_trajviz/attribution.html", "agent_trajviz/awe.html", "agent_trajviz/dashboard.html",
        "agent_trajviz/figures.html", "agent_trajviz/llm.html", "index.html",
    ]
    parser = builder._Links()
    api = (output / "api/agent_trajviz.html").read_text()
    parser.feed(api)
    assert set(tv.__all__) <= parser.ids
    assert "Trajectory.from_steps" in parser.ids
    assert "VIEW_CATALOG" in parser.ids
    assert 'id="load_plugins"' in api
    assert 'id="_source_views"' not in api
    assert "view-source-button" not in parser.classes
    assert "view-source-toggle-state" not in parser.classes
    for name, schema in tv.view_schemas().items():
        assert json.loads((output / f"schema/{name}.json").read_text()) == schema
    catalog = (output / "data/views.html").read_text()
    assert "Per-step tokens" in catalog
    assert "../schema/token_usage.json" in catalog
    assert "Search" in (output / "api/search.js").read_text() or (output / "api/search.js").stat().st_size > 100
