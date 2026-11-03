"""Release docs use matching sources without exposing ignored local notes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

_PATH = Path(__file__).resolve().parents[1] / ".github/scripts/release_docs_source.py"
_SPEC = importlib.util.spec_from_file_location("release_docs_source", _PATH)
assert _SPEC and _SPEC.loader
source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(source)


def _builder(*, awe: bool = False) -> bytes:
    guides = ("index", "quickstart", "guides", "data-reference", *(("awe",) if awe else ()))
    return f"GUIDES = {guides!r}\n".encode()


def _current(root: Path, *, awe: bool = False) -> None:
    for name in (*source.PAGES, *source.SCRIPTS):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("maintained source", encoding="utf-8")
    (root / source.SCRIPTS[0]).write_bytes(_builder(awe=awe))
    if awe:
        (root / "docs/awe.md").write_text("maintained evaluation guide", encoding="utf-8")
    (root / "docs/private-note.md").write_text("private note", encoding="utf-8")


def test_bootstrap_is_explicit_and_does_not_copy_local_notes(tmp_path):
    repo = tmp_path / "repo"
    _current(repo)
    def git(_root, *args):
        return b"" if args[0] == "ls-tree" else b"abc123"
    with patch.object(source, "_git", side_effect=git):
        result = source.prepare(repo, "v0.3.1", tmp_path / "output")
    assert result["source_mode"] == "initial-release-bootstrap"
    assert not (tmp_path / "output/docs/private-note.md").exists()
    assert (tmp_path / "output/docs/index.md").read_text(encoding="utf-8") == "maintained source"


def test_future_release_cannot_silently_use_current_guides(tmp_path):
    with patch.object(source, "_git", return_value=b""), pytest.raises(ValueError, match="no complete"):
        source.prepare(tmp_path, "v0.4.0", tmp_path / "output")


def test_tagged_sources_are_used_and_private_paths_are_excluded(tmp_path):
    files = [*source.PAGES, *source.SCRIPTS, ".github/docs/site.css", "docs/private-note.md", ".github/scripts/private.py"]
    def git(_root, *args):
        if args[0] == "ls-tree":
            return "\n".join(files).encode("utf-8")
        if args[0] == "show":
            assert args[1].startswith("v0.4.0:")
            if args[1].endswith(source.SCRIPTS[0]):
                return _builder()
            return b"released source"
        return b"release-sha"
    with patch.object(source, "_git", side_effect=git):
        result = source.prepare(tmp_path, "v0.4.0", tmp_path / "output")
    assert result["source_mode"] == "release-tag"
    assert (tmp_path / "output/docs/index.md").read_bytes() == b"released source"
    assert (tmp_path / "output/.github/docs/site.css").is_file()
    assert not (tmp_path / "output/docs/private-note.md").exists()
    assert not (tmp_path / "output/.github/scripts/private.py").exists()


def test_reviewed_documentation_correction_can_use_current_guides(tmp_path):
    repo = tmp_path / "repo"
    _current(repo, awe=True)
    def git(_root, *args):
        return "\n".join((*source.PAGES, *source.SCRIPTS)).encode("utf-8") if args[0] == "ls-tree" else b"current-sha"
    with patch.object(source, "_git", side_effect=git):
        result = source.prepare(repo, "v0.4.0", tmp_path / "output", current_guides=True)
    assert result["source_mode"] == "documentation-update"
    assert result["docs_source"] == "current-sha"
    assert (tmp_path / "output/docs/awe.md").read_text(encoding="utf-8") == "maintained evaluation guide"
    assert not (tmp_path / "output/docs/private-note.md").exists()


@pytest.mark.parametrize("guide_present", [True, False])
def test_release_builder_required_evaluation_guide_is_copied_or_rejected(tmp_path, guide_present):
    files = [*source.PAGES, *source.SCRIPTS, "docs/private-note.md"]
    if guide_present:
        files.append("docs/awe.md")
    def git(_root, *args):
        if args[0] == "ls-tree":
            return "\n".join(files).encode("utf-8")
        if args[0] == "show":
            return _builder(awe=True) if args[1].endswith(source.SCRIPTS[0]) else b"released guide"
        return b"release-sha"
    with patch.object(source, "_git", side_effect=git):
        if not guide_present:
            with pytest.raises(ValueError, match="missing a guide"):
                source.prepare(tmp_path, "v0.4.0", tmp_path / "output")
            return
        result = source.prepare(tmp_path, "v0.4.0", tmp_path / "output")
    assert result["source_mode"] == "release-tag"
    assert (tmp_path / "output/docs/awe.md").read_bytes() == b"released guide"
    assert not (tmp_path / "output/docs/private-note.md").exists()


def test_optional_guide_does_not_make_historical_release_incomplete(tmp_path):
    files = [*source.PAGES, *source.SCRIPTS]
    def git(_root, *args):
        if args[0] == "ls-tree":
            return "\n".join(files).encode("utf-8")
        if args[0] == "show":
            return _builder() if args[1].endswith(source.SCRIPTS[0]) else b"historical source"
        return b"historical-sha"
    with patch.object(source, "_git", side_effect=git):
        result = source.prepare(tmp_path, "v0.3.2", tmp_path / "output")
    assert result["source_mode"] == "release-tag"
    assert not (tmp_path / "output/docs/awe.md").exists()


@pytest.mark.parametrize("tag", ["main", "v0.3.1; touch other", "v../notes", "--all"] )
def test_unsafe_tag_is_rejected_before_git(tmp_path, tag):
    with patch.object(source, "_git") as git:
        with pytest.raises(ValueError, match="tag"):
            source.prepare(tmp_path, tag, tmp_path / "output")
        git.assert_not_called()


def test_existing_source_output_is_preserved(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        source.prepare(tmp_path, "v0.3.1", output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
