"""Local documentation must stay outside the repository's tracked files."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def repository_git():
    executable = shutil.which("git")
    if executable is None:
        pytest.skip("Repository policy requires Git")

    def run(*arguments, **kwargs):
        return subprocess.run(
            [executable, "-C", str(ROOT), *arguments],
            capture_output=True, text=True, check=False, timeout=10, **kwargs,
        )

    result = run("rev-parse", "--show-toplevel")
    if result.returncode or Path(result.stdout.strip()).resolve() != ROOT:
        pytest.skip("Repository policy requires this project's Git checkout")
    return run


def test_local_docs_are_absent_from_the_git_index(repository_git):
    result = repository_git("ls-files", "--cached", "--full-name", "-z", "--", "docs/")
    assert result.returncode == 0, result.stderr
    tracked = [path for path in result.stdout.split("\0") if path]
    assert not tracked, f"docs/ must remain local; tracked paths: {tracked}"


def test_local_docs_are_ignored_by_repository_policy(repository_git):
    # The path need not exist: this checks what Git would do with a newly
    # created local document, without touching any preserved private files.
    result = repository_git(
        "check-ignore", "--no-index", "--verbose", "-z", "--stdin",
        input="docs/__trajviz_local_policy_probe__.md\0",
    )
    assert result.returncode == 0, "New local docs/ files must be ignored by Git"
    fields = result.stdout.split("\0")
    assert (ROOT / fields[0]).resolve() == ROOT / ".gitignore", (
        "The docs/ ignore policy must come from the tracked root .gitignore"
    )


def test_public_documentation_build_templates_remain_trackable(repository_git):
    result = repository_git(
        "-c", f"core.excludesFile={os.devnull}",
        "check-ignore", "--no-index", ".github/docs/__trajviz_template_probe__.css",
    )
    assert result.returncode == 1, "Only root docs/ is private; documentation build assets must remain trackable"
