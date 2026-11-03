"""Select maintained documentation sources for a published package version."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

PAGES = ("docs/index.md", "docs/quickstart.md", "docs/guides.md", "docs/data-reference.md")
OPTIONAL_PAGES = ("docs/awe.md",)
SCRIPTS = (".github/scripts/build_docs.py", ".github/scripts/check_docs_examples.py")


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def _allowed(name: str) -> bool:
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts and "\\" not in name
            and (name in (*PAGES, *OPTIONAL_PAGES) or name in SCRIPTS or name.startswith(".github/docs/")))


def _required_pages(builder: bytes) -> tuple[str, ...]:
    """Read the selected builder's declared guides without executing source code."""
    for node in ast.parse(builder).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "GUIDES"
                                               for target in node.targets):
            guides = ast.literal_eval(node.value)
            if not isinstance(guides, (tuple, list)) or any(not isinstance(name, str) for name in guides):
                raise ValueError("Documentation builder must declare literal guide names")
            names = tuple(f"docs/{name}.md" for name in guides)
            if any(name not in (*PAGES, *OPTIONAL_PAGES) for name in names):
                raise ValueError("Documentation builder requires a guide outside the maintained allowlist")
            return tuple(dict.fromkeys((*PAGES, *names)))
    return PAGES


def prepare(root: Path, tag: str, output: Path, bootstrap_version: str = "0.3.1", *, current_guides: bool = False) -> dict:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError("Documentation release tag must be vX.Y.Z")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Documentation source output must be empty")
    names = _git(root, "ls-tree", "-r", "--name-only", tag, "--", "docs", ".github/scripts", ".github/docs")
    release_files = [name for name in names.decode("utf-8").splitlines() if _allowed(name)]
    complete = all(name in release_files for name in (*PAGES, *SCRIPTS))
    output.mkdir(parents=True, exist_ok=True)
    if complete and not current_guides:
        required = _required_pages(_git(root, "show", f"{tag}:{SCRIPTS[0]}"))
        if any(name not in release_files for name in required):
            raise ValueError("The published tag is missing a guide required by its documentation builder")
        for name in release_files:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_git(root, "show", f"{tag}:{name}"))
        mode = "release-tag"
        source = _git(root, "rev-parse", f"{tag}^{{commit}}").decode("utf-8").strip()
    elif current_guides or tag == f"v{bootstrap_version}":
        # This existing release predates the docs site. Its initial guides are
        # validated against the published package before they are archived.
        required = _required_pages((root / SCRIPTS[0]).read_bytes())
        files = [root / name for name in (*required, *SCRIPTS)]
        files.extend(root / name for name in OPTIONAL_PAGES if name not in required and (root / name).exists())
        templates = root / ".github/docs"
        if templates.is_dir():
            files.extend(path for path in templates.rglob("*") if path.is_file())
        for path in files:
            if not path.is_file() or path.is_symlink():
                raise ValueError("A maintained documentation source is missing or linked")
            target = output / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        mode = "documentation-update" if current_guides else "initial-release-bootstrap"
        source = _git(root, "rev-parse", "HEAD").decode("utf-8").strip()
    else:
        raise ValueError("The published tag has no complete documentation sources")
    result = {"package_tag": tag, "docs_source": source, "source_mode": mode}
    (output / "build-info.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-version", default="0.3.1")
    parser.add_argument("--current-guides", action="store_true", help="Use reviewed current guides for a documentation correction")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.repository_root.resolve(), args.tag, args.output.resolve(), args.bootstrap_version, current_guides=args.current_guides)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Documentation sources: {exc}\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
