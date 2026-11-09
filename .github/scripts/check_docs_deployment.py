"""Verify that live Pages content matches the uploaded release documentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
_SHA = re.compile(r"(?:[a-f0-9]{40}|[a-f0-9]{64})\Z")
_MODES = {"release-tag", "documentation-update", "initial-release-bootstrap"}
_MAX_BYTES = 2 * 1024 * 1024


class VerificationError(ValueError):
    """The expected documentation is not available at the public URL."""


def _validate(expected: dict) -> dict:
    if not isinstance(expected.get("version"), str) or not _VERSION.fullmatch(expected["version"]):
        raise VerificationError("Expected version must be a plain X.Y.Z release.")
    if not isinstance(expected.get("docs_source"), str) or not _SHA.fullmatch(expected["docs_source"]):
        raise VerificationError("Expected documentation source must be a commit SHA.")
    if expected.get("source_mode") not in _MODES:
        raise VerificationError("Expected documentation source mode is invalid.")
    if not isinstance(expected.get("index_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", expected["index_sha256"]):
        raise VerificationError("Expected index digest must be SHA256.")
    return expected


def _provenance(expected: dict) -> dict:
    return {
        "package_tag": "v" + expected["version"],
        "docs_source": expected["docs_source"],
        "source_mode": expected["source_mode"],
    }


def _check_versions(value: object, version: str) -> None:
    if not isinstance(value, dict) or value.get("stable") != version or value.get("dev") != "dev":
        raise VerificationError("versions.json does not advertise the expected stable release.")
    versions = value.get("versions")
    if (
        not isinstance(versions, list)
        or version not in versions
        or any(not isinstance(v, str) or not _VERSION.fullmatch(v) for v in versions)
    ):
        raise VerificationError("versions.json has an invalid release list.")


def expectations(site: Path, version: str) -> dict:
    """Read the exact artifact's release provenance and landing-page digest."""
    _check_versions(json.loads((site / "versions.json").read_text()), version)
    provenance = json.loads((site / "stable/build-info.json").read_text())
    expected = _validate(
        {
            "version": version,
            "docs_source": provenance.get("docs_source"),
            "source_mode": provenance.get("source_mode"),
            "index_sha256": hashlib.sha256((site / "stable/index.html").read_bytes()).hexdigest(),
        }
    )
    if provenance != _provenance(expected):
        raise VerificationError("Stable artifact provenance does not match the requested release.")
    if json.loads((site / version / "build-info.json").read_text()) != provenance:
        raise VerificationError("Stable and versioned artifact provenance differ.")
    return expected


def _base_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise VerificationError("Pages URL must be an HTTP(S) site URL without credentials or query parameters.")
    return url.rstrip("/") + "/"


def _fetch(base: str, path: str, deadline: float, attempt: int) -> bytes:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise VerificationError("Verification deadline reached.")
    request = Request(
        base + path + f"?trajviz_verify={attempt}-{time.monotonic_ns()}",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Accept-Encoding": "identity"},
    )
    try:
        with urlopen(request, timeout=min(10.0, remaining)) as response:  # noqa: S310 - public Pages or local HTTP fixture
            data = response.read(_MAX_BYTES + 1)
    except HTTPError as exc:
        raise VerificationError(f"{path} returned HTTP {exc.code}.") from None
    except (URLError, OSError, HTTPException):
        raise VerificationError(f"{path} is temporarily unavailable.") from None
    if len(data) > _MAX_BYTES:
        raise VerificationError(f"{path} exceeds the response size limit.")
    return data


def _json(data: bytes, path: str) -> object:
    try:
        return json.loads(data)
    except (ValueError, UnicodeError):
        raise VerificationError(f"{path} is not valid JSON.") from None


def verify(url: str, expected: dict, *, timeout: float = 720, interval: float = 15) -> int:
    """Retry stale or unavailable content through a bounded CDN propagation window."""
    expected, base = _validate(expected), _base_url(url)
    if not 0 < timeout <= 900 or not 0 < interval <= 60:
        raise VerificationError("Verification timeout must be in (0,900] and interval in (0,60].")
    deadline, attempt, last = time.monotonic() + timeout, 0, "No response received."
    while time.monotonic() < deadline:
        attempt += 1
        try:
            _check_versions(
                _json(_fetch(base, "versions.json", deadline, attempt), "versions.json"), expected["version"]
            )
            for path in ("stable/build-info.json", expected["version"] + "/build-info.json"):
                if _json(_fetch(base, path, deadline, attempt), path) != _provenance(expected):
                    raise VerificationError(f"{path} does not match the uploaded release provenance.")
            data = _fetch(base, "stable/index.html", deadline, attempt)
            if hashlib.sha256(data).hexdigest() != expected["index_sha256"]:
                raise VerificationError("stable/index.html does not match the uploaded release content.")
            return attempt
        except VerificationError as exc:
            last = str(exc)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval, remaining))
    raise VerificationError(f"Live documentation did not match within {timeout:g}s: {last}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("expected")
    record.add_argument("--site", required=True, type=Path)
    record.add_argument("--version", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--url", required=True)
    for name in ("version", "docs-source", "source-mode", "index-sha256"):
        check.add_argument("--" + name, required=True)
    check.add_argument("--timeout", type=float, default=720)
    check.add_argument("--interval", type=float, default=15)
    args = parser.parse_args(argv)
    try:
        if args.command == "expected":
            expected = expectations(args.site, args.version)
            if os.environ.get("GITHUB_OUTPUT"):
                with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
                    handle.write("".join(f"{key}={value}\n" for key, value in expected.items()))
            print(json.dumps(expected))
        else:
            expected = {key: getattr(args, key) for key in ("version", "docs_source", "source_mode", "index_sha256")}
            attempts = verify(args.url, expected, timeout=args.timeout, interval=args.interval)
            print(f"Verified live documentation {args.version} and its recorded provenance ({attempts} attempt(s)).")
    except (VerificationError, OSError, ValueError, TypeError) as exc:
        print(f"Documentation verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
