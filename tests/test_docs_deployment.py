"""Public Pages verification detects stale content despite a successful deploy API."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_docs_deployment", _ROOT / ".github/scripts/check_docs_deployment.py"
)
assert _SPEC and _SPEC.loader
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)
_VERSION, _SOURCE = "0.4.1", "a" * 40
_PROVENANCE = {"package_tag": "v" + _VERSION, "docs_source": _SOURCE, "source_mode": "release-tag"}
_INDEX = b"<!DOCTYPE html><h1>Released package0.4.1</h1>"
_EXPECTED = {
    "version": _VERSION,
    "docs_source": _SOURCE,
    "source_mode": "release-tag",
    "index_sha256": hashlib.sha256(_INDEX).hexdigest(),
}


def _payloads():
    return {
        "versions.json": json.dumps({"stable": _VERSION, "versions": [_VERSION, "0.4.0"], "dev": "dev"}).encode(),
        "stable/build-info.json": json.dumps(_PROVENANCE).encode(),
        _VERSION + "/build-info.json": json.dumps(_PROVENANCE).encode(),
        "stable/index.html": _INDEX,
    }


@contextlib.contextmanager
def _server(respond):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, dict(self.headers)))
            path = urlsplit(self.path).path.removeprefix("/project/")
            status, data = respond(path, seen)
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/project/", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_expectations_are_taken_from_assembled_artifact_and_expose_safe_job_outputs(tmp_path, monkeypatch):
    for name, data in _payloads().items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    assert check.expectations(tmp_path, _VERSION) == _EXPECTED
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert check.main(["expected", "--site", str(tmp_path), "--version", _VERSION]) == 0
    assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == _EXPECTED
    # A wrong release provenance must fail before changing the job outputs.
    before = output.read_text()
    (tmp_path / "stable/build-info.json").write_text(json.dumps(_PROVENANCE | {"package_tag": "v0.4.0"}))
    assert check.main(["expected", "--site", str(tmp_path), "--version", _VERSION]) == 1
    assert output.read_text() == before


def test_real_http_verifies_nonroot_pages_url_provenance_and_landing_page():
    values = _payloads()
    with _server(lambda path, _seen: (200, values[path])) as (url, seen):
        assert check.verify(url, _EXPECTED, timeout=2, interval=0.01) == 1
    assert [urlsplit(path).path for path, _ in seen] == ["/project/" + path for path in values]
    assert all("trajviz_verify=" in path for path, _ in seen)
    assert all(
        headers["Cache-Control"] == "no-cache" and headers["Accept-Encoding"] == "identity" for _, headers in seen
    )


@pytest.mark.parametrize("initial", ["stale", "unavailable"])
def test_real_http_retries_stale_cache_or_transient_http_error_until_correct(initial):
    values = _payloads()

    def respond(path, seen):
        if len(seen) == 1:
            return (
                (503, b"retry")
                if initial == "unavailable"
                else (200, b'{"stable":"0.4.0","versions":["0.4.0"],"dev":"dev"}')
            )
        return 200, values[path]

    with _server(respond) as (url, seen):
        assert check.verify(url, _EXPECTED, timeout=2, interval=0.01) == 2
    assert seen[0][0] != seen[1][0]


@pytest.mark.parametrize("broken", ["manifest", "provenance", "versioned", "html", "json"])
def test_real_http_never_reports_success_for_stale_or_mixed_artifact(broken):
    values = _payloads()
    if broken == "manifest":
        values["versions.json"] = b'{"stable":"0.4.0","versions":["0.4.0"],"dev":"dev"}'
    elif broken in {"provenance", "versioned"}:
        key = "stable/build-info.json" if broken == "provenance" else _VERSION + "/build-info.json"
        values[key] = json.dumps(_PROVENANCE | {"docs_source": "b" * 40}).encode()
    elif broken == "html":
        values["stable/index.html"] = b"old0.4.0 guide despite fresh JSON"
    else:
        values["versions.json"] = b"not JSON"
    with (
        _server(lambda path, _seen: (200, values[path])) as (url, _seen),
        pytest.raises(check.VerificationError, match="did not match within"),
    ):
        check.verify(url, _EXPECTED, timeout=0.12, interval=0.02)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "https://user:secret@example.org/", "https://example.org/?key=secret"]
)
def test_unsafe_site_urls_are_rejected_before_network(url, monkeypatch):
    monkeypatch.setattr(check, "_fetch", lambda *_args: pytest.fail("request must not be made"))
    with pytest.raises(check.VerificationError, match="Pages URL"):
        check.verify(url, _EXPECTED)


def test_workflow_routes_release_through_main_dispatch_and_verifies_actual_content():
    workflow = yaml.safe_load((_ROOT / ".github/workflows/docs.yml").read_text())
    jobs = workflow["jobs"]
    gateway = jobs["release_refresh"]
    assert "github.event_name == 'release'" in gateway["if"]
    assert gateway["permissions"] == {"actions": "write", "contents": "read"}
    assert gateway["env"]["GH_TOKEN"] == "${{ github.token }}"
    command = gateway["steps"][0]["run"]
    assert 'gh workflow run docs.yml --repo "$GITHUB_REPOSITORY" --ref main' in command
    assert "release_version=${RELEASE_TAG#v}" in command
    assert "github.event_name == 'release'" not in jobs["build"]["if"]
    assert "github.event_name == 'workflow_dispatch'" in jobs["build"]["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in jobs["build"]["if"]
    assert jobs["deploy"]["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    check_step = jobs["deploy"]["steps"][-1]
    assert "check_docs_deployment.py verify" in check_step["run"]
    assert "--timeout 720 --interval 15" in check_step["run"]
    assert "INDEX_SHA256" in check_step["env"] and "DOCS_SOURCE" in check_step["env"]
    assert not any("GITHUB_SHA" in step.get("env", {}) for job in jobs.values() for step in job["steps"])


def test_truncated_http_transport_response_is_retried_without_false_green(monkeypatch):
    from http.client import IncompleteRead

    values = _payloads()
    original = check.urlopen
    calls = []

    def transient(request, **kwargs):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise IncompleteRead(b"truncated", 100)
        return original(request, **kwargs)

    monkeypatch.setattr(check, "urlopen", transient)
    with _server(lambda path, _seen: (200, values[path])) as (url, _seen):
        assert check.verify(url, _EXPECTED, timeout=2, interval=0.01) == 2
    assert len(calls) == 5
