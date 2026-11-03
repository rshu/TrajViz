"""Public saved-result CLI and offline HTML exports."""
import json

from agent_trajviz.awe import load_run, write_report
from agent_trajviz.awe.__main__ import main


def _source(tmp_path):
    source = tmp_path / "faults.json"
    source.write_text(json.dumps([{"case_id": "custom__case-1", "agent": "custom", "instance_id": "case-1",
                                   "blame_status": "unattributed", "faults": [],
                                   "task": {"problem_statement": "<script>untrusted()</script>"}}]))
    return source


def test_inspect_export_and_roundtrip_without_backend(tmp_path, capsys):
    source = _source(tmp_path)
    assert main(["inspect", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["cases"] == 1
    output = tmp_path / "run.zip"
    assert main(["export", str(source), "--output", str(output)]) == 0
    assert load_run(output).cases == load_run(source).cases
    envelope = tmp_path / "run.json"
    assert main(["inspect", str(output), "--output", str(envelope)]) == 0
    assert load_run(envelope).cases == load_run(source).cases


def test_report_escapes_source_and_keeps_evidence(tmp_path):
    result = write_report(load_run(_source(tmp_path)), tmp_path / "run.html")
    document = result.read_text()
    assert "&lt;script&gt;untrusted()&lt;/script&gt;" in document
    assert "<script>untrusted()" not in document
    assert "custom__case-1" in document and "Saved run provenance" in document


def test_bundled_operation_catalog_needs_no_external_checkout(capsys):
    assert main(["operations"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {"pipeline", "judge", "arbiter", "rsi_prepare", "preflight"} <= {row["id"] for row in rows}
    assert all(row["available"] for row in rows)
