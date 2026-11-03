"""Exercise bundled evaluation resources from a core-only installed wheel."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import agent_trajviz
from agent_trajviz.awe import EvaluationSettings, available_operations, load_run, run_operation, write_report
from agent_trajviz.awe import _engine


def main() -> None:
    engine = Path(_engine.__file__).parent
    required = ["configs/countermeasures.json", "configs/rsi_adapters.json", "configs/taxonomy.slice.yaml", "templates/attribution.html",
                "schemas/normalized_trajectory.schema.json", "fixtures/fault_injection/python/evaluate.py",
                "fixtures/fault_injection/javascript/evaluate.js", "scripts/run_awe.py"]
    for name in required:
        assert (engine / name).is_file(), f"Missing installed resource: {name}"
    operations = available_operations()
    assert len(operations) >= 40 and all(row["available"] for row in operations)
    with tempfile.TemporaryDirectory(prefix="trajviz-installed-evaluation-") as tmp:
        root = Path(tmp)
        source = root / "faults.json"
        source.write_text(json.dumps([{"case_id": "sample__task", "agent": "sample", "instance_id": "task",
                                      "blame_status": "unattributed", "faults": [],
                                      "task": {"problem_statement": "<script>inert()</script>"},
                                      "future_extension": {"preserved": True}}]), encoding="utf-8")
        run = load_run(source)
        exported = run.export(root / "saved.zip")
        assert load_run(exported).raw_artifacts == run.raw_artifacts
        assert load_run(run.write(root / "saved.json")).cases == run.cases
        report = write_report(run, root / "report.html").read_text(encoding="utf-8")
        assert "&lt;script&gt;inert()&lt;/script&gt;" in report and "<script>inert()" not in report
        reference = root / "input benchmark"
        reference.mkdir()
        settings = EvaluationSettings(reference_root=reference, output_root=root / "outputs")
        catalog = run_operation("catalog", settings)
        assert catalog.success, catalog.reason
        controlled = run_operation("fault_injection", settings, payload={"language": "python"})
        assert controlled.success, controlled.reason
        controls = json.loads((Path(controlled.output_dir) / "fault-injection.json").read_text(encoding="utf-8"))
        assert controls["summary"]["passed"] and controls["summary"]["cases_passed"] > 0
        print(json.dumps({"version": agent_trajviz.__version__, "package": agent_trajviz.__file__,
                          "bundled_resources": len(required), "operations": len(operations),
                          "controlled_cases": controls["summary"]["cases_passed"],
                          "saved_roundtrip": True, "offline_report": True}))


if __name__ == "__main__":
    main()
