"""The persisted normalized-trajectory format must match its schema (review 2.10:
the schema was documentation-only and had drifted from the code — `delegate`
missing from the enum, field names differing, an envelope no code produced).

Stdlib-only structural check: every field the adapters emit must be declared in
the schema with a compatible enum, and every canonical action any adapter can
emit must be in the schema's enum.

Run:  python3 -m unittest tests.test_schema_conformance -v
"""

from __future__ import annotations

from pathlib import Path as _EnginePath
from agent_trajviz.awe import _engine as _engine_package


import dataclasses
import json
import os
import unittest


from agent_trajviz.awe._engine import trajectory
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory

_ENGINE_ROOT = _EnginePath(_engine_package.__file__).resolve().parent

SCHEMA_PATH = os.path.join(str(_ENGINE_ROOT), "schemas", "normalized_trajectory.schema.json")


class TestSchemaConformance(unittest.TestCase):
    def setUp(self):
        with open(SCHEMA_PATH) as f:
            self.schema = json.load(f)
        self.step_props = self.schema["$defs"]["step"]["properties"]

    def test_step_fields_match_dataclass(self):
        code_fields = {f.name for f in dataclasses.fields(NormalizedStep)}
        schema_fields = set(self.step_props)
        self.assertEqual(code_fields, schema_fields, "NormalizedStep fields and schema step properties drifted")

    def test_canonical_action_enum_covers_code_constants(self):
        enum = set(self.step_props["canonical_action"]["enum"])
        code_actions = {
            trajectory.SEARCH,
            trajectory.READ,
            trajectory.EDIT,
            trajectory.WRITE,
            trajectory.RUN_COMMAND,
            trajectory.RUN_TEST,
            trajectory.DELEGATE,
            trajectory.PLAN,
            trajectory.OTHER,
            trajectory.NONE,
        }
        self.assertEqual(enum, code_actions)

    def test_observability_enum_matches(self):
        enum = set(self.step_props["observability"]["enum"])
        self.assertEqual(enum, {trajectory.OBSERVED, trajectory.HIDDEN, trajectory.INFERRED})

    def test_persisted_envelope_matches_schema(self):
        traj = NormalizedTrajectory("a", "i", [NormalizedStep(step_id=0, turn_id=0, actor="agent", type="message")])
        persisted = {"schema_version": trajectory.SCHEMA_VERSION, **traj.to_dict()}
        self.assertEqual(set(persisted), set(self.schema["properties"]))
        step = persisted["steps"][0]
        self.assertTrue(
            set(step) <= set(self.step_props), f"undeclared step fields: {set(step) - set(self.step_props)}"
        )
        for req in self.schema["$defs"]["step"]["required"]:
            self.assertIn(req, step)


if __name__ == "__main__":
    unittest.main()
