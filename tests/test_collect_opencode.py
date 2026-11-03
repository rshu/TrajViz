from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trajviz.collect import opencode as consolidator


SCHEMA = """
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
    slug TEXT,
    directory TEXT,
    title TEXT,
    version TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    time_created INTEGER,
    time_updated INTEGER,
    time_compacting INTEGER,
    time_archived INTEGER,
    summary_diffs TEXT,
    revert TEXT,
    permission TEXT
);
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
"""


class OpenCodeConsolidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _make_database(self) -> tuple[Path, list[dict]]:
        database = self.root / "opencode.db"
        connection = sqlite3.connect(database)
        connection.executescript(SCHEMA)
        connection.execute(
            """
            INSERT INTO session VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                "ses_root", "project", "workspace", None, "root", "/workspace",
                "Root", "test", 0, 0, 0, 1_000, 2_000, None, None,
                None, None, None,
            ),
        )
        token_objects = [
            {
                "total": 100,
                "input": 20,
                "output": 10,
                "reasoning": 0,
                "cache": {"read": 70, "write": 0},
            },
            {
                "total": 80,
                "input": 10,
                "output": 5,
                "reasoning": 0,
                "cache": {"read": 65, "write": 0},
            },
        ]
        for index, tokens in enumerate(token_objects):
            created = 1_100 + index * 100
            payload = {
                "role": "assistant",
                "agent": "build",
                "tokens": tokens,
                "time": {"created": created, "completed": created + 50},
            }
            connection.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                (
                    f"msg_{index}", "ses_root", created, created + 50,
                    json.dumps(payload),
                ),
            )
        connection.commit()
        connection.close()
        return database, token_objects

    def _run_main(self, database: Path, output: Path) -> int:
        with (
            patch.dict(os.environ, {"OPENCODE_DATABASE": str(database)}),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return consolidator.main(["ses_root", str(output)])

    def test_preserves_per_message_token_objects(self) -> None:
        database, expected_tokens = self._make_database()
        output = self.root / "trajectory.json"

        self.assertEqual(self._run_main(database, output), 0)

        result = json.loads(output.read_text(encoding="utf-8"))
        actual_tokens = [
            message["info"]["tokens"] for message in result["messages"]
        ]
        self.assertEqual(actual_tokens, expected_tokens)

    def test_converted_usage_keeps_reported_totals_including_cache(self) -> None:
        import agent_trajviz as tv

        database, tokens = self._make_database()
        output = self.root / "trajectory.json"
        self.assertEqual(self._run_main(database, output), 0)
        run = tv.load(output)
        expected = sum(item["total"] for item in tokens)
        self.assertEqual(run.metrics["tokens"]["total"], expected)
        self.assertEqual(run.raw["token_usage"]["total_tokens"], expected)

    def test_refuses_to_overwrite_source_database(self) -> None:
        database, _ = self._make_database()
        original_header = database.read_bytes()[:16]

        self.assertEqual(self._run_main(database, database), 1)
        self.assertEqual(database.read_bytes()[:16], original_header)
        check = sqlite3.connect(database)
        try:
            self.assertEqual(check.execute("SELECT count(*) FROM message").fetchone()[0], 2)
        finally:
            check.close()

    def test_failed_export_replacement_preserves_existing_output_and_source(self) -> None:
        database, _ = self._make_database()
        database_bytes = database.read_bytes()
        output = self.root / "trajectory.json"
        output.write_bytes(b"previous export")
        with patch("agent_trajviz._fileio.os.replace", side_effect=OSError("controlled export replacement failure")):
            self.assertEqual(self._run_main(database, output), 1)
        self.assertEqual(output.read_bytes(), b"previous export")
        self.assertEqual(database.read_bytes(), database_bytes)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["opencode.db", "trajectory.json"])

    def test_preserves_native_part_runtime_timing_and_fills_row_fields(self) -> None:
        database, _ = self._make_database()
        payload = {"type": "reasoning", "text": "Review the local implementation.",
                   "time": {"start": 1111, "end": 1222, "created": 1100}}
        with sqlite3.connect(database) as connection:
            message_id = connection.execute("SELECT id FROM message ORDER BY id LIMIT 1").fetchone()[0]
            connection.execute("INSERT INTO part VALUES (?,?,?,?,?,?)",
                               ("part-timing", message_id, "ses_root", 1000, 1300, json.dumps(payload)))
        output = self.root / "trajectory.json"
        self.assertEqual(self._run_main(database, output), 0)
        result = json.loads(output.read_text())
        part = next(part for message in result["messages"] for part in message["parts"]
                    if part["id"] == "part-timing")
        self.assertEqual(part["time"], {"start": 1111, "end": 1222, "created": 1100, "updated": 1300})
        with sqlite3.connect(database) as connection:
            stored = json.loads(connection.execute("SELECT data FROM part WHERE id='part-timing'").fetchone()[0])
        self.assertEqual(stored, payload)


class OpenCodeCaptureCompletionTests(unittest.TestCase):
    @staticmethod
    def _assistant(session_id: str, *, finish=None, part_types=("step-start", "reasoning")) -> dict:
        return {
            "info": {"role": "assistant", "sessionID": session_id, "finish": finish,
                     "time": {"created": 1000, "completed": 1000}},
            "parts": [{"type": part_type} for part_type in part_types],
        }

    def _load(self, messages: list[dict]):
        import agent_trajviz as tv

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text(json.dumps({"info": {"id": "ses_root"}, "messages": messages}))
            return tv.load(path)

    def test_finished_child_does_not_hide_unfinished_parent(self) -> None:
        run = self._load([
            self._assistant("ses_root"),
            self._assistant("ses_child", finish="stop", part_types=("step-start", "text", "step-finish")),
        ])
        self.assertEqual(len(run.warnings), 1)
        self.assertIn("OpenCode session ses_root", run.warnings[0])
        self.assertIn("it may be incomplete", run.warnings[0])

    def test_only_last_assistant_in_each_session_is_checked(self) -> None:
        run = self._load([
            self._assistant("ses_root"),
            self._assistant("ses_root", finish="stop"),
        ])
        self.assertEqual(run.warnings, ())

    def test_native_finish_evidence_or_absent_start_does_not_warn(self) -> None:
        for finish, part_types in [
            ("stop", ("step-start", "reasoning")),
            (None, ("step-start", "step-finish")),
            (None, ("step_start", "step_finish")),
            (None, ("text",)),
        ]:
            with self.subTest(finish=finish, part_types=part_types):
                self.assertEqual(self._load([
                    self._assistant("ses_root", finish=finish, part_types=part_types),
                ]).warnings, ())

    def test_missing_session_id_falls_back_to_root_and_warning_is_deduplicated(self) -> None:
        from agent_trajviz.insight.formats.opencode import _convert_opencode_metadata

        message = self._assistant("", part_types=("step_start", "reasoning"))
        del message["info"]["sessionID"]
        raw = {"info": {"id": "ses_root"}, "messages": [message], "_load_warnings": ["existing warning"]}
        _convert_opencode_metadata(raw)
        _convert_opencode_metadata(raw)
        self.assertEqual(len(raw["_load_warnings"]), 2)
        self.assertEqual(raw["_load_warnings"][0], "existing warning")
        self.assertIn("OpenCode session ses_root", raw["_load_warnings"][1])


class OpenCodeDatabasePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _env(self, **env: str):
        return patch.dict(os.environ, env, clear=True)

    def test_open_code_database_env_wins(self) -> None:
        custom = self.root / "custom.db"
        custom.write_bytes(b"x")
        xdg = self.root / "xdg"
        (xdg / "opencode").mkdir(parents=True)
        (xdg / "opencode" / "opencode.db").write_bytes(b"y")
        with self._env(OPENCODE_DATABASE=str(custom), XDG_DATA_HOME=str(xdg)):
            self.assertEqual(consolidator.get_db_path(), custom)

    def test_open_code_db_absolute(self) -> None:
        custom = self.root / "native.db"
        with self._env(OPENCODE_DB=str(custom)):
            self.assertEqual(consolidator.get_db_path(), custom)

    def test_open_code_db_relative_joins_data_dir(self) -> None:
        xdg = self.root / "xdg"
        with self._env(OPENCODE_DB="opencode-dev.db", XDG_DATA_HOME=str(xdg)):
            self.assertEqual(
                consolidator.get_db_path(),
                xdg / "opencode" / "opencode-dev.db",
            )

    def test_xdg_data_home(self) -> None:
        xdg = self.root / "xdg"
        db = xdg / "opencode" / "opencode.db"
        db.parent.mkdir(parents=True)
        db.write_bytes(b"x")
        with self._env(XDG_DATA_HOME=str(xdg)):
            self.assertEqual(consolidator.get_db_path(), db)

    def test_open_code_data_dir(self) -> None:
        data = self.root / "portable"
        db = data / "opencode.db"
        data.mkdir()
        db.write_bytes(b"x")
        with self._env(OPENCODE_DATA_DIR=str(data), XDG_DATA_HOME=str(self.root / "xdg")):
            self.assertEqual(consolidator.get_db_path(), db)

    def test_default_home_local_share_when_missing(self) -> None:
        home = self.root / "home"
        home.mkdir()
        with self._env(), patch.object(Path, "home", return_value=home):
            self.assertEqual(
                consolidator.get_db_path(),
                home / ".local" / "share" / "opencode" / "opencode.db",
            )

    def test_windows_uses_localappdata_when_default_missing(self) -> None:
        home = self.root / "home"
        home.mkdir()
        local = self.root / "AppData" / "Local"
        db = local / "opencode" / "opencode.db"
        db.parent.mkdir(parents=True)
        db.write_bytes(b"x")
        with (
            self._env(LOCALAPPDATA=str(local), APPDATA=str(self.root / "Roaming")),
            patch.object(Path, "home", return_value=home),
            patch.object(consolidator, "_on_windows", return_value=True),
        ):
            self.assertEqual(consolidator.get_db_path(), db)

    def test_windows_uses_appdata_when_localappdata_missing(self) -> None:
        home = self.root / "home"
        home.mkdir()
        roaming = self.root / "AppData" / "Roaming"
        db = roaming / "opencode" / "opencode.db"
        db.parent.mkdir(parents=True)
        db.write_bytes(b"x")
        with (
            self._env(
                LOCALAPPDATA=str(self.root / "AppData" / "Local"),
                APPDATA=str(roaming),
            ),
            patch.object(Path, "home", return_value=home),
            patch.object(consolidator, "_on_windows", return_value=True),
        ):
            self.assertEqual(consolidator.get_db_path(), db)

    def test_windows_prefers_xdg_default_when_both_exist(self) -> None:
        home = self.root / "home"
        default = home / ".local" / "share" / "opencode" / "opencode.db"
        default.parent.mkdir(parents=True)
        default.write_bytes(b"x")
        local = self.root / "AppData" / "Local"
        other = local / "opencode" / "opencode.db"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"y")
        with (
            self._env(LOCALAPPDATA=str(local)),
            patch.object(Path, "home", return_value=home),
            patch.object(consolidator, "_on_windows", return_value=True),
        ):
            self.assertEqual(consolidator.get_db_path(), default)

    def test_non_windows_ignores_localappdata(self) -> None:
        home = self.root / "home"
        home.mkdir()
        local = self.root / "AppData" / "Local"
        db = local / "opencode" / "opencode.db"
        db.parent.mkdir(parents=True)
        db.write_bytes(b"x")
        with (
            self._env(LOCALAPPDATA=str(local)),
            patch.object(Path, "home", return_value=home),
            patch.object(consolidator, "_on_windows", return_value=False),
        ):
            self.assertEqual(
                consolidator.get_db_path(),
                home / ".local" / "share" / "opencode" / "opencode.db",
            )

    def test_expanduser_on_open_code_database(self) -> None:
        home = self.root / "home"
        home.mkdir()
        with self._env(
            OPENCODE_DATABASE="~/db/opencode.db",
            HOME=str(home),
            USERPROFILE=str(home),
        ):
            self.assertEqual(consolidator.get_db_path(), home / "db" / "opencode.db")

    def test_database_uri_escapes_spaces_and_opens_readonly(self) -> None:
        folder = self.root / "user name"
        folder.mkdir()
        db = folder / "opencode.db"
        seeded = sqlite3.connect(db)
        seeded.execute("CREATE TABLE t (id INTEGER)")
        seeded.commit()
        seeded.close()

        uri = consolidator._database_uri(db)
        self.assertTrue(uri.startswith("file:"))
        self.assertTrue(uri.endswith("?mode=ro"))
        self.assertIn("%20", uri)

        opened = sqlite3.connect(uri, uri=True)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                opened.execute("CREATE TABLE w (id INTEGER)")
        finally:
            opened.close()





class OpenCodeInterruptedFamilyTests(unittest.TestCase):
    setUp = OpenCodeConsolidatorTests.setUp
    tearDown = OpenCodeConsolidatorTests.tearDown
    _make_database = OpenCodeConsolidatorTests._make_database
    _run_main = OpenCodeConsolidatorTests._run_main

    def test_parent_rows_include_active_child_and_preserve_empty_descendant(self):
        import agent_trajviz as tv

        database, _ = self._make_database()
        with sqlite3.connect(database) as conn:
            for identity, parent in [("ses_child", "ses_root"), ("ses_empty", "ses_child"), ("ses_unrelated", None)]:
                conn.execute("INSERT INTO session (id,parent_id,title,directory,time_created,time_updated) VALUES (?,?,?,?,?,?)",
                             (identity, parent, identity, "/workspace", 1200, 1800))
            # No agent label and no completed Task metadata: hierarchy is still observed.
            conn.execute("INSERT INTO message VALUES (?,?,?,?,?)", ("child_message", "ses_child", 1300, 1600,
                         json.dumps({"role": "assistant", "time": {"created": 1300, "completed": 1600}})))
            conn.execute("INSERT INTO part VALUES (?,?,?,?,?,?)", ("active_task", "msg_0", "ses_root", 1200, 1800,
                         json.dumps({"type": "tool", "tool": "task", "state": {"status": "running", "input": {}}})))
        output = self.root / "family.json"
        self.assertEqual(self._run_main(database, output), 0)
        raw = json.loads(output.read_text())
        self.assertEqual({m["info"]["sessionID"] for m in raw["messages"]}, {"ses_root", "ses_child"})
        self.assertEqual({entry["info"]["id"] for entry in raw["session_manifest"]}, {"ses_root", "ses_child", "ses_empty"})
        self.assertEqual(raw["statistics"]["subagent_sessions"], 2)
        child = next(m["info"] for m in raw["messages"] if m["info"]["sessionID"] == "ses_child")
        self.assertTrue(child["isSubAgent"])
        self.assertEqual(child["parentSessionID"], "ses_root")
        self.assertEqual(child["sessionDepth"], 1)
        run = tv.load(output)
        self.assertEqual(run.raw["metadata"]["sub_agent_count"], 2)
        self.assertEqual(run.metrics["tool_success"], 0)
        self.assertIsNone(run.metrics["tool_success_rate"])
        child_step = next(s for s in run.steps if s["session_id"] == "ses_child")
        self.assertTrue(child_step["is_sub_agent"])
        self.assertEqual(child_step["parent_session_id"], "ses_root")

    def test_direct_native_pending_or_running_tools_have_unknown_outcomes(self):
        import agent_trajviz as tv

        for status in ["pending", "running"]:
            with self.subTest(status=status):
                path = self.root / f"{status}.json"
                path.write_text(json.dumps({"info": {"id": "root"}, "messages": [{"info": {"role": "assistant", "sessionID": "root"},
                    "parts": [{"type": "tool", "tool": "bash", "state": {"status": status, "input": {"command": "pytest"}}}]}]}))
                run = tv.load(path)
                self.assertEqual(run.metrics["tool_call_count"], 1)
                self.assertEqual(run.metrics["tool_success"], 0)
                self.assertIsNone(run.metrics["tool_success_rate"])

    def test_conflicting_native_parent_excludes_metadata_target_and_deduplicates_child(self):
        database, _ = self._make_database()
        with sqlite3.connect(database) as conn:
            for identity, parent in [("ses_child", "ses_root"), ("ses_outside", "another_root"), ("ses_legacy", None)]:
                conn.execute("INSERT INTO session (id,parent_id,title,directory,time_created,time_updated) VALUES (?,?,?,?,?,?)",
                             (identity, parent, identity, "/workspace", 1200, 1800))
                conn.execute("INSERT INTO message VALUES (?,?,?,?,?)", ("msg_" + identity, identity, 1300, 1600,
                             json.dumps({"role": "assistant", "finish": "stop"})))
                conn.execute("INSERT INTO part VALUES (?,?,?,?,?,?)", ("part_" + identity, "msg_0", "ses_root", 1100, 1500,
                             json.dumps({"type": "tool", "tool": "task", "state": {"status": "completed", "metadata": {"sessionId": identity}}})))
        output = self.root / "selected.json"
        self.assertEqual(self._run_main(database, output), 0)
        raw = json.loads(output.read_text())
        identities = [entry["info"]["id"] for entry in raw["session_manifest"]]
        self.assertEqual(set(identities), {"ses_root", "ses_child", "ses_legacy"})
        self.assertEqual(identities.count("ses_child"), 1)
        self.assertNotIn("ses_outside", {m["info"]["sessionID"] for m in raw["messages"]})

    def test_unfinished_child_has_no_invented_completion_or_duration_after_public_load(self):
        import agent_trajviz as tv

        database, _ = self._make_database()
        with sqlite3.connect(database) as conn:
            conn.execute("INSERT INTO session (id,parent_id,title,directory,time_created,time_updated) VALUES (?,?,?,?,?,?)",
                         ("ses_child", "ses_root", "reviewer", "/workspace", 1200, 2000))
            conn.execute("INSERT INTO message VALUES (?,?,?,?,?)", ("unfinished", "ses_child", 1500, 1750,
                         json.dumps({"role": "assistant", "time": {"created": 1500}})))
            for index, kind in enumerate(["step-start", "reasoning"]):
                conn.execute("INSERT INTO part VALUES (?,?,?,?,?,?)", (f"unfinished_{index}", "unfinished", "ses_child", 1500, 1750,
                             json.dumps({"type": kind, "text": "Reviewing."})))
        output = self.root / "unfinished.json"
        self.assertEqual(self._run_main(database, output), 0)
        raw = json.loads(output.read_text())
        child = next(m for m in raw["messages"] if m["info"]["id"] == "unfinished")
        self.assertNotIn("completed", child["info"]["time"])
        self.assertEqual(child["info"]["time"]["updated"], 1750)
        run = tv.load(output)
        child_step = next(s for s in run.steps if s["id"] == "unfinished")
        self.assertIsNone(child_step["time_completed_ms"])
        self.assertIsNone(child_step["duration"])
        self.assertTrue(any("ses_child" in warning and "incomplete" in warning for warning in run.warnings))

    def test_completed_message_still_uses_legacy_row_time_fallback(self):
        database, _ = self._make_database()
        with sqlite3.connect(database) as conn:
            conn.execute("UPDATE message SET data=? WHERE id='msg_0'", (json.dumps({"role": "assistant", "finish": "stop"}),))
        output = self.root / "finished.json"
        self.assertEqual(self._run_main(database, output), 0)
        raw = json.loads(output.read_text())
        message = next(m for m in raw["messages"] if m["info"]["id"] == "msg_0")
        self.assertEqual(message["info"]["time"]["completed"], 1150)


if __name__ == "__main__":
    unittest.main()
