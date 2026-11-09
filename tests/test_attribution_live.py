"""Executed checks of the attribution attribution seam and its content-identity gate.

attribution's immutable reference trajectories and cached verdicts predate native
Claude ingestion. A test-only bridge supplies normalized native records with
the original archived source identity, without retaining a legacy production
loader. Native file/tree/ZIP identity is exercised in test_claude_native.py;
the DSH ZIP checks below continue to exercise its production loader directly.

What each group protects:

* reference identity — ``_source_sha256`` identifies the exact bytes attribution
  parses, so a reference file diagnoses, while a
  BOM-prefixed or one-byte-flipped copy is refused (lenient decoding must never
  leak into identity).
* mid-diagnosis mutation — the post-diagnosis recheck refuses when the
  canonical file changes, or disappears, while ``case_record`` is running.
* instance-id validation — encoded traversal, absolute paths and NUL bytes are
  refused before any filesystem access, with no file-existence oracle.
* reference-root isolation — concurrent diagnoses with different roots never see
  each other's gold, and a root is never inherited from a previous caller.
* verdict provenance — each of the four stamped fields independently gates the
  judge and the arbiter, the arbiter candidate glob is an exact match (not a
  prefix), and a rejection is surfaced in the rendered HTML rather than
  silently shaping a confident verdict.
* the Attribution tab — its handler survived the ``ui/`` split: it derives
  (agent, instance) from a reference path and degrades without fabricating.

The engine and reference fixture are required during collection. Only legacy
schema-20 verdict expectations are version-gated; identity and isolation always run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import unittest
import zipfile
from dataclasses import asdict
from functools import wraps
from html import escape
from pathlib import Path

from agent_trajviz.insight import attribution
from agent_trajviz.insight.loaders import load_trajectory
from agent_trajviz.insight.parser import parse_steps
from agent_trajviz.insight.rendering import build_attribution_html
from agent_trajviz.insight.session import LoadedSession, session_from_run
from claude_native_helpers import native_events

GOLD_AGENT = "claude_code"
GOLD_INST = "astropy__astropy-13033"          # deductive-only case
ARB_INST = "django__django-11477"             # arbiter-refuted case
PROVENANCE_FIELDS = ("trajectory_sha256", "requirements_sha256",
                     "prompt_version", "evidence_schema")


def _reference_present() -> bool:
    if not attribution.ATTRIBUTION_AVAILABLE:
        return False
    from agent_trajviz.awe._engine import config
    return config.requirements_path(GOLD_INST).is_file()


requires_backend = unittest.skipUnless(
    attribution.ATTRIBUTION_AVAILABLE, "attribution (awe) not importable")
requires_reference = unittest.skipUnless(
    _reference_present(), "reference fixture not present")


def _legacy_layer_controls() -> bool:
    if not attribution.ATTRIBUTION_AVAILABLE:
        return False
    from agent_trajviz.awe._engine.dossier import case_record
    return attribution._supports_layer_controls(case_record)


requires_legacy_verdicts = unittest.skipUnless(
    _legacy_layer_controls(),
    "schema-20 cached-verdict goldens exercise the legacy per-call dossier API")


def _canonical(agent: str, inst: str) -> Path:
    path = attribution._canonical_trajectory_path(agent, inst)
    if path is None:
        raise AssertionError("Missing canonical trajectory in the reference fixture")
    return Path(path)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _archived_reference_session(path) -> LoadedSession:
    """Test-only bridge for attribution's byte-pinned pre-native reference evidence.

    Production loading deliberately rejects that retired export format. The
    seam tests supply normalized native records while preserving the original
    source bytes/hash that the external attribution cache is pinned to. Native file
    and bundle loading/identity are exercised in test_claude_native.py.
    """
    from agent_trajviz import Trajectory
    from agent_trajviz.insight.formats.claude_code import convert_claude_code_events

    path = Path(path)
    original = path.read_bytes()
    doc = json.loads(original.decode("utf-8-sig"))
    session_id = (doc.get("session") or {}).get("id", "reference")
    raw = convert_claude_code_events(native_events(doc["trajectory"], session_id=session_id))
    raw["_source_path"] = str(path)
    raw["_source_sha256"] = hashlib.sha256(original).hexdigest()
    run = Trajectory(format="claude_code", steps=parse_steps(raw), source=str(path),
                     sha256=raw["_source_sha256"], raw=raw)
    return session_from_run(run)


class _RootRestoringTest(unittest.TestCase):
    """Every diagnosis rewrites attribution's process-global root; put it back."""

    def setUp(self):
        # Restoring on cleanup is not enough: tests in OTHER files rebind the
        # same process-global root and do not put it back, so start from a
        # known root as well as leaving one behind.
        attribution.configure(attribution._DEFAULT_ROOT)
        self.addCleanup(attribution.configure, attribution._DEFAULT_ROOT)
        self.tmp = Path(tempfile.mkdtemp(prefix="attr-live-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _reference_copy(self, name: str = "reference") -> Path:
        root = self.tmp / name
        shutil.copytree(attribution._DEFAULT_ROOT / "data", root / "data")
        return root


# --------------------------------------------------------------- load identity
@requires_backend
@requires_reference
class ArchivedReferenceIdentityTests(_RootRestoringTest):
    """The gate compares normalized archived evidence against attribution's source."""

    def test_reference_bridge_sha_is_the_sha_of_the_bytes_attribution_parses(self):
        """``_source_sha256`` must be the raw bytes of the canonical file.

        The read path decodes ``utf-8-sig`` and may rewrite ``file_path`` before
        hashing; if either ever leaked into the hash (hashing decoded text, or
        hashing a different file than the one displayed) every reference diagnosis
        would silently refuse while looking exactly like a healthy gate.
        """
        canon = _canonical(GOLD_AGENT, GOLD_INST)
        raw = _archived_reference_session(canon).raw
        self.assertNotIn("_error", raw)
        self.assertEqual(raw["_source_path"], str(canon))
        self.assertEqual(raw["_source_sha256"], _sha(canon))

    def test_loaded_reference_trajectory_passes_the_identity_gate(self):
        """End-to-end seam: loader output -> diagnose() must produce a verdict."""
        canon = _canonical(GOLD_AGENT, GOLD_INST)
        raw = _archived_reference_session(canon).raw
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                                   source_path=raw["_source_path"],
                                   expected_sha=raw["_source_sha256"])
        self.assertTrue(res.available, res.reason)
        self.assertEqual(res.mode, "reference")
        self.assertTrue(res.scorecard)

    def test_one_byte_flip_in_a_copy_is_refused(self):
        """Identity is CONTENT identity: a copy that differs anywhere must not
        borrow the canonical run's reference patch and test outcome."""
        canon = _canonical(GOLD_AGENT, GOLD_INST)
        data = bytearray(canon.read_bytes())
        idx = data.index(b"a")
        data[idx] = ord("b")
        flipped = self.tmp / "flipped.json"
        flipped.write_bytes(bytes(data))

        raw = _archived_reference_session(flipped).raw
        self.assertNotIn("_error", raw)
        self.assertNotEqual(raw["_source_sha256"], _sha(canon))
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                                   source_path=raw["_source_path"],
                                   expected_sha=raw["_source_sha256"])
        self.assertFalse(res.available)
        self.assertIn("does not match the canonical", res.reason or "")
        self.assertEqual(res.faults, [])
        self.assertIsNone(res.primary)

    def test_bom_prefixed_copy_loads_but_is_refused(self):
        """``utf-8-sig`` makes a BOM'd file LOAD; it must not make it PASS.

        Lenient decoding is a parsing convenience. If the hash were taken over
        the decoded text instead of the bytes, a BOM'd (or re-encoded) file
        would be accepted as the canonical run and be diagnosed against gold it
        does not belong to.
        """
        canon = _canonical(GOLD_AGENT, GOLD_INST)
        bom = self.tmp / "bom.json"
        bom.write_bytes(b"\xef\xbb\xbf" + canon.read_bytes())

        raw = _archived_reference_session(bom).raw
        self.assertNotIn("_error", raw, "utf-8-sig decode should still parse a BOM")
        self.assertTrue(parse_steps(raw), "the BOM'd copy must still be displayable")
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                                   source_path=raw["_source_path"],
                                   expected_sha=raw["_source_sha256"])
        self.assertFalse(res.available)
        self.assertIn("does not match the canonical", res.reason or "")

    def test_unreadable_source_without_a_captured_sha_refuses(self):
        """A displayed source whose identity cannot be established at all must
        refuse, not silently skip the gate and diagnose the reference copy."""
        missing = self.tmp / "vanished.json"
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                                   source_path=str(missing))
        self.assertFalse(res.available)
        self.assertIn("content identity", res.reason or "")
        self.assertEqual(res.faults, [])


# ------------------------------------------------------- mid-diagnosis changes
@requires_backend
@requires_reference
class MidDiagnosisMutationTests(_RootRestoringTest):
    """``case_record``/``detect`` re-read the reference data AFTER the identity gate."""

    def _race(self, mutate):
        # Hook the function this backend generation actually calls. Modern
        # objective analysis bypasses case_record, so patching it would make a
        # mutation test pass through without ever mutating the evidence.
        if _legacy_layer_controls():
            import agent_trajviz.awe._engine.dossier as module
            name = "case_record"
        else:
            import agent_trajviz.awe._engine.detect as module
            name = "detect"
        original = getattr(module, name)

        @wraps(original)
        def racing(*args, **kwargs):
            out = original(*args, **kwargs)
            mutate()
            return out

        setattr(module, name, racing)
        self.addCleanup(setattr, module, name, original)

    def test_canonical_change_during_diagnosis_is_refused(self):
        """An external writer (a ``git pull`` of the reference data) swapping the file between
        the gate and the read would produce a verdict describing bytes the gate
        never approved and the UI never displayed."""
        root = self._reference_copy()
        target = root / "data" / "trajectory" / GOLD_AGENT / f"{GOLD_INST}.json"
        loaded_sha = _sha(target)

        def mutate():
            payload = json.loads(target.read_text())
            payload["_raced"] = True
            target.write_text(json.dumps(payload))

        self._race(mutate)
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                                   expected_sha=loaded_sha, reference_root=root)
        self.assertFalse(res.available)
        self.assertIn("changed while the diagnosis was running", res.reason or "")
        self.assertEqual(res.faults, [])

    def test_canonical_deletion_during_diagnosis_is_refused(self):
        """Deletion is a change too: an unreadable canonical file must not
        recheck as 'unchanged' just because hashing it now returns None."""
        root = self._reference_copy()
        target = root / "data" / "trajectory" / GOLD_AGENT / f"{GOLD_INST}.json"
        loaded_sha = _sha(target)

        self._race(target.unlink)
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST,
                                   expected_sha=loaded_sha, reference_root=root)
        self.assertFalse(res.available)
        self.assertIn("changed while the diagnosis was running", res.reason or "")


# ------------------------------------------------------- instance-id validation
@requires_backend
class InstanceIdValidationTests(_RootRestoringTest):
    """``instance_id`` must never reach path construction."""

    HOSTILE = (
        "../../../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "%2e%2e%2f%2e%2e%2fetc",
        "/etc/passwd",
        "/Users/someone/.ssh/id_rsa",
        f"../{GOLD_AGENT}/{GOLD_INST}",
        "..",
        f"{GOLD_INST}/",
        f"{GOLD_INST}%00.json",
        "x\x00y",
        "~/secrets",
        "  ",
    )

    def test_hostile_instance_ids_are_refused(self):
        for evil in self.HOSTILE:
            with self.subTest(instance_id=evil):
                res = attribution.diagnose(agent=GOLD_AGENT, instance_id=evil)
                self.assertFalse(res.available)
                self.assertIn("invalid instance id", res.reason or "")
                self.assertEqual(res.faults, [])
                self.assertIsNone(res.primary)

    def test_rejection_is_not_a_file_existence_oracle(self):
        """Every rejected id gets the SAME message, so a caller cannot probe the
        filesystem by comparing the refusal for a path that exists against one
        that does not."""
        reasons = {attribution.diagnose(agent=GOLD_AGENT, instance_id=e).reason
                   for e in ("/etc/passwd", "/definitely/not/here/xyz",
                             "../../etc/hosts", "../../nope/nope")}
        self.assertEqual(len(reasons), 1, reasons)


# ------------------------------------------------------------- root isolation
@requires_backend
@requires_reference
class ReferenceRootIsolationTests(_RootRestoringTest):
    """A request's reference root is explicit and never inherited."""

    SENTINEL = "ROOT-B-SENTINEL-TASK-STATEMENT"

    def _root_with_sentinel_gold(self) -> Path:
        root = self._reference_copy("rootB")
        req = root / "data" / "requirements" / f"{GOLD_INST}.json"
        payload = json.loads(req.read_text())
        payload["problem_statement"] = self.SENTINEL
        req.write_text(json.dumps(payload))
        return root

    def test_concurrent_diagnoses_with_different_roots_do_not_leak(self):
        """Two viewers pointed at different corpora must never be served each
        other's gold: attribution's root is process-global, so only holding the module
        lock across configure+diagnose keeps a request's root its own."""
        root_a = self._reference_copy("rootA")
        root_b = self._root_with_sentinel_gold()
        seen: dict[str, set] = {"A": set(), "B": set()}
        failures: list[str] = []
        start = threading.Barrier(2)

        def run(tag: str, root: Path):
            try:
                start.wait(timeout=30)
                for _ in range(6):
                    res = attribution.diagnose(agent=GOLD_AGENT,
                                               instance_id=GOLD_INST,
                                               reference_root=root)
                    statement = ((res.task or {}).get("problem_statement") or "")
                    seen[tag].add(self.SENTINEL in statement)
            except Exception as exc:  # pragma: no cover - reported as a failure
                failures.append(f"{tag}: {exc!r}")

        threads = [threading.Thread(target=run, args=("A", root_a)),
                   threading.Thread(target=run, args=("B", root_b))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        self.assertEqual(failures, [])
        self.assertEqual(seen["A"], {False}, "root A saw root B's gold")
        self.assertEqual(seen["B"], {True}, "root B did not see its own gold")

    def test_root_is_not_inherited_from_a_previous_caller(self):
        """``configure()`` by an earlier caller must not decide the next
        diagnosis: a caller that passes no root gets the import-time default."""
        from agent_trajviz.awe._engine import config
        root_b = self._root_with_sentinel_gold()
        attribution.configure(root_b)
        self.assertEqual(config.DATA, root_b.resolve() / "data")

        env_before = dict(os.environ)
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST)
        self.assertEqual(config.DATA, attribution._DEFAULT_ROOT.resolve() / "data")
        self.assertNotIn(self.SENTINEL,
                         (res.task or {}).get("problem_statement") or "")
        # Configuration goes through attribution's attributes; the process
        # environment is never used as a side channel.
        self.assertEqual(dict(os.environ), env_before)


# --------------------------------------------------------- verdict provenance
@requires_backend
@requires_reference
@requires_legacy_verdicts
class VerdictProvenanceTests(_RootRestoringTest):
    """A cached LLM verdict shapes a diagnosis only with FULL provenance."""

    def setUp(self):
        super().setUp()
        import agent_trajviz.awe._engine.arbiter as arbiter
        from agent_trajviz.awe._engine import config
        self.arbiter_mod = arbiter
        self.config_mod = config
        self.addCleanup(setattr, config, "JUDGE_CACHE_DIR", config.JUDGE_CACHE_DIR)
        self.addCleanup(setattr, arbiter, "ARBITER_CACHE_DIR", arbiter.ARBITER_CACHE_DIR)
        self.judge_src = config.JUDGE_CACHE_DIR
        self.arbiter_src = arbiter.ARBITER_CACHE_DIR

    def _cache_copy(self, suffix: str) -> tuple[Path, Path]:
        judge = self.tmp / f"judge-{suffix}"
        arb = self.tmp / f"arbiter-{suffix}"
        shutil.copytree(self.judge_src, judge)
        shutil.copytree(self.arbiter_src, arb)
        self.config_mod.JUDGE_CACHE_DIR = judge
        self.arbiter_mod.ARBITER_CACHE_DIR = arb
        return judge, arb

    def test_baseline_vendored_verdicts_verify(self):
        """Guards the other tests in this class: if the vendored verdicts ever
        stop verifying, every 'field X disables the layer' assertion below would
        pass for the wrong reason."""
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=ARB_INST)
        self.assertTrue(res.available, res.reason)
        self.assertEqual(res.notes, [])
        self.assertEqual(res.blame_status, "refuted_unattributed")

    def _tamper(self, directory: Path, field: str) -> int:
        count = 0
        for path in directory.rglob("*.json"):
            record = json.loads(path.read_text())
            record[field] = 999999 if field == "evidence_schema" else "TAMPERED"
            path.write_text(json.dumps(record))
            count += 1
        return count

    def test_every_provenance_field_disables_the_judge(self):
        """The trajectory is not the only prompt input: the task statement, the
        prompt text and the evidence schema all shape a verdict, so any of them
        drifting must disable the judge rather than reuse a stale opinion."""
        for field in PROVENANCE_FIELDS:
            with self.subTest(field=field):
                judge, _ = self._cache_copy(f"j-{field}")
                self.assertTrue(self._tamper(judge, field))
                res = attribution.diagnose(agent=GOLD_AGENT, instance_id=ARB_INST)
                self.assertTrue(res.available, res.reason)
                self.assertFalse(res.used_judge)
                self.assertTrue(any("judge verdict ignored" in n for n in res.notes),
                                res.notes)

    def test_every_provenance_field_disables_the_arbiter(self):
        """An unverifiable refutation must not erase a rule-elected primary."""
        for field in PROVENANCE_FIELDS:
            with self.subTest(field=field):
                _, arb = self._cache_copy(f"a-{field}")
                self.assertTrue(self._tamper(arb, field))
                res = attribution.diagnose(agent=GOLD_AGENT, instance_id=ARB_INST)
                self.assertTrue(res.available, res.reason)
                self.assertEqual(res.blame_status, "primary")
                self.assertTrue(any("arbiter verdict ignored" in n for n in res.notes),
                                res.notes)

    def test_arbiter_candidate_glob_is_exact_not_prefix(self):
        """A bare ``{id}*`` glob would let an unrelated instance whose id merely
        EXTENDS this one (proj-1147 vs proj-11477) disable a perfectly good
        arbiter verdict; candidates must be ``{id}.json`` / ``{id}__*.json``."""
        _, arb = self._cache_copy("glob")
        agent_dir = next(p for p in arb.rglob(GOLD_AGENT) if p.is_dir())
        intruder = agent_dir / f"{ARB_INST}X__code_editing__relevant_change_omitted.json"
        intruder.write_text(json.dumps({
            "agent": GOLD_AGENT, "instance_id": f"{ARB_INST}X",
            "trajectory_sha256": "0" * 64, "requirements_sha256": "0" * 64,
            "prompt_version": "deadbeefcafe", "evidence_schema": 0,
            "verdict": {"decision": "refute"},
        }))
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=ARB_INST)
        self.assertTrue(res.available, res.reason)
        self.assertEqual(res.notes, [],
                         "a prefix-extending sibling disabled this case's arbiter")
        self.assertEqual(res.blame_status, "refuted_unattributed")

    def test_provenance_rejection_is_surfaced_in_the_rendered_html(self):
        """Degrading honestly means SAYING so: when the LLM layers are dropped
        the reader must see it next to the verdict, not just in the dataclass."""
        judge, _ = self._cache_copy("render")
        self._tamper(judge, "trajectory_sha256")
        res = attribution.diagnose(agent=GOLD_AGENT, instance_id=ARB_INST)
        self.assertTrue(res.notes)
        html = build_attribution_html(asdict(res))
        for note in res.notes:
            self.assertIn(escape(note), html)


# ----------------------------------------------------- modern model isolation
@requires_backend
@requires_reference
@unittest.skipIf(_legacy_layer_controls(), "legacy cached-verdict behavior is tested above")
class ModernVerdictIsolationTests(_RootRestoringTest):
    """The modern objective bridge must not consume unverifiable model caches."""

    def test_objective_diagnosis_never_reads_model_verdicts(self):
        import agent_trajviz.awe._engine.judge as judge
        import agent_trajviz.awe._engine.arbiter as arbiter

        def forbidden(*args, **kwargs):
            self.fail("Objective analysis tried to read a cached model verdict")

        for module, name in ((judge, "verdict_cached"), (arbiter, "auto_adjudicate")):
            original = getattr(module, name)
            setattr(module, name, forbidden)
            self.addCleanup(setattr, module, name, original)
        result = attribution.diagnose(agent=GOLD_AGENT, instance_id=ARB_INST)
        self.assertTrue(result.available, result.reason)
        self.assertFalse(result.used_judge)
        self.assertIsNone(result.arbiter)
        self.assertEqual(result.primary, {"capability": "code_editing", "error_type": "relevant_change_omitted"})


# ------------------------------------------------------------- zip identity
class ZipSourceIdentityTests(unittest.TestCase):
    """The zip branch hashes the ARCHIVE, so every merged member is covered."""

    def _events(self, child_text: str) -> tuple[list[dict], list[dict]]:
        def evt(kind, seq, time, data):
            return {"type": kind, "seq": seq, "time": time, "data": data}

        def session(**extra):
            base = {"type": "session", "version": 0, "id": "session-parent",
                    "createdAt": 1_787_623_647_203, "cwd": "/p",
                    "delegationDepth": 0, "agentPreset": "standard"}
            base.update(extra)
            return base

        def user(seq, time, text, mid):
            return evt("user/message", seq, time, {
                "content": [{"type": "text", "text": text}],
                "source": {"kind": "user"}, "role": "user", "id": mid})

        def assistant(seq, time, text, mid):
            return evt("assistant/message", seq, time, {
                "turn": 1, "step": 1,
                "message": {"role": "assistant",
                            "content": [{"type": "text", "text": text}],
                            "source": {"kind": "model", "provider": "deepseek-official",
                                       "model": "deepseek-v4-pro"},
                            "id": mid},
                "usage": {"inputTokens": 10, "outputTokens": 3,
                          "cacheReadTokens": 0, "reasoningTokens": 0}})

        parent = [session(), user(1, 1_787_623_647_300, "do it", "u1"),
                  assistant(2, 1_787_623_647_500, "parent done", "a1")]
        child = [session(id="child-1", origin="subagent",
                         parentSession="session-parent", seedLength=1,
                         createdAt=1_787_623_648_050),
                 user(2, 1_787_623_648_200, "explore", "cu"),
                 assistant(3, 1_787_623_648_400, child_text, "ca")]
        return parent, child

    def _build_zip(self, tmp: Path, name: str, child_text: str) -> Path:
        parent, child = self._events(child_text)
        stage = tmp / f"stage-{name}"
        (stage / "export" / "subagents" / "child-1").mkdir(parents=True)
        parent_path = stage / "export" / "session.jsonl"
        child_path = stage / "export" / "subagents" / "child-1" / "session.jsonl"
        parent_path.write_text("".join(json.dumps(e) + "\n" for e in parent))
        child_path.write_text("".join(json.dumps(e) + "\n" for e in child))
        archive = tmp / f"{name}.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(parent_path, "export/session.jsonl")
            zf.write(child_path, "export/subagents/child-1/session.jsonl")
        return archive

    def test_zip_sha_is_the_archive_and_covers_subagent_members(self):
        """``_source_path`` points at the zip, so ``_source_sha256`` must be the
        zip's bytes: a subagent member is displayed content, and an identity
        that ignored it would call two different sessions the same run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            first = self._build_zip(tmp, "first", "CHILD-ALPHA")
            second = self._build_zip(tmp, "second", "CHILD-BETA-DIFFERENT-TEXT")

            raw_first = load_trajectory(str(first))
            raw_second = load_trajectory(str(second))
            self.assertNotIn("_error", raw_first)
            self.assertEqual(raw_first["_source_path"], str(first))
            self.assertEqual(raw_first["_source_sha256"], _sha(first))
            self.assertNotEqual(raw_first["_source_sha256"],
                                raw_second["_source_sha256"],
                                "a changed subagent member left the identity unchanged")
            self.assertNotEqual(parse_steps(raw_first), parse_steps(raw_second))


# ------------------------------------------------------------ Attribution tab
@requires_backend
@requires_reference
class AttributionTabTests(_RootRestoringTest):
    """The tab handler after the ui/ split, driven with a real reference path."""

    @staticmethod
    def _handler(name: str):
        from agent_trajviz.insight.insight import build_ui
        app = build_ui()
        for block_fn in app.fns.values():
            fn = getattr(block_fn, "fn", None)
            if getattr(fn, "__name__", "") == name:
                return fn
        raise AssertionError(f"{name} is not wired into build_ui()")

    def _session_raw(self, path) -> dict:
        session = _archived_reference_session(path)
        self.assertIsInstance(session, LoadedSession, getattr(session, "message", ""))
        return session.raw

    def test_reference_path_auto_derives_agent_and_instance(self):
        """A reference file (.../trajectory/<agent>/<id>.json) must diagnose with no
        overrides typed — that auto-derivation is the tab's whole ergonomics."""
        handler = self._handler("on_diagnose")
        raw = self._session_raw(_canonical(GOLD_AGENT, GOLD_INST))
        self.assertIn("_source_sha256", raw,
                      "the load path must hand the tab the displayed bytes' identity")

        html, status = handler(raw, "", "", "")
        self.assertIn("score-dim-grid", html)
        self.assertIn("Primary cause", html)
        self.assertIn(f"{GOLD_AGENT}/{GOLD_INST}", status)

    def test_uploaded_native_format_derives_agent_without_cache_folder_identity(self):
        """A cache folder is incidental; recognized session content supplies the agent.

        A matching canonical source and test evidence still gate the diagnosis.
        """
        handler = self._handler("on_diagnose")
        upload_dir = self.tmp / "0123456789abcdef"
        upload_dir.mkdir()
        upload = upload_dir / f"{GOLD_INST}.json"
        shutil.copyfile(_canonical(GOLD_AGENT, GOLD_INST), upload)

        raw = self._session_raw(upload)
        html, status = handler(raw, "", "", "")
        self.assertIn("score-dim-grid", html)
        self.assertIn("Primary cause", html)
        self.assertIn(f"{GOLD_AGENT}/{GOLD_INST}", status)
        self.assertNotIn(upload_dir.name, status)

        # Explicit reference identities remain supported.
        html_ok, status_ok = handler(raw, GOLD_AGENT, GOLD_INST, "")
        self.assertIn("score-dim-grid", html_ok)
        self.assertIn(f"{GOLD_AGENT}/{GOLD_INST}", status_ok)

    def test_no_loaded_trajectory_is_reported_not_diagnosed(self):
        handler = self._handler("on_diagnose")
        html, status = handler({}, "", "", "")
        self.assertIn("Load a trajectory", html)
        self.assertIn("No trajectory loaded", status)

    def test_autoload_replaces_stale_overrides_with_current_session_identity(self):
        """The overrides name a CASE; carrying them across a new load would
        attribute the newly loaded trajectory to the previous case."""
        handler = self._handler("on_diagnose_autoload")
        raw = self._session_raw(_canonical(GOLD_AGENT, GOLD_INST))
        outputs = handler(raw, "")
        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[2], GOLD_AGENT)
        self.assertEqual(outputs[3], GOLD_INST)
        self.assertNotIn("score-dim-grid", outputs[0])
        self.assertNotIn("Primary cause", outputs[0])


if __name__ == "__main__":
    unittest.main()
