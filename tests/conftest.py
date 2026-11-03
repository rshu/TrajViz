"""Hermetic inputs for the bundled attribution engine; no external checkout."""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_FIX = _TESTS / "fixtures"
_REFERENCE = _FIX / "reference"
_CACHE = _FIX / "attribution_cache" / "cache"

# Pin inputs before the facade captures its default reference root. Original
# schema-20 verdict fixtures remain available only to explicit legacy-contract
# checks; modern verification tests generate their own current fingerprints.
os.environ["TRAJVIZ_REFERENCE_ROOT"] = str(_REFERENCE)
os.environ["AWE_JUDGE_MODEL"] = "deepseek-v4-pro"
os.environ["AWE_STRICT_VERDICT_PROVENANCE"] = "1"

from agent_trajviz.awe._engine import adapters as _ad
from agent_trajviz.awe._engine import arbiter as _arb
from agent_trajviz.awe._engine import config as _cfg
from agent_trajviz.awe._engine import trajsignals as _ts

# All writable engine state belongs to this test process. Reference artifacts
# and recorded verdicts are read-only; no user cache or neighbouring data leaks
# into per-test outcomes, grading overrides or normalized trajectory evidence.
_TMP_CACHE = Path(tempfile.mkdtemp(prefix="trajviz-evaluation-tests-"))
atexit.register(shutil.rmtree, _TMP_CACHE, ignore_errors=True)
_cfg.PROJECT_DATA = _TMP_CACHE
_cfg.JUDGE_CACHE_DIR = _CACHE / "judge"
_arb.ARBITER_CACHE_DIR = _CACHE / "arbiter"
_ad._NORMTRAJ_CACHE = _TMP_CACHE / "normtraj"
_ts._CACHE = _TMP_CACHE / "trajsig"


def pytest_collection_finish(session):
    # The engine is shipped in every installation. Missing code or fixture data
    # must fail collection rather than turn integration checks into silent skips.
    assert (_REFERENCE / "data" / "requirements" / "astropy__astropy-13033.json").is_file(), \
        "The bundled attribution reference fixture is missing"
    from agent_trajviz.insight import attribution
    assert attribution.ATTRIBUTION_AVAILABLE, attribution._IMPORT_ERROR
