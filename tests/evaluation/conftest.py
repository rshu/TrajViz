"""Run evaluator regression controls with private inputs and no network."""

from __future__ import annotations

import importlib
import socket

import pytest


@pytest.fixture(autouse=True)
def isolated_evaluation_inputs(tmp_path, monkeypatch):
    from agent_trajviz.awe._engine import config

    data = tmp_path / "benchmark" / "data"
    project = tmp_path / "project"
    project.mkdir()
    for name in ("requirements", "patch", "trajectory", "labels"):
        (data / name).mkdir(parents=True)
    for name, value in {
        "ARGUS_ROOT": data.parent,
        "DATA": data,
        "REQUIREMENTS_DIR": data / "requirements",
        "PATCH_DIR": data / "patch",
        "TRAJECTORY_DIR": data / "trajectory",
        "LABELS_DIR": data / "labels",
        "PROJECT_DATA": project,
        "OUTPUT_DATA": project,
        "JUDGE_CACHE_DIR": project / "cache" / "judge",
        "OPENROUTER_API_KEY": "",
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    modules = [
        importlib.import_module("agent_trajviz.awe._engine." + name)
        for name in (
            "adapters",
            "arbiter",
            "benchmark",
            "countermeasures",
            "execution_context",
            "gold",
            "outcomes",
            "scout",
            "trajsignals",
        )
    ]
    for module_name, name, path in (
        ("adapters", "_NORMTRAJ_CACHE", project / "cache" / "normtraj"),
        ("trajsignals", "_CACHE", project / "cache" / "trajsig"),
        ("arbiter", "ARBITER_CACHE_DIR", project / "cache" / "arbiter"),
    ):
        module = importlib.import_module("agent_trajviz.awe._engine." + module_name)
        monkeypatch.setattr(module, name, path)

    def clear_caches():
        for module in modules:
            for value in list(vars(module).values()):
                clear = getattr(value, "cache_clear", None)
                if callable(clear):
                    clear()

    def no_network(*args, **kwargs):
        raise AssertionError("Evaluation unit tests cannot make network requests")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    clear_caches()
    yield
    clear_caches()
