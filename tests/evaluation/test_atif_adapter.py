from agent_trajviz.awe._engine.adapters import adapt_atif, adapt_pi
from agent_trajviz.awe._engine.trajectory import EDIT, READ, RUN_TEST


def test_atif_adapter_normalizes_workspace_paths_tools_and_results():
    raw = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "harbor-agent", "extra": {"cwds": ["/workspace/repo"]}},
        "steps": [
            {"step_id": 1, "source": "user", "message": "Work in `/workspace/repo`."},
            {
                "step_id": 2,
                "source": "agent",
                "reasoning_content": "Inspect before editing.",
                "tool_calls": [
                    {
                        "tool_call_id": "r1",
                        "function_name": "Read",
                        "arguments": {"file_path": "/workspace/repo/src/a.py", "offset": 3},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "r1",
                            "content": "print('a')",
                            "extra": {"tool_result_is_error": False},
                        }
                    ]
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "e1",
                        "function_name": "apply_patch",
                        "arguments": {"patchText": "*** Update File: src/a.py\n@@\n-old\n+new"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "e1", "content": "Done"}]},
            },
            {
                "step_id": 4,
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "t1",
                        "function_name": "bash",
                        "arguments": {"command": "cd src && python -m pytest test_a.py"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "t1",
                            "content": "1 failed",
                            "extra": {"tool_result_is_error": True},
                        }
                    ]
                },
            },
        ],
    }

    traj = adapt_atif(raw, "harbor", "trial")
    calls = traj.tool_calls()
    assert calls[0].canonical_action == READ
    assert calls[0].files == ["src/a.py"]
    assert calls[0].args["offset"] == 3
    assert calls[1].canonical_action == EDIT
    assert calls[1].files == ["src/a.py"]
    assert calls[2].canonical_action == RUN_TEST
    assert calls[2].test_run is True
    results = [s for s in traj.steps if s.type == "tool_result"]
    assert results[-1].call_id == "t1"
    assert results[-1].tool_error is True


def test_atif_adapter_uses_prompt_workspace_when_agent_cwd_is_missing():
    raw = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "mini-swe-agent"},
        "steps": [
            {"source": "user", "message": "You are working in `/workspace/project`."},
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "r",
                        "function_name": "read",
                        "arguments": {"path": "/workspace/project/pkg/mod.py"},
                    }
                ],
            },
        ],
    }
    traj = adapt_atif(raw, "harbor", "trial")
    assert traj.tool_calls()[0].files == ["pkg/mod.py"]


def test_atif_adapter_detects_multilanguage_test_commands():
    commands = [
        "cd pkg && go test ./...",
        "timeout 1200 mvn -pl core -Dtest=ConfigTest test",
        "./gradlew :module:test",
        "cargo test --workspace",
        "./python -m test test_math",
        "npm run test:unit",
    ]
    raw = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "agent", "extra": {"cwds": ["/workspace/repo"]}},
        "steps": [
            {
                "source": "agent",
                "tool_calls": [{"tool_call_id": str(i), "function_name": "bash", "arguments": {"command": command}}],
            }
            for i, command in enumerate(commands)
        ],
    }
    traj = adapt_atif(raw, "harbor", "trial")
    assert all(call.test_run for call in traj.tool_calls())


def test_pi_v3_jsonl_adapter():
    lines = [
        {"type": "session", "version": 3, "cwd": "/workspace/demo"},
        {"type": "turn_start"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "inspect then test"},
                    {
                        "type": "toolCall",
                        "id": "c1",
                        "name": "bash",
                        "arguments": {"command": "cd /workspace/demo && go test ./pkg/..."},
                    },
                ],
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "c1",
            "toolName": "bash",
            "result": {"content": [{"type": "text", "text": "ok"}]},
            "isError": False,
        },
    ]
    traj = adapt_pi(lines, "pi", "demo")
    assert any(step.type == "reasoning" for step in traj.steps)
    call = next(step for step in traj.steps if step.type == "tool_call")
    assert call.canonical_action == RUN_TEST
    assert call.test_run is True
    assert call.args["command"].startswith("cd /workspace/demo")
    result = next(step for step in traj.steps if step.type == "tool_result")
    assert result.call_id == "c1"
    assert result.tool_error is False
