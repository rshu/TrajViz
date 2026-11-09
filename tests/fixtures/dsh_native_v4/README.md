# Sanitized DSH native v4 excerpts

These excerpts come from an isolated coding benchmark run on 2026-10-26 with
DeepSeek Harness 0.2.1-rc.2, using deepseek-flash. The complete run
finished successfully with a parent and two reviewer sessions. Each excerpt
keeps its first model request, measured tool events, native token counters,
header lineage, and child catalog entries. Content, tool arguments, identifiers,
and workspace paths are replaced. Streaming content deltas are omitted.

Tests encode the readable JSONL into concatenated Zstandard frames, matching
the released native writer.

Official format source:
https://github.com/deepseek-ai/deepseek-harness/blob/dsh-v0.2.1-rc.2/packages/session/session-persistence-jsonl/README.md
