"""Turn an agent's local session store into a file TrajViz reads.

Run ``agent-trajviz-collect <tool> --help`` (or ``python -m
agent_trajviz.collect <tool> --help``) for each tool:

* ``codex``    — a native Codex rollout and its verified child sessions;
* ``cursor``   — a Cursor agent chat: transcript JSONL plus the Composer
  ``state.vscdb`` row, with reachable sub-agent transcripts;
* ``opencode`` — an OpenCode session and its child sessions from ``opencode.db``;
* ``codearts`` — a CodeArts session from ``opencode.db`` or a session folder.

The tools only read the source stores and never overwrite their inputs.
"""
