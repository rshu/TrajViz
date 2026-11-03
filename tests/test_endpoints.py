"""Anthropic-flavoured base URLs must reach the Messages endpoint in either documented form."""

from __future__ import annotations

import pytest

from agent_trajviz._endpoints import anthropic_messages_url


@pytest.mark.parametrize("base_url,expected", [
    # The official SDKs' base, which leaves /v1/messages to the client.
    ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
    # An explicit /v1 base is not doubled, with or without a trailing slash.
    ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"),
    ("https://api.deepseek.com/anthropic/v1/", "https://api.deepseek.com/anthropic/v1/messages"),
])
def test_messages_url_accepts_the_base_with_or_without_v1(base_url, expected):
    assert anthropic_messages_url(base_url) == expected
