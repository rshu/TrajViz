"""Endpoint URL construction shared by the dashboard analysis client and the labeler."""

from __future__ import annotations


def anthropic_messages_url(base_url: str) -> str:
    """The Messages endpoint for an Anthropic-flavoured *base_url*.

    The official SDKs take ``https://api.anthropic.com`` and add ``/v1/messages``
    themselves. Accept that form and the explicit ``.../v1`` form (DeepSeek and
    other compatible gateways serve both), so neither 404s::

        >>> anthropic_messages_url("https://api.anthropic.com")
        'https://api.anthropic.com/v1/messages'
        >>> anthropic_messages_url("https://api.deepseek.com/anthropic/v1/")
        'https://api.deepseek.com/anthropic/v1/messages'
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"
