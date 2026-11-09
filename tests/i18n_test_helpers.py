"""Compare presentation contracts independently of localization wrappers."""

import re

_OWNED_TEXT = re.compile(r'<span\s+data-tv-i18n="[^"]*"[^>]*>(.*?)</span>', re.DOTALL)


def unmark_html(value: str) -> str:
    """Keep English text and original tags; unwrap only generated i18n spans."""
    return _OWNED_TEXT.sub(r"\1", value)
