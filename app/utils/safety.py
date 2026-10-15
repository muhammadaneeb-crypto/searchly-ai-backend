"""Helpers for treating web pages and uploaded files as UNTRUSTED data."""
import re

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)",
    r"forget (all |everything |your )?(previous |prior )?(instructions|rules)",
    r"you are now\b",
    r"new instructions?\s*:",
    r"(reveal|show|print|repeat|leak) (me )?(your |the )?(system )?(prompt|instructions|api key|secret)",
    r"\bsystem prompt\b",
    r"^\s*(system|assistant)\s*:",
    r"do not (cite|mention) (any )?sources",
    r"override (your|the) (rules|instructions|safety)",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)
_TAG_RE = re.compile(r"</?\s*(source|evidence|document|system|assistant|user)\b[^>]*>?", re.IGNORECASE)
REMOVED = "[removed: text that looked like an instruction aimed at the AI]"


def neutralize(text: str) -> tuple[str, bool]:
    """Return (cleaned_text, was_suspicious).

    Lines that look like instructions to an AI are replaced with a marker, and
    fake delimiter tags (like </source>) are defanged so the page cannot escape
    its evidence block. The system prompt ALSO tells the model to ignore
    instructions inside evidence; this is a second layer, not the only one.
    """
    flagged = False
    lines = []
    for line in text.splitlines():
        if _TAG_RE.search(line):
            flagged = True
            line = _TAG_RE.sub("[tag removed]", line)
        if _INJECTION_RE.search(line):
            flagged = True
            lines.append(REMOVED)
        else:
            lines.append(line)
    return "\n".join(lines), flagged


def safe_attr(value: str, limit: int = 200) -> str:
    """Make a string safe to place inside an XML-style attribute."""
    return re.sub(r'["<>\r\n]+', " ", value or "")[:limit].strip()
