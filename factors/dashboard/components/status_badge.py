"""Status style metadata independent of Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

from .formatters import status_text


@dataclass(frozen=True)
class StatusStyle:
    text: str
    tone: str


TONES = {
    "fresh": "positive",
    "favorable": "positive",
    "limited_support": "positive",
    "neutral": "neutral",
    "strong": "neutral",
    "weak": "neutral",
    "mixed": "warning",
    "partial": "warning",
    "rich": "warning",
    "cheap": "neutral",
    "unfavorable": "negative",
    "insufficient": "negative",
    "unavailable": "negative",
    "stale": "negative",
}


def status_style(value: object) -> StatusStyle:
    code = str(value)
    return StatusStyle(status_text(value), TONES.get(code, "neutral"))

