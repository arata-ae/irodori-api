from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DurationEstimate:
    seconds: float
    method: str
    units: int


def estimate_seconds(
    text: str,
    *,
    min_seconds: float = 2.0,
    scale: float = 1.25,
    phonemes_per_second: float = 11.0,
    chars_per_second: float = 7.0,
    padding_seconds: float = 0.6,
    max_seconds: float = 30.0,
) -> DurationEstimate:
    normalized = "".join(ch for ch in str(text) if not ch.isspace())
    try:
        import pyopenjtalk

        phonemes = pyopenjtalk.g2p(text, kana=False).split()
        units = len(phonemes)
        seconds = units / phonemes_per_second + padding_seconds
        method = "pyopenjtalk"
    except Exception:
        units = len(normalized)
        seconds = units / chars_per_second + padding_seconds
        method = "chars"

    seconds = max(float(min_seconds), float(seconds) * float(scale))
    seconds = min(float(max_seconds), seconds)
    return DurationEstimate(seconds=seconds, method=method, units=units)


def split_text(text: str, *, min_chars: int = 80) -> list[str]:
    raw = str(text).strip()
    if len(raw) <= min_chars:
        return [raw] if raw else []

    chunks: list[str] = []
    start = 0
    boundaries = set("。、，,．.!！?？\n\r")
    for index, char in enumerate(raw):
        if char in boundaries and index - start + 1 >= min_chars:
            chunk = raw[start : index + 1].strip()
            if chunk:
                chunks.append(chunk)
            start = index + 1
    tail = raw[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks or [raw]
