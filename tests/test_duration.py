from __future__ import annotations

from api.duration import estimate_seconds, split_text


def test_estimate_seconds_has_minimum():
    estimate = estimate_seconds("短い", min_seconds=2.0)

    assert estimate.seconds >= 2.0
    assert estimate.units > 0


def test_split_text_keeps_short_text_as_one_chunk():
    assert split_text("こんにちは。", min_chars=80) == ["こんにちは。"]


def test_split_text_uses_punctuation_boundaries():
    chunks = split_text("あ" * 80 + "。" + "い" * 80 + "。", min_chars=40)

    assert len(chunks) == 2
