from __future__ import annotations

import torch
import pytest

from api.audio import CONTENT_TYPES, encode_audio, normalize_response_format


def test_normalize_response_format_defaults_to_wav():
    assert normalize_response_format(None) == "wav"
    assert normalize_response_format("wave") == "wav"


def test_normalize_response_format_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_response_format("zip")


def test_encode_wav_returns_header():
    audio = torch.zeros(2400)

    data = encode_audio(audio, 24000, "wav")

    assert data.startswith(b"RIFF")


def test_content_types_include_openai_formats():
    for fmt in ["wav", "mp3", "flac", "opus", "aac", "pcm"]:
        assert fmt in CONTENT_TYPES
