from __future__ import annotations

import pytest

from api.config import Settings
from api.voices import VoiceRegistry


def test_resolve_none_voice_uses_no_ref(tmp_path):
    registry = VoiceRegistry(Settings(voices_dir=tmp_path, _env_file=None))

    voice = registry.resolve("none")

    assert voice.no_ref is True
    assert voice.ref_wav is None


def test_write_and_list_voice(tmp_path):
    registry = VoiceRegistry(Settings(voices_dir=tmp_path, _env_file=None))

    written = registry.write_file(filename="speaker.wav", data=b"riff", voice_id=None)

    assert written.voice_id == "speaker"
    assert registry.list()[0].voice_id == "speaker"


def test_write_voice_rejects_bad_id(tmp_path):
    registry = VoiceRegistry(Settings(voices_dir=tmp_path, _env_file=None))

    with pytest.raises(ValueError):
        registry.write_file(filename="speaker.wav", data=b"riff", voice_id="../bad")


def test_delete_voice(tmp_path):
    registry = VoiceRegistry(Settings(voices_dir=tmp_path, _env_file=None))
    registry.write_file(filename="speaker.wav", data=b"riff")

    assert registry.delete("speaker") is True
    assert registry.delete("speaker") is False
