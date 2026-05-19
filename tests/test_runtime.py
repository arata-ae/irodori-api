from __future__ import annotations

import threading

import pytest

from irodori_tts_lite import checkpoint_loader, weights
from api import runtime as runtime_module
from api.config import Settings
from api.runtime import RuntimeLoadTimeoutError, RuntimeManager


def test_default_checkpoint_source_uses_archive():
    assert weights.DEFAULT_REPO == "arata-ae/irodori-archive"
    assert weights.DEFAULT_DIT_FILE == "Irodori-TTS-Lite-int4/dit_int4.safetensors"
    assert weights.DEFAULT_DACVAE_FILE == "Irodori-TTS-Lite-int4/dacvae_int4.safetensors"


def test_default_codec_device_is_cpu():
    assert Settings(_env_file=None).codec_device == "cpu"


def test_runtime_configures_lite_patch_before_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "dit_int4.safetensors"
    checkpoint.write_bytes(b"test")
    calls = []
    loaded_runtime = object()

    monkeypatch.setattr(runtime_module.irodori_tts_lite, "configure", lambda **kwargs: calls.append(("configure", kwargs)))
    monkeypatch.setattr(runtime_module.irodori_tts_lite, "patch", lambda: calls.append(("patch", {})))
    monkeypatch.setattr(runtime_module.InferenceRuntime, "from_key", staticmethod(lambda key: calls.append(("from_key", key)) or loaded_runtime))

    manager = RuntimeManager(
        Settings(checkpoint=str(checkpoint), model_device="cpu", codec_device="cpu", _env_file=None)
    )

    assert manager.get() is loaded_runtime
    assert calls[0][0] == "configure"
    assert calls[0][1]["force_fp16"] is True
    assert calls[1][0] == "patch"
    assert calls[2][1].checkpoint == str(checkpoint)


def test_runtime_resolves_hf_lite_checkpoint_when_local_checkpoint_is_unset(monkeypatch):
    manager = RuntimeManager(
        Settings(hf_checkpoint="owner/repo", hf_checkpoint_file="dit.safetensors", _env_file=None)
    )

    def fake_resolve(arg, *, default_repo=None, default_filename=None):
        assert arg is None
        assert default_repo == "owner/repo"
        assert default_filename == "dit.safetensors"
        return "/cache/dit.safetensors"

    monkeypatch.setattr(runtime_module.irodori_tts_lite, "resolve_checkpoint", fake_resolve)

    assert manager._resolve_checkpoint_path() == "/cache/dit.safetensors"


def test_runtime_resolves_auto_devices_before_building_key(tmp_path, monkeypatch):
    checkpoint = tmp_path / "dit_int4.safetensors"
    checkpoint.write_bytes(b"test")
    calls = []
    monkeypatch.setattr(runtime_module.irodori_tts_lite, "configure", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_module.irodori_tts_lite, "patch", lambda: None)
    monkeypatch.setattr(
        runtime_module.InferenceRuntime,
        "from_key",
        staticmethod(lambda key: calls.append(key) or object()),
    )
    manager = RuntimeManager(
        Settings(checkpoint=str(checkpoint), model_device="auto", codec_device="auto", _env_file=None)
    )

    manager.get()

    assert calls[0].model_device == "mps"
    assert calls[0].codec_device == "mps"


def test_runtime_load_timeout_while_another_thread_is_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "dit_int4.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        model_load_timeout=0.05,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    started = threading.Event()
    release = threading.Event()
    loaded_runtime = object()
    errors: list[BaseException] = []

    monkeypatch.setattr(runtime_module.irodori_tts_lite, "configure", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_module.irodori_tts_lite, "patch", lambda: None)

    def fake_from_key(_key):
        started.set()
        release.wait(timeout=2)
        return loaded_runtime

    monkeypatch.setattr(runtime_module.InferenceRuntime, "from_key", staticmethod(fake_from_key))

    def load_runtime():
        try:
            assert manager.get() is loaded_runtime
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=load_runtime)
    thread.start()
    assert started.wait(timeout=1)

    with pytest.raises(RuntimeLoadTimeoutError):
        manager.get()

    release.set()
    thread.join(timeout=2)
    assert errors == []
    assert manager.is_loaded
    assert not manager.is_loading


def test_patched_load_clears_stale_quantized_state(tmp_path, monkeypatch):
    checkpoint_loader._PENDING_SWAPS["stale"] = {"entry": {}, "state": {}}
    checkpoint_loader._PENDING_EXTRA["stale"] = {"entry": {}, "tensors": {}}
    monkeypatch.setattr(checkpoint_loader, "_orig_load", lambda _path: ({"plain.weight": object()}, {}, {}))

    checkpoint_loader._patched_load(tmp_path / "plain.safetensors")

    assert checkpoint_loader._PENDING_SWAPS == {}
    assert checkpoint_loader._PENDING_EXTRA == {}
