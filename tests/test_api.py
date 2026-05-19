from __future__ import annotations

from io import BytesIO

import torch
from fastapi.testclient import TestClient
from irodori_tts.inference_runtime import SamplingResult

from api import app as main


class FakeRuntime:
    def __init__(self) -> None:
        self.requests = []

    def synthesize(self, req, *, log_fn=None):
        self.requests.append(req)
        if log_fn is not None:
            log_fn("fake")
        audio = torch.zeros(2400)
        return SamplingResult(audio, [audio], 24000, [], 0.1, req.seed or 1234)


def test_health():
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]["id"] == main.settings.model_name
    assert body["runtime"]["loaded"] is False
    assert body["defaults"]["voice"] == "none"


def test_models_lists_lite_model():
    response = TestClient(main.app).get("/v1/models")

    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == main.settings.model_name
    assert data[0]["owned_by"] == "irodori-api"


def test_auth_required_when_api_key_is_configured(monkeypatch):
    monkeypatch.setattr(main.settings, "api_key", "secret")
    client = TestClient(main.app)

    missing = client.get("/v1/models")
    wrong = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    ok = client.get("/v1/models", headers={"Authorization": "Bearer secret"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 200


def test_voice_upload_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "voices_dir", tmp_path)
    monkeypatch.setattr(main.voice_registry, "root", tmp_path)
    client = TestClient(main.app)

    created = client.post(
        "/v1/audio/voices",
        files={"file": ("speaker.wav", BytesIO(b"riff"), "audio/wav")},
    )
    listed = client.get("/v1/audio/voices")
    deleted = client.delete("/v1/audio/voices/speaker")

    assert created.status_code == 201
    assert listed.json()["data"][0]["id"] == "speaker"
    assert deleted.json()["deleted"] is True


def test_speech_uses_defaults(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setattr(main.runtime_manager, "get", lambda: fake)
    monkeypatch.setattr(main.settings, "default_auto_seconds", False)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={"input": "こんにちは。", "irodori": {"seed": 42}},
    )

    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert response.headers["x-irodori-seed"] == "42"
    assert fake.requests[0].num_steps == main.settings.default_num_steps
    assert fake.requests[0].seed == 42


def test_speech_rejects_unknown_model(monkeypatch):
    monkeypatch.setattr(main.runtime_manager, "get", lambda: FakeRuntime())

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={"model": "other", "input": "こんにちは。"},
    )

    assert response.status_code == 400


def test_speech_rejects_streaming(monkeypatch):
    monkeypatch.setattr(main.runtime_manager, "get", lambda: FakeRuntime())

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={"input": "こんにちは。", "stream_format": "sse"},
    )

    assert response.status_code == 400
