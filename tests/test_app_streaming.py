from __future__ import annotations

import asyncio
import base64
import json
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from api import app as api_app


@dataclass
class FakeResult:
    audio: torch.Tensor
    audios: list[torch.Tensor]
    sample_rate: int
    total_to_decode: int
    used_seed: int | None


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.model_cfg = SimpleNamespace(use_duration_predictor=False)

    def synthesize(self, req, log_fn):
        self.calls.append((req.text, req.seed))
        log_fn("fake synth")
        audio = torch.tensor([0.25, -0.25], dtype=torch.float32)
        return FakeResult(
            audio=audio,
            audios=[audio],
            sample_rate=2,
            total_to_decode=audio.numel(),
            used_seed=req.seed,
        )


class StreamingSpeechTests(unittest.TestCase):
    def test_stream_speech_lines_yields_one_audio_event_per_text_chunk(self) -> None:
        runtime = FakeRuntime()
        request = api_app.SpeechRequest(
            input="one. two.",
            response_format="pcm",
            irodori=api_app.IrodoriOptions(
                chunking_enabled=True,
                chunk_min_chars=1,
                seed=7,
            ),
        )

        events = asyncio.run(
            collect_events(
                api_app._stream_speech_lines(
                    runtime,
                    request,
                    "pcm",
                    1.0,
                )
            )
        )

        self.assertEqual([call[0] for call in runtime.calls], ["one.", "two."])
        self.assertEqual([event["type"] for event in events], ["chunk", "chunk", "done"])
        self.assertEqual(events[0]["text"], "one.")
        self.assertEqual(events[1]["text"], "two.")
        self.assertEqual(events[0]["seed"], 7)
        self.assertEqual(events[1]["seed"], 7)
        self.assertEqual(events[0]["format"], "pcm")
        self.assertEqual(base64.b64decode(events[0]["audio"]), encode_pcm([0.25, -0.25]))
        self.assertEqual(events[2]["chunks"], 2)

    def test_chunked_synthesis_without_seed_reuses_one_seed_for_the_whole_request(self) -> None:
        runtime = FakeRuntime()
        request = api_app.SpeechRequest(
            input="one. two.",
            irodori=api_app.IrodoriOptions(
                chunking_enabled=True,
                chunk_min_chars=1,
            ),
        )

        with patch("api.app.secrets.randbits", return_value=99):
            api_app._synthesize_sync(runtime, request)

        self.assertEqual([call[0] for call in runtime.calls], ["one.", "two."])
        self.assertEqual({call[1] for call in runtime.calls}, {99})

    def test_normalize_stream_format_rejects_unknown_formats(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported stream_format"):
            api_app.normalize_stream_format("websocket")


async def collect_events(lines) -> list[dict]:
    return [json.loads(line) async for line in lines]


def encode_pcm(samples: list[float]) -> bytes:
    tensor = torch.tensor(samples, dtype=torch.float32).clamp(-1.0, 1.0)
    return (tensor * 32767.0).to(torch.int16).numpy().tobytes()
