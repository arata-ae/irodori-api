from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec

import torch


@dataclass(frozen=True)
class RuntimeKey:
    checkpoint: str
    model_device: str
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    model_precision: str = "fp32"
    codec_device: str = "cpu"
    codec_precision: str = "fp32"
    codec_deterministic_encode: bool = True
    codec_deterministic_decode: bool = True
    compile_model: bool = False
    compile_dynamic: bool = False


@dataclass
class SamplingRequest:
    text: str
    ref_wav: str | None = None
    ref_latent: str | None = None
    no_ref: bool = False
    seconds: float | None = None
    duration_scale: float = 1.0
    min_seconds: float = 0.5
    max_seconds: float = 30.0
    max_ref_seconds: float | None = 30.0
    num_steps: int = 40
    num_candidates: int = 1
    decode_mode: str = "sequential"
    seed: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class SamplingResult:
    audio: torch.Tensor
    audios: list[torch.Tensor]
    sample_rate: int
    stage_timings: list[tuple[str, float]]
    total_to_decode: float
    used_seed: int


class InferenceRuntime:
    @classmethod
    def from_key(cls, key: RuntimeKey):
        runtime = cls()
        runtime.key = key
        return runtime

    def synthesize(self, req: SamplingRequest, *, log_fn=None) -> SamplingResult:
        if log_fn is not None:
            log_fn("fake synthesize")
        sample_rate = 24000
        seconds = req.seconds or 0.1
        audio = torch.zeros(int(sample_rate * seconds), dtype=torch.float32)
        return SamplingResult(audio, [audio], sample_rate, [], seconds, req.seed or 1234)


def default_runtime_device() -> str:
    return "mps"


def _install_fake_upstream_modules() -> None:
    root = types.ModuleType("irodori_tts")
    root.__spec__ = ModuleSpec("irodori_tts", loader=None)
    inference_runtime = types.ModuleType("irodori_tts.inference_runtime")
    inference_runtime.RuntimeKey = RuntimeKey
    inference_runtime.SamplingRequest = SamplingRequest
    inference_runtime.SamplingResult = SamplingResult
    inference_runtime.InferenceRuntime = InferenceRuntime
    inference_runtime.default_runtime_device = default_runtime_device
    inference_runtime.__spec__ = ModuleSpec("irodori_tts.inference_runtime", loader=None)
    infer = types.ModuleType("infer")
    infer.__spec__ = ModuleSpec("infer", loader=None)
    sys.modules.setdefault("irodori_tts", root)
    sys.modules.setdefault("irodori_tts.inference_runtime", inference_runtime)
    sys.modules.setdefault("infer", infer)


_install_fake_upstream_modules()
