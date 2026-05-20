from __future__ import annotations

from io import BytesIO

import soundfile as sf
import torch

CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg; codecs=opus",
    "pcm": "audio/L16",
}

_SOUNDFILE_FORMATS = {
    "wav": ("WAV", "PCM_16"),
    "mp3": ("MP3", None),
    "flac": ("FLAC", None),
    "opus": ("OGG", "OPUS"),
}


def normalize_response_format(value: str | None, default: str = "wav") -> str:
    fmt = (value or default).strip().lower()
    if fmt == "wave":
        fmt = "wav"
    if fmt not in CONTENT_TYPES:
        supported = ", ".join(sorted(CONTENT_TYPES))
        raise ValueError(
            f"Unsupported response_format={value!r}. Supported formats: {supported}.",
        )
    return fmt


def _audio_to_numpy(audio: torch.Tensor):
    data = audio.detach().cpu()
    if data.ndim == 2 and data.shape[0] <= 8:
        data = data.transpose(0, 1)
    elif data.ndim == 1:
        pass
    elif data.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {tuple(data.shape)}")
    return data.numpy()


def encode_audio(audio: torch.Tensor, sample_rate: int, response_format: str) -> bytes:
    fmt = normalize_response_format(response_format)
    data = _audio_to_numpy(audio)
    if fmt == "pcm":
        clipped = torch.as_tensor(data).clamp(-1.0, 1.0)
        return (clipped * 32767.0).to(torch.int16).numpy().tobytes()

    buffer = BytesIO()
    sf_format, subtype = _SOUNDFILE_FORMATS[fmt]
    sf.write(buffer, data, int(sample_rate), format=sf_format, subtype=subtype)
    return buffer.getvalue()
