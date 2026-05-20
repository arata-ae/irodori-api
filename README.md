# Irodori-API

OpenAI-compatible local TTS API for the quantized Irodori-TTS Lite runtime.

This package contains:

- `api/`: FastAPI server
- `irodori_tts_lite/`: int4 checkpoint loader and runtime patches
- `irodori_tts/`: runtime modules used by inference

## Quick Start

```bash
uv sync
uv run irodori-api
```

Health check:

```bash
curl http://127.0.0.1:8088/health
```

Generate speech:

```bash
curl http://127.0.0.1:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"こんにちは。これはIrodori-APIのテストです。"}' \
  --output speech.wav
```

Generate speech with Irodori options:

```bash
curl http://127.0.0.1:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "irodori-tts-lite",
    "input": "こんにちは。",
    "irodori": {
      "num_steps": 24,
      "seed": 1234
    }
  }' \
  --output speech.wav
```

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",
    api_key="not-used",
)

with client.audio.speech.with_streaming_response.create(
    model="irodori-tts-lite",
    voice="none",
    input="こんにちは。",
    response_format="wav",
) as response:
    response.stream_to_file("speech.wav")
```

## Irodori Options

Supported response formats: `wav`, `mp3`, `flac`, `opus`, `pcm`.

| Option             | Description                                        |
| ------------------ | -------------------------------------------------- |
| `num_steps`        | Sampling steps. Defaults to `40`.                  |
| `seconds`          | Manual target duration before runtime clamping.    |
| `seed`             | Reproducible generation seed.                      |
| `duration_scale`   | Scales automatic duration estimates.               |
| `chunking_enabled` | Splits long text into chunks before synthesis.     |
| `chunk_min_chars`  | Minimum chunk size when chunking is enabled.       |
| `decode_mode`      | `sequential` or `batch`.                           |

## Endpoints

| Endpoint                             | Description             |
| ------------------------------------ | ----------------------- |
| `GET /health`                        | Runtime status          |
| `GET /v1/models`                     | Model list              |
| `POST /v1/audio/speech`              | Speech synthesis        |
| `GET /v1/audio/voices`               | List reference voices   |
| `POST /v1/audio/voices`              | Upload reference voice  |
| `GET /v1/audio/voices/{voice_id}`    | Get voice metadata      |
| `PUT /v1/audio/voices/{voice_id}`    | Replace reference voice |
| `DELETE /v1/audio/voices/{voice_id}` | Delete reference voice  |

## Configuration

All settings use the `IRODORI_` environment prefix.

| Variable                     | Default                                      |
| ---------------------------- | -------------------------------------------- |
| `IRODORI_HOST`               | `127.0.0.1`                                  |
| `IRODORI_PORT`               | `8088`                                       |
| `IRODORI_API_KEY`            | unset                                        |
| `IRODORI_CHECKPOINT`         | unset                                        |
| `IRODORI_HF_CHECKPOINT`      | `arata-ae/irodori-archive`                   |
| `IRODORI_HF_CHECKPOINT_FILE` | `Irodori-TTS-Lite-int4/dit_int4.safetensors` |
| `IRODORI_MODEL_NAME`         | `irodori-tts-lite`                           |
| `IRODORI_MODEL_DEVICE`       | `auto`                                       |
| `IRODORI_CODEC_DEVICE`       | `cpu`                                        |
| `IRODORI_VOICES_DIR`         | `voices`                                     |
| `IRODORI_DEFAULT_VOICE`      | `none`                                       |
| `IRODORI_DEFAULT_NUM_STEPS`  | `40`                                         |
| `IRODORI_CODEC_INT4`         | `false`                                      |

The default checkpoint is downloaded from Hugging Face on first startup unless `IRODORI_CHECKPOINT` points to a local file.

## Notes

- Python 3.10 is the supported runtime.
- CUDA is the practical target for int4 inference.
- macOS uses eager-dequant fallbacks because Triton is not installed there.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon.
- Streaming synthesis is not implemented.

## Credits

Runtime work is based on [kizuna-intelligence/Irodori-TTS-Lite](https://github.com/kizuna-intelligence/Irodori-TTS-Lite), [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS), and [Aratako/Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server).

Third-party license notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
