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
    "input": "こんにちは。",
    "irodori": {
      "num_steps": 24,
      "seed": 1234
    }
  }' \
  --output speech.wav
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
| `IRODORI_CHECKPOINT_FILE`    | `arata-ae/irodori-archive/Irodori-TTS-Lite-int4/dit_int4.safetensors` |
| `IRODORI_MODEL_DEVICE`       | `auto`                                       |
| `IRODORI_CODEC_DEVICE`       | `cpu`                                        |
| `IRODORI_VOICES_DIR`         | `voices`                                     |
| `IRODORI_DEFAULT_VOICE`      | `none`                                       |
| `IRODORI_DEFAULT_NUM_STEPS`  | `40`                                         |
| `IRODORI_CODEC_INT4`         | `false`                                      |
| `IRODORI_PACK_RTN_EXTRAS`    | `true`                                       |
| `IRODORI_HF_DURATION_DONOR`  | `arata-ae/irodori-archive/Irodori-TTS-500M-v3-int4/model.safetensors` |

`IRODORI_CHECKPOINT_FILE` accepts either a Hugging Face checkpoint spec in `<org>/<repo>/<filename>` form or a local file path.

To run v3 directly:

```bash
IRODORI_CHECKPOINT_FILE=arata-ae/irodori-archive/Irodori-TTS-500M-v3-int4/model.safetensors \
uv run irodori-api
```

You can also hot-swap per request with the OpenAI `model` field. Omitting it uses `IRODORI_CHECKPOINT_FILE`; `irodori-tts-v2` and `irodori-tts-v3` lazy-load and cache their checkpoints on first use.

### Duration donor for v2

The default `IRODORI_CHECKPOINT_FILE` is the v2 Lite checkpoint. Because v2 does not include a learned duration predictor, Irodori-API also defaults `IRODORI_HF_DURATION_DONOR` to the v3 int4 checkpoint from `arata-ae/irodori-archive`. On startup, the loader first reads the main checkpoint config. If `use_duration_predictor` is false, it opens the donor checkpoint and copies only its duration config and `duration_predictor.*` tensors into the runtime.

If you set `IRODORI_CHECKPOINT_FILE` to the v3 checkpoint, the main model already has `use_duration_predictor=true`, so the donor is skipped.

When a duration predictor is available, requests without explicit `irodori.seconds` use model-predicted duration. `irodori.duration_scale`, `irodori.min_seconds`, and `irodori.max_seconds` still apply. If you set `irodori.seconds`, that manual duration wins. Set `IRODORI_HF_DURATION_DONOR=` to disable donor grafting for v2.

## Notes

- Python 3.10 is the supported runtime.
- CUDA is the practical target for int4 inference.
- macOS uses eager-dequant fallbacks because Triton is not installed there.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon.
- Streaming synthesis is not implemented.

## Credits

Runtime work is based on [kizuna-intelligence/Irodori-TTS-Lite](https://github.com/kizuna-intelligence/Irodori-TTS-Lite), [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS), and [Aratako/Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server).

Third-party license notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
