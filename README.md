# Irodori-API

Credit: Irodori-API is a hard fork of [kizuna-intelligence/Irodori-TTS-Lite](https://github.com/kizuna-intelligence/Irodori-TTS-Lite), with API/server implementation adapted from [Aratako/Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server).

Irodori-API is a lightweight local API server for running a 4-bit quantized Irodori-TTS DiT checkpoint.

It loads the 4-bit checkpoint directly, patches the upstream `irodori_tts.inference_runtime`, and exposes an OpenAI-compatible HTTP API for apps such as Electron.

## Highlights

- Runs the DiT checkpoint from a **279 MB** int4 safetensors file instead of the original **1.88 GB** fp32 checkpoint.
- Uses about **552 MB peak GPU memory** for the DiT model alone in the measured CUDA setup.
- Can optionally quantize the DACVAE codec conv layers too, bringing the measured end-to-end peak to about **1 GB**.
- Keeps OneCompression out of runtime dependencies; quantized weights plus this runtime are enough.
- Uses Triton fused int4 linear kernels on Linux/CUDA. macOS can sync/import the package, but it falls back to eager dequantization and is much slower.
- Provides an OpenAI-compatible API server: `POST /v1/audio/speech`.

## Requirements

- Python 3.10
- `uv`
- A sibling checkout of upstream [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS), or an explicit upstream path
- CUDA GPU for practical int4 inference

Expected local layout:

```text
parent/
├── Irodori-TTS/
└── Irodori-API/
```

The int4 weights are downloaded from [arata-ae/irodori-archive](https://huggingface.co/arata-ae/irodori-archive) during startup unless you pass a local checkpoint.

## Install

```bash
uv sync --python 3.10
```

## API Server

Start the OpenAI-compatible API:

```bash
uv run --python 3.10 --with-editable ../Irodori-TTS \
  irodori-api --host 127.0.0.1 --port 8088
```

The server always loads the runtime during startup so the first speech request does not pay the model load cost.

If upstream Irodori-TTS is elsewhere:

```bash
IRODORI_UPSTREAM_PATH=/path/to/Irodori-TTS \
uv run --python 3.10 --with-editable /path/to/Irodori-TTS \
  irodori-api --host 127.0.0.1 --port 8088
```

Health check:

```bash
curl http://127.0.0.1:8088/health
```

Generate speech with the defaults:

```bash
curl http://127.0.0.1:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "こんにちは。これはIrodori-APIのテストです。"
  }' \
  --output speech.wav
```

The default API behavior is tuned for the common Lite setup:

- model id: `irodori-tts-lite` (you may omit `model` in direct HTTP calls)
- voice: `none`, for no-reference / baked-speaker checkpoints
- audio format: `wav`
- RF sampling steps: `40`, the quality-oriented runtime default
- duration: estimated from `pyopenjtalk` phoneme count, with a character-count fallback and safety multiplier
- runtime loading: preloaded at server startup

These defaults make the smallest useful request just `{"input": "..."}` while keeping the runtime's quality-oriented sampling default. The server starts warm, uses the baked/no-reference voice path, returns a simple WAV file, and picks a conservative duration estimate because the Lite checkpoint has no learned duration predictor. `pyopenjtalk` is installed by default, so the API uses the phoneme estimator out of the box.

For faster local app responses, reduce the RF sampling steps per request:

```json
{
  "input": "こんにちは。",
  "irodori": {
    "num_steps": 16
  }
}
```

Or set it once for the server:

```bash
IRODORI_DEFAULT_NUM_STEPS=16
```

Generate and auto-play on macOS:

```bash
curl -sS http://127.0.0.1:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "こんにちは。短いテストです。",
    "irodori": {
      "seconds": 3.0
    }
  }' \
  --output /tmp/irodori_tts.wav \
&& afplay /tmp/irodori_tts.wav
```

## API Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /health` | Server status and runtime configuration. |
| `GET /v1/models` | Lists the configured model id. |
| `POST /v1/audio/speech` | Synthesizes speech and returns audio bytes. |
| `GET /v1/audio/voices` | Lists reference voices. |
| `POST /v1/audio/voices` | Uploads a reference voice file. |
| `GET /v1/audio/voices/{voice_id}` | Returns reference voice metadata. |
| `PUT /v1/audio/voices/{voice_id}` | Replaces a reference voice file. |
| `DELETE /v1/audio/voices/{voice_id}` | Deletes a reference voice file. |

Streaming synthesis is not implemented. Requests return one complete audio response.

## Speech Request

```json
{
  "input": "こんにちは。"
}
```

Full request with optional overrides:

```json
{
  "model": "irodori-tts-lite",
  "input": "こんにちは。",
  "voice": "speaker",
  "response_format": "wav",
  "irodori": {
    "seconds": 3.0,
    "num_steps": 24,
    "seed": 1234
  }
}
```

Use a fixed seed when you want repeatable character output for the same text, voice, and sampling settings:

```json
{
  "input": "こんにちは。今日もよろしくね。",
  "voice": "none",
  "irodori": {
    "seed": 424242,
    "num_steps": 40
  }
}
```

For strongest consistency, keep the same `voice`, `seed`, `num_steps`, duration settings, and text normalization. Exact output can still vary across hardware/backend changes.

Supported `response_format` values:

- `wav`
- `mp3`
- `flac`
- `opus`
- `aac`
- `pcm`

The Lite checkpoint does not include a duration predictor. The API therefore estimates `seconds` when `seconds` is omitted. It uses `pyopenjtalk` phoneme count when available and falls back to character count only when `pyopenjtalk` cannot be imported or fails. Long inputs are split at punctuation by default, and each chunk gets its own automatic duration estimate. A request-level `irodori.seconds` applies to the whole request and disables chunking. If output cuts off early, set `irodori.seconds` explicitly or increase `IRODORI_DEFAULT_AUTO_SECONDS_SCALE`.

## Voice Files

Put reference audio files in `voices/`. The file stem becomes the voice id:

```text
voices/
└── speaker.wav
```

Then use:

```json
"voice": "speaker"
```

Use `"voice": "none"` for no-reference / baked-speaker checkpoints.

## OpenAI SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",
    api_key="not-used",
)

with client.audio.speech.with_streaming_response.create(
    model="irodori-tts-lite",
    voice="none",
    input="こんにちは。Electronからもこの形で呼べます。",
    response_format="wav",
) as response:
    response.stream_to_file("speech.wav")
```

## API Configuration

All settings use the `IRODORI_` environment prefix.

| Variable | Default | Description |
| --- | --- | --- |
| `IRODORI_HOST` | `0.0.0.0` | Default API server host. |
| `IRODORI_PORT` | `8088` | Default API server port. |
| `IRODORI_API_KEY` | unset | If set, requests must use `Authorization: Bearer ...`. |
| `IRODORI_UPSTREAM_PATH` | sibling `../Irodori-TTS` | Upstream checkout path. |
| `IRODORI_CHECKPOINT` | unset | Local checkpoint path or `hf://org/repo/file`. |
| `IRODORI_HF_CHECKPOINT` | `arata-ae/irodori-archive` | Default checkpoint repo. |
| `IRODORI_HF_CHECKPOINT_FILE` | `Irodori-TTS-Lite-int4/dit_int4.safetensors` | Default checkpoint file. |
| `IRODORI_MODEL_NAME` | `irodori-tts-lite` | Accepted API model id. |
| `IRODORI_MODEL_DEVICE` | `auto` | Let upstream choose the model device. |
| `IRODORI_CODEC_DEVICE` | `cpu` | Codec device. CPU is the conservative default for macOS/MPS. |
| `IRODORI_MODEL_PRECISION` | `fp32` | Model precision passed to upstream. |
| `IRODORI_CODEC_PRECISION` | `fp32` | Codec precision passed to upstream. |
| `IRODORI_VOICES_DIR` | `voices` | Reference voice directory. |
| `IRODORI_DEFAULT_VOICE` | `none` | Default voice id when requests omit `voice`. |
| `IRODORI_DEFAULT_RESPONSE_FORMAT` | `wav` | Default audio format. |
| `IRODORI_DEFAULT_NUM_STEPS` | `40` | Default RF sampling steps. Lower values such as `16` are faster but may reduce stability or quality. |
| `IRODORI_DEFAULT_AUTO_SECONDS` | `true` | Estimate duration when request omits `seconds`. |
| `IRODORI_DEFAULT_AUTO_MIN_SECONDS` | `2.0` | Minimum auto-estimated duration before scaling. |
| `IRODORI_DEFAULT_AUTO_SECONDS_SCALE` | `1.25` | Safety multiplier for auto-estimated duration. |
| `IRODORI_DEFAULT_PHONEMES_PER_SECOND` | `11.0` | `pyopenjtalk` phoneme-count duration heuristic. Lower means longer. |
| `IRODORI_DEFAULT_CHARS_PER_SECOND` | `7.0` | Character-count fallback duration heuristic. Lower means longer. |
| `IRODORI_DEFAULT_DURATION_PADDING_SECONDS` | `0.6` | Extra seconds added to the automatic duration estimate. |
| `IRODORI_DEFAULT_MAX_SECONDS` | `30.0` | Maximum accepted/generated duration unless overridden in code. |
| `IRODORI_FORCE_FP16` | `true` | Cast non-int4 model parts to fp16. |
| `IRODORI_CODEC_INT4` | `false` | Quantize DACVAE codec conv layers at load time. |
| `IRODORI_MAX_CONCURRENT_SYNTHESIS` | `1` | Synthesis concurrency limit. |

## Notes and Limitations

- Linux/CUDA is the intended inference target. macOS does not install Triton and uses eager-dequant fallbacks.
- On Apple Silicon, use `PYTORCH_ENABLE_MPS_FALLBACK=1`. The default model device `auto` resolves to `mps` when available, while the codec stays on CPU by default.
- CPU inference is not supported for the DiT model. The codec can be offloaded to CPU.
- The Lite checkpoint has no learned duration predictor, so either set `seconds` explicitly or use the API phoneme-count duration heuristic.
- `stream_format: "sse"` is rejected because streaming synthesis is not implemented.
- `FusedInt4Linear` has a known fp32-input fallback shape-check limitation; use `force_fp16=True` for inference.

## Project Layout

```text
.
├── api/                      # OpenAI-compatible API server
├── irodori_tts_lite/         # Int4 checkpoint runtime patch
├── tools/
├── docs/
├── pyproject.toml
└── uv.lock
```

## Development

```bash
uv sync --python 3.10 --group dev
uv run --group dev ruff check
uv run --group dev --with-editable ../Irodori-TTS pytest
```

Smoke-test the API command without loading the model:

```bash
uv run --with-editable ../Irodori-TTS irodori-api --help
```

## Upstream Projects

- TTS pipeline: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- DACVAE codec: [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)

This project quantizes and runs those upstream artifacts.

## License

[MIT License](LICENSE)

`irodori_tts_lite/fused_int4_linear.py` is vendored from OneCompression and keeps its original license notice.

Copyright (c) 2025-2026 Kizuna Intelligence contributors.
