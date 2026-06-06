from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from typing import Any, AsyncIterator, Literal

import irodori_tts_lite
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .audio import CONTENT_TYPES, encode_audio, normalize_response_format
from .config import get_settings
from .duration import estimate_seconds, split_text
from .runtime_imports import require_runtime_package
from .voices import VoiceRegistry

_settings = get_settings()
require_runtime_package()

from irodori_tts.inference_runtime import SamplingRequest  # noqa: E402

from .runtime import RuntimeLoadTimeoutError, RuntimeManager  # noqa: E402

logger = logging.getLogger(__name__)

settings = _settings
runtime_manager = RuntimeManager(settings)
voice_registry = VoiceRegistry(settings)
_synthesis_semaphore: asyncio.Semaphore | None = None
_synthesis_semaphore_limit: int | None = None
STREAM_CONTENT_TYPES = {"ndjson": "application/x-ndjson"}
STREAM_FORMAT_ALIASES = {"jsonl": "ndjson", "ndjson": "ndjson"}
_runtime_warmup_task: asyncio.Task | None = None
_runtime_warmup_error: str | None = None


class IrodoriOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    ref_wav: str | None = None
    ref_latent: str | None = None
    no_ref: bool | None = None
    seconds: float | None = None
    duration_scale: float | None = None
    min_seconds: float | None = None
    max_seconds: float | None = None
    max_ref_seconds: float | None = None
    ref_normalize_db: float | None = None
    ref_ensure_max: bool | None = None
    num_steps: int | None = None
    t_schedule_mode: Literal["linear", "sway"] | None = None
    sway_coeff: float | None = None
    num_candidates: int | None = None
    decode_mode: Literal["sequential", "batch"] | None = None
    cfg_scale_text: float | None = None
    cfg_scale_caption: float | None = None
    cfg_scale_speaker: float | None = None
    cfg_guidance_mode: Literal["independent", "joint", "alternating"] | None = None
    cfg_scale: float | None = None
    cfg_min_t: float | None = None
    cfg_max_t: float | None = None
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    context_kv_cache: bool | None = None
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    seed: int | None = None
    trim_tail: bool | None = None
    tail_window_size: int | None = None
    tail_std_threshold: float | None = None
    tail_mean_threshold: float | None = None
    max_text_len: int | None = None
    lora_adapter: str | None = None
    chunking_enabled: bool | None = None
    chunk_min_chars: int | None = None


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str = Field(min_length=1, max_length=4096)
    voice: str | dict[str, Any] | None = None
    response_format: str | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: str | None = None
    irodori: IrodoriOptions = Field(default_factory=IrodoriOptions)


def startup() -> None:
    voices_dir = voice_registry.ensure_dir()
    logger.info("voices directory: %s", voices_dir)


async def warm_default_runtime() -> None:
    global _runtime_warmup_error
    _runtime_warmup_error = None
    logger.info("warming default runtime")
    try:
        await asyncio.to_thread(runtime_manager.get)
    except Exception as exc:
        _runtime_warmup_error = str(exc)
        logger.exception("default runtime warmup failed")
    else:
        logger.info("default runtime warmup completed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _runtime_warmup_task
    startup()
    _runtime_warmup_task = asyncio.create_task(warm_default_runtime())
    yield
    if _runtime_warmup_task is not None and not _runtime_warmup_task.done():
        _runtime_warmup_task.cancel()


app = FastAPI(title="Irodori-API", version="0.1.0", lifespan=lifespan)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def openai_error_response(message: str, *, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


def cuda_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_built": bool(torch.backends.cuda.is_built()),
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    try:
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["device_count"] = int(torch.cuda.device_count())
        if status["cuda_available"] and status["device_count"] > 0:
            status["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        status["error"] = str(exc)
    return status


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return openai_error_response(
        str(exc.detail),
        status_code=int(exc.status_code),
        error_type="invalid_request_error" if exc.status_code < 500 else "server_error",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return openai_error_response(str(exc), status_code=422, error_type="invalid_request_error")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled api error", exc_info=(type(exc), exc, exc.__traceback__))
    return openai_error_response(
        "Internal server error.",
        status_code=500,
        error_type="server_error",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    voices_dir = settings.voices_dir.expanduser()
    status = "ok"
    if _runtime_warmup_error is not None:
        status = "error"
    elif not runtime_manager.is_loaded:
        status = "loading"
    return {
        "status": status,
        "model": {
            "id": settings.effective_model_name,
            "checkpoint_file": settings.checkpoint_file,
            "model_device": settings.model_device,
            "codec_device": settings.codec_device,
            "model_precision": settings.model_precision,
            "codec_precision": settings.codec_precision,
            "use_fused": settings.use_fused,
            "force_fp16": settings.force_fp16,
            "disable_eager_dequant": settings.disable_eager_dequant,
            "codec_int4": settings.codec_int4,
            "codec_int4_groupsize": settings.codec_int4_groupsize,
            "pack_rtn_extras": settings.pack_rtn_extras,
            "hf_duration_donor": settings.hf_duration_donor,
            "compile_model": settings.compile_model,
            "compile_dynamic": settings.compile_dynamic,
        },
        "runtime": {
            "loaded": runtime_manager.is_loaded,
            "loading": runtime_manager.is_loading,
            "warmup_error": _runtime_warmup_error,
            "cuda": cuda_status(),
            "checkpoint": runtime_manager.checkpoint_path,
            "checkpoints": runtime_manager.checkpoint_paths,
            "load_timeout": settings.model_load_timeout,
            "max_concurrent_synthesis": settings.max_concurrent_synthesis,
            "synthesis_wait_timeout": settings.synthesis_wait_timeout,
        },
        "voices": {
            "dir": str(voices_dir),
            "dir_exists": voices_dir.is_dir(),
            "files": len(voice_registry.list_files()) if voices_dir.is_dir() else 0,
        },
        "defaults": {
            "voice": settings.default_voice,
            "response_format": settings.default_response_format,
            "num_steps": settings.default_num_steps,
            "auto_seconds": settings.default_auto_seconds,
            "auto_min_seconds": settings.default_auto_min_seconds,
            "auto_seconds_scale": settings.default_auto_seconds_scale,
            "phonemes_per_second": settings.default_phonemes_per_second,
            "chars_per_second": settings.default_chars_per_second,
            "duration_padding_seconds": settings.default_duration_padding_seconds,
            "chunking_enabled": settings.default_chunking_enabled,
            "chunk_min_chars": settings.default_chunk_min_chars,
        },
    }


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "irodori-tts-v2",
                "object": "model",
                "created": 0,
                "owned_by": "irodori-api",
            },
            {
                "id": "irodori-tts-v3",
                "object": "model",
                "created": 0,
                "owned_by": "irodori-api",
            }
        ],
    }


@app.get("/v1/audio/voices")
def list_voices() -> dict[str, Any]:
    return {"object": "list", "data": [voice.metadata() for voice in voice_registry.list()]}


@app.post("/v1/audio/voices", status_code=201)
async def upload_voice(file: UploadFile = File(...), voice_id: str | None = Form(default=None)):
    try:
        voice_file = voice_registry.write_file(
            filename=file.filename or "",
            data=await file.read(),
            voice_id=voice_id,
            replace=False,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(status_code=201, content=voice_file.metadata())


@app.get("/v1/audio/voices/{voice_id}")
def get_voice_file(voice_id: str) -> dict[str, Any]:
    voice = voice_registry.get_file(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id!r} was not found.")
    return voice.metadata()


@app.put("/v1/audio/voices/{voice_id}")
async def replace_voice(voice_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        voice = voice_registry.write_file(
            filename=file.filename or f"{voice_id}.wav",
            data=await file.read(),
            voice_id=voice_id,
            replace=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return voice.metadata()


@app.delete("/v1/audio/voices/{voice_id}")
def delete_voice(voice_id: str) -> dict[str, Any]:
    if not voice_registry.delete(voice_id):
        raise HTTPException(status_code=404, detail=f"Voice {voice_id!r} was not found.")
    return {"id": voice_id, "deleted": True}


def _request_kwargs(req: SpeechRequest, text: str, seconds: float | None) -> dict[str, Any]:
    voice = voice_registry.resolve(req.voice)
    options = req.irodori
    kwargs: dict[str, Any] = {
        "text": text,
        "ref_wav": options.ref_wav or voice.ref_wav,
        "ref_latent": options.ref_latent or voice.ref_latent,
        "no_ref": voice.no_ref if options.no_ref is None else bool(options.no_ref),
        "seconds": seconds,
        "duration_scale": options.duration_scale or 1.0,
        "min_seconds": options.min_seconds or settings.default_auto_min_seconds,
        "max_seconds": options.max_seconds or settings.default_max_seconds,
        "num_steps": options.num_steps or settings.default_num_steps,
        "t_schedule_mode": options.t_schedule_mode or settings.default_t_schedule_mode,
        "sway_coeff": (
            settings.default_sway_coeff
            if options.sway_coeff is None
            else options.sway_coeff
        ),
        "cfg_guidance_mode": options.cfg_guidance_mode or settings.default_cfg_guidance_mode,
        "seed": options.seed,
    }
    if options.cfg_scale is not None:
        kwargs["cfg_scale"] = options.cfg_scale
    elif (
        settings.default_cfg_scale is not None
        and options.cfg_scale_text is None
        and options.cfg_scale_caption is None
        and options.cfg_scale_speaker is None
    ):
        kwargs["cfg_scale"] = settings.default_cfg_scale
    for name, value in options.model_dump(exclude_none=True).items():
        if name not in kwargs and hasattr(SamplingRequest, "__dataclass_fields__"):
            if name in SamplingRequest.__dataclass_fields__:
                kwargs[name] = value
    return kwargs


async def _get_synthesis_semaphore() -> asyncio.Semaphore:
    global _synthesis_semaphore, _synthesis_semaphore_limit
    limit = max(1, int(settings.max_concurrent_synthesis))
    if _synthesis_semaphore is None or _synthesis_semaphore_limit != limit:
        _synthesis_semaphore = asyncio.Semaphore(limit)
        _synthesis_semaphore_limit = limit
    return _synthesis_semaphore


def _concat_results(results: list[Any]):
    first = results[0]
    if len(results) == 1:
        return first
    audio = torch.cat([result.audio.detach().cpu().reshape(-1) for result in results], dim=0)
    updates: dict[str, Any] = {
        "audio": audio,
        "audios": [audio],
        "total_to_decode": sum(r.total_to_decode for r in results),
    }
    if hasattr(first, "stage_timings"):
        updates["stage_timings"] = [
            (f"chunk{index}.{name}", seconds)
            for index, result in enumerate(results)
            for name, seconds in getattr(result, "stage_timings", [])
        ]
    return replace(first, **updates)


def _stage_timings(result: Any) -> list[tuple[str, float]]:
    timings = getattr(result, "stage_timings", None)
    if not timings:
        return []
    return [(str(name), float(seconds)) for name, seconds in timings]


def _timing_headers(result: Any) -> dict[str, str]:
    headers: dict[str, str] = {}
    total_to_decode = getattr(result, "total_to_decode", None)
    if total_to_decode is not None:
        headers["X-Irodori-Total-To-Decode"] = f"{float(total_to_decode):.6f}"
    timings = _stage_timings(result)
    if timings:
        headers["X-Irodori-Stage-Timings"] = ",".join(
            f"{name};dur={seconds:.6f}" for name, seconds in timings
        )
    return headers


def _request_with_seed(req: SpeechRequest) -> SpeechRequest:
    if req.irodori.seed is not None:
        return req
    irodori = req.irodori.model_copy(update={"seed": int(secrets.randbits(63))})
    return req.model_copy(update={"irodori": irodori})


def _request_texts(req: SpeechRequest) -> list[str]:
    chunking_enabled = (
        settings.default_chunking_enabled
        if req.irodori.chunking_enabled is None
        else bool(req.irodori.chunking_enabled)
    )
    chunk_min_chars = req.irodori.chunk_min_chars or settings.default_chunk_min_chars
    texts = [req.input]
    if req.irodori.seconds is None and chunking_enabled:
        texts = split_text(req.input, min_chars=chunk_min_chars)
    return texts


def _runtime_uses_duration_predictor(runtime) -> bool:
    return bool(getattr(getattr(runtime, "model_cfg", None), "use_duration_predictor", False))


def _chunk_seconds(
    req: SpeechRequest,
    text: str,
    *,
    use_model_duration_predictor: bool,
) -> float | None:
    seconds = req.irodori.seconds
    if seconds is None and settings.default_auto_seconds and not use_model_duration_predictor:
        estimate = estimate_seconds(
            text,
            min_seconds=req.irodori.min_seconds or settings.default_auto_min_seconds,
            scale=settings.default_auto_seconds_scale,
            phonemes_per_second=settings.default_phonemes_per_second,
            chars_per_second=settings.default_chars_per_second,
            padding_seconds=settings.default_duration_padding_seconds,
            max_seconds=req.irodori.max_seconds or settings.default_max_seconds,
        )
        seconds = estimate.seconds
    return seconds


def _synthesize_text_sync(
    runtime,
    req: SpeechRequest,
    text: str,
    *,
    use_model_duration_predictor: bool,
):
    sampling_req = SamplingRequest(
        **_request_kwargs(
            req,
            text,
            _chunk_seconds(
                req,
                text,
                use_model_duration_predictor=use_model_duration_predictor,
            ),
        )
    )
    return runtime.synthesize(sampling_req, log_fn=partial(logger.info, "irodori runtime: %s"))


def _synthesize_sync(runtime, req: SpeechRequest):
    request = _request_with_seed(req)
    use_model_duration_predictor = _runtime_uses_duration_predictor(runtime)
    results = [
        _synthesize_text_sync(
            runtime,
            request,
            text,
            use_model_duration_predictor=use_model_duration_predictor,
        )
        for text in _request_texts(request)
    ]
    return _concat_results(results)


def normalize_stream_format(value: str | None) -> str | None:
    if value is None:
        return None
    fmt = str(value).strip().lower()
    stream_format = STREAM_FORMAT_ALIASES.get(fmt)
    if stream_format is None:
        supported = ", ".join(sorted(STREAM_FORMAT_ALIASES))
        raise ValueError(
            f"Unsupported stream_format={value!r}. Supported formats: {supported}.",
        )
    return stream_format


def _ndjson_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _stream_error_line(message: str, *, error_type: str) -> str:
    return _ndjson_line(
        {
            "type": "error",
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            },
        }
    )


async def _stream_speech_lines(
    runtime,
    req: SpeechRequest,
    response_format: str,
    started: float,
) -> AsyncIterator[str]:
    request = _request_with_seed(req)
    use_model_duration_predictor = _runtime_uses_duration_predictor(runtime)
    texts = _request_texts(request)
    semaphore = await _get_synthesis_semaphore()
    total_audio_seconds = 0.0
    total_bytes = 0

    async with semaphore:
        for index, text in enumerate(texts):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        _synthesize_text_sync,
                        runtime,
                        request,
                        text,
                        use_model_duration_predictor=use_model_duration_predictor,
                    ),
                    timeout=float(settings.synthesis_wait_timeout),
                )
            except asyncio.TimeoutError:
                yield _stream_error_line("Synthesis timed out.", error_type="server_error")
                return
            except (FileNotFoundError, ValueError) as exc:
                yield _stream_error_line(str(exc), error_type="invalid_request_error")
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("streaming speech synthesis failed")
                yield _stream_error_line("Internal server error.", error_type="server_error")
                return

            audio_bytes = encode_audio(result.audio, result.sample_rate, response_format)
            audio_seconds = float(result.audio.numel()) / float(result.sample_rate)
            total_audio_seconds += audio_seconds
            total_bytes += len(audio_bytes)
            payload: dict[str, Any] = {
                "type": "chunk",
                "index": index,
                "chunks": len(texts),
                "text": text,
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
                "format": response_format,
                "mime_type": CONTENT_TYPES[response_format],
                "audio_seconds": audio_seconds,
                "sample_rate": int(result.sample_rate),
                "seed": getattr(result, "used_seed", request.irodori.seed),
                "total_to_decode": float(getattr(result, "total_to_decode", 0.0)),
            }
            timings = _stage_timings(result)
            if timings:
                payload["stage_timings"] = {
                    name: seconds
                    for name, seconds in timings
                }
            yield _ndjson_line(payload)

    elapsed = time.perf_counter() - started
    logger.info(
        "streaming speech synthesis completed: elapsed=%.2fs audio_seconds=%.2f bytes=%s seed=%s chunks=%s",
        elapsed,
        total_audio_seconds,
        total_bytes,
        request.irodori.seed,
        len(texts),
    )
    yield _ndjson_line(
        {
            "type": "done",
            "chunks": len(texts),
            "elapsed_seconds": elapsed,
            "audio_seconds": total_audio_seconds,
            "bytes": total_bytes,
            "seed": request.irodori.seed,
        }
    )


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest) -> Response:
    requested_model = str(req.model or "").strip()
    try:
        checkpoint_file = (
            irodori_tts_lite.checkpoint_file_from_model_name(requested_model)
            if requested_model
            else settings.checkpoint_file
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.speed != 1.0:
        raise HTTPException(status_code=400, detail="Speed is not supported by this runtime.")

    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input must not be blank.")

    try:
        response_format = normalize_response_format(
            req.response_format,
            settings.default_response_format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        stream_format = normalize_stream_format(req.stream_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        runtime = await asyncio.to_thread(runtime_manager.get, checkpoint_file)
    except RuntimeLoadTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Runtime load timed out.") from exc

    request = _request_with_seed(req.model_copy(update={"input": text}))
    started = time.perf_counter()
    if stream_format is not None:
        return StreamingResponse(
            _stream_speech_lines(runtime, request, response_format, started),
            media_type=STREAM_CONTENT_TYPES[stream_format],
            headers={
                "X-Irodori-Stream-Format": stream_format,
                "X-Irodori-Chunked": "true",
            },
        )

    semaphore = await _get_synthesis_semaphore()
    async with semaphore:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_synthesize_sync, runtime, request),
                timeout=float(settings.synthesis_wait_timeout),
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Synthesis timed out.") from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    audio_bytes = encode_audio(result.audio, result.sample_rate, response_format)
    logger.info(
        "speech synthesis completed: elapsed=%.2fs audio_seconds=%.2f bytes=%s seed=%s",
        time.perf_counter() - started,
        float(result.audio.numel()) / float(result.sample_rate),
        len(audio_bytes),
        getattr(result, "used_seed", None),
    )
    headers = {
        "X-Irodori-Seed": str(getattr(result, "used_seed", "")),
        **_timing_headers(result),
    }
    return Response(
        content=audio_bytes,
        media_type=CONTENT_TYPES[response_format],
        headers=headers,
    )
