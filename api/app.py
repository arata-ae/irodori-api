from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from typing import Any, Literal

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from .audio import CONTENT_TYPES, encode_audio, normalize_response_format
from .config import get_settings
from .duration import estimate_seconds, split_text
from .upstream import ensure_upstream_path, require_upstream
from .voices import VoiceRegistry

_settings = get_settings()
if _settings.upstream_path:
    ensure_upstream_path(_settings.upstream_path)
require_upstream()

from irodori_tts.inference_runtime import SamplingRequest  # noqa: E402

from .runtime import RuntimeManager  # noqa: E402

logger = logging.getLogger(__name__)

settings = _settings
runtime_manager = RuntimeManager(settings)
voice_registry = VoiceRegistry(settings)
_synthesis_semaphore: asyncio.Semaphore | None = None
_synthesis_semaphore_limit: int | None = None


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
    logger.info("loading runtime during startup")
    runtime_manager.get()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup()
    yield


app = FastAPI(title="Irodori-API", version="0.1.0", lifespan=lifespan)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if settings.api_key is None:
        return
    if authorization != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key.")


def openai_error_response(message: str, *, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


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
    return openai_error_response(str(exc), status_code=500, error_type="server_error")


@app.get("/health")
def health() -> dict[str, Any]:
    voices_dir = settings.voices_dir.expanduser()
    return {
        "status": "ok",
        "model": {
            "id": settings.model_name,
            "hf_checkpoint": settings.hf_checkpoint,
            "hf_checkpoint_file": settings.hf_checkpoint_file,
            "model_device": settings.model_device,
            "codec_device": settings.codec_device,
            "model_precision": settings.model_precision,
            "codec_precision": settings.codec_precision,
            "use_fused": settings.use_fused,
            "force_fp16": settings.force_fp16,
            "disable_eager_dequant": settings.disable_eager_dequant,
            "codec_int4": settings.codec_int4,
            "codec_int4_groupsize": settings.codec_int4_groupsize,
            "compile_model": settings.compile_model,
            "compile_dynamic": settings.compile_dynamic,
        },
        "runtime": {
            "loaded": runtime_manager.is_loaded,
            "loading": runtime_manager.is_loading,
            "checkpoint": runtime_manager.checkpoint_path,
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


@app.get("/v1/models", dependencies=[Depends(require_auth)])
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "irodori-api",
            }
        ],
    }


@app.get("/v1/audio/voices", dependencies=[Depends(require_auth)])
def list_voices() -> dict[str, Any]:
    return {"object": "list", "data": [voice.metadata() for voice in voice_registry.list()]}


@app.post("/v1/audio/voices", status_code=201, dependencies=[Depends(require_auth)])
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


@app.get("/v1/audio/voices/{voice_id}", dependencies=[Depends(require_auth)])
def get_voice_file(voice_id: str) -> dict[str, Any]:
    voice = voice_registry.get_file(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id!r} was not found.")
    return voice.metadata()


@app.put("/v1/audio/voices/{voice_id}", dependencies=[Depends(require_auth)])
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


@app.delete("/v1/audio/voices/{voice_id}", dependencies=[Depends(require_auth)])
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
        "seed": options.seed,
    }
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
    return replace(first, audio=audio, audios=[audio], total_to_decode=sum(r.total_to_decode for r in results))


def _synthesize_sync(runtime, req: SpeechRequest):
    manual_seconds = req.irodori.seconds
    chunking_enabled = (
        settings.default_chunking_enabled
        if req.irodori.chunking_enabled is None
        else bool(req.irodori.chunking_enabled)
    )
    chunk_min_chars = req.irodori.chunk_min_chars or settings.default_chunk_min_chars
    texts = [req.input]
    if manual_seconds is None and chunking_enabled:
        texts = split_text(req.input, min_chars=chunk_min_chars)

    results = []
    for text in texts:
        seconds = manual_seconds
        if seconds is None and settings.default_auto_seconds:
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
        sampling_req = SamplingRequest(**_request_kwargs(req, text, seconds))
        results.append(runtime.synthesize(sampling_req, log_fn=partial(logger.info, "irodori runtime: %s")))
    return _concat_results(results)


@app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
async def create_speech(req: SpeechRequest) -> Response:
    if req.stream_format is not None:
        raise HTTPException(status_code=400, detail="Streaming synthesis is not implemented.")
    if req.model not in {None, settings.model_name}:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model!r}")

    response_format = normalize_response_format(req.response_format, settings.default_response_format)
    runtime = runtime_manager.get()
    semaphore = await _get_synthesis_semaphore()
    started = time.perf_counter()
    async with semaphore:
        result = await asyncio.wait_for(
            asyncio.to_thread(_synthesize_sync, runtime, req),
            timeout=float(settings.synthesis_wait_timeout),
        )
    audio_bytes = encode_audio(result.audio, result.sample_rate, response_format)
    logger.info(
        "speech synthesis completed: elapsed=%.2fs audio_seconds=%.2f bytes=%s seed=%s",
        time.perf_counter() - started,
        float(result.audio.numel()) / float(result.sample_rate),
        len(audio_bytes),
        getattr(result, "used_seed", None),
    )
    headers = {"X-Irodori-Seed": str(getattr(result, "used_seed", ""))}
    return Response(content=audio_bytes, media_type=CONTENT_TYPES[response_format], headers=headers)
