from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from irodori_tts_lite.weights import DEFAULT_DIT_FILE, DEFAULT_REPO

DEFAULT_UPSTREAM_PATH = Path(__file__).resolve().parents[2] / "Irodori-TTS"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRODORI_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8088
    api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=list)
    upstream_path: Path | None = DEFAULT_UPSTREAM_PATH

    checkpoint: str | None = None
    hf_checkpoint: str = DEFAULT_REPO
    hf_checkpoint_file: str = DEFAULT_DIT_FILE
    model_name: str = "irodori-tts-lite"

    model_device: str = "auto"
    codec_device: str = "cpu"
    model_precision: str = "fp32"
    codec_precision: str = "fp32"
    use_fused: bool = True
    force_fp16: bool = True
    disable_eager_dequant: bool = False
    codec_int4: bool = False
    codec_int4_groupsize: int = 32
    compile_model: bool = False
    compile_dynamic: bool = False

    voices_dir: Path = Path("voices")
    default_voice: str = "none"
    default_response_format: str = "wav"
    default_num_steps: int = 40
    default_auto_seconds: bool = True
    default_auto_min_seconds: float = 2.0
    default_auto_seconds_scale: float = 1.25
    default_phonemes_per_second: float = 11.0
    default_chars_per_second: float = 7.0
    default_duration_padding_seconds: float = 0.6
    default_max_seconds: float = 30.0
    default_chunking_enabled: bool = True
    default_chunk_min_chars: int = 80

    model_load_timeout: float = 300.0
    synthesis_wait_timeout: float = 600.0
    max_concurrent_synthesis: int = 1


def get_settings() -> Settings:
    return Settings()
