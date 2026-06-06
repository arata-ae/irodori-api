from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from irodori_tts_lite.weights import (
    DEFAULT_CHECKPOINT_FILE,
    DEFAULT_HF_DURATION_DONOR,
    model_name_from_checkpoint_file,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRODORI_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8088
    cors_origins: list[str] = Field(default_factory=list)

    checkpoint_file: str = DEFAULT_CHECKPOINT_FILE

    @property
    def effective_model_name(self) -> str:
        return model_name_from_checkpoint_file(self.checkpoint_file)

    model_device: str = "auto"
    codec_device: str = "cpu"
    model_precision: str = "fp32"
    codec_precision: str = "fp32"
    use_fused: bool = True
    force_fp16: bool = True
    disable_eager_dequant: bool = False
    codec_int4: bool = False
    codec_int4_groupsize: int = 32
    pack_rtn_extras: bool = True
    hf_duration_donor: str | None = DEFAULT_HF_DURATION_DONOR
    compile_model: bool = False
    compile_dynamic: bool = False

    voices_dir: Path = Path("voices")
    default_voice: str = "none"
    default_response_format: str = "wav"
    default_num_steps: int = 24
    default_t_schedule_mode: str = "linear"
    default_sway_coeff: float = -1.0
    default_cfg_guidance_mode: str = "independent"
    default_cfg_scale: float | None = None
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
