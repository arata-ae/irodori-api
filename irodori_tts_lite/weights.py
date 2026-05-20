"""Resolve checkpoint paths, optionally downloading from Hugging Face."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO = "arata-ae/irodori-archive"
V2_DIT_FILE = "Irodori-TTS-Lite-int4/dit_int4.safetensors"
V3_DIT_FILE = "Irodori-TTS-500M-v3-int4/model.safetensors"
DEFAULT_DIT_FILE = V2_DIT_FILE
DEFAULT_CHECKPOINT_FILE = f"{DEFAULT_REPO}/{V2_DIT_FILE}"
DEFAULT_DACVAE_FILE = "Irodori-TTS-Lite-int4/dacvae_int4.safetensors"
DEFAULT_HF_DURATION_DONOR = f"{DEFAULT_REPO}/{V3_DIT_FILE}"
V2_MODEL_NAME = "irodori-tts-v2"
V3_MODEL_NAME = "irodori-tts-v3"
MODEL_CHECKPOINT_FILES = {
    V2_MODEL_NAME: DEFAULT_CHECKPOINT_FILE,
    V3_MODEL_NAME: f"{DEFAULT_REPO}/{V3_DIT_FILE}",
}


def _hf_download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _split_hf_spec(spec: str) -> tuple[str, str]:
    value = str(spec).strip()
    if value.startswith("hf://"):
        value = value[len("hf://"):]
    parts = value.split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Bad Hugging Face checkpoint spec {spec!r}; "
            "expected <org>/<repo>/<filename>"
        )
    return f"{parts[0]}/{parts[1]}", parts[2]


def resolve_hf_checkpoint(spec: str) -> str:
    repo_id, filename = _split_hf_spec(spec)
    return _hf_download(repo_id, filename)


def model_name_from_checkpoint_file(value: str) -> str:
    checkpoint = str(value).strip()
    if checkpoint.startswith("hf://"):
        checkpoint = checkpoint[len("hf://"):]
    if checkpoint == DEFAULT_CHECKPOINT_FILE or checkpoint.endswith(f"/{V2_DIT_FILE}"):
        return V2_MODEL_NAME
    if checkpoint == DEFAULT_HF_DURATION_DONOR or checkpoint.endswith(f"/{V3_DIT_FILE}"):
        return V3_MODEL_NAME
    if "v3" in checkpoint.lower():
        return V3_MODEL_NAME
    return V2_MODEL_NAME


def checkpoint_file_from_model_name(model_name: str) -> str:
    value = str(model_name).strip()
    try:
        return MODEL_CHECKPOINT_FILES[value]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name!r}") from exc


def resolve_checkpoint(
    arg: str | None,
    *,
    default_filename: str = DEFAULT_DIT_FILE,
    default_repo: str = DEFAULT_REPO,
) -> str:
    """Return a local path to the checkpoint.

    Resolution order:
      - `hf://<org>/<repo>/<filename>` URI → `hf_hub_download(repo, filename)`
      - `<org>/<repo>/<filename>` spec     → `hf_hub_download(repo, filename)`
      - None / empty                       → `hf_hub_download(default_repo, default_filename)`
      - Otherwise a local path             → returned as-is (must exist).
    """
    if not arg:
        return _hf_download(default_repo, default_filename)

    if arg.startswith("hf://"):
        return resolve_hf_checkpoint(arg)

    path = Path(os.path.expanduser(arg))
    if path.exists():
        return str(path)
    if not path.is_absolute() and len(str(arg).split("/", 2)) == 3:
        return resolve_hf_checkpoint(arg)
    raise FileNotFoundError(f"checkpoint not found: {path}")
