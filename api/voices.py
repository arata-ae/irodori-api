from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceFile:
    voice_id: str
    path: Path

    @property
    def ref_wav(self) -> str:
        return str(self.path)

    @property
    def ref_latent(self) -> None:
        return None

    @property
    def no_ref(self) -> bool:
        return False

    def metadata(self) -> dict[str, object]:
        return {
            "id": self.voice_id,
            "object": "voice",
            "ref_wav": self.ref_wav,
            "ref_latent": None,
            "no_ref": False,
        }


@dataclass(frozen=True)
class VoiceSpec:
    voice_id: str
    ref_wav: str | None = None
    ref_latent: str | None = None
    no_ref: bool = False


class VoiceRegistry:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.root = Path(settings.voices_dir)

    def ensure_dir(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def list_files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(path for path in self.root.iterdir() if path.is_file())

    def list(self) -> list[VoiceFile]:
        return [VoiceFile(path.stem, path) for path in self.list_files()]

    def get_file(self, voice_id: str) -> VoiceFile | None:
        self.validate_voice_id(voice_id, require_exists=False)
        for path in self.list_files():
            if path.stem == voice_id:
                return VoiceFile(voice_id, path)
        return None

    def resolve(self, value: str | dict | None) -> VoiceSpec:
        if isinstance(value, dict):
            voice_id = str(value.get("id") or value.get("voice") or "inline")
            return VoiceSpec(
                voice_id=voice_id,
                ref_wav=value.get("ref_wav"),
                ref_latent=value.get("ref_latent"),
                no_ref=bool(value.get("no_ref", False)),
            )
        voice_id = str(value or self.settings.default_voice)
        if voice_id in {"", "none", "no_ref", "no-ref"}:
            return VoiceSpec(voice_id="none", no_ref=True)
        voice = self.get_file(voice_id)
        if voice is None:
            raise FileNotFoundError(f"Voice {voice_id!r} was not found.")
        return VoiceSpec(voice_id=voice_id, ref_wav=voice.ref_wav)

    def validate_voice_id(self, voice_id: str, *, require_exists: bool = True) -> None:
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", voice_id) is None:
            raise ValueError("voice_id may only contain letters, numbers, dot, dash, and underscore.")
        if require_exists and self.get_file(voice_id) is None:
            raise FileNotFoundError(f"Voice {voice_id!r} was not found.")

    def write_file(
        self,
        *,
        filename: str,
        data: bytes,
        voice_id: str | None = None,
        replace: bool = False,
    ) -> VoiceFile:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}:
            raise ValueError("Unsupported voice file extension.")
        target_id = voice_id or Path(filename).stem
        self.validate_voice_id(target_id, require_exists=False)
        self.ensure_dir()
        path = self.root / f"{target_id}{suffix}"
        if path.exists() and not replace:
            raise FileExistsError(f"Voice {target_id!r} already exists.")
        path.write_bytes(data)
        return VoiceFile(target_id, path)

    def delete(self, voice_id: str) -> bool:
        voice = self.get_file(voice_id)
        if voice is None:
            return False
        voice.path.unlink()
        return True
