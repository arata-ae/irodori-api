from __future__ import annotations

import importlib.util


def require_runtime_package() -> None:
    missing = [name for name in ("irodori_tts",) if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Irodori-TTS runtime package is not available "
            f"(missing: {', '.join(missing)}). "
            "Install irodori-api with its packaged irodori_tts runtime."
        )
