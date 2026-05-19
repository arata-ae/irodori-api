from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_upstream_path(path: str | Path | None) -> None:
    if path is None or str(path).strip() == "":
        return
    upstream_path = Path(path).expanduser().resolve()
    if not upstream_path.is_dir():
        return
    path_text = str(upstream_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def require_upstream() -> None:
    missing = [
        name
        for name in ("irodori_tts", "infer")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Upstream Irodori-TTS is not available "
            f"(missing: {', '.join(missing)}). "
            "Set IRODORI_UPSTREAM_PATH or run with --with-editable ../Irodori-TTS."
        )
