from __future__ import annotations

import logging
import threading
import time

import irodori_tts_lite
import torch
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

try:
    from irodori_tts.inference_runtime import default_runtime_device as _default_runtime_device
except ImportError:
    _default_runtime_device = None

logger = logging.getLogger(__name__)


class RuntimeLoadTimeoutError(TimeoutError):
    pass


class RuntimeManager:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._runtime = None
        self._checkpoint_path: str | None = None
        self._loading = False
        self._condition = threading.Condition()

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    def _resolve_checkpoint_path(self) -> str:
        checkpoint = self.settings.checkpoint
        filename = str(self.settings.hf_checkpoint_file).strip()
        if checkpoint is None and not str(self.settings.hf_checkpoint).strip():
            raise ValueError("Set IRODORI_CHECKPOINT or IRODORI_HF_CHECKPOINT.")
        if checkpoint is None and filename == "":
            raise ValueError("Set IRODORI_HF_CHECKPOINT_FILE.")
        return irodori_tts_lite.resolve_checkpoint(
            checkpoint,
            default_repo=self.settings.hf_checkpoint,
            default_filename=filename,
        )

    @staticmethod
    def _resolve_device(value: str) -> str:
        device = str(value).strip().lower()
        if device != "auto":
            return str(value)
        if _default_runtime_device is not None:
            return str(_default_runtime_device())
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def get(self):
        with self._condition:
            if self._runtime is not None:
                return self._runtime
            if self._loading:
                deadline = time.monotonic() + float(self.settings.model_load_timeout)
                while self._loading and self._runtime is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeLoadTimeoutError("Timed out waiting for runtime load.")
                    self._condition.wait(timeout=remaining)
                if self._runtime is None:
                    raise RuntimeLoadTimeoutError("Runtime load failed.")
                return self._runtime
            self._loading = True

        try:
            logger.info("loading runtime")
            started = time.perf_counter()
            checkpoint_path = self._resolve_checkpoint_path()
            logger.info("checkpoint resolved: %s", checkpoint_path)
            irodori_tts_lite.configure(
                use_fused=self.settings.use_fused,
                force_fp16=self.settings.force_fp16,
                disable_eager=self.settings.disable_eager_dequant,
                codec_int4=self.settings.codec_int4,
                codec_int4_groupsize=self.settings.codec_int4_groupsize,
            )
            irodori_tts_lite.patch()
            model_device = self._resolve_device(self.settings.model_device)
            codec_device = self._resolve_device(self.settings.codec_device)
            logger.info("runtime devices resolved: model=%s codec=%s", model_device, codec_device)
            key = RuntimeKey(
                checkpoint=checkpoint_path,
                model_device=model_device,
                codec_device=codec_device,
                model_precision=self.settings.model_precision,
                codec_precision=self.settings.codec_precision,
                compile_model=self.settings.compile_model,
                compile_dynamic=self.settings.compile_dynamic,
            )
            runtime = InferenceRuntime.from_key(key)
            logger.info("runtime loaded in %.2fs", time.perf_counter() - started)
        except Exception:
            with self._condition:
                self._loading = False
                self._condition.notify_all()
            raise

        with self._condition:
            self._runtime = runtime
            self._checkpoint_path = checkpoint_path
            self._loading = False
            self._condition.notify_all()
            return runtime
