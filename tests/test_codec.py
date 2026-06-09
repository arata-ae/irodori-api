from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

import torch

from irodori_tts.codec import DACVAECodec

ORIGINAL_IMPORT = builtins.__import__


class CodecNormalizationTests(unittest.TestCase):
    def test_loudness_normalization_does_not_import_scipy_backed_loudness(self) -> None:
        waveform = torch.tensor([0.05, -0.05, 0.025, -0.025], dtype=torch.float32)

        with patch("builtins.__import__", side_effect=reject_loudness_imports):
            normalized = DACVAECodec._normalize_loudness(
                waveform,
                sample_rate=24_000,
                target_db=-16.0,
            )

        self.assertTrue(torch.isfinite(normalized).all())
        self.assertLessEqual(float(normalized.abs().max()), 1.0)
        self.assertGreater(float(normalized.abs().max()), float(waveform.abs().max()))

    def test_loudness_normalization_keeps_hot_audio_peak_safe(self) -> None:
        waveform = torch.tensor([2.0, -1.5, 0.5, -0.5], dtype=torch.float32)

        normalized = DACVAECodec._normalize_loudness(
            waveform,
            sample_rate=24_000,
            target_db=-16.0,
        )

        self.assertTrue(torch.isfinite(normalized).all())
        self.assertLessEqual(float(normalized.abs().max()), 1.0)


def reject_loudness_imports(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in {"audiotools", "pyloudnorm", "scipy"}:
        raise ImportError(name)
    return ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


if __name__ == "__main__":
    unittest.main()
