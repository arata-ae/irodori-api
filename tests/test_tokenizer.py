from __future__ import annotations

import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from irodori_tts.tokenizer import (
    PretrainedTextTokenizer,
    _disable_transformers_sklearn_probe,
)


class FakeTokenizer:
    padding_side = "left"
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"
    bos_token_id = 2

    def __len__(self) -> int:
        return 3

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class PretrainedTextTokenizerTests(unittest.TestCase):
    def test_disables_transformers_sklearn_probe(self) -> None:
        from transformers.utils import import_utils

        import_utils._sklearn_available = True

        _disable_transformers_sklearn_probe()

        self.assertFalse(import_utils._sklearn_available)

    def test_from_pretrained_disables_sklearn_before_importing_tokenizer(self) -> None:
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = Mock()
        fake_transformers.AutoTokenizer.from_pretrained.return_value = FakeTokenizer()

        with (
            patch.dict("sys.modules", {"transformers": fake_transformers}),
            patch("irodori_tts.tokenizer._disable_transformers_sklearn_probe") as disable_probe,
        ):
            tokenizer = PretrainedTextTokenizer.from_pretrained("local/tokenizer")

        disable_probe.assert_called_once_with()
        fake_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "local/tokenizer",
            use_fast=True,
            trust_remote_code=False,
            local_files_only=False,
        )
        self.assertEqual(tokenizer.vocab_size, 3)


if __name__ == "__main__":
    unittest.main()
