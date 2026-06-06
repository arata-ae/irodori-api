from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from irodori_tts.rf import sample_euler_rf_cfg


class FakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.cfg = SimpleNamespace(
            patched_latent_dim=2,
            use_caption_condition=False,
            use_speaker_condition=False,
        )
        self.forward_batch_sizes: list[int] = []
        self.cache_batch_sizes: list[int] = []

    def encode_conditions(
        self,
        *,
        text_input_ids,
        text_mask,
        ref_latent,
        ref_mask,
        caption_input_ids,
        caption_mask,
        speaker_state_override,
        speaker_mask_override,
        speaker_uncond_mode,
    ):
        batch = text_input_ids.shape[0]
        text_state = torch.ones((batch, 1, 2), dtype=self.dtype)
        return text_state, text_mask[:, :1], None, None, None, None

    def build_context_kv_cache(
        self,
        *,
        text_state,
        speaker_state,
        caption_state,
    ):
        self.cache_batch_sizes.append(int(text_state.shape[0]))
        return []

    def forward_with_encoded_conditions(
        self,
        *,
        x_t,
        t,
        text_state,
        text_mask,
        speaker_state,
        speaker_mask,
        caption_state,
        caption_mask,
        context_kv_cache,
    ):
        self.forward_batch_sizes.append(int(x_t.shape[0]))
        text_strength = text_mask.to(dtype=x_t.dtype).sum(dim=1)[:, None, None]
        return torch.ones_like(x_t) * text_strength


class RFSamplerTests(unittest.TestCase):
    def test_joint_cfg_uses_one_batched_forward_per_active_step(self) -> None:
        model = FakeModel()

        result = sample_euler_rf_cfg(
            model=model,
            text_input_ids=torch.ones((1, 3), dtype=torch.long),
            text_mask=torch.ones((1, 3), dtype=torch.bool),
            ref_latent=None,
            ref_mask=None,
            sequence_length=2,
            num_steps=4,
            cfg_guidance_mode="joint",
            cfg_scale=3.0,
            cfg_min_t=0.0,
            cfg_max_t=1.0,
            seed=1,
        )

        self.assertEqual(model.forward_batch_sizes, [2, 2, 2, 2])
        self.assertEqual(model.cache_batch_sizes, [1, 2])
        self.assertEqual(tuple(result.shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
