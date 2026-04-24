from __future__ import annotations

from contextlib import nullcontext
import unittest

from wancanvas.backbones.wan_loader import WanLoader
from wancanvas.models.wan_outpaint_wrapper import WanForwardRequest, WanOutpaintWrapper

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class _FakeTransformer:
    dtype = torch.float32 if torch is not None else None

    class config:
        in_channels = 48
        text_dim = 4096

    def __init__(self) -> None:
        self.calls = []

    def cache_context(self, _: str):
        return nullcontext()

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, attention_kwargs=None, return_dict=False):
        self.calls.append({
            'hidden_states': hidden_states.detach().clone(),
            'timestep': timestep.detach().clone(),
            'encoder_hidden_states': encoder_hidden_states.detach().clone(),
        })
        scale = encoder_hidden_states.mean(dim=(1, 2)).view(hidden_states.shape[0], 1, 1, 1, 1)
        return (hidden_states + scale,)


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = _FakeTransformer()
        self._execution_device = torch.device('cpu')

    def encode_prompt(self, **kwargs):
        batch = 1
        prompt = torch.full((batch, 4, 4096), 0.25, dtype=torch.float32)
        negative = torch.zeros_like(prompt)
        return prompt, negative


@unittest.skipIf(torch is None, 'torch not installed')
class WanForwardConditioningTest(unittest.TestCase):
    def test_forward_uses_real_encoder_hidden_states_and_known_region_latent_blend(self) -> None:
        wrapper = WanOutpaintWrapper(WanLoader())
        runtime = _FakePipeline()
        noisy_latents = torch.ones(1, 48, 2, 4, 4)
        condition_latents = torch.full_like(noisy_latents, 3.0)
        latent_mask = torch.ones(1, 1, 2, 4, 4)
        latent_mask[:, :, :, :2, :2] = 0.0
        request = WanForwardRequest(
            prompt='expand',
            prompt_is_placeholder=True,
            noisy_latents=noisy_latents,
            timesteps=torch.tensor([5], dtype=torch.int64),
            layout_tokens=torch.randn(1, 2, 4096),
            geometry_tokens=torch.randn(1, 1, 4096),
            mask_tokens=torch.randn(1, 1, 4096),
            condition_latents=condition_latents,
            latent_mask=latent_mask,
        )

        output = wrapper.forward(request, runtime=runtime, guidance_scale=1.0, do_classifier_free_guidance=False)

        self.assertEqual(tuple(output.conditioned_prompt_embeds.shape), (1, 8, 4096))
        self.assertEqual(tuple(output.raw_condition_tokens.shape), (1, 4, 4096))
        self.assertTrue(torch.allclose(output.latents, noisy_latents))
        self.assertTrue(torch.allclose(output.latent_model_input[:, :, :, :2, :2], condition_latents[:, :, :, :2, :2]))
        self.assertTrue(torch.allclose(output.latent_model_input[:, :, :, 2:, 2:], noisy_latents[:, :, :, 2:, 2:]))
        transformer_call = runtime.transformer.calls[-1]
        self.assertEqual(tuple(transformer_call['encoder_hidden_states'].shape), (1, 8, 4096))
        self.assertEqual(transformer_call['timestep'].ndim, 2)
        self.assertTrue((transformer_call['timestep'] == 0).any())


if __name__ == '__main__':
    unittest.main()
