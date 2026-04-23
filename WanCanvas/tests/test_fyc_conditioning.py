from __future__ import annotations

import unittest

from wancanvas.models.fyc_conditioning import FYCConditioningBuilder, FYCConditioningConfig
from wancanvas.models.layout_encoder import LayoutEncoderConfig
from wancanvas.models.geometry_encoder import GeometryEncoderConfig
from wancanvas.models.mask_summary import MaskSummaryConfig
from wancanvas.models.wan_outpaint_wrapper import WanForwardRequest, WanOutpaintWrapper
from wancanvas.backbones.wan_loader import WanLoader

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, 'torch not installed')
class FYCConditioningBuilderTest(unittest.TestCase):
    def test_builder_preserves_le_rre_mask_semantics(self) -> None:
        builder = FYCConditioningBuilder(
            FYCConditioningConfig(
                layout=LayoutEncoderConfig(hidden_dim=16, token_dim=32, token_count=3),
                geometry=GeometryEncoderConfig(hidden_dim=8, token_dim=32, token_count=2),
                mask=MaskSummaryConfig(token_dim=32, token_count=1),
                include_mask_summary=True,
            )
        )
        anchor_video = torch.randn(2, 5, 3, 16, 16)
        relative_position = torch.randn(2, 6)
        known_mask = torch.randint(0, 2, (2, 5, 1, 16, 16)).float()
        prompt_embeds = torch.randn(2, 4, 32)

        output = builder.encode(
            anchor_video=anchor_video,
            relative_position=relative_position,
            known_mask=known_mask,
            prompt_embeds=prompt_embeds,
        )

        self.assertEqual(output.bundle.order, ['text', 'layout', 'geometry', 'mask'])
        self.assertEqual(output.metadata['semantic_roles'], ['text', 'layout_encoder', 'relative_region_embedding', 'known_region_mask_summary'])
        self.assertEqual(tuple(output.layout.tokens.shape), (2, 3, 32))
        self.assertEqual(tuple(output.geometry.tokens.shape), (2, 2, 32))
        self.assertEqual(tuple(output.mask.tokens.shape), (2, 1, 32))
        self.assertEqual(tuple(output.concatenated_tokens.shape), (2, 10, 32))
        self.assertTrue(output.metadata['preserves_fyc_v1_rre'])

    def test_wrapper_reports_condition_shapes_and_semantics(self) -> None:
        builder = FYCConditioningBuilder(
            FYCConditioningConfig(
                layout=LayoutEncoderConfig(hidden_dim=8, token_dim=16, token_count=2),
                geometry=GeometryEncoderConfig(hidden_dim=8, token_dim=16, token_count=2),
                mask=MaskSummaryConfig(token_dim=16, token_count=1),
                include_mask_summary=True,
            )
        )
        output = builder.encode(
            anchor_video=torch.randn(1, 4, 3, 8, 8),
            relative_position=torch.randn(1, 6),
            known_mask=torch.randint(0, 2, (1, 4, 1, 8, 8)).float(),
            prompt_embeds=torch.randn(1, 3, 16),
        )
        wrapper = WanOutpaintWrapper(WanLoader())
        prepared = wrapper.prepare_inputs(
            WanForwardRequest(
                noisy_latents=torch.zeros(1, 4, 16, 8, 8),
                timesteps=torch.tensor([999], dtype=torch.int64),
                prompt_embeds=torch.randn(1, 3, 16),
                layout_tokens=output.layout.tokens,
                geometry_tokens=output.geometry.tokens,
                mask_tokens=output.mask.tokens,
                known_region_state={'mode': 'overwrite'},
            )
        )
        bundle = prepared['condition_bundle']
        self.assertEqual(bundle['semantic_roles'], ['text', 'layout_encoder', 'relative_region_embedding', 'known_region_mask_summary'])
        self.assertEqual(bundle['token_shapes']['layout'], [1, 2, 16])
        self.assertEqual(bundle['token_shapes']['geometry'], [1, 2, 16])
        self.assertEqual(bundle['token_shapes']['mask'], [1, 1, 16])
        self.assertEqual(bundle['concat_shape'], [1, 8, 16])
        self.assertTrue(prepared['has_known_region_state'])
        self.assertTrue(all(prepared['request_contract']['checks'].values()))


if __name__ == '__main__':
    unittest.main()
