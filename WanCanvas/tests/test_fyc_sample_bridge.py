from __future__ import annotations

import unittest

from wancanvas.data.outpaint_dataset import CropReference, DatasetRecord, WanCanvasDataset
from wancanvas.data.samplers import AnchorTargetSamplingConfig
from wancanvas.models.fyc_sample_bridge import FYCSampleToWanBridge

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, 'torch not installed')
class FYCSampleToWanBridgeTest(unittest.TestCase):
    def test_crop_reference_sample_bridges_to_wrapper_payload(self) -> None:
        dataset = WanCanvasDataset(
            records=[DatasetRecord(source_id='sample', prompt='expand snowy forest', frame_height=720, frame_width=1280, frame_count=8)],
            sampling_config=AnchorTargetSamplingConfig(target_size=(512, 512), anchor_size=(384, 384), seed=5),
        )
        sample = dataset[0]
        self.assertIsInstance(sample.anchor_video, CropReference)
        bridge = FYCSampleToWanBridge()
        output = bridge.build(sample)
        bundle = output.wrapper_payload['condition_bundle']
        self.assertEqual(bundle['order'], ['text', 'layout', 'geometry', 'mask'])
        self.assertEqual(bundle['semantic_roles'], ['text', 'layout_encoder', 'relative_region_embedding', 'known_region_mask_summary'])
        self.assertEqual(bundle['token_shapes']['layout'], [1, 8, 1024])
        self.assertEqual(bundle['token_shapes']['geometry'], [1, 4, 1024])
        self.assertEqual(bundle['token_shapes']['mask'], [1, 1, 1024])
        self.assertEqual(bundle['concat_shape'], [1, 29, 1024])
        self.assertEqual(output.to_dict()['request']['noisy_latents_shape'], [1, 8, 16, 64, 64])
        self.assertEqual(output.to_dict()['request']['timesteps_shape'], [1])
        self.assertTrue(all(output.wrapper_payload['request_contract']['checks'].values()))
        self.assertEqual(output.metadata['anchor_meta']['anchor_source'], 'crop-reference-synthetic')
        self.assertTrue(output.request.known_region_state['anchor_region'] is not None)

    def test_real_tensor_anchor_is_preserved(self) -> None:
        frames = torch.randn(8, 3, 32, 32)

        def frame_loader(_: DatasetRecord):
            return frames

        def cropper(video: torch.Tensor, region):
            return video[:, :, region.top:region.bottom, region.left:region.right]

        dataset = WanCanvasDataset(
            records=[DatasetRecord(source_id='tensor', prompt='expand lake', frame_height=32, frame_width=32, frame_count=8)],
            sampling_config=AnchorTargetSamplingConfig(target_size=(16, 16), anchor_size=(16, 16), seed=9),
            frame_loader=frame_loader,
            cropper=cropper,
        )
        sample = dataset[0]
        bridge = FYCSampleToWanBridge()
        output = bridge.build(sample)
        self.assertEqual(output.metadata['anchor_meta']['anchor_source'], 'tensor-frames')
        self.assertEqual(output.wrapper_payload['condition_bundle']['token_shapes']['layout'][0], 1)
        self.assertTrue(all(output.wrapper_payload['request_contract']['checks'].values()))


if __name__ == '__main__':
    unittest.main()
