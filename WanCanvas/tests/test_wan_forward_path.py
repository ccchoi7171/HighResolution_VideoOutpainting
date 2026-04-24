from __future__ import annotations

import unittest

from wancanvas.backbones.wan_loader import WanLoader
from wancanvas.data.outpaint_dataset import DatasetRecord, WanCanvasDataset
from wancanvas.data.samplers import AnchorTargetSamplingConfig
from wancanvas.models.fyc_sample_bridge import FYCSampleToWanBridge
from wancanvas.models.wan_outpaint_wrapper import WanOutpaintWrapper

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, 'torch not installed')
class WanForwardPathTest(unittest.TestCase):
    def test_wrapper_forward_uses_image_conditioning_stream(self) -> None:
        runtime = WanLoader().load_pipeline(runtime_variant='smoke', device='cpu', torch_dtype='float32')
        dataset = WanCanvasDataset(
            records=[DatasetRecord(source_id='forward-demo', prompt='expand the frame', frame_height=64, frame_width=64, frame_count=9)],
            sampling_config=AnchorTargetSamplingConfig(target_size=(32, 32), anchor_size=(32, 32), seed=13),
        )
        sample = dataset[0]
        wrapper = WanOutpaintWrapper(WanLoader())
        request = FYCSampleToWanBridge(wrapper=wrapper).build(sample).request
        forward = wrapper.forward(request, runtime=runtime, guidance_scale=1.0, do_classifier_free_guidance=False).to_dict()

        self.assertEqual(forward['metadata']['consumption_path'], 'encoder_hidden_states_image')
        self.assertEqual(forward['noise_pred_shape'], [1, 16, 3, 4, 4])
        self.assertEqual(forward['projected_condition_tokens_shape'][-1], 64)


if __name__ == '__main__':
    unittest.main()
