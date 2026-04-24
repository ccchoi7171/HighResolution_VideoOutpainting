from __future__ import annotations

import unittest

from wancanvas.backbones.wan_loader import WanLoader
from wancanvas.data.outpaint_dataset import DatasetRecord, WanCanvasDataset
from wancanvas.data.samplers import AnchorTargetSamplingConfig
from wancanvas.models.wan_outpaint_wrapper import WanOutpaintWrapper
from wancanvas.train import SmokeTrainer

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, 'torch not installed')
class SmokeTrainerTest(unittest.TestCase):
    def test_smoke_trainer_runs_forward_backward_and_optimizer_step(self) -> None:
        runtime = WanLoader().load_pipeline(runtime_variant='smoke', device='cpu', torch_dtype='float32')
        frames = torch.linspace(0.0, 1.0, 9 * 3 * 64 * 64, dtype=torch.float32).view(9, 3, 64, 64)

        def frame_loader(_: DatasetRecord):
            return frames

        def cropper(video: torch.Tensor, region):
            return video[:, :, region.top:region.bottom, region.left:region.right]

        dataset = WanCanvasDataset(
            records=[DatasetRecord(source_id='train-demo', prompt='expand the frame', frame_height=64, frame_width=64, frame_count=9)],
            sampling_config=AnchorTargetSamplingConfig(target_size=(32, 32), anchor_size=(32, 32), seed=11),
            frame_loader=frame_loader,
            cropper=cropper,
        )
        sample = dataset[0]
        wrapper = WanOutpaintWrapper(WanLoader())
        trainer = SmokeTrainer(wrapper=wrapper)
        report = trainer.run_once(sample, runtime=runtime).to_dict()

        self.assertGreater(report['loss'], 0.0)
        self.assertGreater(report['updated_parameter_count'], 0)
        self.assertIn('condition_video_shape', report['request_summary'])
        self.assertEqual(report['forward_summary']['metadata']['consumption_path'], 'encoder_hidden_states_image')
        self.assertIsNotNone(report['forward_summary']['projected_condition_tokens_shape'])


if __name__ == '__main__':
    unittest.main()
