from __future__ import annotations

from contextlib import nullcontext
import unittest

from wancanvas.backbones.wan_loader import WanLoader
from wancanvas.data.outpaint_dataset import DatasetRecord, WanCanvasDataset
from wancanvas.data.samplers import AnchorTargetSamplingConfig
from wancanvas.models.wan_outpaint_wrapper import WanOutpaintWrapper
from wancanvas.train import SmokeTrainer

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


class _FakeScheduler:
    order = 1

    class config:
        num_train_timesteps = 20
        expand_timesteps = True

    def __init__(self) -> None:
        self.timesteps = torch.tensor([10, 5, 1], dtype=torch.float32)
        self.sigmas = torch.tensor([1.0, 0.5, 0.1], dtype=torch.float32)

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        self.timesteps = torch.linspace(10, 1, steps=num_inference_steps, device=device)
        self.sigmas = torch.linspace(1.0, 0.1, steps=num_inference_steps, device=device)

    def step(self, model_output, timestep, sample, return_dict=False):
        return (sample - 0.1 * model_output,)

    def scale_noise(self, sample, timestep, noise):
        sigma = self._sigma_for(timestep, sample)
        return sigma * noise + (1.0 - sigma) * sample

    def index_for_timestep(self, timestep, schedule_timesteps):
        timestep = float(timestep.item() if hasattr(timestep, 'item') else timestep)
        values = schedule_timesteps.detach().cpu().tolist()
        return min(range(len(values)), key=lambda idx: abs(values[idx] - timestep))

    def _sigma_for(self, timestep, sample):
        if timestep.ndim == 0:
            timestep = timestep.view(1)
        schedule_timesteps = self.timesteps.to(sample.device)
        sigma_values = self.sigmas.to(sample.device, sample.dtype)
        index = self.index_for_timestep(timestep[0], schedule_timesteps)
        sigma = sigma_values[index].view(1, 1, 1, 1, 1)
        return sigma


class _FakeTransformer:
    dtype = torch.float32

    class config:
        in_channels = 48
        text_dim = 4096

    def cache_context(self, _: str):
        return nullcontext()

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, attention_kwargs=None, return_dict=False):
        scale = encoder_hidden_states.mean(dim=(1, 2)).view(hidden_states.shape[0], 1, 1, 1, 1)
        timestep_scale = timestep.float().mean(dim=1, keepdim=False).view(hidden_states.shape[0], 1, 1, 1, 1) / 10.0
        return (hidden_states * 0.1 + scale + timestep_scale,)


class _FakeVAE:
    dtype = torch.float32

    class config:
        z_dim = 48
        latents_mean = [0.0] * 48
        latents_std = [1.0] * 48

    def encode(self, video):
        pooled = F.adaptive_avg_pool3d(video, output_size=(video.shape[2], max(video.shape[3] // 8, 1), max(video.shape[4] // 8, 1)))
        latents = pooled.mean(dim=1, keepdim=True).repeat(1, 48, 1, 1, 1)
        return type('EncodeOutput', (), {'latents': latents})()

    def decode(self, latents, return_dict=False):
        video = F.interpolate(latents[:, :3], size=(latents.shape[2], latents.shape[3] * 8, latents.shape[4] * 8), mode='trilinear', align_corners=False)
        return (video,)


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = _FakeTransformer()
        self.scheduler = _FakeScheduler()
        self.vae = _FakeVAE()
        self._execution_device = torch.device('cpu')

    def encode_prompt(self, **kwargs):
        prompt = torch.ones(1, 4, 4096, dtype=torch.float32)
        negative = torch.zeros_like(prompt)
        return prompt, negative

    def prepare_latents(self, batch_size, num_channels_latents, height, width, num_frames, dtype, device, generator, latents=None):
        return torch.zeros(batch_size, num_channels_latents, num_frames, height // 8, width // 8, dtype=dtype, device=device)


@unittest.skipIf(torch is None, 'torch not installed')
class SmokeTrainerTest(unittest.TestCase):
    def test_smoke_trainer_runs_forward_backward_and_optimizer_step(self) -> None:
        frames = torch.randn(5, 3, 32, 32)

        def frame_loader(_: DatasetRecord):
            return frames

        def cropper(video: torch.Tensor, region):
            return video[:, :, region.top:region.bottom, region.left:region.right]

        dataset = WanCanvasDataset(
            records=[DatasetRecord(source_id='train-demo', prompt='expand the frame', frame_height=32, frame_width=32, frame_count=5)],
            sampling_config=AnchorTargetSamplingConfig(target_size=(16, 16), anchor_size=(16, 16), seed=11),
            frame_loader=frame_loader,
            cropper=cropper,
        )
        trainer = SmokeTrainer(wrapper=WanOutpaintWrapper(WanLoader()))
        report = trainer.run_once(dataset[0], runtime=_FakePipeline())

        self.assertGreater(report.loss, 0.0)
        self.assertGreaterEqual(report.grad_norm, 0.0)
        self.assertGreater(report.updated_parameter_count, 0)
        self.assertEqual(report.optimizer_lr, 1e-4)
        self.assertEqual(report.forward_summary['conditioned_prompt_embeds_shape'][-1], 4096)
        self.assertEqual(report.request_summary['condition_video_shape'][0], 1)
        self.assertEqual(set(report.loss_components.keys()), {'diffusion', 'known_region', 'seam'})


if __name__ == '__main__':
    unittest.main()
