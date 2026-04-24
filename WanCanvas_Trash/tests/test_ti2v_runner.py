from __future__ import annotations

from contextlib import nullcontext
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio

from wancanvas.inference.outpaint_runner import OutpaintInferenceConfig, _ensure_output_dirs, _normalize_num_frames, _resolve_run_name, run_outpaint_inference

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


class _FakeScheduler:
    order = 1

    class config:
        num_train_timesteps = 4
        expand_timesteps = True

    def __init__(self) -> None:
        self.timesteps = torch.tensor([4.0], dtype=torch.float32)
        self.sigmas = torch.tensor([0.5], dtype=torch.float32)

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        self.timesteps = torch.linspace(4.0, 1.0, steps=num_inference_steps, device=device)
        self.sigmas = torch.linspace(0.5, 0.1, steps=num_inference_steps, device=device)

    def step(self, model_output, timestep, sample, return_dict=False):
        return (sample - 0.05 * model_output,)

    def index_for_timestep(self, timestep, schedule_timesteps):
        timestep = float(timestep.item() if hasattr(timestep, 'item') else timestep)
        values = schedule_timesteps.detach().cpu().tolist()
        return min(range(len(values)), key=lambda idx: abs(values[idx] - timestep))


class _FakeTransformer:
    dtype = torch.float32

    class config:
        in_channels = 48
        text_dim = 4096

    def cache_context(self, _: str):
        return nullcontext()

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, attention_kwargs=None, return_dict=False):
        scale = encoder_hidden_states.mean(dim=(1, 2)).view(hidden_states.shape[0], 1, 1, 1, 1)
        return (hidden_states * 0.05 + scale,)


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


class _LoadedFake:
    def __init__(self) -> None:
        self.pipeline = _FakePipeline()

    def summary(self):
        return {'pipeline_class': 'FakePipeline'}


class OutpaintRunnerTest(unittest.TestCase):
    def test_run_name_is_stable(self) -> None:
        config = OutpaintInferenceConfig(prompt="Polar bear walking in snow", source_video_tensor=torch.zeros(5, 3, 8, 8))
        self.assertTrue(_resolve_run_name(config).startswith("outpaint-polar-bear"))

    def test_num_frames_are_normalized_to_valid_wan_count(self) -> None:
        self.assertEqual(_normalize_num_frames(7), 5)
        self.assertEqual(_normalize_num_frames(8), 9)
        self.assertEqual(_normalize_num_frames(9), 9)

    def test_output_dirs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root, original_dir, samples_dir, logs_dir = _ensure_output_dirs(tmpdir, "demo-run")
            self.assertTrue(run_root.exists())
            self.assertTrue(original_dir.exists())
            self.assertTrue(samples_dir.exists())
            self.assertTrue(logs_dir.exists())

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_outpaint_runner_writes_artifacts_with_fake_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OutpaintInferenceConfig(
                prompt='expand snowy forest',
                source_video_tensor=torch.rand(5, 3, 8, 8),
                output_root=tmpdir,
                canvas_height=16,
                canvas_width=16,
                num_frames=5,
                num_inference_steps=1,
                tile_height=16,
                tile_width=16,
                overlap_height=0,
                overlap_width=0,
                rounds=1,
            )
            artifacts = run_outpaint_inference(config, runtime=_LoadedFake())
            self.assertTrue(Path(artifacts.sample_video_path).exists())
            self.assertTrue(Path(artifacts.metadata_path).exists())
            self.assertTrue(Path(artifacts.plan_path).exists())
            reader = imageio.get_reader(artifacts.sample_video_path)
            try:
                first_frame = reader.get_data(0)
            finally:
                reader.close()
            self.assertGreater(float(first_frame[4:12, 4:12].mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
