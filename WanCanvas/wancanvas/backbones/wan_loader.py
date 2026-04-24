from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import math

from ..config_schema import ModelConfig, RuntimeConfig
from .runtime_env import RuntimeInspection, inspect_diffusers_runtime, stable_release_allowed

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


@dataclass(slots=True)
class WanLoaderReport:
    runtime: dict[str, Any]
    model: dict[str, Any]
    ready_for_download: bool
    strict_dense_mvp: bool
    download_attempted: bool = False
    download_skipped_reason: str | None = None
    loaded_pipeline_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoadedWanPipeline:
    pipeline: Any
    runtime: RuntimeInspection
    model_id: str
    pipeline_class: str
    device: str
    torch_dtype: str
    flow_shift: float | None
    runtime_variant: str = "pretrained"

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "pipeline_class": self.pipeline_class,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "flow_shift": self.flow_shift,
            "diffusers_version": self.runtime.diffusers_version,
            "runtime_variant": self.runtime_variant,
        }


class SmokeWanPipeline:
    def __init__(
        self,
        *,
        device: str,
        torch_dtype: Any,
        scheduler_name: str | None,
        flow_shift: float | None,
    ) -> None:
        if torch is None or F is None:
            raise RuntimeError("torch is required for the smoke Wan runtime")

        import diffusers

        execution_device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            execution_device = "cpu"
        self._execution_device = torch.device(execution_device)
        if self._execution_device.type == "cpu" and torch_dtype in {torch.float16, torch.bfloat16}:
            torch_dtype = torch.float32

        self.latent_channels = 16
        self.text_embed_dim = 128
        self.image_embed_dim = 64
        self.vae_scale_factor_spatial = 8
        self.vae_scale_factor_temporal = 4
        self.transformer = diffusers.WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=(self.latent_channels * 2) + 1,
            out_channels=self.latent_channels,
            text_dim=self.text_embed_dim,
            freq_dim=32,
            ffn_dim=192,
            num_layers=2,
            image_dim=self.image_embed_dim,
            rope_max_seq_len=1024,
        ).to(self._execution_device, dtype=torch_dtype)

        resolved_scheduler_name = scheduler_name or "FlowMatchEulerDiscreteScheduler"
        scheduler_cls = getattr(diffusers, resolved_scheduler_name, None)
        if scheduler_cls is None:
            raise RuntimeError(f"Requested scheduler class is unavailable: {resolved_scheduler_name}")
        if resolved_scheduler_name == "FlowMatchEulerDiscreteScheduler":
            self.scheduler = scheduler_cls(shift=flow_shift or 5.0)
        elif resolved_scheduler_name == "UniPCMultistepScheduler":
            self.scheduler = scheduler_cls(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=flow_shift or 5.0)
        else:
            self.scheduler = scheduler_cls()

    @staticmethod
    def _normalize_prompt_list(prompt: str | list[str] | None, *, batch_size: int | None = None) -> list[str]:
        if prompt is None:
            size = batch_size or 1
            return [""] * size
        if isinstance(prompt, str):
            return [prompt]
        return list(prompt)

    def _embed_text(self, prompts: list[str], *, dtype: torch.dtype) -> torch.Tensor:
        seq_len = 16
        embeddings = torch.zeros(
            len(prompts),
            seq_len,
            self.text_embed_dim,
            device=self._execution_device,
            dtype=dtype,
        )
        base = torch.linspace(0.05, 1.0, self.text_embed_dim, device=self._execution_device, dtype=dtype)
        for batch_index, prompt in enumerate(prompts):
            codes = [ord(char) % 251 for char in prompt] or [0]
            for token_index in range(seq_len):
                code = codes[token_index % len(codes)]
                rolled = torch.roll(base, shifts=code % self.text_embed_dim)
                embeddings[batch_index, token_index] = rolled * (0.2 + (code / 300.0))
            embeddings[batch_index, :, 0] = float(len(prompt)) / max(len(codes), 1)
        return embeddings

    def encode_prompt(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del max_sequence_length
        if prompt_embeds is None:
            prompts = self._normalize_prompt_list(prompt)
            prompt_embeds = self._embed_text(prompts, dtype=dtype or self.transformer.dtype)
        else:
            prompt_embeds = prompt_embeds.to(device=device or self._execution_device, dtype=dtype or prompt_embeds.dtype)

        if num_videos_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

        if not do_classifier_free_guidance:
            return prompt_embeds, negative_prompt_embeds

        if negative_prompt_embeds is None:
            negative_prompts = self._normalize_prompt_list(negative_prompt, batch_size=prompt_embeds.shape[0])
            negative_prompt_embeds = self._embed_text(negative_prompts, dtype=prompt_embeds.dtype)
        else:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=device or self._execution_device,
                dtype=dtype or negative_prompt_embeds.dtype,
            )
        if num_videos_per_prompt > 1 and negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
        last_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del last_image
        latent_frames = ((num_frames - 1) // self.vae_scale_factor_temporal) + 1
        latent_height = max(height // self.vae_scale_factor_spatial, 1)
        latent_width = max(width // self.vae_scale_factor_spatial, 1)
        shape = (batch_size, num_channels_latents, latent_frames, latent_height, latent_width)
        if latents is None:
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError("generator list length must match batch size")
                parts = [
                    torch.randn((1, *shape[1:]), generator=single, device=device, dtype=dtype)
                    for single in generator
                ]
                return torch.cat(parts, dim=0)
            return torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return latents.to(device=device, dtype=dtype)

    def encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        if torch is None or F is None:
            raise RuntimeError("torch is required for smoke latent encoding")
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError("video tensor must be [F,C,H,W] or [B,F,C,H,W]")
        video = video.to(device=self._execution_device, dtype=torch.float32)
        video = video.permute(0, 2, 1, 3, 4)
        latent_frames = ((video.shape[2] - 1) // self.vae_scale_factor_temporal) + 1
        latent_height = max(video.shape[-2] // self.vae_scale_factor_spatial, 1)
        latent_width = max(video.shape[-1] // self.vae_scale_factor_spatial, 1)
        latents = F.interpolate(video, size=(latent_frames, latent_height, latent_width), mode="trilinear", align_corners=False)
        repeat_factor = math.ceil(self.latent_channels / latents.shape[1])
        latents = latents.repeat(1, repeat_factor, 1, 1, 1)[:, : self.latent_channels]
        return latents

    def decode_latents(self, latents: torch.Tensor, *, output_type: str = "tensor") -> Any:
        if torch is None or F is None:
            raise RuntimeError("torch is required for smoke latent decoding")
        frames = ((latents.shape[2] - 1) * self.vae_scale_factor_temporal) + 1
        height = latents.shape[3] * self.vae_scale_factor_spatial
        width = latents.shape[4] * self.vae_scale_factor_spatial
        video = latents[:, :3]
        video = F.interpolate(video, size=(frames, height, width), mode="trilinear", align_corners=False)
        video = video - video.amin(dim=(2, 3, 4), keepdim=True)
        video = video / torch.clamp(video.amax(dim=(2, 3, 4), keepdim=True), min=1e-6)
        video = video.permute(0, 2, 1, 3, 4)
        if output_type == "tensor":
            return video
        return video.detach().cpu().numpy()


class WanLoader:
    """Runtime-aware loader facade for the WanCanvas outpainting stack."""

    def __init__(self, runtime_config: RuntimeConfig | None = None, model_config: ModelConfig | None = None) -> None:
        self.runtime_config = runtime_config or RuntimeConfig()
        self.model_config = model_config or ModelConfig()
        self.runtime_config.validate()
        self.model_config.validate()

    def inspect_runtime(self) -> RuntimeInspection:
        return inspect_diffusers_runtime()

    def build_model_bundle(self) -> dict[str, Any]:
        self.model_config.validate()
        return {
            "base_model_id": self.model_config.base_model_id,
            "base_pipeline_class": self.model_config.base_pipeline_class,
            "vace_reference_model_id": self.model_config.vace_reference_model_id,
            "vace_pipeline_class": self.model_config.vace_pipeline_class,
            "strict_dense_mvp": self.model_config.strict_dense_mvp,
            "transformer_2": self.model_config.transformer_2,
            "boundary_ratio": self.model_config.boundary_ratio,
        }

    def smoke_validate(self, *, download_model: bool = False, strict_runtime: bool = False) -> WanLoaderReport:
        inspection = self.inspect_runtime()
        ready = stable_release_allowed(inspection)
        if strict_runtime and not ready:
            reason = "required Wan classes are not available in the current diffusers installation"
        elif download_model and not ready:
            reason = "runtime is not ready for model download"
        else:
            reason = "architecture/runtime smoke validation" if not download_model else None
        return WanLoaderReport(
            runtime=inspection.to_dict(),
            model=self.build_model_bundle(),
            ready_for_download=ready,
            strict_dense_mvp=self.model_config.strict_dense_mvp,
            download_attempted=download_model and ready,
            download_skipped_reason=reason,
            loaded_pipeline_class=self.model_config.base_pipeline_class if ready else None,
        )

    def _resolve_torch_dtype(self, torch_dtype: str | Any) -> tuple[Any, str]:
        if torch is None:
            raise RuntimeError("torch is required to resolve runtime dtypes")
        if isinstance(torch_dtype, str):
            normalized = torch_dtype.lower()
            mapping = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if normalized not in mapping:
                raise ValueError(f"Unsupported torch dtype string: {torch_dtype}")
            return mapping[normalized], normalized
        return torch_dtype, getattr(torch_dtype, "name", str(torch_dtype))

    @staticmethod
    def _model_cache_roots(cache_dir: str | None) -> list[Path]:
        roots: list[Path] = []
        if cache_dir:
            roots.append(Path(cache_dir).expanduser())
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        return roots

    @classmethod
    def _has_local_model_snapshot(cls, model_id: str, cache_dir: str | None) -> bool:
        model_slug = "models--" + model_id.replace("/", "--")
        for root in cls._model_cache_roots(cache_dir):
            snapshot_root = root / model_slug / "snapshots"
            if not snapshot_root.exists():
                continue
            if any(candidate.is_file() for candidate in snapshot_root.glob("*/model_index.json")):
                return True
        return False

    def _load_pretrained_pipeline(
        self,
        *,
        model_id: str | None,
        pipeline_class_name: str | None,
        device: str,
        torch_dtype: str | Any,
        cache_dir: str | None,
        local_files_only: bool,
        flow_shift: float | None,
        scheduler_name: str | None,
        use_fp32_vae: bool,
        enable_model_cpu_offload: bool,
    ) -> LoadedWanPipeline:
        report = self.smoke_validate(download_model=True, strict_runtime=True)
        if not report.ready_for_download:
            raise RuntimeError(
                "Wan runtime is not ready inside the current environment. Install a compatible diffusers build first."
            )

        import diffusers

        resolved_dtype, dtype_name = self._resolve_torch_dtype(torch_dtype)
        resolved_model_id = model_id or self.model_config.base_model_id
        resolved_pipeline_class = pipeline_class_name or self.model_config.base_pipeline_class

        if not hasattr(diffusers, resolved_pipeline_class):
            raise RuntimeError(f"Requested pipeline class is unavailable: {resolved_pipeline_class}")
        pipeline_cls = getattr(diffusers, resolved_pipeline_class)

        common_kwargs: dict[str, Any] = {
            "torch_dtype": resolved_dtype,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if use_fp32_vae and hasattr(diffusers, "AutoencoderKLWan"):
            vae = diffusers.AutoencoderKLWan.from_pretrained(
                resolved_model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            common_kwargs["vae"] = vae

        pipe = pipeline_cls.from_pretrained(resolved_model_id, **common_kwargs)
        if scheduler_name is not None:
            if not hasattr(diffusers, scheduler_name):
                raise RuntimeError(f"Requested scheduler class is unavailable: {scheduler_name}")
            scheduler_cls = getattr(diffusers, scheduler_name)
            if scheduler_name == "FlowMatchEulerDiscreteScheduler":
                pipe.scheduler = scheduler_cls(shift=flow_shift or 5.0)
            elif scheduler_name == "UniPCMultistepScheduler":
                pipe.scheduler = scheduler_cls(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=flow_shift or 5.0)
            else:
                pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        elif flow_shift is not None and hasattr(diffusers, "UniPCMultistepScheduler"):
            scheduler_config = getattr(getattr(pipe, "scheduler", None), "config", None)
            current_flow_shift = getattr(scheduler_config, "flow_shift", None)
            if current_flow_shift != flow_shift:
                pipe.scheduler = diffusers.UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    prediction_type="flow_prediction",
                    use_flow_sigmas=True,
                    flow_shift=flow_shift,
                )

        if enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        return LoadedWanPipeline(
            pipeline=pipe,
            runtime=self.inspect_runtime(),
            model_id=resolved_model_id,
            pipeline_class=resolved_pipeline_class,
            device=device,
            torch_dtype=dtype_name,
            flow_shift=flow_shift,
            runtime_variant="pretrained",
        )

    def _build_smoke_runtime(
        self,
        *,
        model_id: str | None,
        pipeline_class_name: str | None,
        device: str,
        torch_dtype: str | Any,
        flow_shift: float | None,
        scheduler_name: str | None,
    ) -> LoadedWanPipeline:
        resolved_dtype, dtype_name = self._resolve_torch_dtype(torch_dtype)
        runtime = SmokeWanPipeline(
            device=device,
            torch_dtype=resolved_dtype,
            scheduler_name=scheduler_name,
            flow_shift=flow_shift,
        )
        resolved_model_id = model_id or self.model_config.base_model_id
        resolved_pipeline_class = pipeline_class_name or self.model_config.base_pipeline_class
        return LoadedWanPipeline(
            pipeline=runtime,
            runtime=self.inspect_runtime(),
            model_id=resolved_model_id,
            pipeline_class=f"{resolved_pipeline_class}[smoke]",
            device=str(runtime._execution_device),
            torch_dtype=dtype_name,
            flow_shift=flow_shift,
            runtime_variant="smoke",
        )

    def load_pipeline(
        self,
        *,
        model_id: str | None = None,
        pipeline_class_name: str | None = None,
        device: str = "cuda",
        torch_dtype: str | Any = "bfloat16",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        flow_shift: float | None = 5.0,
        scheduler_name: str | None = None,
        use_fp32_vae: bool = True,
        enable_model_cpu_offload: bool = False,
        runtime_variant: str = "auto",
    ) -> LoadedWanPipeline:
        if runtime_variant not in {"auto", "pretrained", "smoke"}:
            raise ValueError(f"Unsupported runtime_variant: {runtime_variant}")

        resolved_model_id = model_id or self.model_config.base_model_id
        if runtime_variant == "smoke":
            return self._build_smoke_runtime(
                model_id=resolved_model_id,
                pipeline_class_name=pipeline_class_name,
                device=device,
                torch_dtype=torch_dtype,
                flow_shift=flow_shift,
                scheduler_name=scheduler_name,
            )

        if runtime_variant == "auto":
            has_local_snapshot = self._has_local_model_snapshot(resolved_model_id, cache_dir)
            if not has_local_snapshot:
                return self._build_smoke_runtime(
                    model_id=resolved_model_id,
                    pipeline_class_name=pipeline_class_name,
                    device=device,
                    torch_dtype=torch_dtype,
                    flow_shift=flow_shift,
                    scheduler_name=scheduler_name,
                )
            try:
                return self._load_pretrained_pipeline(
                    model_id=resolved_model_id,
                    pipeline_class_name=pipeline_class_name,
                    device=device,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    flow_shift=flow_shift,
                    scheduler_name=scheduler_name,
                    use_fp32_vae=use_fp32_vae,
                    enable_model_cpu_offload=enable_model_cpu_offload,
                )
            except Exception:
                return self._build_smoke_runtime(
                    model_id=resolved_model_id,
                    pipeline_class_name=pipeline_class_name,
                    device=device,
                    torch_dtype=torch_dtype,
                    flow_shift=flow_shift,
                    scheduler_name=scheduler_name,
                )

        return self._load_pretrained_pipeline(
            model_id=resolved_model_id,
            pipeline_class_name=pipeline_class_name,
            device=device,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            flow_shift=flow_shift,
            scheduler_name=scheduler_name,
            use_fp32_vae=use_fp32_vae,
            enable_model_cpu_offload=enable_model_cpu_offload,
        )
