from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = None

from ..backbones.wan_loader import WanLoader
from ..config_schema import ModelConfig
from .condition_adapter import ConditionAdapter


@dataclass(slots=True)
class WanForwardRequest:
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    noisy_latents: Any = None
    timesteps: Any = None
    prompt_embeds: Any = None
    negative_prompt_embeds: Any = None
    prompt_is_placeholder: bool = False
    layout_tokens: Any = None
    geometry_tokens: Any = None
    mask_tokens: Any = None
    condition_video: Any = None
    target_video: Any = None
    known_mask: Any = None
    condition_latents: Any = None
    target_latents: Any = None
    latent_mask: Any = None
    known_region_state: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WanForwardOutput:
    noise_pred: Any
    latents: Any
    latent_model_input: Any
    latent_mask: Any
    condition_latents: Any
    target_latents: Any
    timestep: Any
    conditioned_prompt_embeds: Any
    conditioned_negative_prompt_embeds: Any
    raw_condition_tokens: Any
    projected_condition_tokens: Any
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        def shape_of(value: Any) -> list[int] | None:
            shape = getattr(value, "shape", None)
            if shape is None:
                return None
            return [int(dim) for dim in shape]

        return {
            "noise_pred_shape": shape_of(self.noise_pred),
            "latents_shape": shape_of(self.latents),
            "latent_model_input_shape": shape_of(self.latent_model_input),
            "latent_mask_shape": shape_of(self.latent_mask),
            "condition_latents_shape": shape_of(self.condition_latents),
            "target_latents_shape": shape_of(self.target_latents),
            "timestep_shape": shape_of(self.timestep),
            "conditioned_prompt_embeds_shape": shape_of(self.conditioned_prompt_embeds),
            "conditioned_negative_prompt_embeds_shape": shape_of(self.conditioned_negative_prompt_embeds),
            "raw_condition_tokens_shape": shape_of(self.raw_condition_tokens),
            "projected_condition_tokens_shape": shape_of(self.projected_condition_tokens),
            "metadata": dict(self.metadata),
        }


class WanOutpaintWrapper(nn.Module if nn is not None else object):
    def __init__(
        self,
        loader: WanLoader,
        *,
        model_config: ModelConfig | None = None,
        condition_adapter: ConditionAdapter | None = None,
        fyc_token_dim: int = 1024,
    ) -> None:
        if nn is not None:
            super().__init__()
        self.loader = loader
        self.model_config = model_config or loader.model_config
        self.condition_adapter = condition_adapter or ConditionAdapter()
        self.fyc_token_dim = fyc_token_dim
        self.fyc_token_projector = None if nn is None else nn.Identity()
        self._loaded_runtime = None

    @staticmethod
    def _shape_of(value: Any) -> list[int] | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return [int(dim) for dim in shape]

    @staticmethod
    def _dtype_of(value: Any) -> str | None:
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            return None
        return str(dtype)

    @staticmethod
    def _is_tensor(value: Any) -> bool:
        return torch is not None and isinstance(value, torch.Tensor)

    @staticmethod
    def _ensure_torch() -> None:
        if torch is None or F is None:
            raise RuntimeError("torch is required for WanOutpaintWrapper")

    def _resolve_pipeline(self, runtime: Any = None) -> Any:
        resolved = runtime or self._loaded_runtime
        if resolved is None:
            resolved = self.load_runtime()
        return resolved.pipeline if hasattr(resolved, "pipeline") else resolved

    def load_runtime(self, **kwargs: Any) -> Any:
        self._loaded_runtime = self.loader.load_pipeline(**kwargs)
        return self._loaded_runtime

    @staticmethod
    def _cache_context(transformer: Any, name: str):
        cache_context = getattr(transformer, "cache_context", None)
        if cache_context is None:
            return nullcontext()
        return cache_context(name)

    @staticmethod
    def _ensure_bfchw(video: Any) -> torch.Tensor:
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"expected a torch.Tensor video, got {type(video)!r}")
        if video.ndim == 4:
            return video.unsqueeze(0)
        if video.ndim == 5:
            return video
        raise ValueError("video tensor must be [F,C,H,W] or [B,F,C,H,W]")

    def _ensure_latent_bcfhw(self, latents: Any) -> torch.Tensor:
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"expected a torch.Tensor latent tensor, got {type(latents)!r}")
        if latents.ndim != 5:
            raise ValueError("latent tensor must be rank 5")
        if latents.shape[1] <= 8 and latents.shape[2] >= 8:
            return latents.permute(0, 2, 1, 3, 4)
        return latents

    @staticmethod
    def _ensure_channel_first_mask(mask: Any) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0).unsqueeze(2)
        elif mask.ndim == 4:
            if mask.shape[1] == 1:
                mask = mask.unsqueeze(0)
            elif mask.shape[2] == 1:
                mask = mask.permute(0, 2, 1, 3, 4)
                return mask
            else:
                raise ValueError("4D mask must be [F,1,H,W] or [B,F,H,W]")
        elif mask.ndim == 5:
            if mask.shape[1] == 1:
                return mask
            if mask.shape[2] == 1:
                return mask.permute(0, 2, 1, 3, 4)
            raise ValueError("5D mask must be [B,1,F,H,W] or [B,F,1,H,W]")
        else:
            raise ValueError("mask must be 2D, 3D, 4D, or 5D")
        return mask

    def _match_latent_mask(self, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        mask = self._ensure_channel_first_mask(mask).to(device=reference.device, dtype=reference.dtype)
        batch, _, latent_frames, latent_h, latent_w = reference.shape
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1, -1, -1)
        if mask.shape[2] != latent_frames:
            frame_indices = torch.linspace(0, mask.shape[2] - 1, steps=latent_frames, device=mask.device)
            frame_indices = frame_indices.round().long()
            mask = mask.index_select(2, frame_indices)
        if mask.shape[-2:] != (latent_h, latent_w):
            mask_2d = mask.permute(0, 2, 1, 3, 4).reshape(mask.shape[0] * mask.shape[2], 1, mask.shape[3], mask.shape[4])
            mask_2d = F.interpolate(mask_2d, size=(latent_h, latent_w), mode="nearest")
            mask = mask_2d.view(mask.shape[0], latent_frames, 1, latent_h, latent_w).permute(0, 2, 1, 3, 4)
        return mask.clamp(0.0, 1.0)

    def encode_video_to_latents(self, video: Any, runtime: Any = None) -> torch.Tensor:
        self._ensure_torch()
        pipe = self._resolve_pipeline(runtime)
        if hasattr(pipe, "encode_video_to_latents"):
            return pipe.encode_video_to_latents(video).to(dtype=torch.float32)

        from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

        video_bfchw = self._ensure_bfchw(video)
        video_bcfhw = video_bfchw.permute(0, 2, 1, 3, 4).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
        encoded = retrieve_latents(pipe.vae.encode(video_bcfhw), sample_mode="argmax")
        latents_mean = torch.tensor(pipe.vae.config.latents_mean, device=encoded.device, dtype=encoded.dtype).view(1, pipe.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std, device=encoded.device, dtype=encoded.dtype).view(1, pipe.vae.config.z_dim, 1, 1, 1)
        return ((encoded - latents_mean) * latents_std).to(torch.float32)

    def decode_latents(self, latents: torch.Tensor, runtime: Any = None, *, output_type: str = "tensor") -> Any:
        self._ensure_torch()
        pipe = self._resolve_pipeline(runtime)
        if hasattr(pipe, "decode_latents"):
            return pipe.decode_latents(latents, output_type=output_type)

        denormalized = latents.to(device=pipe._execution_device, dtype=pipe.vae.dtype)
        latents_mean = torch.tensor(pipe.vae.config.latents_mean, device=denormalized.device, dtype=denormalized.dtype).view(1, pipe.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std, device=denormalized.device, dtype=denormalized.dtype).view(1, pipe.vae.config.z_dim, 1, 1, 1)
        denormalized = denormalized / latents_std + latents_mean
        decoded = pipe.vae.decode(denormalized, return_dict=False)[0]
        if output_type == "tensor":
            return decoded.permute(0, 2, 1, 3, 4)
        if hasattr(pipe, "video_processor"):
            return pipe.video_processor.postprocess_video(decoded, output_type=output_type)
        return decoded

    def _build_condition_bundle(self, request: WanForwardRequest) -> Any:
        return self.condition_adapter.build_bundle(
            layout_tokens=request.layout_tokens,
            geometry_tokens=request.geometry_tokens,
            mask_tokens=request.mask_tokens,
        )

    def _project_condition_tokens(self, fyc_tokens: torch.Tensor, *, target_dim: int) -> torch.Tensor:
        if fyc_tokens.shape[-1] == target_dim:
            return fyc_tokens
        if nn is None:
            raise RuntimeError("token projection requires torch.nn")
        projector = self.fyc_token_projector
        if not isinstance(projector, nn.Linear) or projector.in_features != fyc_tokens.shape[-1] or projector.out_features != target_dim:
            projector = nn.Linear(fyc_tokens.shape[-1], target_dim, bias=False)
            self.fyc_token_projector = projector
        projector = projector.to(device=fyc_tokens.device, dtype=fyc_tokens.dtype)
        self.fyc_token_projector = projector
        return projector(fyc_tokens)

    def _resolve_prompt_embeddings(
        self,
        request: WanForwardRequest,
        *,
        runtime: Any = None,
        do_classifier_free_guidance: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._ensure_torch()
        pipe = self._resolve_pipeline(runtime)
        use_runtime_encoding = request.prompt_embeds is None or request.prompt_is_placeholder
        if use_runtime_encoding:
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=request.prompt or request.extras.get("prompt", ""),
                negative_prompt=request.negative_prompt or request.extras.get("negative_prompt", ""),
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                device=getattr(pipe, "_execution_device", None),
                dtype=getattr(pipe.transformer, "dtype", torch.float32),
            )
        else:
            prompt_embeds = request.prompt_embeds
            negative_prompt_embeds = request.negative_prompt_embeds
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                _, negative_prompt_embeds = pipe.encode_prompt(
                    prompt=request.prompt or request.extras.get("prompt", ""),
                    negative_prompt=request.negative_prompt or request.extras.get("negative_prompt", ""),
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=1,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=None,
                    device=getattr(pipe, "_execution_device", None),
                    dtype=getattr(pipe.transformer, "dtype", torch.float32),
                )
        prompt_embeds = prompt_embeds.to(device=pipe._execution_device, dtype=pipe.transformer.dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=pipe._execution_device, dtype=pipe.transformer.dtype)
        return prompt_embeds, negative_prompt_embeds

    def _resolve_condition_latents(
        self,
        request: WanForwardRequest,
        *,
        runtime: Any = None,
        reference_latents: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        self._ensure_torch()
        if request.condition_latents is not None:
            latents = self._ensure_latent_bcfhw(request.condition_latents)
        elif request.condition_video is not None:
            latents = self.encode_video_to_latents(request.condition_video, runtime=runtime)
        else:
            latents = None
        if latents is None:
            return None
        if reference_latents is not None:
            latents = latents.to(device=reference_latents.device, dtype=reference_latents.dtype)
            if latents.shape != reference_latents.shape:
                raise ValueError(
                    f"condition_latents shape {tuple(latents.shape)} must match latents {tuple(reference_latents.shape)}"
                )
        return latents

    def _resolve_target_latents(self, request: WanForwardRequest, *, runtime: Any = None) -> torch.Tensor | None:
        self._ensure_torch()
        if request.target_latents is not None:
            return self._ensure_latent_bcfhw(request.target_latents)
        if request.target_video is None:
            return None
        return self.encode_video_to_latents(request.target_video, runtime=runtime)

    def _resolve_noisy_latents(self, request: WanForwardRequest, *, runtime: Any = None) -> torch.Tensor:
        self._ensure_torch()
        if request.noisy_latents is not None:
            return self._ensure_latent_bcfhw(request.noisy_latents)

        pipe = self._resolve_pipeline(runtime)
        frame_count = int(request.extras.get("frame_count"))
        target_hw = request.extras.get("aligned_target_hw", request.extras.get("target_hw"))
        if target_hw is None:
            raise ValueError("request.extras must include target_hw or aligned_target_hw when noisy_latents is omitted")
        height = int(target_hw[0])
        width = int(target_hw[1])
        latent_channels = int(getattr(getattr(pipe, "transformer", None), "config", None).out_channels)
        latents = pipe.prepare_latents(
            1,
            latent_channels,
            height,
            width,
            frame_count,
            torch.float32,
            pipe._execution_device,
            None,
            None,
        )
        if isinstance(latents, tuple):
            latents = latents[0]
        return self._ensure_latent_bcfhw(latents)

    def _resolve_latent_mask(self, request: WanForwardRequest, *, latents: torch.Tensor) -> torch.Tensor:
        self._ensure_torch()
        source = request.latent_mask if request.latent_mask is not None else request.known_mask
        if source is None:
            return torch.ones((latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]), device=latents.device, dtype=latents.dtype)
        return self._match_latent_mask(source, latents)

    @staticmethod
    def _build_transformer_inputs(
        latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
        latent_mask: torch.Tensor,
        *,
        target_channels: int,
    ) -> torch.Tensor:
        if target_channels <= latents.shape[1]:
            return latents

        expected_condition_channels = target_channels - latents.shape[1]
        if condition_latents is None:
            condition_latents = torch.zeros_like(latents)
        available = torch.cat([latent_mask, condition_latents], dim=1)
        if available.shape[1] < expected_condition_channels:
            padding = torch.zeros(
                available.shape[0],
                expected_condition_channels - available.shape[1],
                available.shape[2],
                available.shape[3],
                available.shape[4],
                device=available.device,
                dtype=available.dtype,
            )
            available = torch.cat([available, padding], dim=1)
        else:
            available = available[:, :expected_condition_channels]
        return torch.cat([latents, available], dim=1)

    @staticmethod
    def _merge_output_latents(
        latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        if condition_latents is None:
            return latents
        return condition_latents * (1.0 - latent_mask) + latents * latent_mask

    @staticmethod
    def _expand_timestep_for_transformer(timesteps: torch.Tensor, batch_size: int) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.view(1)
        if timesteps.ndim == 1 and timesteps.numel() == 1:
            timesteps = timesteps.expand(batch_size)
        if timesteps.ndim not in {1, 2}:
            raise ValueError("timesteps must be scalar, 1D, or 2D")
        return timesteps

    @staticmethod
    def _scheduler_sigma(scheduler: Any, timestep: torch.Tensor, *, sample: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep.view(1)
        if timestep.ndim == 2:
            timestep = timestep[:, 0]
        schedule_timesteps = scheduler.timesteps.to(sample.device)
        sigma_values = scheduler.sigmas.to(device=sample.device, dtype=sample.dtype)
        indices = [scheduler.index_for_timestep(t, schedule_timesteps) for t in timestep]
        sigma = sigma_values[indices].flatten()
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def reconstruct_clean_latents(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        *,
        runtime: Any = None,
    ) -> torch.Tensor:
        pipe = self._resolve_pipeline(runtime)
        sigma = self._scheduler_sigma(pipe.scheduler, timestep, sample=noisy_latents)
        denominator = torch.clamp(1.0 - sigma, min=1e-6)
        return (noisy_latents - sigma * noise_pred) / denominator

    def validate_request_contract(self, request: WanForwardRequest) -> dict[str, Any]:
        prompt_shape = self._shape_of(request.prompt_embeds)
        latents_shape = self._shape_of(request.noisy_latents)
        timesteps_shape = self._shape_of(request.timesteps)
        checks: dict[str, bool] = {
            "prompt_or_text_available": bool((request.prompt and str(request.prompt).strip()) or self._is_tensor(request.prompt_embeds)),
            "prompt_embeds_optional_or_rank_3": request.prompt_embeds is None or bool(prompt_shape and len(prompt_shape) == 3),
            "noisy_latents_optional_or_rank_5": request.noisy_latents is None or bool(latents_shape and len(latents_shape) == 5),
            "timesteps_optional_or_tensor": request.timesteps is None or self._is_tensor(request.timesteps),
            "timesteps_optional_or_supported_rank": request.timesteps is None or bool(self._is_tensor(request.timesteps) and request.timesteps.ndim in {0, 1, 2}),
            "layout_tokens_optional_or_rank_3": request.layout_tokens is None or bool(self._shape_of(request.layout_tokens) and len(self._shape_of(request.layout_tokens)) == 3),
            "geometry_tokens_optional_or_rank_3": request.geometry_tokens is None or bool(self._shape_of(request.geometry_tokens) and len(self._shape_of(request.geometry_tokens)) == 3),
            "mask_tokens_optional_or_rank_3": request.mask_tokens is None or bool(self._shape_of(request.mask_tokens) and len(self._shape_of(request.mask_tokens)) == 3),
            "videos_optional_or_tensor": all(value is None or self._is_tensor(value) for value in (request.condition_video, request.target_video)),
            "known_region_state_optional_or_valid": request.known_region_state is None or (
                isinstance(request.known_region_state, dict) and request.known_region_state.get("mode") in {"overwrite", "blend"}
            ),
        }
        return {
            "shapes": {
                "prompt_embeds": prompt_shape,
                "negative_prompt_embeds": self._shape_of(request.negative_prompt_embeds),
                "noisy_latents": latents_shape,
                "timesteps": timesteps_shape,
                "layout_tokens": self._shape_of(request.layout_tokens),
                "geometry_tokens": self._shape_of(request.geometry_tokens),
                "mask_tokens": self._shape_of(request.mask_tokens),
                "condition_video": self._shape_of(request.condition_video),
                "target_video": self._shape_of(request.target_video),
                "known_mask": self._shape_of(request.known_mask),
                "condition_latents": self._shape_of(request.condition_latents),
                "target_latents": self._shape_of(request.target_latents),
                "latent_mask": self._shape_of(request.latent_mask),
            },
            "dtypes": {
                "prompt_embeds": self._dtype_of(request.prompt_embeds),
                "negative_prompt_embeds": self._dtype_of(request.negative_prompt_embeds),
                "noisy_latents": self._dtype_of(request.noisy_latents),
                "timesteps": self._dtype_of(request.timesteps),
                "layout_tokens": self._dtype_of(request.layout_tokens),
                "geometry_tokens": self._dtype_of(request.geometry_tokens),
                "mask_tokens": self._dtype_of(request.mask_tokens),
                "condition_video": self._dtype_of(request.condition_video),
                "target_video": self._dtype_of(request.target_video),
                "known_mask": self._dtype_of(request.known_mask),
                "condition_latents": self._dtype_of(request.condition_latents),
                "target_latents": self._dtype_of(request.target_latents),
                "latent_mask": self._dtype_of(request.latent_mask),
            },
            "checks": checks,
        }

    def prepare_inputs(self, request: WanForwardRequest) -> dict[str, Any]:
        bundle = self._build_condition_bundle(request)
        concatenated = self.condition_adapter.concat_bundle(bundle)
        semantic_roles = {
            "layout": "layout_encoder",
            "geometry": "relative_region_embedding",
            "mask": "known_region_mask_summary",
        }
        return {
            "base_model_id": self.model_config.base_model_id,
            "base_pipeline_class": self.model_config.base_pipeline_class,
            "strict_dense_mvp": self.model_config.strict_dense_mvp,
            "condition_bundle": {
                "order": bundle.order,
                "metadata": bundle.metadata,
                "token_keys": list(bundle.tokens.keys()),
                "semantic_roles": [semantic_roles[name] for name in bundle.order],
                "token_shapes": {name: self._shape_of(bundle.tokens[name]) for name in bundle.order},
                "concat_shape": self._shape_of(concatenated),
                "consumption_path": "encoder_hidden_states_image",
            },
            "conditioning_path": {
                "text_prompt_encoded_at_runtime": request.prompt_embeds is None or request.prompt_is_placeholder,
                "image_conditioning_stream": True,
                "known_region_latent_path": request.condition_video is not None or request.condition_latents is not None,
            },
            "request_contract": self.validate_request_contract(request),
            "has_known_region_state": request.known_region_state is not None,
            "extras": dict(request.extras),
        }

    def dry_run(self, request: WanForwardRequest) -> dict[str, Any]:
        report = self.loader.smoke_validate(download_model=False, strict_runtime=False)
        payload = self.prepare_inputs(request)
        payload["runtime"] = report.runtime
        payload["ready_for_download"] = report.ready_for_download
        payload["download_skipped_reason"] = report.download_skipped_reason
        return payload

    def describe_request(self, request: WanForwardRequest) -> dict[str, Any]:
        return {
            "extras": dict(request.extras),
            "has_known_region_state": request.known_region_state is not None,
            "prompt": request.prompt,
            "prompt_embeds_shape": self._shape_of(request.prompt_embeds),
            "noisy_latents_shape": self._shape_of(request.noisy_latents),
            "timesteps_shape": self._shape_of(request.timesteps),
            "layout_tokens_shape": self._shape_of(request.layout_tokens),
            "geometry_tokens_shape": self._shape_of(request.geometry_tokens),
            "mask_tokens_shape": self._shape_of(request.mask_tokens),
            "condition_video_shape": self._shape_of(request.condition_video),
            "target_video_shape": self._shape_of(request.target_video),
            "known_mask_shape": self._shape_of(request.known_mask),
            "latent_mask_shape": self._shape_of(request.latent_mask),
            "condition_latents_shape": self._shape_of(request.condition_latents),
            "target_latents_shape": self._shape_of(request.target_latents),
        }

    def forward(
        self,
        request: WanForwardRequest,
        *,
        runtime: Any = None,
        guidance_scale: float = 1.0,
        do_classifier_free_guidance: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> WanForwardOutput:
        self._ensure_torch()
        pipe = self._resolve_pipeline(runtime)
        prompt_embeds, negative_prompt_embeds = self._resolve_prompt_embeddings(
            request,
            runtime=pipe,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        latents = self._resolve_noisy_latents(request, runtime=pipe).to(device=pipe._execution_device, dtype=torch.float32)
        condition_latents = self._resolve_condition_latents(request, runtime=pipe, reference_latents=latents)
        target_latents = self._resolve_target_latents(request, runtime=pipe)
        latent_mask = self._resolve_latent_mask(request, latents=latents)
        target_channels = int(getattr(pipe.transformer.config, "in_channels", latents.shape[1]))
        latent_model_input = self._build_transformer_inputs(
            latents,
            condition_latents,
            latent_mask,
            target_channels=target_channels,
        ).to(device=pipe._execution_device, dtype=pipe.transformer.dtype)

        if request.timesteps is None:
            raise ValueError("WanForwardRequest.timesteps must be provided for forward()")
        timestep = request.timesteps.to(device=pipe._execution_device)
        timestep = self._expand_timestep_for_transformer(timestep, latents.shape[0])

        bundle = self._build_condition_bundle(request)
        raw_condition_tokens = self.condition_adapter.concat_bundle(bundle) if bundle.order else None
        projected_condition_tokens = None
        if raw_condition_tokens is not None:
            raw_condition_tokens = raw_condition_tokens.to(device=pipe._execution_device, dtype=pipe.transformer.dtype)
            image_dim = getattr(pipe.transformer.config, "image_dim", None) or raw_condition_tokens.shape[-1]
            projected_condition_tokens = self._project_condition_tokens(raw_condition_tokens, target_dim=image_dim)

        transformer_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "attention_kwargs": attention_kwargs,
            "return_dict": False,
        }
        if projected_condition_tokens is not None:
            transformer_kwargs["encoder_hidden_states_image"] = projected_condition_tokens

        with self._cache_context(pipe.transformer, "cond"):
            noise_pred = pipe.transformer(**transformer_kwargs)[0]
        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            uncond_kwargs = dict(transformer_kwargs)
            uncond_kwargs["encoder_hidden_states"] = negative_prompt_embeds
            with self._cache_context(pipe.transformer, "uncond"):
                noise_uncond = pipe.transformer(**uncond_kwargs)[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        return WanForwardOutput(
            noise_pred=noise_pred,
            latents=latents,
            latent_model_input=latent_model_input,
            latent_mask=latent_mask,
            condition_latents=condition_latents,
            target_latents=target_latents,
            timestep=timestep,
            conditioned_prompt_embeds=prompt_embeds,
            conditioned_negative_prompt_embeds=negative_prompt_embeds,
            raw_condition_tokens=raw_condition_tokens,
            projected_condition_tokens=projected_condition_tokens,
            metadata={
                "consumption_path": "encoder_hidden_states_image" if projected_condition_tokens is not None else "text_only",
                "known_region_latent_path": condition_latents is not None,
                "guidance_scale": guidance_scale,
                "do_classifier_free_guidance": do_classifier_free_guidance,
                "prompt_token_count": int(prompt_embeds.shape[1]),
                "image_condition_token_count": int(projected_condition_tokens.shape[1]) if projected_condition_tokens is not None else 0,
            },
        )

    def generate(
        self,
        request: WanForwardRequest,
        *,
        runtime: Any = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        attention_kwargs: dict[str, Any] | None = None,
        output_type: str = "tensor",
    ) -> dict[str, Any]:
        self._ensure_torch()
        pipe = self._resolve_pipeline(runtime)
        latents = self._resolve_noisy_latents(request, runtime=pipe).to(device=pipe._execution_device, dtype=torch.float32)
        pipe.scheduler.set_timesteps(num_inference_steps, device=pipe._execution_device)
        working_request = WanForwardRequest(**{field: getattr(request, field) for field in request.__dataclass_fields__})
        working_request.noisy_latents = latents
        last_forward = None
        for timestep in pipe.scheduler.timesteps:
            working_request.timesteps = timestep.view(1)
            last_forward = self.forward(
                working_request,
                runtime=pipe,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=guidance_scale > 1.0,
                attention_kwargs=attention_kwargs,
            )
            latents = pipe.scheduler.step(last_forward.noise_pred, timestep, latents, return_dict=False)[0]
            latents = self._merge_output_latents(latents, last_forward.condition_latents, last_forward.latent_mask)
            working_request.noisy_latents = latents
        if last_forward is None:
            raise RuntimeError("generate() requires at least one inference step")
        video = latents if output_type == "latent" else self.decode_latents(latents, runtime=pipe, output_type=output_type)
        return {
            "frames": video,
            "latents": latents,
            "forward": last_forward.to_dict(),
        }
