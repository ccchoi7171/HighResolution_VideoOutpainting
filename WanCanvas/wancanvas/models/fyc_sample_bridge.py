from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from ..data.contracts import FYCOutpaintSample
from ..data.outpaint_dataset import CropReference
from ..pipelines.size_alignment import SizeAlignmentRule, snap_spatial_size
from ..utils.latent_ops import estimate_latent_frames, estimate_latent_hw
from .fyc_conditioning import FYCConditioningBuilder, FYCConditioningOutput
from .wan_outpaint_wrapper import WanForwardRequest, WanOutpaintWrapper


@dataclass(slots=True)
class FYCSampleBridgeConfig:
    token_dim: int = 1024
    synthetic_anchor_fill: float = 0.0
    synthetic_target_fill: float = 0.0
    latent_channels: int = 16
    timestep_value: int = 999
    vae_scale_factor_temporal: int = 4

    def validate(self) -> None:
        if self.token_dim <= 0 or self.latent_channels <= 0:
            raise ValueError("token_dim and latent_channels must be positive")


@dataclass(slots=True)
class FYCSampleBridgeOutput:
    conditioning: FYCConditioningOutput
    request: WanForwardRequest
    wrapper_payload: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditioning": self.conditioning.to_dict(),
            "request": {
                "extras": dict(self.request.extras),
                "has_known_region_state": self.request.known_region_state is not None,
                "noisy_latents_shape": list(self.request.noisy_latents.shape) if hasattr(self.request.noisy_latents, "shape") else None,
                "timesteps_shape": list(self.request.timesteps.shape) if hasattr(self.request.timesteps, "shape") else None,
                "prompt_embeds_shape": list(self.request.prompt_embeds.shape) if hasattr(self.request.prompt_embeds, "shape") else None,
                "condition_video_shape": list(self.request.condition_video.shape) if hasattr(self.request.condition_video, "shape") else None,
                "target_video_shape": list(self.request.target_video.shape) if hasattr(self.request.target_video, "shape") else None,
                "known_mask_shape": list(self.request.known_mask.shape) if hasattr(self.request.known_mask, "shape") else None,
                "latent_mask_shape": list(self.request.latent_mask.shape) if hasattr(self.request.latent_mask, "shape") else None,
            },
            "wrapper_payload": self.wrapper_payload,
            "metadata": dict(self.metadata),
        }


class FYCSampleToWanBridge:
    def __init__(
        self,
        *,
        conditioning_builder: FYCConditioningBuilder | None = None,
        wrapper: WanOutpaintWrapper | None = None,
        config: FYCSampleBridgeConfig | None = None,
    ) -> None:
        self.conditioning_builder = conditioning_builder or FYCConditioningBuilder()
        self.wrapper = wrapper or WanOutpaintWrapper.__new__(WanOutpaintWrapper)
        if not hasattr(self.wrapper, "condition_adapter"):
            from ..backbones.wan_loader import WanLoader

            self.wrapper = WanOutpaintWrapper(WanLoader())
        self.config = config or FYCSampleBridgeConfig()
        self.config.validate()

    @staticmethod
    def _ensure_torch() -> None:
        if torch is None or F is None:
            raise RuntimeError("torch is required for FYCSampleToWanBridge")

    def _video_tensor(self, value: Any, *, fill_value: float, source_label: str) -> tuple[torch.Tensor, dict[str, Any]]:
        self._ensure_torch()
        if isinstance(value, torch.Tensor):
            if value.ndim == 4:
                return value.unsqueeze(0).float(), {f"{source_label}_source": "tensor-frames", "source": f"{source_label}-tensor-frames"}
            if value.ndim == 5:
                return value.float(), {f"{source_label}_source": "tensor-batch", "source": f"{source_label}-tensor-batch"}
            raise ValueError(f"{source_label} tensor must be [F,C,H,W] or [B,F,C,H,W]")
        if isinstance(value, CropReference):
            region = value.region
            tensor = torch.full(
                (1, value.frame_count, 3, region.height, region.width),
                fill_value=float(fill_value),
                dtype=torch.float32,
            )
            return tensor, {
                f"{source_label}_source": "crop-reference-synthetic",
                "source": f"{source_label}-crop-reference-synthetic",
                "source_id": value.source_id,
                "region": asdict(region),
            }
        raise TypeError(f"Unsupported {source_label} type: {type(value)!r}")

    def _known_mask_tensor(self, sample: FYCOutpaintSample) -> torch.Tensor:
        self._ensure_torch()
        mask = sample.known_mask
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 4:
                return mask.unsqueeze(0).float()
            if mask.ndim == 5:
                return mask.float()
            raise ValueError("known_mask tensor must be [F,1,H,W] or [B,F,1,H,W]")
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        if mask_tensor.ndim != 2:
            raise ValueError("known_mask list must be 2D [H,W]")
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return mask_tensor.repeat(1, sample.frame_count, 1, 1, 1)

    def _relative_position_tensor(self, sample: FYCOutpaintSample) -> torch.Tensor:
        self._ensure_torch()
        return torch.tensor([list(sample.relative_position_norm)], dtype=torch.float32)

    def _target_hw(self, sample: FYCOutpaintSample) -> tuple[int, int]:
        target_region = sample.canvas_meta.target_region
        if target_region is not None:
            return target_region.height, target_region.width
        mask = sample.known_mask
        if isinstance(mask, torch.Tensor):
            return int(mask.shape[-2]), int(mask.shape[-1])
        return len(mask), len(mask[0])

    def _latent_request_tensors(self, sample: FYCOutpaintSample) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._ensure_torch()
        target_hw = self._target_hw(sample)
        rule = SizeAlignmentRule()
        aligned_target_hw = snap_spatial_size(*target_hw, rule, mode="ceil")
        latent_hw = estimate_latent_hw(*aligned_target_hw, vae_scale_factor_spatial=rule.vae_scale_factor_spatial)
        latent_frames = estimate_latent_frames(sample.frame_count, vae_scale_factor_temporal=self.config.vae_scale_factor_temporal)
        noisy_latents = torch.zeros(
            (1, self.config.latent_channels, latent_frames, latent_hw[0], latent_hw[1]),
            dtype=torch.float32,
        )
        timesteps = torch.tensor([self.config.timestep_value], dtype=torch.int64)
        metadata = {
            "target_hw": list(target_hw),
            "aligned_target_hw": list(aligned_target_hw),
            "latent_hw": list(latent_hw),
            "latent_frames": latent_frames,
            "latent_channels": self.config.latent_channels,
            "timestep_value": self.config.timestep_value,
        }
        return noisy_latents, timesteps, metadata

    def _known_region_state(self, sample: FYCOutpaintSample) -> dict[str, Any]:
        anchor_region = sample.canvas_meta.anchor_region
        target_region = sample.canvas_meta.target_region
        return {
            "mode": "overwrite",
            "canvas_height": sample.canvas_meta.canvas_height,
            "canvas_width": sample.canvas_meta.canvas_width,
            "anchor_region": asdict(anchor_region) if anchor_region else None,
            "target_region": asdict(target_region) if target_region else None,
        }

    def _build_condition_video(
        self,
        *,
        sample: FYCOutpaintSample,
        anchor_video: torch.Tensor,
        target_video: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self._ensure_torch()
        condition_video = torch.zeros_like(target_video)
        anchor_region = sample.canvas_meta.anchor_region
        target_region = sample.canvas_meta.target_region
        if anchor_region is None or target_region is None:
            return condition_video, {"copied_overlap": None}
        overlap = anchor_region.intersection(target_region)
        if overlap is None:
            return condition_video, {"copied_overlap": None}
        anchor_local = overlap.to_local(anchor_region)
        target_local = overlap.to_local(target_region)
        condition_video[:, :, :, target_local.top:target_local.bottom, target_local.left:target_local.right] = anchor_video[
            :, :, :, anchor_local.top:anchor_local.bottom, anchor_local.left:anchor_local.right
        ]
        return condition_video, {
            "copied_overlap": asdict(overlap),
            "anchor_overlap_local": asdict(anchor_local),
            "target_overlap_local": asdict(target_local),
        }

    def _build_latent_mask(self, known_mask: torch.Tensor, request_tensor_meta: dict[str, Any]) -> torch.Tensor:
        self._ensure_torch()
        latent_frames = request_tensor_meta["latent_frames"]
        latent_h, latent_w = request_tensor_meta["latent_hw"]
        mask = known_mask.permute(0, 2, 1, 3, 4)
        if mask.shape[2] != latent_frames:
            frame_indices = torch.linspace(0, mask.shape[2] - 1, steps=latent_frames).round().long()
            mask = mask.index_select(2, frame_indices)
        mask_2d = mask.permute(0, 2, 1, 3, 4).reshape(mask.shape[0] * mask.shape[2], 1, mask.shape[3], mask.shape[4])
        mask_2d = F.interpolate(mask_2d, size=(latent_h, latent_w), mode="nearest")
        return mask_2d.view(mask.shape[0], latent_frames, 1, latent_h, latent_w).permute(0, 2, 1, 3, 4)

    def build(self, sample: FYCOutpaintSample) -> FYCSampleBridgeOutput:
        anchor_video, anchor_meta = self._video_tensor(sample.anchor_video, fill_value=self.config.synthetic_anchor_fill, source_label="anchor")
        target_video, target_meta = self._video_tensor(sample.target_video, fill_value=self.config.synthetic_target_fill, source_label="target")
        known_mask = self._known_mask_tensor(sample)
        condition_video, condition_meta = self._build_condition_video(sample=sample, anchor_video=anchor_video, target_video=target_video)
        relative_position = self._relative_position_tensor(sample)
        noisy_latents, timesteps, request_tensor_meta = self._latent_request_tensors(sample)
        latent_mask = self._build_latent_mask(known_mask, request_tensor_meta)

        conditioning = self.conditioning_builder.encode(
            anchor_video=anchor_video,
            relative_position=relative_position,
            known_mask=known_mask,
            prompt_embeds=None,
        )
        request = WanForwardRequest(
            prompt=sample.prompt,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            layout_tokens=conditioning.layout.tokens if conditioning.layout is not None else None,
            geometry_tokens=conditioning.geometry.tokens if conditioning.geometry is not None else None,
            mask_tokens=conditioning.mask.tokens if conditioning.mask is not None else None,
            condition_video=condition_video,
            target_video=target_video,
            known_mask=known_mask,
            latent_mask=latent_mask,
            known_region_state=self._known_region_state(sample),
            extras={
                "prompt": sample.prompt,
                "fps": sample.fps,
                "frame_count": sample.frame_count,
                "relative_position_raw": list(sample.relative_position_raw),
                "relative_position_norm": list(sample.relative_position_norm),
                "target_hw": request_tensor_meta["target_hw"],
                "aligned_target_hw": request_tensor_meta["aligned_target_hw"],
                "latent_hw": request_tensor_meta["latent_hw"],
                "latent_frames": request_tensor_meta["latent_frames"],
                "latent_channels": request_tensor_meta["latent_channels"],
            },
        )
        wrapper_payload = self.wrapper.prepare_inputs(request)
        metadata = {
            "anchor_meta": anchor_meta,
            "target_meta": target_meta,
            "condition_meta": condition_meta,
            "mask_generate_ratio": conditioning.metadata.get("mask_aux", {}).get("generate_ratio") if conditioning.mask else None,
            "request_tensors": request_tensor_meta,
            "canvas_meta": {
                "canvas_height": sample.canvas_meta.canvas_height,
                "canvas_width": sample.canvas_meta.canvas_width,
                "anchor_region": asdict(sample.canvas_meta.anchor_region) if sample.canvas_meta.anchor_region else None,
                "target_region": asdict(sample.canvas_meta.target_region) if sample.canvas_meta.target_region else None,
            },
        }
        return FYCSampleBridgeOutput(
            conditioning=conditioning,
            request=request,
            wrapper_payload=wrapper_payload,
            metadata=metadata,
        )
