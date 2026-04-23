from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..data.contracts import FYCOutpaintSample, Rect
from ..data.outpaint_dataset import CropReference
from ..pipelines.size_alignment import SizeAlignmentRule, snap_spatial_size
from ..utils.latent_ops import estimate_latent_hw
from .fyc_conditioning import FYCConditioningBuilder, FYCConditioningOutput
from .wan_outpaint_wrapper import WanForwardRequest, WanOutpaintWrapper


@dataclass(slots=True)
class FYCSampleBridgeConfig:
    prompt_token_count: int = 16
    token_dim: int = 1024
    synthetic_anchor_fill: float = 0.0
    latent_channels: int = 16
    timestep_value: int = 999

    def validate(self) -> None:
        if self.prompt_token_count <= 0 or self.token_dim <= 0 or self.latent_channels <= 0:
            raise ValueError('prompt_token_count, token_dim, and latent_channels must be positive')


@dataclass(slots=True)
class FYCSampleBridgeOutput:
    conditioning: FYCConditioningOutput
    request: WanForwardRequest
    wrapper_payload: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'conditioning': self.conditioning.to_dict(),
            'request': {
                'extras': dict(self.request.extras),
                'has_known_region_state': self.request.known_region_state is not None,
                'noisy_latents_shape': list(self.request.noisy_latents.shape) if hasattr(self.request.noisy_latents, 'shape') else None,
                'timesteps_shape': list(self.request.timesteps.shape) if hasattr(self.request.timesteps, 'shape') else None,
                'prompt_embeds_shape': list(self.request.prompt_embeds.shape) if hasattr(self.request.prompt_embeds, 'shape') else None,
            },
            'wrapper_payload': self.wrapper_payload,
            'metadata': dict(self.metadata),
        }


class FYCSampleToWanBridge:
    """End-to-end bridge from the FYC sample contract to Wan-side wrapper payloads.

    This proves that the repository-level FYC sample structure can be transformed into
    the DiT-facing condition/request surface without inventing a second parallel contract.
    """

    def __init__(
        self,
        *,
        conditioning_builder: FYCConditioningBuilder | None = None,
        wrapper: WanOutpaintWrapper | None = None,
        config: FYCSampleBridgeConfig | None = None,
    ) -> None:
        self.conditioning_builder = conditioning_builder or FYCConditioningBuilder()
        self.wrapper = wrapper or WanOutpaintWrapper.__new__(WanOutpaintWrapper)
        if not hasattr(self.wrapper, 'condition_adapter'):
            # When caller doesn't provide a real wrapper, create a standard instance.
            from ..backbones.wan_loader import WanLoader
            self.wrapper = WanOutpaintWrapper(WanLoader())
        self.config = config or FYCSampleBridgeConfig()
        self.config.validate()

    @staticmethod
    def _ensure_torch() -> None:
        if torch is None:
            raise RuntimeError('torch is required for FYCSampleToWanBridge')

    def _prompt_embeds(self, sample: FYCOutpaintSample) -> 'torch.Tensor':
        self._ensure_torch()
        batch = 1
        # Deterministic placeholder embed that still keeps explicit shape contract.
        base = torch.zeros(batch, self.config.prompt_token_count, self.config.token_dim, dtype=torch.float32)
        if sample.prompt:
            base[:, 0, 0] = float(len(sample.prompt))
        return base

    def _anchor_video_tensor(self, sample: FYCOutpaintSample) -> tuple['torch.Tensor', dict[str, Any]]:
        self._ensure_torch()
        anchor = sample.anchor_video
        if isinstance(anchor, torch.Tensor):
            if anchor.ndim == 4:
                return anchor.unsqueeze(0), {'anchor_source': 'tensor-frames'}
            if anchor.ndim == 5:
                return anchor, {'anchor_source': 'tensor-batch'}
            raise ValueError('anchor tensor must be [F,C,H,W] or [B,F,C,H,W]')
        if isinstance(anchor, CropReference):
            region = anchor.region
            tensor = torch.full(
                (1, anchor.frame_count, 3, region.height, region.width),
                fill_value=float(self.config.synthetic_anchor_fill),
                dtype=torch.float32,
            )
            return tensor, {
                'anchor_source': 'crop-reference-synthetic',
                'source_id': anchor.source_id,
                'region': asdict(region),
            }
        raise TypeError(f'Unsupported anchor_video type: {type(anchor)!r}')

    def _known_mask_tensor(self, sample: FYCOutpaintSample) -> 'torch.Tensor':
        self._ensure_torch()
        mask = sample.known_mask
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 4:
                return mask.unsqueeze(0)
            if mask.ndim == 5:
                return mask
            raise ValueError('known_mask tensor must be [F,1,H,W] or [B,F,1,H,W]')
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        if mask_tensor.ndim != 2:
            raise ValueError('known_mask list must be 2D [H,W]')
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask_tensor = mask_tensor.repeat(1, sample.frame_count, 1, 1, 1)
        return mask_tensor

    def _relative_position_tensor(self, sample: FYCOutpaintSample) -> 'torch.Tensor':
        self._ensure_torch()
        return torch.tensor([list(sample.relative_position_norm)], dtype=torch.float32)

    @staticmethod
    def _target_hw(sample: FYCOutpaintSample) -> tuple[int, int]:
        target_region = sample.canvas_meta.target_region
        if target_region is not None:
            return target_region.height, target_region.width
        mask = sample.known_mask
        if torch is not None and isinstance(mask, torch.Tensor):
            return int(mask.shape[-2]), int(mask.shape[-1])
        return len(mask), len(mask[0])

    def _latent_request_tensors(self, sample: FYCOutpaintSample) -> tuple['torch.Tensor', 'torch.Tensor', dict[str, Any]]:
        self._ensure_torch()
        target_hw = self._target_hw(sample)
        rule = SizeAlignmentRule()
        aligned_target_hw = snap_spatial_size(*target_hw, rule, mode='ceil')
        latent_hw = estimate_latent_hw(*aligned_target_hw, vae_scale_factor_spatial=rule.vae_scale_factor_spatial)
        noisy_latents = torch.zeros(
            (1, sample.frame_count, self.config.latent_channels, latent_hw[0], latent_hw[1]),
            dtype=torch.float32,
        )
        timesteps = torch.tensor([self.config.timestep_value], dtype=torch.int64)
        metadata = {
            'target_hw': list(target_hw),
            'aligned_target_hw': list(aligned_target_hw),
            'latent_hw': list(latent_hw),
            'latent_channels': self.config.latent_channels,
            'timestep_value': self.config.timestep_value,
        }
        return noisy_latents, timesteps, metadata

    def _known_region_state(self, sample: FYCOutpaintSample) -> dict[str, Any]:
        anchor_region = sample.canvas_meta.anchor_region
        target_region = sample.canvas_meta.target_region
        return {
            'mode': 'overwrite',
            'canvas_height': sample.canvas_meta.canvas_height,
            'canvas_width': sample.canvas_meta.canvas_width,
            'anchor_region': asdict(anchor_region) if anchor_region else None,
            'target_region': asdict(target_region) if target_region else None,
        }

    def build(self, sample: FYCOutpaintSample) -> FYCSampleBridgeOutput:
        anchor_video, anchor_meta = self._anchor_video_tensor(sample)
        known_mask = self._known_mask_tensor(sample)
        relative_position = self._relative_position_tensor(sample)
        prompt_embeds = self._prompt_embeds(sample)
        noisy_latents, timesteps, request_tensor_meta = self._latent_request_tensors(sample)

        conditioning = self.conditioning_builder.encode(
            anchor_video=anchor_video,
            relative_position=relative_position,
            known_mask=known_mask,
            prompt_embeds=prompt_embeds,
        )
        request = WanForwardRequest(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            layout_tokens=conditioning.layout.tokens if conditioning.layout is not None else None,
            geometry_tokens=conditioning.geometry.tokens if conditioning.geometry is not None else None,
            mask_tokens=conditioning.mask.tokens if conditioning.mask is not None else None,
            known_region_state=self._known_region_state(sample),
            extras={
                'prompt': sample.prompt,
                'fps': sample.fps,
                'frame_count': sample.frame_count,
                'relative_position_raw': list(sample.relative_position_raw),
                'relative_position_norm': list(sample.relative_position_norm),
                'target_hw': request_tensor_meta['target_hw'],
                'aligned_target_hw': request_tensor_meta['aligned_target_hw'],
                'latent_hw': request_tensor_meta['latent_hw'],
                'latent_channels': request_tensor_meta['latent_channels'],
            },
        )
        wrapper_payload = self.wrapper.prepare_inputs(request)
        metadata = {
            'anchor_meta': anchor_meta,
            'mask_generate_ratio': conditioning.metadata.get('mask_aux', {}).get('generate_ratio') if conditioning.mask else None,
            'request_tensors': request_tensor_meta,
            'canvas_meta': {
                'canvas_height': sample.canvas_meta.canvas_height,
                'canvas_width': sample.canvas_meta.canvas_width,
                'anchor_region': asdict(sample.canvas_meta.anchor_region) if sample.canvas_meta.anchor_region else None,
                'target_region': asdict(sample.canvas_meta.target_region) if sample.canvas_meta.target_region else None,
            },
        }
        return FYCSampleBridgeOutput(
            conditioning=conditioning,
            request=request,
            wrapper_payload=wrapper_payload,
            metadata=metadata,
        )
