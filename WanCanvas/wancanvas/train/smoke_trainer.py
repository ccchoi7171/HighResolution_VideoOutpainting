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
from ..models.fyc_sample_bridge import FYCSampleToWanBridge
from ..models.wan_outpaint_wrapper import WanForwardRequest, WanOutpaintWrapper


@dataclass(slots=True)
class SmokeTrainConfig:
    learning_rate: float = 1e-4
    scheduler_train_steps: int = 20
    gradient_clip_norm: float = 1.0
    diffusion_loss_weight: float = 1.0
    known_region_loss_weight: float = 1.0
    seam_loss_weight: float = 0.25
    seed: int = 7
    negative_prompt: str = ""

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        if self.scheduler_train_steps <= 0:
            raise ValueError('scheduler_train_steps must be positive')
        if self.gradient_clip_norm <= 0:
            raise ValueError('gradient_clip_norm must be positive')


@dataclass(slots=True)
class SmokeTrainReport:
    loss: float
    loss_components: dict[str, float]
    grad_norm: float
    optimizer_lr: float
    updated_parameter_count: int
    request_summary: dict[str, Any]
    forward_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmokeTrainer:
    def __init__(
        self,
        wrapper: WanOutpaintWrapper,
        *,
        bridge: FYCSampleToWanBridge | None = None,
        config: SmokeTrainConfig | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.bridge = bridge or FYCSampleToWanBridge(wrapper=wrapper)
        self.config = config or SmokeTrainConfig()
        self.config.validate()
        self.optimizer = None

    @staticmethod
    def _ensure_torch() -> None:
        if torch is None or F is None:
            raise RuntimeError('torch is required for SmokeTrainer')

    @staticmethod
    def _module_parameters(module: Any) -> list[torch.nn.Parameter]:
        if module is None or not hasattr(module, 'parameters'):
            return []
        return [param for param in module.parameters() if param.requires_grad]

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        self._ensure_torch()
        builder = self.bridge.conditioning_builder
        params: list[torch.nn.Parameter] = []
        for module in (
            getattr(builder, 'layout_encoder', None),
            getattr(builder, 'geometry_encoder', None),
            getattr(builder, 'mask_encoder', None),
            getattr(self.wrapper, 'fyc_token_projector', None),
        ):
            params.extend(self._module_parameters(module))
        unique: list[torch.nn.Parameter] = []
        seen = set()
        for param in params:
            if id(param) in seen:
                continue
            seen.add(id(param))
            unique.append(param)
        return unique

    @staticmethod
    def _seam_mask(latent_mask: torch.Tensor) -> torch.Tensor:
        band = latent_mask.permute(0, 2, 1, 3, 4).reshape(latent_mask.shape[0] * latent_mask.shape[2], 1, latent_mask.shape[3], latent_mask.shape[4])
        dilated = F.max_pool2d(band, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-band, kernel_size=3, stride=1, padding=1)
        seam = (dilated - eroded).clamp(0.0, 1.0)
        return seam.view(latent_mask.shape[0], latent_mask.shape[2], 1, latent_mask.shape[3], latent_mask.shape[4]).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = torch.clamp(mask.sum(), min=1.0)
        return (loss_map * mask).sum() / weight

    def _ensure_optimizer(self) -> Any:
        if self.optimizer is not None:
            return self.optimizer
        parameters = self._trainable_parameters()
        if not parameters:
            raise RuntimeError('No trainable parameters were found for the smoke trainer')
        self.optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        return self.optimizer

    def run_once(self, sample: FYCOutpaintSample, *, runtime: Any = None) -> SmokeTrainReport:
        self._ensure_torch()
        torch.manual_seed(self.config.seed)
        bridge_output = self.bridge.build(sample)
        request = bridge_output.request
        request.negative_prompt = self.config.negative_prompt
        request.prompt_is_placeholder = True

        pipe = self.wrapper._resolve_pipeline(runtime)
        pipe.scheduler.set_timesteps(self.config.scheduler_train_steps, device=pipe._execution_device)
        timestep = pipe.scheduler.timesteps[len(pipe.scheduler.timesteps) // 2].view(1)
        target_latents = self.wrapper.encode_video_to_latents(request.target_video, runtime=pipe).to(device=pipe._execution_device, dtype=torch.float32)
        condition_latents = self.wrapper.encode_video_to_latents(request.condition_video, runtime=pipe).to(device=pipe._execution_device, dtype=torch.float32)
        noise = torch.randn_like(target_latents)
        noisy_latents = pipe.scheduler.scale_noise(target_latents, timestep, noise)

        request = WanForwardRequest(**{field: getattr(request, field) for field in request.__dataclass_fields__})
        request.target_latents = target_latents
        request.condition_latents = condition_latents
        request.noisy_latents = noisy_latents
        request.timesteps = timestep

        optimizer = self._ensure_optimizer()
        optimizer.zero_grad(set_to_none=True)
        forward = self.wrapper.forward(request, runtime=pipe, guidance_scale=1.0, do_classifier_free_guidance=False)
        pred_x0 = self.wrapper.reconstruct_clean_latents(forward.latents, forward.noise_pred, request.timesteps, runtime=pipe)

        generation_mask = forward.latent_mask
        preserve_mask = 1.0 - generation_mask
        seam_mask = self._seam_mask(generation_mask)

        diffusion_loss_map = (forward.noise_pred - noise).pow(2)
        diffusion_loss = self._masked_mean(diffusion_loss_map, generation_mask.expand_as(diffusion_loss_map))
        known_region_loss_map = (pred_x0 - target_latents).pow(2)
        known_region_loss = self._masked_mean(known_region_loss_map, preserve_mask.expand_as(known_region_loss_map))
        seam_loss_map = (pred_x0 - target_latents).abs()
        seam_loss = self._masked_mean(seam_loss_map, seam_mask.expand_as(seam_loss_map))

        total_loss = (
            self.config.diffusion_loss_weight * diffusion_loss
            + self.config.known_region_loss_weight * known_region_loss
            + self.config.seam_loss_weight * seam_loss
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self.config.gradient_clip_norm)
        optimizer.step()

        return SmokeTrainReport(
            loss=float(total_loss.detach().cpu()),
            loss_components={
                'diffusion': float(diffusion_loss.detach().cpu()),
                'known_region': float(known_region_loss.detach().cpu()),
                'seam': float(seam_loss.detach().cpu()),
            },
            grad_norm=float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            optimizer_lr=float(optimizer.param_groups[0]['lr']),
            updated_parameter_count=sum(param.grad is not None for param in self._trainable_parameters()),
            request_summary=self.wrapper.describe_request(request),
            forward_summary=forward.to_dict(),
        )
