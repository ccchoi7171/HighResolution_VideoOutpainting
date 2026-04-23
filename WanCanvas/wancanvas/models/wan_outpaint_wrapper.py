from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..backbones.wan_loader import WanLoader
from ..config_schema import ModelConfig
from .condition_adapter import ConditionAdapter


@dataclass(slots=True)
class WanForwardRequest:
    noisy_latents: Any
    timesteps: Any
    prompt_embeds: Any
    layout_tokens: Any = None
    geometry_tokens: Any = None
    mask_tokens: Any = None
    known_region_state: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


class WanOutpaintWrapper:
    def __init__(
        self,
        loader: WanLoader,
        *,
        model_config: ModelConfig | None = None,
        condition_adapter: ConditionAdapter | None = None,
    ) -> None:
        self.loader = loader
        self.model_config = model_config or loader.model_config
        self.condition_adapter = condition_adapter or ConditionAdapter()

    @staticmethod
    def _shape_of(value: Any) -> list[int] | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return [int(dim) for dim in shape]

    @staticmethod
    def _dtype_of(value: Any) -> str | None:
        dtype = getattr(value, 'dtype', None)
        if dtype is None:
            return None
        return str(dtype)

    @staticmethod
    def _is_tensor(value: Any) -> bool:
        return torch is not None and isinstance(value, torch.Tensor)

    def validate_request_contract(self, request: WanForwardRequest) -> dict[str, Any]:
        prompt_is_tensor = self._is_tensor(request.prompt_embeds)
        latents_is_tensor = self._is_tensor(request.noisy_latents)
        timesteps_is_tensor = self._is_tensor(request.timesteps)
        prompt_shape = self._shape_of(request.prompt_embeds)
        latents_shape = self._shape_of(request.noisy_latents)
        timesteps_shape = self._shape_of(request.timesteps)

        checks: dict[str, bool] = {
            'prompt_embeds_is_tensor': prompt_is_tensor,
            'noisy_latents_is_tensor': latents_is_tensor,
            'timesteps_is_tensor': timesteps_is_tensor,
            'prompt_embeds_rank_3': bool(prompt_shape and len(prompt_shape) == 3),
            'noisy_latents_rank_5': bool(latents_shape and len(latents_shape) == 5),
            'timesteps_rank_scalar_or_1d': bool(
                timesteps_is_tensor and request.timesteps.ndim in {0, 1}
            ),
        }

        batch_size = prompt_shape[0] if prompt_shape and len(prompt_shape) == 3 else None
        checks['batch_size_known'] = batch_size is not None
        if batch_size is not None and latents_shape and len(latents_shape) == 5:
            checks['latents_batch_matches_prompt'] = latents_shape[0] == batch_size
        else:
            checks['latents_batch_matches_prompt'] = False
        if batch_size is not None and timesteps_is_tensor and request.timesteps.ndim == 1:
            checks['timesteps_batch_matches_or_broadcastable'] = request.timesteps.numel() in {1, batch_size}
        elif timesteps_is_tensor and request.timesteps.ndim == 0:
            checks['timesteps_batch_matches_or_broadcastable'] = True
        else:
            checks['timesteps_batch_matches_or_broadcastable'] = False

        token_dim_candidates: list[int] = []
        if prompt_shape and len(prompt_shape) == 3:
            token_dim_candidates.append(prompt_shape[-1])
        optional_tokens = {
            'layout_tokens': request.layout_tokens,
            'geometry_tokens': request.geometry_tokens,
            'mask_tokens': request.mask_tokens,
        }
        for name, tokens in optional_tokens.items():
            token_shape = self._shape_of(tokens)
            token_is_tensor = self._is_tensor(tokens)
            checks[f'{name}_optional_or_tensor'] = tokens is None or token_is_tensor
            checks[f'{name}_optional_or_rank_3'] = tokens is None or bool(token_shape and len(token_shape) == 3)
            if tokens is not None and batch_size is not None and token_shape and len(token_shape) == 3:
                checks[f'{name}_batch_matches_prompt'] = token_shape[0] == batch_size
                token_dim_candidates.append(token_shape[-1])
            else:
                checks[f'{name}_batch_matches_prompt'] = tokens is None
        checks['token_dim_consistent'] = len(set(token_dim_candidates)) == 1 if token_dim_candidates else False

        extras_frame_count = request.extras.get('frame_count')
        if extras_frame_count is not None and latents_shape and len(latents_shape) == 5:
            checks['frame_count_matches_latents'] = latents_shape[1] == int(extras_frame_count)
        else:
            checks['frame_count_matches_latents'] = extras_frame_count is None

        known_region_state = request.known_region_state
        checks['known_region_state_present'] = known_region_state is not None
        checks['known_region_mode_valid'] = isinstance(known_region_state, dict) and known_region_state.get('mode') == 'overwrite'

        return {
            'shapes': {
                'prompt_embeds': prompt_shape,
                'noisy_latents': latents_shape,
                'timesteps': timesteps_shape,
                'layout_tokens': self._shape_of(request.layout_tokens),
                'geometry_tokens': self._shape_of(request.geometry_tokens),
                'mask_tokens': self._shape_of(request.mask_tokens),
            },
            'dtypes': {
                'prompt_embeds': self._dtype_of(request.prompt_embeds),
                'noisy_latents': self._dtype_of(request.noisy_latents),
                'timesteps': self._dtype_of(request.timesteps),
                'layout_tokens': self._dtype_of(request.layout_tokens),
                'geometry_tokens': self._dtype_of(request.geometry_tokens),
                'mask_tokens': self._dtype_of(request.mask_tokens),
            },
            'checks': checks,
        }

    def prepare_inputs(self, request: WanForwardRequest) -> dict[str, Any]:
        bundle = self.condition_adapter.build_bundle(
            text_tokens=request.prompt_embeds,
            layout_tokens=request.layout_tokens,
            geometry_tokens=request.geometry_tokens,
            mask_tokens=request.mask_tokens,
        )
        concatenated = self.condition_adapter.concat_bundle(bundle)
        semantic_roles = {
            "text": "text",
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
        known_region_state = request.known_region_state
        return {
            'extras': dict(request.extras),
            'has_known_region_state': known_region_state is not None,
            'known_region_mode': (
                known_region_state.get('mode')
                if isinstance(known_region_state, dict)
                else None
            ),
            'prompt_embeds_shape': self._shape_of(request.prompt_embeds),
            'noisy_latents_shape': self._shape_of(request.noisy_latents),
            'timesteps_shape': self._shape_of(request.timesteps),
            'layout_tokens_shape': self._shape_of(request.layout_tokens),
            'geometry_tokens_shape': self._shape_of(request.geometry_tokens),
            'mask_tokens_shape': self._shape_of(request.mask_tokens),
            'prompt_embeds_dtype': self._dtype_of(request.prompt_embeds),
            'noisy_latents_dtype': self._dtype_of(request.noisy_latents),
            'timesteps_dtype': self._dtype_of(request.timesteps),
        }

    def forward(self, request: WanForwardRequest) -> Any:
        raise RuntimeError(
            "WanOutpaintWrapper.forward is intentionally disabled during architecture-only bring-up. "
            f"Use dry_run() first. Prepared payload: {self.describe_request(request)}"
        )
