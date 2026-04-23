from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..utils.masks import validate_binary_mask


@dataclass(slots=True)
class KnownRegionState:
    mask_latent: Any
    known_latents: Any
    mode: str = "overwrite"
    blend_schedule: dict[str, Any] = field(default_factory=lambda: {"kind": "cosine"})


@dataclass(slots=True)
class PreserveAction:
    mode: str
    preserve_fraction: float
    step_index: int
    total_steps: int
    blend_alpha: float


def _blend_alpha(step_index: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    progress = step_index / max(total_steps - 1, 1)
    return 0.5 * (1.0 + (1.0 - progress))


def describe_preserve_action(mask_2d: list[list[int]], *, mode: str, step_index: int, total_steps: int) -> PreserveAction:
    validate_binary_mask(mask_2d)
    flat = [value for row in mask_2d for value in row]
    preserve_fraction = flat.count(0) / len(flat)
    alpha = 1.0 if mode == "overwrite" else _blend_alpha(step_index, total_steps)
    return PreserveAction(
        mode=mode,
        preserve_fraction=preserve_fraction,
        step_index=step_index,
        total_steps=total_steps,
        blend_alpha=alpha,
    )


def apply_known_region(current_latents: Any, state: KnownRegionState, *, step_index: int, total_steps: int) -> Any:
    if torch is None:
        raise RuntimeError("torch is required to apply known-region updates")
    preserve = 1.0 - state.mask_latent
    if state.mode == "overwrite":
        return current_latents * state.mask_latent + state.known_latents * preserve
    if state.mode == "blend":
        alpha = _blend_alpha(step_index, total_steps)
        blended = alpha * state.known_latents + (1.0 - alpha) * current_latents
        return current_latents * state.mask_latent + blended * preserve
    raise ValueError(f"Unsupported known-region mode: {state.mode}")
