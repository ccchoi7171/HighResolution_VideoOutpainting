from __future__ import annotations

from dataclasses import dataclass
import math

from ..utils.latent_ops import alignment_quantum


@dataclass(frozen=True, slots=True)
class SizeAlignmentRule:
    vae_scale_factor_spatial: int = 8
    patch_size: int = 2

    @property
    def quantum(self) -> int:
        return alignment_quantum(self.vae_scale_factor_spatial, self.patch_size)


def validate_spatial_size(height: int, width: int, rule: SizeAlignmentRule) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if height % rule.quantum != 0:
        errors.append(f"height={height} is not divisible by quantum={rule.quantum}")
    if width % rule.quantum != 0:
        errors.append(f"width={width} is not divisible by quantum={rule.quantum}")
    return (not errors), errors


def snap_spatial_size(height: int, width: int, rule: SizeAlignmentRule, *, mode: str = "ceil") -> tuple[int, int]:
    quantum = rule.quantum
    if mode == "ceil":
        snap = lambda value: int(math.ceil(value / quantum) * quantum)
    elif mode == "floor":
        snap = lambda value: int(math.floor(value / quantum) * quantum)
    else:
        raise ValueError("mode must be 'ceil' or 'floor'")
    return snap(height), snap(width)
