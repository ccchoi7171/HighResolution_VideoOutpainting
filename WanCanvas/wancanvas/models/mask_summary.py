from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(slots=True)
class MaskSummaryConfig:
    token_dim: int = 1024
    token_count: int = 1

    def validate(self) -> None:
        if self.token_dim <= 0 or self.token_count <= 0:
            raise ValueError("token_dim and token_count must be positive")


@dataclass(slots=True)
class MaskSummaryOutput:
    tokens: Any
    aux: dict[str, Any]


if nn is None:

    class SimpleMaskSummaryEncoder:
        def __init__(self, config: MaskSummaryConfig | None = None) -> None:
            self.config = config or MaskSummaryConfig()
            self.config.validate()

        def forward(self, *_: Any, **__: Any) -> MaskSummaryOutput:
            raise RuntimeError("torch is required to run SimpleMaskSummaryEncoder.forward")

else:

    class SimpleMaskSummaryEncoder(nn.Module):
        def __init__(self, config: MaskSummaryConfig | None = None) -> None:
            super().__init__()
            self.config = config or MaskSummaryConfig()
            self.config.validate()
            self.proj = nn.Linear(4, self.config.token_dim * self.config.token_count)

        def forward(self, mask: "torch.Tensor") -> MaskSummaryOutput:
            if mask.ndim != 5:
                raise ValueError("mask summary expects [B, F, 1, H, W]")
            if mask.shape[2] != 1:
                raise ValueError("mask summary expects a single channel mask")
            batch = mask.shape[0]
            coverage = mask.mean(dim=(1, 2, 3, 4), keepdim=False)
            mask_min = mask.amin(dim=(1, 2, 3, 4))
            mask_max = mask.amax(dim=(1, 2, 3, 4))
            frame_mean = mask.mean(dim=(2, 3, 4)).mean(dim=1)
            features = torch.stack([coverage, mask_min, mask_max, frame_mean], dim=1)
            tokens = self.proj(features).view(batch, self.config.token_count, self.config.token_dim)
            return MaskSummaryOutput(tokens=tokens, aux={"generate_ratio": coverage.tolist()})
