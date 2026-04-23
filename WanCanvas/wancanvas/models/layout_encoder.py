from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None
    nn = None


@dataclass(slots=True)
class LayoutEncoderConfig:
    input_channels: int = 3
    hidden_dim: int = 128
    token_dim: int = 1024
    token_count: int = 8

    def validate(self) -> None:
        for field_name in ("input_channels", "hidden_dim", "token_dim", "token_count"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


@dataclass(slots=True)
class LayoutEncoderOutput:
    tokens: Any
    aux: dict[str, Any]


if nn is None:

    class SimpleLayoutEncoder:  # pragma: no cover - simple runtime stub
        def __init__(self, config: LayoutEncoderConfig | None = None) -> None:
            self.config = config or LayoutEncoderConfig()
            self.config.validate()

        @staticmethod
        def is_torch_available() -> bool:
            return False

        def describe_output_shape(self, batch_size: int) -> tuple[int, int, int]:
            return (batch_size, self.config.token_count, self.config.token_dim)

        def forward(self, *_: Any, **__: Any) -> LayoutEncoderOutput:
            raise RuntimeError("torch is required to run SimpleLayoutEncoder.forward")

else:

    class SimpleLayoutEncoder(nn.Module):
        def __init__(self, config: LayoutEncoderConfig | None = None) -> None:
            super().__init__()
            self.config = config or LayoutEncoderConfig()
            self.config.validate()
            self.frame_projection = nn.Linear(self.config.input_channels, self.config.hidden_dim)
            self.token_projection = nn.Linear(self.config.hidden_dim, self.config.token_dim)

        @staticmethod
        def is_torch_available() -> bool:
            return True

        def describe_output_shape(self, batch_size: int) -> tuple[int, int, int]:
            return (batch_size, self.config.token_count, self.config.token_dim)

        def forward(self, video: "torch.Tensor") -> LayoutEncoderOutput:
            if video.ndim != 5:
                raise ValueError("layout encoder expects [B, F, C, H, W]")
            batch, frames, channels, _, _ = video.shape
            if channels != self.config.input_channels:
                raise ValueError(
                    f"expected {self.config.input_channels} channels, got {channels}"
                )
            pooled = video.mean(dim=(-1, -2))
            projected = torch.relu(self.frame_projection(pooled))
            if frames >= self.config.token_count:
                indices = torch.linspace(0, frames - 1, steps=self.config.token_count).long().to(video.device)
                sampled = projected.index_select(1, indices)
            else:
                indices = torch.arange(frames, device=video.device)
                pad = projected[:, -1:, :].expand(batch, self.config.token_count - frames, self.config.hidden_dim)
                sampled = torch.cat([projected, pad], dim=1)
            tokens = self.token_projection(sampled)
            return LayoutEncoderOutput(tokens=tokens, aux={"frame_indices": indices.tolist()})
