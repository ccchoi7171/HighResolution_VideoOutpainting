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
class GeometryEncoderConfig:
    input_dim: int = 6
    hidden_dim: int = 64
    token_dim: int = 1024
    token_count: int = 4

    def validate(self) -> None:
        for field_name in ("input_dim", "hidden_dim", "token_dim", "token_count"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


@dataclass(slots=True)
class GeometryEncoderOutput:
    tokens: Any
    aux: dict[str, Any]


if nn is None:

    class SimpleGeometryEncoder:
        def __init__(self, config: GeometryEncoderConfig | None = None) -> None:
            self.config = config or GeometryEncoderConfig()
            self.config.validate()

        @staticmethod
        def is_torch_available() -> bool:
            return False

        def describe_output_shape(self, batch_size: int) -> tuple[int, int, int]:
            return (batch_size, self.config.token_count, self.config.token_dim)

        def forward(self, *_: Any, **__: Any) -> GeometryEncoderOutput:
            raise RuntimeError("torch is required to run SimpleGeometryEncoder.forward")

else:

    class SimpleGeometryEncoder(nn.Module):
        def __init__(self, config: GeometryEncoderConfig | None = None) -> None:
            super().__init__()
            self.config = config or GeometryEncoderConfig()
            self.config.validate()
            self.mlp = nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, self.config.token_dim * self.config.token_count),
            )

        @staticmethod
        def is_torch_available() -> bool:
            return True

        def describe_output_shape(self, batch_size: int) -> tuple[int, int, int]:
            return (batch_size, self.config.token_count, self.config.token_dim)

        def forward(self, geometry: "torch.Tensor") -> GeometryEncoderOutput:
            if geometry.ndim != 2 or geometry.shape[-1] != self.config.input_dim:
                raise ValueError(
                    f"geometry encoder expects [B, {self.config.input_dim}]"
                )
            batch = geometry.shape[0]
            tokens = self.mlp(geometry).view(batch, self.config.token_count, self.config.token_dim)
            return GeometryEncoderOutput(tokens=tokens, aux={"input_dim": self.config.input_dim})
