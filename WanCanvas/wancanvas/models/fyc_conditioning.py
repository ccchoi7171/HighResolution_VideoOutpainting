from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .condition_adapter import ConditionAdapter, ConditionBundle
from .geometry_encoder import GeometryEncoderConfig, GeometryEncoderOutput, SimpleGeometryEncoder
from .layout_encoder import LayoutEncoderConfig, LayoutEncoderOutput, SimpleLayoutEncoder
from .mask_summary import MaskSummaryConfig, MaskSummaryOutput, SimpleMaskSummaryEncoder


@dataclass(slots=True)
class FYCConditioningConfig:
    layout: LayoutEncoderConfig = field(default_factory=LayoutEncoderConfig)
    geometry: GeometryEncoderConfig = field(default_factory=GeometryEncoderConfig)
    mask: MaskSummaryConfig = field(default_factory=MaskSummaryConfig)
    include_mask_summary: bool = True

    def validate(self) -> None:
        self.layout.validate()
        self.geometry.validate()
        self.mask.validate()


@dataclass(slots=True)
class FYCConditioningOutput:
    bundle: ConditionBundle
    concatenated_tokens: Any
    layout: LayoutEncoderOutput | None
    geometry: GeometryEncoderOutput | None
    mask: MaskSummaryOutput | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_order": list(self.bundle.order),
            "metadata": dict(self.metadata),
        }


class FYCConditioningBuilder:
    """Compose FYC-side conditions into a DiT-facing token bundle.

    This is the explicit proof surface for the architecture claim that:
    - LE (layout encoder) produces anchor/layout tokens
    - RRE (relative region embedding) produces geometry tokens from the 6-field FYC relation
    - known-region state can be summarized into mask tokens
    - all three become an ordered conditioning bundle consumable by the Wan-side wrapper
    """

    def __init__(
        self,
        config: FYCConditioningConfig | None = None,
        *,
        condition_adapter: ConditionAdapter | None = None,
        layout_encoder: SimpleLayoutEncoder | None = None,
        geometry_encoder: SimpleGeometryEncoder | None = None,
        mask_encoder: SimpleMaskSummaryEncoder | None = None,
    ) -> None:
        self.config = config or FYCConditioningConfig()
        self.config.validate()
        self.condition_adapter = condition_adapter or ConditionAdapter()
        self.layout_encoder = layout_encoder or SimpleLayoutEncoder(self.config.layout)
        self.geometry_encoder = geometry_encoder or SimpleGeometryEncoder(self.config.geometry)
        self.mask_encoder = mask_encoder or SimpleMaskSummaryEncoder(self.config.mask)

    @staticmethod
    def _shape_of(value: Any) -> list[int] | None:
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return [int(dim) for dim in shape]

    def encode(
        self,
        *,
        anchor_video: Any,
        relative_position: Any,
        known_mask: Any | None = None,
        prompt_embeds: Any = None,
    ) -> FYCConditioningOutput:
        layout_output = self.layout_encoder(anchor_video)
        geometry_output = self.geometry_encoder(relative_position)
        mask_output = None
        if self.config.include_mask_summary and known_mask is not None:
            mask_output = self.mask_encoder(known_mask)

        bundle = self.condition_adapter.build_bundle(
            text_tokens=prompt_embeds,
            layout_tokens=layout_output.tokens,
            geometry_tokens=geometry_output.tokens,
            mask_tokens=mask_output.tokens if mask_output is not None else None,
        )
        concatenated = self.condition_adapter.concat_bundle(bundle)
        semantic_roles = {
            "text": "text",
            "layout": "layout_encoder",
            "geometry": "relative_region_embedding",
            "mask": "known_region_mask_summary",
        }
        metadata = {
            "semantic_roles": [semantic_roles[name] for name in bundle.order],
            "layout_shape": self._shape_of(layout_output.tokens),
            "geometry_shape": self._shape_of(geometry_output.tokens),
            "mask_shape": self._shape_of(mask_output.tokens) if mask_output is not None else None,
            "concat_shape": self._shape_of(concatenated),
            "layout_aux": dict(layout_output.aux),
            "geometry_aux": dict(geometry_output.aux),
            "mask_aux": dict(mask_output.aux) if mask_output is not None else None,
            "preserves_fyc_v1_rre": True,
        }
        return FYCConditioningOutput(
            bundle=bundle,
            concatenated_tokens=concatenated,
            layout=layout_output,
            geometry=geometry_output,
            mask=mask_output,
            metadata=metadata,
        )
