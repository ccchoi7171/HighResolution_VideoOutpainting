from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class RuntimeConfig:
    conda_env: str = "wancanvas"
    diffusers_install: str = "source"
    allow_stable_fallback: bool = True
    source_install_hint: str = "pip install git+https://github.com/huggingface/diffusers"

    def validate(self) -> None:
        if self.diffusers_install not in {"source", "stable"}:
            raise ValueError(f"Unsupported diffusers_install: {self.diffusers_install}")


@dataclass(slots=True)
class ModelConfig:
    base_model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    base_pipeline_class: str = "WanPipeline"
    vace_reference_model_id: str = "Wan-AI/Wan2.1-VACE-14B-diffusers"
    vace_pipeline_class: str = "WanVACEPipeline"
    strict_dense_mvp: bool = True
    transformer_2: str | None = None
    boundary_ratio: float | None = None

    def validate(self) -> None:
        if self.strict_dense_mvp and (self.transformer_2 is not None or self.boundary_ratio is not None):
            raise ValueError(
                "TI2V-5B MVP must not require transformer_2 or boundary_ratio. "
                "Disable strict_dense_mvp only for a future A14B upgrade config."
            )
        if self.boundary_ratio is not None and not 0.0 < self.boundary_ratio < 1.0:
            raise ValueError("boundary_ratio must be between 0 and 1")


@dataclass(slots=True)
class ConditionConfig:
    layout_token_dim: int = 1024
    layout_token_count: int = 8
    geometry_token_dim: int = 1024
    geometry_token_count: int = 4
    geometry_version: str = "v1"
    use_mask_summary: bool = True

    def validate(self) -> None:
        if self.geometry_version not in {"v1", "v1.1"}:
            raise ValueError(f"Unsupported geometry_version: {self.geometry_version}")
        for field_name in ("layout_token_dim", "layout_token_count", "geometry_token_dim", "geometry_token_count"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


@dataclass(slots=True)
class WindowConfig:
    tile_height: int = 720
    tile_width: int = 720
    overlap_height: int = 180
    overlap_width: int = 180
    rounds: int = 1

    def validate(self) -> None:
        for field_name in ("tile_height", "tile_width", "rounds"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.overlap_height < 0 or self.overlap_width < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap_height >= self.tile_height or self.overlap_width >= self.tile_width:
            raise ValueError("overlap must be smaller than the tile size")


@dataclass(slots=True)
class KnownRegionConfig:
    mode: str = "overwrite"
    blend_schedule: dict[str, Any] = field(default_factory=lambda: {"kind": "cosine"})

    def validate(self) -> None:
        if self.mode not in {"overwrite", "blend"}:
            raise ValueError(f"Unsupported known-region mode: {self.mode}")


@dataclass(slots=True)
class TrainSkeletonConfig:
    dry_run_only: bool = True
    trainable_modules: tuple[str, ...] = (
        "layout_encoder",
        "geometry_encoder",
        "condition_adapter",
        "wan_outpaint_wrapper",
    )

    def validate(self) -> None:
        if not self.trainable_modules:
            raise ValueError("trainable_modules must not be empty")


@dataclass(slots=True)
class WanCanvasConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    conditions: ConditionConfig = field(default_factory=ConditionConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    known_region: KnownRegionConfig = field(default_factory=KnownRegionConfig)
    train: TrainSkeletonConfig = field(default_factory=TrainSkeletonConfig)

    def validate(self) -> None:
        self.runtime.validate()
        self.model.validate()
        self.conditions.validate()
        self.window.validate()
        self.known_region.validate()
        self.train.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "WanCanvasConfig":
        def pick(section: str, factory: Any) -> Any:
            values = dict(mapping.get(section, {}))
            return factory(**values)

        config = cls(
            runtime=pick("runtime", RuntimeConfig),
            model=pick("model", ModelConfig),
            conditions=pick("conditions", ConditionConfig),
            window=pick("window", WindowConfig),
            known_region=pick("known_region", KnownRegionConfig),
            train=pick("train", TrainSkeletonConfig),
        )
        config.validate()
        return config
