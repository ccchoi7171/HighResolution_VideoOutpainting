"""WanCanvas architecture scaffold."""

from .config_schema import (
    ConditionConfig,
    KnownRegionConfig,
    ModelConfig,
    RuntimeConfig,
    TrainSkeletonConfig,
    WanCanvasConfig,
    WindowConfig,
)
from .inference import Ti2VInferenceConfig, Ti2VRunArtifacts, run_ti2v_inference

__all__ = [
    "ConditionConfig",
    "KnownRegionConfig",
    "ModelConfig",
    "RuntimeConfig",
    "TrainSkeletonConfig",
    "WanCanvasConfig",
    "WindowConfig",
    "Ti2VInferenceConfig",
    "Ti2VRunArtifacts",
    "run_ti2v_inference",
]
