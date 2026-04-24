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
from .inference import OutpaintInferenceConfig, OutpaintRunArtifacts, run_outpaint_inference
from .train import SmokeTrainConfig, SmokeTrainReport, SmokeTrainer

__all__ = [
    "ConditionConfig",
    "KnownRegionConfig",
    "ModelConfig",
    "RuntimeConfig",
    "TrainSkeletonConfig",
    "WanCanvasConfig",
    "WindowConfig",
    "OutpaintInferenceConfig",
    "OutpaintRunArtifacts",
    "run_outpaint_inference",
    "SmokeTrainConfig",
    "SmokeTrainReport",
    "SmokeTrainer",
]
