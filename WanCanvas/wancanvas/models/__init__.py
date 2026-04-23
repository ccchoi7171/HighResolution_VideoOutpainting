from __future__ import annotations

from importlib import import_module

__all__ = [
    "ConditionAdapter",
    "ConditionBundle",
    "FYCConditioningBuilder",
    "FYCConditioningConfig",
    "FYCConditioningOutput",
    "FYCSampleBridgeConfig",
    "FYCSampleBridgeOutput",
    "FYCSampleToWanBridge",
    "GeometryEncoderConfig",
    "GeometryEncoderOutput",
    "LayoutEncoderConfig",
    "LayoutEncoderOutput",
    "MaskSummaryConfig",
    "MaskSummaryOutput",
    "SimpleGeometryEncoder",
    "SimpleLayoutEncoder",
    "SimpleMaskSummaryEncoder",
    "WanForwardRequest",
    "WanOutpaintWrapper",
]

_MODULE_BY_NAME = {
    "ConditionAdapter": ".condition_adapter",
    "ConditionBundle": ".condition_adapter",
    "FYCConditioningBuilder": ".fyc_conditioning",
    "FYCConditioningConfig": ".fyc_conditioning",
    "FYCConditioningOutput": ".fyc_conditioning",
    "FYCSampleBridgeConfig": ".fyc_sample_bridge",
    "FYCSampleBridgeOutput": ".fyc_sample_bridge",
    "FYCSampleToWanBridge": ".fyc_sample_bridge",
    "GeometryEncoderConfig": ".geometry_encoder",
    "GeometryEncoderOutput": ".geometry_encoder",
    "LayoutEncoderConfig": ".layout_encoder",
    "LayoutEncoderOutput": ".layout_encoder",
    "MaskSummaryConfig": ".mask_summary",
    "MaskSummaryOutput": ".mask_summary",
    "SimpleGeometryEncoder": ".geometry_encoder",
    "SimpleLayoutEncoder": ".layout_encoder",
    "SimpleMaskSummaryEncoder": ".mask_summary",
    "WanForwardRequest": ".wan_outpaint_wrapper",
    "WanOutpaintWrapper": ".wan_outpaint_wrapper",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
