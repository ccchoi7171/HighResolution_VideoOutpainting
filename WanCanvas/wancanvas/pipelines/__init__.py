from __future__ import annotations

from importlib import import_module

__all__ = [
    'KnownRegionState',
    'MultiRoundOutpaintRequest',
    'OutpaintRequest',
    'PreserveAction',
    'SizeAlignmentRule',
    'TilePlan',
    'WanOutpaintPipeline',
    'WindowScheduler',
    'apply_known_region',
    'describe_preserve_action',
    'gaussian_weights_2d',
    'snap_spatial_size',
    'validate_spatial_size',
]

_MODULE_BY_NAME = {
    'KnownRegionState': '.known_region',
    'PreserveAction': '.known_region',
    'apply_known_region': '.known_region',
    'describe_preserve_action': '.known_region',
    'gaussian_weights_2d': '.overlap_merge',
    'SizeAlignmentRule': '.size_alignment',
    'snap_spatial_size': '.size_alignment',
    'validate_spatial_size': '.size_alignment',
    'MultiRoundOutpaintRequest': '.wan_outpaint_pipeline',
    'OutpaintRequest': '.wan_outpaint_pipeline',
    'WanOutpaintPipeline': '.wan_outpaint_pipeline',
    'TilePlan': '.window_scheduler',
    'WindowScheduler': '.window_scheduler',
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
