from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(slots=True)
class ConditionBundle:
    tokens: dict[str, Any] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ConditionAdapter:
    def __init__(self, token_order: tuple[str, ...] = ("text", "layout", "geometry", "mask")) -> None:
        self.token_order = token_order

    def build_bundle(
        self,
        *,
        text_tokens: Any = None,
        layout_tokens: Any = None,
        geometry_tokens: Any = None,
        mask_tokens: Any = None,
    ) -> ConditionBundle:
        candidates = {
            "text": text_tokens,
            "layout": layout_tokens,
            "geometry": geometry_tokens,
            "mask": mask_tokens,
        }
        order = [name for name in self.token_order if candidates.get(name) is not None]
        tokens = {name: candidates[name] for name in order}
        metadata = {"condition_order": order, "condition_count": len(order)}
        return ConditionBundle(tokens=tokens, order=order, metadata=metadata)

    def concat_bundle(self, bundle: ConditionBundle) -> Any:
        if not bundle.order:
            return None
        first = bundle.tokens[bundle.order[0]]
        if torch is None or first is None or not hasattr(first, "ndim"):
            return [bundle.tokens[name] for name in bundle.order]
        return torch.cat([bundle.tokens[name] for name in bundle.order], dim=1)
