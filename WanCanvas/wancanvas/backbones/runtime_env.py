from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import importlib.util
from pathlib import Path
from typing import Any


_REQUIRED_WAN_CLASSES = ("WanImageToVideoPipeline", "WanVACEPipeline", "WanTransformer3DModel")


@dataclass(slots=True)
class RuntimeInspection:
    diffusers_available: bool
    diffusers_version: str | None
    diffusers_path: str | None
    install_mode: str
    available_classes: dict[str, bool] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def missing_classes(self) -> list[str]:
        return [name for name, available in self.available_classes.items() if not available]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_classes"] = self.missing_classes
        return payload


def _guess_install_mode(module_file: str | None) -> str:
    if not module_file:
        return "missing"
    if "site-packages" in module_file:
        return "stable"
    return "source"


def inspect_diffusers_runtime(required_classes: tuple[str, ...] = _REQUIRED_WAN_CLASSES) -> RuntimeInspection:
    spec = importlib.util.find_spec("diffusers")
    if spec is None:
        return RuntimeInspection(
            diffusers_available=False,
            diffusers_version=None,
            diffusers_path=None,
            install_mode="missing",
            available_classes={name: False for name in required_classes},
            errors={"diffusers": "Package not installed in the current environment."},
            notes=["Install diffusers source/main in the wancanvas env for runtime bring-up."],
        )

    diffusers = importlib.import_module("diffusers")
    module_file = getattr(diffusers, "__file__", None)
    inspection = RuntimeInspection(
        diffusers_available=True,
        diffusers_version=getattr(diffusers, "__version__", None),
        diffusers_path=str(Path(module_file).resolve()) if module_file else None,
        install_mode=_guess_install_mode(module_file),
    )
    for class_name in required_classes:
        inspection.available_classes[class_name] = hasattr(diffusers, class_name)
        if not inspection.available_classes[class_name]:
            inspection.errors[class_name] = f"{class_name} not found in diffusers"
    if inspection.install_mode == "source":
        inspection.notes.append("diffusers appears to come from a source checkout or editable install")
    else:
        inspection.notes.append("diffusers appears to come from a site-packages installation")
    return inspection


def stable_release_allowed(inspection: RuntimeInspection) -> bool:
    return inspection.diffusers_available and not inspection.missing_classes
