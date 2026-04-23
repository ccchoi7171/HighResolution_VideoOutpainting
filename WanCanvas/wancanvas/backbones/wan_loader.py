from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..config_schema import ModelConfig, RuntimeConfig
from .runtime_env import RuntimeInspection, inspect_diffusers_runtime, stable_release_allowed


@dataclass(slots=True)
class WanLoaderReport:
    runtime: dict[str, Any]
    model: dict[str, Any]
    ready_for_download: bool
    strict_dense_mvp: bool
    download_attempted: bool = False
    download_skipped_reason: str | None = None
    loaded_pipeline_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoadedWanPipeline:
    pipeline: Any
    runtime: RuntimeInspection
    model_id: str
    pipeline_class: str
    device: str
    torch_dtype: str
    flow_shift: float | None

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "pipeline_class": self.pipeline_class,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "flow_shift": self.flow_shift,
            "diffusers_version": self.runtime.diffusers_version,
        }


class WanLoader:
    """Runtime-aware loader facade for the TI2V-5B-first WanCanvas stack.

    The loader can operate in two modes:
    1. architecture-only smoke inspection (`smoke_validate`)
    2. real runtime loading for verified inference bring-up (`load_pipeline`)

    The MVP keeps the base model on Wan2.2 TI2V-5B and treats any second-transformer
    / boundary-ratio configuration as an optional future A14B extension point.
    """

    def __init__(self, runtime_config: RuntimeConfig | None = None, model_config: ModelConfig | None = None) -> None:
        self.runtime_config = runtime_config or RuntimeConfig()
        self.model_config = model_config or ModelConfig()
        self.runtime_config.validate()
        self.model_config.validate()

    def inspect_runtime(self) -> RuntimeInspection:
        return inspect_diffusers_runtime()

    def build_model_bundle(self) -> dict[str, Any]:
        self.model_config.validate()
        return {
            "base_model_id": self.model_config.base_model_id,
            "base_pipeline_class": self.model_config.base_pipeline_class,
            "vace_reference_model_id": self.model_config.vace_reference_model_id,
            "vace_pipeline_class": self.model_config.vace_pipeline_class,
            "strict_dense_mvp": self.model_config.strict_dense_mvp,
            "transformer_2": self.model_config.transformer_2,
            "boundary_ratio": self.model_config.boundary_ratio,
        }

    def smoke_validate(self, *, download_model: bool = False, strict_runtime: bool = False) -> WanLoaderReport:
        inspection = self.inspect_runtime()
        ready = stable_release_allowed(inspection)
        if strict_runtime and not ready:
            reason = "required Wan classes are not available in the current diffusers installation"
        elif download_model and not ready:
            reason = "runtime is not ready for model download"
        else:
            reason = "architecture-only dry-run" if not download_model else None
        return WanLoaderReport(
            runtime=inspection.to_dict(),
            model=self.build_model_bundle(),
            ready_for_download=ready,
            strict_dense_mvp=self.model_config.strict_dense_mvp,
            download_attempted=download_model and ready,
            download_skipped_reason=reason,
            loaded_pipeline_class=self.model_config.base_pipeline_class if ready else None,
        )

    def _resolve_torch_dtype(self, torch_dtype: str | Any) -> tuple[Any, str]:
        import torch

        if isinstance(torch_dtype, str):
            normalized = torch_dtype.lower()
            mapping = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if normalized not in mapping:
                raise ValueError(f"Unsupported torch dtype string: {torch_dtype}")
            return mapping[normalized], normalized
        return torch_dtype, getattr(torch_dtype, "name", str(torch_dtype))

    def load_pipeline(
        self,
        *,
        model_id: str | None = None,
        pipeline_class_name: str | None = None,
        device: str = "cuda",
        torch_dtype: str | Any = "bfloat16",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        flow_shift: float | None = 5.0,
        scheduler_name: str | None = None,
        use_fp32_vae: bool = True,
        enable_model_cpu_offload: bool = False,
    ) -> LoadedWanPipeline:
        report = self.smoke_validate(download_model=True, strict_runtime=True)
        if not report.ready_for_download:
            raise RuntimeError(
                "Wan runtime is not ready inside the current environment. "
                "Install a compatible diffusers build in the wancanvas env first."
            )

        import diffusers
        import torch

        resolved_dtype, dtype_name = self._resolve_torch_dtype(torch_dtype)
        resolved_model_id = model_id or self.model_config.base_model_id
        resolved_pipeline_class = pipeline_class_name or self.model_config.base_pipeline_class

        if not hasattr(diffusers, resolved_pipeline_class):
            raise RuntimeError(f"Requested pipeline class is unavailable: {resolved_pipeline_class}")
        pipeline_cls = getattr(diffusers, resolved_pipeline_class)

        common_kwargs: dict[str, Any] = {
            "torch_dtype": resolved_dtype,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if use_fp32_vae and hasattr(diffusers, "AutoencoderKLWan"):
            vae = diffusers.AutoencoderKLWan.from_pretrained(
                resolved_model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            common_kwargs["vae"] = vae

        pipe = pipeline_cls.from_pretrained(resolved_model_id, **common_kwargs)
        if scheduler_name is not None:
            if not hasattr(diffusers, scheduler_name):
                raise RuntimeError(f"Requested scheduler class is unavailable: {scheduler_name}")
            scheduler_cls = getattr(diffusers, scheduler_name)
            scheduler_kwargs = {"flow_shift": flow_shift} if flow_shift is not None else {}
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **scheduler_kwargs)
        elif flow_shift is not None and hasattr(diffusers, "UniPCMultistepScheduler"):
            scheduler_config = getattr(getattr(pipe, "scheduler", None), "config", None)
            current_flow_shift = getattr(scheduler_config, "flow_shift", None)
            if current_flow_shift != flow_shift:
                pipe.scheduler = diffusers.UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)

        if enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        return LoadedWanPipeline(
            pipeline=pipe,
            runtime=self.inspect_runtime(),
            model_id=resolved_model_id,
            pipeline_class=resolved_pipeline_class,
            device=device,
            torch_dtype=dtype_name,
            flow_shift=flow_shift,
        )
