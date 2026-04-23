"""Merged dry-run training contracts for WanCanvas.

This module collapses the previous micro-file contract ladder into one readable
surface so the paper/runtime repository keeps the training dry-run logic in a
single place.
"""

from __future__ import annotations


# === merged from losses.py ===

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class LossTargets:
    diffusion_weight: float = 1.0
    known_region_weight: float = 1.0
    seam_weight: float = 0.25


def describe_loss_targets(targets: LossTargets | None = None) -> dict[str, Any]:
    return asdict(targets or LossTargets())

# === merged from batch_contract.py ===

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..data.contracts import FYCOutpaintSample
from ..models.fyc_sample_bridge import FYCSampleBridgeOutput
from ..pipelines.size_alignment import SizeAlignmentRule, snap_spatial_size, validate_spatial_size
from ..utils.latent_ops import estimate_latent_hw


@dataclass(slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    semantic_role: str
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingBatchContract:
    target_hw: tuple[int, int]
    aligned_target_hw: tuple[int, int]
    latent_hw: tuple[int, int]
    latent_channels: int
    frame_count: int
    input_specs: dict[str, TensorSpec]
    target_specs: dict[str, TensorSpec]
    preserve_state: dict[str, Any]
    invariants: dict[str, bool]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def spec_to_dict(spec: TensorSpec) -> dict[str, Any]:
            payload = asdict(spec)
            payload['shape'] = list(spec.shape)
            payload['notes'] = list(spec.notes)
            return payload

        return {
            'target_hw': list(self.target_hw),
            'aligned_target_hw': list(self.aligned_target_hw),
            'latent_hw': list(self.latent_hw),
            'latent_channels': self.latent_channels,
            'frame_count': self.frame_count,
            'input_specs': {name: spec_to_dict(spec) for name, spec in self.input_specs.items()},
            'target_specs': {name: spec_to_dict(spec) for name, spec in self.target_specs.items()},
            'preserve_state': dict(self.preserve_state),
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingBatchContractBuilder:
    def __init__(self, *, rule: SizeAlignmentRule | None = None, latent_channels: int = 16) -> None:
        if latent_channels <= 0:
            raise ValueError('latent_channels must be positive')
        self.rule = rule or SizeAlignmentRule()
        self.latent_channels = latent_channels

    @staticmethod
    def _mask_hw(mask: Any) -> tuple[int, int]:
        if torch is not None and isinstance(mask, torch.Tensor):
            if mask.ndim == 5:
                return int(mask.shape[-2]), int(mask.shape[-1])
            if mask.ndim == 4:
                return int(mask.shape[-2]), int(mask.shape[-1])
            raise ValueError('known_mask tensor must be [F,1,H,W] or [B,F,1,H,W]')
        if not mask or not mask[0]:
            raise ValueError('known_mask list must be non-empty')
        return len(mask), len(mask[0])

    @staticmethod
    def _known_mask_summary(mask: Any) -> dict[str, int]:
        if torch is not None and isinstance(mask, torch.Tensor):
            values = mask.detach().cpu().to(dtype=torch.int32).reshape(-1).tolist()
        else:
            values = [value for row in mask for value in row]
        return {'known': values.count(0), 'generate': values.count(1)}

    def build(self, sample: FYCOutpaintSample, bridge: FYCSampleBridgeOutput) -> TrainingBatchContract:
        target_hw = (
            sample.canvas_meta.target_region.height if sample.canvas_meta.target_region is not None else self._mask_hw(sample.known_mask)[0],
            sample.canvas_meta.target_region.width if sample.canvas_meta.target_region is not None else self._mask_hw(sample.known_mask)[1],
        )
        is_aligned, alignment_errors = validate_spatial_size(*target_hw, self.rule)
        aligned_target_hw = snap_spatial_size(*target_hw, self.rule, mode='ceil') if not is_aligned else target_hw
        latent_hw = estimate_latent_hw(*aligned_target_hw, vae_scale_factor_spatial=self.rule.vae_scale_factor_spatial)

        bundle = bridge.wrapper_payload['condition_bundle']
        text_shape = tuple(bundle['token_shapes']['text'])
        layout_shape = tuple(bundle['token_shapes']['layout'])
        geometry_shape = tuple(bundle['token_shapes']['geometry'])
        mask_shape = tuple(bundle['token_shapes']['mask'])
        latent_shape = (1, sample.frame_count, self.latent_channels, latent_hw[0], latent_hw[1])
        known_mask_latent_shape = (1, sample.frame_count, 1, latent_hw[0], latent_hw[1])
        known_mask_summary = self._known_mask_summary(sample.known_mask)

        input_specs = {
            'noisy_latents': TensorSpec(latent_shape, 'float32', 'diffusion_model_input'),
            'timesteps': TensorSpec((1,), 'int64', 'diffusion_timestep'),
            'prompt_embeds': TensorSpec(text_shape, 'float32', 'text_condition_tokens'),
            'layout_tokens': TensorSpec(layout_shape, 'float32', 'layout_encoder_tokens'),
            'geometry_tokens': TensorSpec(geometry_shape, 'float32', 'relative_region_embedding_tokens'),
            'mask_tokens': TensorSpec(mask_shape, 'float32', 'known_region_mask_summary_tokens'),
        }
        target_specs = {
            'noise_target': TensorSpec(latent_shape, 'float32', 'diffusion_supervision_target'),
            'known_mask_latent': TensorSpec(known_mask_latent_shape, 'float32', 'known_region_preserve_mask'),
            'known_latents': TensorSpec(latent_shape, 'float32', 'known_region_latent_reference'),
        }
        invariants = {
            'target_size_aligned': is_aligned,
            'latent_hw_positive': latent_hw[0] > 0 and latent_hw[1] > 0,
            'condition_order_valid': bundle['order'] == ['text', 'layout', 'geometry', 'mask'],
            'semantic_roles_valid': bundle['semantic_roles'] == [
                'text',
                'layout_encoder',
                'relative_region_embedding',
                'known_region_mask_summary',
            ],
            'token_dim_consistent': len({text_shape[-1], layout_shape[-1], geometry_shape[-1], mask_shape[-1]}) == 1,
            'frame_count_positive': sample.frame_count > 0,
            'relative_position_length_valid': len(sample.relative_position_raw) == 6 and len(sample.relative_position_norm) == 6,
            'known_mask_has_generate_region': known_mask_summary['generate'] > 0,
            'known_region_state_present': bridge.request.known_region_state is not None,
        }
        metadata = {
            'alignment_quantum': self.rule.quantum,
            'alignment_errors': alignment_errors,
            'snap_applied': aligned_target_hw != target_hw,
            'relative_position_raw': list(sample.relative_position_raw),
            'relative_position_norm': list(sample.relative_position_norm),
            'known_mask_summary': known_mask_summary,
            'condition_bundle_order': list(bundle['order']),
        }
        return TrainingBatchContract(
            target_hw=target_hw,
            aligned_target_hw=aligned_target_hw,
            latent_hw=latent_hw,
            latent_channels=self.latent_channels,
            frame_count=sample.frame_count,
            input_specs=input_specs,
            target_specs=target_specs,
            preserve_state={
                'mode': bridge.request.known_region_state.get('mode'),
                'known_mask_latent_shape': list(known_mask_latent_shape),
                'known_latents_shape': list(latent_shape),
            },
            invariants=invariants,
            metadata=metadata,
        )

# === merged from denoising_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any, Sequence

try:
    import diffusers
except ImportError:  # pragma: no cover
    diffusers = None

from ..pipelines.known_region import describe_preserve_action


@dataclass(slots=True)
class DenoisingStepSpec:
    step_index: int
    timestep: float
    preserve_fraction: float
    blend_alpha: float


@dataclass(slots=True)
class DenoisingLoopContract:
    scheduler_name: str
    mode: str
    flow_shift: float | None
    steps: list[DenoisingStepSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'scheduler_name': self.scheduler_name,
            'mode': self.mode,
            'flow_shift': self.flow_shift,
            'steps': [asdict(step) for step in self.steps],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class DenoisingLoopContractBuilder:
    def generate_timesteps(
        self,
        *,
        scheduler_name: str = 'FlowMatchEulerDiscreteScheduler',
        num_inference_steps: int,
        flow_shift: float | None = 5.0,
        device: str = 'cpu',
    ) -> list[float]:
        if num_inference_steps <= 0:
            raise ValueError('num_inference_steps must be positive')
        if diffusers is None:
            raise RuntimeError('diffusers is required to generate scheduler timesteps')
        if not hasattr(diffusers, scheduler_name):
            raise RuntimeError(f'Unavailable scheduler class: {scheduler_name}')
        scheduler_cls = getattr(diffusers, scheduler_name)
        if scheduler_name == 'FlowMatchEulerDiscreteScheduler':
            scheduler = scheduler_cls(shift=flow_shift or 1.0)
        else:
            scheduler = scheduler_cls()
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = getattr(scheduler, 'timesteps', None)
        if timesteps is None:
            raise RuntimeError(f'{scheduler_name} did not expose timesteps after set_timesteps')
        return [float(x) for x in timesteps.tolist()]

    def build(
        self,
        *,
        mask_2d: list[list[int]],
        timesteps: Sequence[float],
        mode: str = 'overwrite',
        scheduler_name: str = 'FlowMatchEulerDiscreteScheduler',
        flow_shift: float | None = 5.0,
        source: str = 'explicit',
    ) -> DenoisingLoopContract:
        if not timesteps:
            raise ValueError('timesteps must not be empty')
        total_steps = len(timesteps)
        steps: list[DenoisingStepSpec] = []
        preserve_fractions: list[float] = []
        blend_alphas: list[float] = []
        for index, timestep in enumerate(timesteps):
            action = describe_preserve_action(mask_2d, mode=mode, step_index=index, total_steps=total_steps)
            preserve_fractions.append(action.preserve_fraction)
            blend_alphas.append(action.blend_alpha)
            steps.append(
                DenoisingStepSpec(
                    step_index=index,
                    timestep=float(timestep),
                    preserve_fraction=action.preserve_fraction,
                    blend_alpha=action.blend_alpha,
                )
            )

        descending = all(timesteps[idx] > timesteps[idx + 1] for idx in range(total_steps - 1))
        invariants = {
            'step_count_positive': total_steps > 0,
            'timesteps_strictly_descending': descending,
            'first_timestep_greater_than_last': float(timesteps[0]) > float(timesteps[-1]),
            'preserve_fraction_constant': len({round(value, 8) for value in preserve_fractions}) == 1,
            'overwrite_alpha_all_one': mode != 'overwrite' or all(abs(alpha - 1.0) < 1e-6 for alpha in blend_alphas),
            'blend_alpha_non_increasing': mode != 'blend' or all(
                blend_alphas[idx] >= blend_alphas[idx + 1] for idx in range(total_steps - 1)
            ),
            'last_timestep_positive': float(timesteps[-1]) > 0.0,
            'verified_mvp_scheduler': scheduler_name == 'FlowMatchEulerDiscreteScheduler',
        }
        metadata = {
            'source': source,
            'num_inference_steps': total_steps,
            'first_timestep': float(timesteps[0]),
            'last_timestep': float(timesteps[-1]),
            'preserve_fraction': preserve_fractions[0],
        }
        return DenoisingLoopContract(
            scheduler_name=scheduler_name,
            mode=mode,
            flow_shift=flow_shift,
            steps=steps,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from forward_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class LossComponentContract:
    name: str
    weight: float
    prediction_spec: TensorSpec
    target_spec: TensorSpec
    mask_spec: TensorSpec | None = None
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingForwardContract:
    model_output_specs: dict[str, TensorSpec]
    auxiliary_specs: dict[str, TensorSpec]
    loss_components: dict[str, LossComponentContract]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        def spec_to_dict(spec: TensorSpec) -> dict[str, Any]:
            payload = asdict(spec)
            payload['shape'] = list(spec.shape)
            payload['notes'] = list(spec.notes)
            return payload

        def component_to_dict(component: LossComponentContract) -> dict[str, Any]:
            return {
                'name': component.name,
                'weight': component.weight,
                'prediction_spec': spec_to_dict(component.prediction_spec),
                'target_spec': spec_to_dict(component.target_spec),
                'mask_spec': spec_to_dict(component.mask_spec) if component.mask_spec is not None else None,
                'notes': list(component.notes),
            }

        return {
            'model_output_specs': {name: spec_to_dict(spec) for name, spec in self.model_output_specs.items()},
            'auxiliary_specs': {name: spec_to_dict(spec) for name, spec in self.auxiliary_specs.items()},
            'loss_components': {name: component_to_dict(component) for name, component in self.loss_components.items()},
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingForwardContractBuilder:
    def build(self, batch_contract: TrainingBatchContract, losses: LossTargets) -> TrainingForwardContract:
        noise_target = batch_contract.target_specs['noise_target']
        known_latents = batch_contract.target_specs['known_latents']
        known_mask_latent = batch_contract.target_specs['known_mask_latent']

        diffusion_prediction = TensorSpec(
            shape=noise_target.shape,
            dtype='float32',
            semantic_role='wan_dit_diffusion_prediction',
            notes=('matches diffusion supervision target shape',),
        )
        scheduler_step_latents = TensorSpec(
            shape=known_latents.shape,
            dtype='float32',
            semantic_role='scheduler_step_latents',
            notes=('post-scheduler latent state before overwrite preserve action',),
        )
        preserve_blend_latents = TensorSpec(
            shape=known_latents.shape,
            dtype='float32',
            semantic_role='preserve_blend_latents',
            notes=('post-overwrite latent state used for preserve-aware supervision',),
        )
        seam_mask_latent = TensorSpec(
            shape=known_mask_latent.shape,
            dtype='float32',
            semantic_role='seam_boundary_mask',
            notes=('derived from the known-region preserve mask boundary band before seam reduction',),
        )

        model_output_specs = {
            'diffusion_prediction': diffusion_prediction,
            'scheduler_step_latents': scheduler_step_latents,
            'preserve_blend_latents': preserve_blend_latents,
        }
        auxiliary_specs = {
            'known_mask_latent': known_mask_latent,
            'seam_mask_latent': seam_mask_latent,
        }
        loss_components = {
            'diffusion': LossComponentContract(
                name='diffusion_mse',
                weight=float(losses.diffusion_weight),
                prediction_spec=diffusion_prediction,
                target_spec=noise_target,
                notes=('primary diffusion supervision in latent prediction space',),
            ),
            'known_region': LossComponentContract(
                name='known_region_preserve',
                weight=float(losses.known_region_weight),
                prediction_spec=preserve_blend_latents,
                target_spec=known_latents,
                mask_spec=known_mask_latent,
                notes=('overwrite-preserved region should remain anchored to known latent reference',),
            ),
            'seam': LossComponentContract(
                name='seam_consistency',
                weight=float(losses.seam_weight),
                prediction_spec=preserve_blend_latents,
                target_spec=known_latents,
                mask_spec=seam_mask_latent,
                notes=('boundary-band auxiliary term derived from known-region mask edges',),
            ),
        }

        invariants = {
            'diffusion_prediction_matches_noise_target': diffusion_prediction.shape == noise_target.shape,
            'scheduler_latents_match_known_latents': scheduler_step_latents.shape == known_latents.shape,
            'preserve_blend_matches_known_latents': preserve_blend_latents.shape == known_latents.shape,
            'known_mask_latent_broadcastable_to_latents': (
                known_mask_latent.shape[0] == known_latents.shape[0]
                and known_mask_latent.shape[1] == known_latents.shape[1]
                and known_mask_latent.shape[2] == 1
                and known_mask_latent.shape[-2:] == known_latents.shape[-2:]
            ),
            'seam_mask_matches_known_mask_shape': seam_mask_latent.shape == known_mask_latent.shape,
            'loss_weights_non_negative': all(component.weight >= 0.0 for component in loss_components.values()),
            'preserve_losses_require_overwrite_mode': batch_contract.preserve_state.get('mode') == 'overwrite',
            'loss_component_names_unique': len({component.name for component in loss_components.values()}) == len(loss_components),
            'forward_outputs_cover_loss_predictions': {
                component.prediction_spec.semantic_role for component in loss_components.values()
            } <= {spec.semantic_role for spec in model_output_specs.values()},
        }
        metadata = {
            'prediction_parameterization_note': 'Exact Wan training parameterization (noise vs velocity) remains deferred, but tensor shapes are locked to the diffusion supervision target.',
            'aligned_target_hw': list(batch_contract.aligned_target_hw),
            'latent_hw': list(batch_contract.latent_hw),
            'frame_count': batch_contract.frame_count,
            'latent_channels': batch_contract.latent_channels,
            'loss_weights': {
                'diffusion': float(losses.diffusion_weight),
                'known_region': float(losses.known_region_weight),
                'seam': float(losses.seam_weight),
            },
        }
        return TrainingForwardContract(
            model_output_specs=model_output_specs,
            auxiliary_specs=auxiliary_specs,
            loss_components=loss_components,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from execution_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class TrainExecutionPhaseSpec:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    readiness_checks: dict[str, bool]
    deferred: bool = False
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingExecutionContract:
    phases: list[TrainExecutionPhaseSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'phases': [
                {
                    **asdict(phase),
                    'inputs': list(phase.inputs),
                    'outputs': list(phase.outputs),
                    'notes': list(phase.notes),
                }
                for phase in self.phases
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingExecutionContractBuilder:
    def build(
        self,
        *,
        batch_contract: TrainingBatchContract,
        forward_contract: TrainingForwardContract,
        denoising_contract: DenoisingLoopContract,
        request_contract: dict[str, Any],
    ) -> TrainingExecutionContract:
        request_checks = dict(request_contract.get('checks', {}))
        forward_invariants = dict(forward_contract.invariants)
        denoising_invariants = dict(denoising_contract.invariants)
        batch_invariants = dict(batch_contract.invariants)
        loss_components = forward_contract.loss_components

        phases = [
            TrainExecutionPhaseSpec(
                name='validate_request',
                inputs=('prompt_embeds', 'layout_tokens', 'geometry_tokens', 'mask_tokens', 'noisy_latents', 'timesteps', 'known_region_state'),
                outputs=('validated_request',),
                readiness_checks={
                    'request_contract_all_true': all(request_checks.values()),
                    'known_region_state_present': request_checks.get('known_region_state_present', False),
                    'known_region_mode_valid': request_checks.get('known_region_mode_valid', False),
                },
                notes=('Verifies the FYC-conditioned Wan request surface before any train-step tensor materialization.',),
            ),
            TrainExecutionPhaseSpec(
                name='materialize_batch_targets',
                inputs=('validated_request', 'known_mask', 'relative_position'),
                outputs=('noise_target', 'known_latents', 'known_mask_latent', 'preserve_state'),
                readiness_checks={
                    'batch_contract_all_true': all(batch_invariants.values()),
                    'latent_target_matches_noise_target': (
                        batch_contract.target_specs['known_latents'].shape
                        == batch_contract.target_specs['noise_target'].shape
                    ),
                    'known_region_mode_is_overwrite': batch_contract.preserve_state.get('mode') == 'overwrite',
                },
                notes=('Locks FYC preserve geometry into aligned latent supervision tensors.',),
            ),
            TrainExecutionPhaseSpec(
                name='prepare_denoising_schedule',
                inputs=('timesteps', 'known_mask_latent'),
                outputs=('scheduler_timesteps', 'preserve_actions'),
                readiness_checks={
                    'denoising_contract_all_true': all(denoising_invariants.values()),
                    'step_count_positive': denoising_invariants.get('step_count_positive', False),
                    'scheduler_verified_mvp': denoising_invariants.get('verified_mvp_scheduler', False),
                },
                notes=('Binds preserve semantics to the verified Wan MVP scheduler loop.',),
            ),
            TrainExecutionPhaseSpec(
                name='wan_forward',
                inputs=('noisy_latents', 'scheduler_timesteps', 'prompt_embeds', 'layout_tokens', 'geometry_tokens', 'mask_tokens'),
                outputs=('diffusion_prediction', 'scheduler_step_latents', 'preserve_blend_latents'),
                readiness_checks={
                    'forward_contract_all_true': all(forward_invariants.values()),
                    'diffusion_prediction_matches_noise_target': forward_invariants.get('diffusion_prediction_matches_noise_target', False),
                    'preserve_blend_matches_known_latents': forward_invariants.get('preserve_blend_matches_known_latents', False),
                },
                notes=('Represents the DiT forward payload expected immediately before any real loss computation.',),
            ),
            TrainExecutionPhaseSpec(
                name='assemble_losses',
                inputs=('diffusion_prediction', 'preserve_blend_latents', 'noise_target', 'known_latents', 'known_mask_latent', 'seam_mask_latent'),
                outputs=('diffusion_loss', 'known_region_loss', 'seam_loss', 'total_loss'),
                readiness_checks={
                    'loss_components_present': sorted(loss_components) == ['diffusion', 'known_region', 'seam'],
                    'loss_weights_non_negative': all(component.weight >= 0.0 for component in loss_components.values()),
                    'seam_mask_matches_known_mask_shape': forward_invariants.get('seam_mask_matches_known_mask_shape', False),
                },
                notes=('Aggregates diffusion/preserve/seam supervision into a pre-backward scalar objective.',),
            ),
            TrainExecutionPhaseSpec(
                name='backward_and_optimizer',
                inputs=('total_loss',),
                outputs=('updated_weights',),
                readiness_checks={
                    'backward_deferred_by_scope': True,
                    'optimizer_deferred_by_scope': True,
                },
                deferred=True,
                notes=(
                    'The current session intentionally stops before backward/optimizer execution.',
                    'This phase remains deferred until real training is explicitly run.',
                ),
            ),
        ]

        phase_names = [phase.name for phase in phases]
        deferred_names = [phase.name for phase in phases if phase.deferred]
        non_deferred = [phase for phase in phases if not phase.deferred]
        invariants = {
            'phase_order_complete': phase_names == [
                'validate_request',
                'materialize_batch_targets',
                'prepare_denoising_schedule',
                'wan_forward',
                'assemble_losses',
                'backward_and_optimizer',
            ],
            'all_non_deferred_phase_checks_true': all(all(phase.readiness_checks.values()) for phase in non_deferred),
            'loss_phase_after_forward': phase_names.index('assemble_losses') > phase_names.index('wan_forward'),
            'denoising_before_forward': phase_names.index('prepare_denoising_schedule') < phase_names.index('wan_forward'),
            'backward_only_phase_deferred': deferred_names == ['backward_and_optimizer'],
            'forward_outputs_cover_losses': forward_invariants.get('forward_outputs_cover_loss_predictions', False),
            'known_region_losses_require_overwrite_mode': forward_invariants.get('preserve_losses_require_overwrite_mode', False),
            'execution_ready_pre_optimizer': all(
                all(phase.readiness_checks.values()) for phase in non_deferred
            ) and deferred_names == ['backward_and_optimizer'],
            'loss_outputs_present': set(phases[4].outputs) == {'diffusion_loss', 'known_region_loss', 'seam_loss', 'total_loss'},
        }
        metadata = {
            'phase_names': phase_names,
            'deferred_phase_names': deferred_names,
            'scheduler_name': denoising_contract.scheduler_name,
            'denoising_step_count': len(denoising_contract.steps),
            'preserve_mode': batch_contract.preserve_state.get('mode'),
            'loss_component_names': sorted(loss_components),
            'execution_scope_note': 'The train-step contract is locked through total-loss assembly; backward/optimizer execution remains intentionally deferred.',
        }
        return TrainingExecutionContract(
            phases=phases,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from update_scope_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


_MODULE_FILE_HINTS = {
    'layout_encoder': ('WanCanvas/wancanvas/models/layout_encoder.py',),
    'geometry_encoder': ('WanCanvas/wancanvas/models/geometry_encoder.py',),
    'mask_summary_encoder': ('WanCanvas/wancanvas/models/mask_summary.py',),
    'condition_adapter': ('WanCanvas/wancanvas/models/condition_adapter.py',),
    'wan_outpaint_wrapper': (
        'WanCanvas/wancanvas/models/wan_outpaint_wrapper.py',
        'WanCanvas/wancanvas/backbones/wan_loader.py',
    ),
}


@dataclass(slots=True)
class UpdateScopeModuleSpec:
    module_name: str
    role: str
    has_parameters: bool
    parameter_names: tuple[str, ...]
    parameter_shapes: dict[str, list[int]]
    parameter_count: int
    parameter_elements: int
    requires_grad_parameter_count: int
    requires_grad_element_count: int
    direct_update_ready: bool
    rationale: str
    file_hints: tuple[str, ...]


@dataclass(slots=True)
class TrainingUpdateScopeContract:
    modules: list[UpdateScopeModuleSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'modules': [
                {
                    **asdict(module),
                    'parameter_names': list(module.parameter_names),
                    'file_hints': list(module.file_hints),
                }
                for module in self.modules
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingUpdateScopeContractBuilder:
    def _iter_named_parameters(self, module: Any) -> list[tuple[str, Any]]:
        if nn is None or not isinstance(module, nn.Module):
            return []
        return list(module.named_parameters())

    def _apply_requires_grad_policy(self, *, module_name: str, module: Any, declared_trainable_modules: tuple[str, ...]) -> None:
        named_parameters = self._iter_named_parameters(module)
        if not named_parameters:
            return
        requires_grad = module_name in declared_trainable_modules
        for _, parameter in named_parameters:
            parameter.requires_grad_(requires_grad)

    def _role_for_module(
        self,
        *,
        module_name: str,
        has_parameters: bool,
        declared_trainable_modules: tuple[str, ...],
    ) -> tuple[str, str]:
        if module_name == 'wan_outpaint_wrapper':
            return (
                'deferred_backbone_binding',
                'The wrapper is a structural integration surface; direct Wan backbone updates remain deferred until real forward/backward execution.',
            )
        if not has_parameters:
            return (
                'structural_only',
                'The module contributes ordering or request semantics but exposes no direct trainable parameters in the current scaffold.',
            )
        if module_name in declared_trainable_modules:
            return (
                'trainable',
                'The module has direct parameters and is declared as trainable for the first real optimization step.',
            )
        return (
            'frozen',
            'The module has direct parameters but is intentionally frozen for the MVP train-entry scope.',
        )

    def build(
        self,
        *,
        declared_trainable_modules: tuple[str, ...],
        module_registry: dict[str, Any],
        wrapper_forward_enabled: bool,
    ) -> TrainingUpdateScopeContract:
        declared_trainable_modules = tuple(declared_trainable_modules)
        modules: list[UpdateScopeModuleSpec] = []
        all_parameter_names: list[str] = []
        trainable_modules: list[str] = []
        frozen_modules: list[str] = []
        structural_modules: list[str] = []
        deferred_modules: list[str] = []
        total_trainable_parameter_count = 0
        total_trainable_element_count = 0

        for module_name, module in module_registry.items():
            self._apply_requires_grad_policy(
                module_name=module_name,
                module=module,
                declared_trainable_modules=declared_trainable_modules,
            )
            named_parameters = self._iter_named_parameters(module)
            has_parameters = bool(named_parameters)
            role, rationale = self._role_for_module(
                module_name=module_name,
                has_parameters=has_parameters,
                declared_trainable_modules=declared_trainable_modules,
            )
            parameter_names = tuple(name for name, _ in named_parameters)
            parameter_shapes = {
                name: [int(dim) for dim in parameter.shape]
                for name, parameter in named_parameters
            }
            parameter_count = len(named_parameters)
            parameter_elements = sum(int(parameter.numel()) for _, parameter in named_parameters)
            requires_grad_parameter_count = sum(int(bool(parameter.requires_grad)) for _, parameter in named_parameters)
            requires_grad_element_count = sum(int(parameter.numel()) for _, parameter in named_parameters if parameter.requires_grad)
            direct_update_ready = role == 'trainable' and has_parameters and requires_grad_parameter_count == parameter_count

            if role == 'trainable':
                trainable_modules.append(module_name)
                total_trainable_parameter_count += parameter_count
                total_trainable_element_count += requires_grad_element_count
            elif role == 'frozen':
                frozen_modules.append(module_name)
            elif role == 'structural_only':
                structural_modules.append(module_name)
            elif role == 'deferred_backbone_binding':
                deferred_modules.append(module_name)

            all_parameter_names.extend(f'{module_name}.{name}' for name in parameter_names)
            modules.append(
                UpdateScopeModuleSpec(
                    module_name=module_name,
                    role=role,
                    has_parameters=has_parameters,
                    parameter_names=parameter_names,
                    parameter_shapes=parameter_shapes,
                    parameter_count=parameter_count,
                    parameter_elements=parameter_elements,
                    requires_grad_parameter_count=requires_grad_parameter_count,
                    requires_grad_element_count=requires_grad_element_count,
                    direct_update_ready=direct_update_ready,
                    rationale=rationale,
                    file_hints=_MODULE_FILE_HINTS.get(module_name, ()),
                )
            )

        module_names = [module.module_name for module in modules]
        trainable_specs = [module for module in modules if module.role == 'trainable']
        frozen_specs = [module for module in modules if module.role == 'frozen']
        mask_summary_spec = next((module for module in modules if module.module_name == 'mask_summary_encoder'), None)
        wrapper_spec = next((module for module in modules if module.module_name == 'wan_outpaint_wrapper'), None)

        invariants = {
            'declared_trainable_modules_covered': all(name in module_names for name in declared_trainable_modules),
            'parameterized_trainable_modules_present': any(module.has_parameters for module in trainable_specs),
            'parameterized_trainable_modules_require_grad': all(
                module.parameter_count == module.requires_grad_parameter_count
                for module in trainable_specs
                if module.has_parameters
            ),
            'parameterized_frozen_modules_require_grad_false': all(
                module.requires_grad_parameter_count == 0
                for module in frozen_specs
                if module.has_parameters
            ),
            'mask_summary_frozen_by_default': bool(
                mask_summary_spec
                and mask_summary_spec.role == 'frozen'
                and mask_summary_spec.requires_grad_parameter_count == 0
            ),
            'structural_parameter_free_modules_accounted_for': set(structural_modules) >= {'condition_adapter'},
            'wrapper_scope_deferred_until_real_train_forward': bool(
                wrapper_spec
                and wrapper_spec.role == 'deferred_backbone_binding'
                and not wrapper_forward_enabled
            ),
            'update_targets_exist_pre_optimizer': total_trainable_parameter_count > 0 and total_trainable_element_count > 0,
            'trainable_modules_match_declared_param_scope': set(trainable_modules) == {
                name
                for name in declared_trainable_modules
                if name in {'layout_encoder', 'geometry_encoder'}
            },
            'all_parameter_names_unique_within_scope': len(all_parameter_names) == len(set(all_parameter_names)),
            'no_unaccounted_parameterized_modules': all(
                module.role in {'trainable', 'frozen'}
                for module in modules
                if module.has_parameters
            ),
        }
        metadata = {
            'declared_trainable_modules': list(declared_trainable_modules),
            'module_names': module_names,
            'trainable_module_names': trainable_modules,
            'frozen_module_names': frozen_modules,
            'structural_module_names': structural_modules,
            'deferred_module_names': deferred_modules,
            'total_trainable_parameter_count': total_trainable_parameter_count,
            'total_trainable_element_count': total_trainable_element_count,
            'wrapper_forward_enabled': wrapper_forward_enabled,
            'scope_note': (
                'The current WanCanvas MVP locks direct updates to local LE/RRE parameterized modules '
                'while keeping mask summary frozen and the Wan wrapper/backbone update lane deferred.'
            ),
        }
        return TrainingUpdateScopeContract(
            modules=modules,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from module_mode_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any

try:
    from torch import nn
except ImportError:  # pragma: no cover
    nn = None



@dataclass(slots=True)
class ModuleModeSpec:
    module_name: str
    role: str
    has_training_flag: bool
    current_mode: str
    pre_enable_mode: str
    enabled_mode: str
    mode_transition_required: bool
    participates_in_real_forward: bool
    manual_enable_flag: str | None
    requires_grad_parameter_count: int
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingModuleModeContract:
    modules: list[ModuleModeSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'modules': [
                {
                    **asdict(module),
                    'notes': list(module.notes),
                }
                for module in self.modules
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingModuleModeContractBuilder:
    @staticmethod
    def _is_nn_module(module: Any) -> bool:
        return nn is not None and isinstance(module, nn.Module)

    def _apply_pre_enable_mode_policy(self, *, role: str, module: Any) -> None:
        if not self._is_nn_module(module):
            return
        if role == 'trainable':
            module.train()
        elif role == 'frozen':
            module.eval()

    def _current_mode(self, module: Any) -> tuple[bool, str]:
        if not self._is_nn_module(module):
            return False, 'not_applicable'
        return True, 'train' if bool(module.training) else 'eval'

    @staticmethod
    def _pre_enable_mode(role: str) -> str:
        if role == 'trainable':
            return 'train'
        if role == 'frozen':
            return 'eval'
        if role == 'deferred_backbone_binding':
            return 'disabled'
        return 'not_applicable'

    @staticmethod
    def _enabled_mode(role: str) -> str:
        if role == 'trainable':
            return 'train'
        if role == 'frozen':
            return 'eval'
        if role == 'deferred_backbone_binding':
            return 'train_forward_enabled'
        return 'not_applicable'

    @staticmethod
    def _manual_enable_flag(role: str) -> str | None:
        if role in {'trainable', 'frozen', 'structural_only', 'deferred_backbone_binding'}:
            return 'real_train_forward_enabled'
        return None

    def build(
        self,
        *,
        update_scope_contract: TrainingUpdateScopeContract,
        enable_gate_contract: TrainingEnableGateContract,
        module_registry: dict[str, Any],
        wrapper_forward_enabled: bool,
        autograd_enabled: bool,
        optimizer_step_enabled: bool,
    ) -> TrainingModuleModeContract:
        module_specs = {module.module_name: module for module in update_scope_contract.modules}
        modules: list[ModuleModeSpec] = []
        train_mode_names: list[str] = []
        eval_mode_names: list[str] = []
        structural_only_names: list[str] = []
        deferred_names: list[str] = []

        for module_name in update_scope_contract.metadata.get('module_names', []):
            module = module_registry[module_name]
            update_spec = module_specs[module_name]
            role = update_spec.role
            self._apply_pre_enable_mode_policy(role=role, module=module)
            has_training_flag, current_mode = self._current_mode(module)
            pre_enable_mode = self._pre_enable_mode(role)
            enabled_mode = self._enabled_mode(role)
            mode_transition_required = enabled_mode != pre_enable_mode
            manual_enable_flag = self._manual_enable_flag(role)
            participates_in_real_forward = role in {
                'trainable',
                'frozen',
                'structural_only',
                'deferred_backbone_binding',
            }

            if pre_enable_mode == 'train':
                train_mode_names.append(module_name)
            elif pre_enable_mode == 'eval':
                eval_mode_names.append(module_name)
            elif role == 'structural_only':
                structural_only_names.append(module_name)
            elif role == 'deferred_backbone_binding':
                deferred_names.append(module_name)

            modules.append(
                ModuleModeSpec(
                    module_name=module_name,
                    role=role,
                    has_training_flag=has_training_flag,
                    current_mode=current_mode,
                    pre_enable_mode=pre_enable_mode,
                    enabled_mode=enabled_mode,
                    mode_transition_required=mode_transition_required,
                    participates_in_real_forward=participates_in_real_forward,
                    manual_enable_flag=manual_enable_flag,
                    requires_grad_parameter_count=update_spec.requires_grad_parameter_count,
                    notes=(
                        'Pre-enable mode policy keeps trainable token producers in train mode, frozen token producers in eval mode, and wrapper forward explicitly disabled.',
                    ),
                )
            )

        trainable_specs = [module for module in modules if module.role == 'trainable']
        frozen_specs = [module for module in modules if module.role == 'frozen']
        wrapper_spec = next((module for module in modules if module.module_name == 'wan_outpaint_wrapper'), None)
        structural_specs = [module for module in modules if module.role == 'structural_only']
        invariants = {
            'module_names_match_update_scope': [module.module_name for module in modules] == update_scope_contract.metadata.get('module_names', []),
            'trainable_modules_in_train_mode_pre_enable': all(
                (module.current_mode == 'train' and module.pre_enable_mode == 'train' and module.requires_grad_parameter_count > 0)
                for module in trainable_specs
            ),
            'frozen_modules_in_eval_mode_pre_enable': all(
                module.current_mode == 'eval' and module.pre_enable_mode == 'eval' and module.requires_grad_parameter_count == 0
                for module in frozen_specs
            ),
            'structural_modules_need_no_mode_flip': all(
                module.pre_enable_mode == 'not_applicable' and module.enabled_mode == 'not_applicable'
                for module in structural_specs
            ),
            'wrapper_forward_disabled_pre_enable': bool(
                wrapper_spec
                and wrapper_spec.role == 'deferred_backbone_binding'
                and wrapper_spec.pre_enable_mode == 'disabled'
                and not wrapper_forward_enabled
            ),
            'autograd_disabled_pre_enable': autograd_enabled is False,
            'optimizer_step_disabled_pre_enable': optimizer_step_enabled is False,
            'real_forward_participants_accounted_for': all(module.participates_in_real_forward for module in modules),
            'manual_enable_flags_match_policy': [module.manual_enable_flag for module in modules] == [
                'real_train_forward_enabled',
                'real_train_forward_enabled',
                'real_train_forward_enabled',
                'real_train_forward_enabled',
                'real_train_forward_enabled',
            ],
            'enable_gate_dependency_matches_policy': enable_gate_contract.invariants.get('only_policy_blockers_remain', False),
            'only_mode_and_policy_toggles_remain': (
                all(
                    [
                        not wrapper_forward_enabled,
                        autograd_enabled is False,
                        optimizer_step_enabled is False,
                        enable_gate_contract.invariants.get('only_policy_blockers_remain', False),
                    ]
                )
            ),
        }
        metadata = {
            'module_names': [module.module_name for module in modules],
            'train_mode_module_names': train_mode_names,
            'eval_mode_module_names': eval_mode_names,
            'structural_only_module_names': structural_only_names,
            'deferred_module_names': deferred_names,
            'wrapper_forward_enabled': wrapper_forward_enabled,
            'autograd_enabled': autograd_enabled,
            'optimizer_step_enabled': optimizer_step_enabled,
            'grad_enabled_required_when_real_train_starts': True,
            'mode_scope_note': (
                'The WanCanvas MVP now locks the pre-enable module mode surface: LE/RRE stay in train mode, the frozen mask summary encoder stays in eval mode, '
                'and wrapper forward/autograd/optimizer step remain explicitly policy-gated.'
            ),
        }
        return TrainingModuleModeContract(
            modules=modules,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from optimizer_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None



@dataclass(slots=True)
class OptimizerGroupSpec:
    group_name: str
    module_name: str
    parameter_names: tuple[str, ...]
    parameter_shapes: dict[str, list[int]]
    parameter_count: int
    parameter_elements: int
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    grad_clip_norm: float
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingOptimizerContract:
    groups: list[OptimizerGroupSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'groups': [
                {
                    **asdict(group),
                    'parameter_names': list(group.parameter_names),
                    'betas': list(group.betas),
                    'notes': list(group.notes),
                }
                for group in self.groups
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingOptimizerContractBuilder:
    def __init__(
        self,
        *,
        optimizer_class_name: str = 'AdamW',
        default_learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        grad_clip_norm: float = 1.0,
        zero_grad_strategy: str = 'set_to_none',
    ) -> None:
        self.optimizer_class_name = optimizer_class_name
        self.default_learning_rate = default_learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.grad_clip_norm = grad_clip_norm
        self.zero_grad_strategy = zero_grad_strategy

    @staticmethod
    def _iter_named_parameters(module: Any) -> list[tuple[str, Any]]:
        if nn is None or not isinstance(module, nn.Module):
            return []
        return list(module.named_parameters())

    def _group_hyperparams(self, module_name: str) -> dict[str, Any]:
        # MVP keeps a simple two-group optimizer surface while still leaving room
        # for later module-specific scheduling.
        lr = self.default_learning_rate
        if module_name == 'geometry_encoder':
            lr = self.default_learning_rate
        return {
            'learning_rate': lr,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'eps': self.eps,
            'grad_clip_norm': self.grad_clip_norm,
        }

    def build(
        self,
        *,
        update_scope_contract: TrainingUpdateScopeContract,
        execution_contract: TrainingExecutionContract,
        module_registry: dict[str, Any],
    ) -> TrainingOptimizerContract:
        module_specs = {module.module_name: module for module in update_scope_contract.modules}
        trainable_module_names = list(update_scope_contract.metadata.get('trainable_module_names', []))
        groups: list[OptimizerGroupSpec] = []
        optimizer_param_groups: list[dict[str, Any]] = []
        optimizer_param_names: list[str] = []
        materialized_optimizer = None

        for module_name in trainable_module_names:
            module = module_registry[module_name]
            named_parameters = [
                (name, parameter)
                for name, parameter in self._iter_named_parameters(module)
                if bool(getattr(parameter, 'requires_grad', False))
            ]
            hyperparams = self._group_hyperparams(module_name)
            parameter_names = tuple(name for name, _ in named_parameters)
            parameter_shapes = {
                name: [int(dim) for dim in parameter.shape]
                for name, parameter in named_parameters
            }
            parameter_count = len(named_parameters)
            parameter_elements = sum(int(parameter.numel()) for _, parameter in named_parameters)
            groups.append(
                OptimizerGroupSpec(
                    group_name=f'{module_name}_group',
                    module_name=module_name,
                    parameter_names=parameter_names,
                    parameter_shapes=parameter_shapes,
                    parameter_count=parameter_count,
                    parameter_elements=parameter_elements,
                    learning_rate=float(hyperparams['learning_rate']),
                    weight_decay=float(hyperparams['weight_decay']),
                    betas=tuple(float(value) for value in hyperparams['betas']),
                    eps=float(hyperparams['eps']),
                    grad_clip_norm=float(hyperparams['grad_clip_norm']),
                    notes=(
                        'Materialized with real requires-grad parameters but intentionally never stepped.',
                        'Optimizer execution remains deferred until real backward is enabled.',
                    ),
                )
            )
            optimizer_param_names.extend(f'{module_name}.{name}' for name in parameter_names)
            optimizer_param_groups.append({
                'params': [parameter for _, parameter in named_parameters],
                'lr': hyperparams['learning_rate'],
                'weight_decay': hyperparams['weight_decay'],
                'betas': hyperparams['betas'],
                'eps': hyperparams['eps'],
            })

        if torch is not None and optimizer_param_groups:
            optimizer_cls = getattr(torch.optim, self.optimizer_class_name)
            materialized_optimizer = optimizer_cls(optimizer_param_groups)

        total_group_parameter_count = sum(group.parameter_count for group in groups)
        total_group_element_count = sum(group.parameter_elements for group in groups)
        expected_trainable_parameter_count = sum(
            module.requires_grad_parameter_count
            for module in update_scope_contract.modules
            if module.role == 'trainable'
        )
        expected_trainable_element_count = sum(
            module.requires_grad_element_count
            for module in update_scope_contract.modules
            if module.role == 'trainable'
        )
        frozen_names = list(update_scope_contract.metadata.get('frozen_module_names', []))
        structural_names = list(update_scope_contract.metadata.get('structural_module_names', []))
        deferred_names = list(update_scope_contract.metadata.get('deferred_module_names', []))
        execution_deferred_phase_names = list(execution_contract.metadata.get('deferred_phase_names', []))

        invariants = {
            'optimizer_groups_present': len(groups) > 0,
            'optimizer_groups_cover_trainable_modules': [group.module_name for group in groups] == trainable_module_names,
            'optimizer_groups_exclude_frozen_modules': all(group.module_name not in frozen_names for group in groups),
            'optimizer_groups_exclude_structural_modules': all(group.module_name not in structural_names for group in groups),
            'optimizer_groups_exclude_deferred_modules': all(group.module_name not in deferred_names for group in groups),
            'optimizer_parameter_names_unique': len(optimizer_param_names) == len(set(optimizer_param_names)),
            'optimizer_parameter_count_matches_update_scope': total_group_parameter_count == expected_trainable_parameter_count,
            'optimizer_parameter_elements_match_update_scope': total_group_element_count == expected_trainable_element_count,
            'all_group_learning_rates_positive': all(group.learning_rate > 0.0 for group in groups),
            'all_group_weight_decay_non_negative': all(group.weight_decay >= 0.0 for group in groups),
            'all_group_betas_valid': all(0.0 <= beta_0 < 1.0 and 0.0 <= beta_1 < 1.0 for group in groups for beta_0, beta_1 in [group.betas]),
            'all_group_eps_positive': all(group.eps > 0.0 for group in groups),
            'materialized_optimizer_created': materialized_optimizer is not None,
            'materialized_optimizer_group_count_matches_contract': (
                materialized_optimizer is not None
                and len(materialized_optimizer.param_groups) == len(groups)
            ),
            'optimizer_state_empty_before_backward': materialized_optimizer is not None and len(materialized_optimizer.state) == 0,
            'backward_phase_still_deferred': execution_deferred_phase_names == ['backward_and_optimizer'],
            'optimizer_step_deferred_by_scope': execution_deferred_phase_names == ['backward_and_optimizer'],
            'optimizer_ready_post_backward': (
                len(groups) > 0
                and total_group_element_count > 0
                and execution_deferred_phase_names == ['backward_and_optimizer']
            ),
        }
        metadata = {
            'optimizer_class_name': self.optimizer_class_name,
            'group_names': [group.group_name for group in groups],
            'group_module_names': [group.module_name for group in groups],
            'group_count': len(groups),
            'zero_grad_strategy': self.zero_grad_strategy,
            'grad_clip_norm': self.grad_clip_norm,
            'default_learning_rate': self.default_learning_rate,
            'default_weight_decay': self.weight_decay,
            'default_betas': list(self.betas),
            'default_eps': self.eps,
            'total_optimizer_parameter_count': total_group_parameter_count,
            'total_optimizer_element_count': total_group_element_count,
            'state_dict_empty_before_backward': bool(materialized_optimizer is not None and len(materialized_optimizer.state) == 0),
            'backward_deferred': execution_deferred_phase_names == ['backward_and_optimizer'],
            'scope_note': (
                'The optimizer surface is materialized with real parameter groups for LE/RRE modules, '
                'but optimizer.step()/zero_grad() remain intentionally outside the current execution scope.'
            ),
        }
        return TrainingOptimizerContract(
            groups=groups,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from backward_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None



@dataclass(slots=True)
class GradientTargetSpec:
    parameter_name: str
    module_name: str
    optimizer_group_name: str
    shape: list[int]
    numel: int
    requires_grad: bool
    grad_is_none_pre_backward: bool


@dataclass(slots=True)
class TrainingBackwardContract:
    gradient_targets: list[GradientTargetSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'gradient_targets': [asdict(target) for target in self.gradient_targets],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingBackwardContractBuilder:
    def __init__(self, *, zero_grad_strategy: str = 'set_to_none') -> None:
        self.zero_grad_strategy = zero_grad_strategy

    @staticmethod
    def _iter_named_parameters(module: Any) -> list[tuple[str, Any]]:
        if nn is None or not isinstance(module, nn.Module):
            return []
        return list(module.named_parameters())

    def _materialize_total_loss_anchor(self, gradient_targets: list[tuple[str, str, str, Any]]) -> Any:
        if torch is None or not gradient_targets:
            return None
        loss = None
        for _, _, _, parameter in gradient_targets:
            term = parameter.reshape(-1)[:1].sum() * 0.0
            loss = term if loss is None else (loss + term)
        return loss

    def build(
        self,
        *,
        update_scope_contract: TrainingUpdateScopeContract,
        optimizer_contract: TrainingOptimizerContract,
        execution_contract: TrainingExecutionContract,
        module_registry: dict[str, Any],
    ) -> TrainingBackwardContract:
        group_name_by_module = {
            group.module_name: group.group_name
            for group in optimizer_contract.groups
        }
        gradient_target_records: list[tuple[str, str, str, Any]] = []
        gradient_targets: list[GradientTargetSpec] = []

        for module_name in optimizer_contract.metadata.get('group_module_names', []):
            module = module_registry[module_name]
            group_name = group_name_by_module[module_name]
            for parameter_name, parameter in self._iter_named_parameters(module):
                if not bool(getattr(parameter, 'requires_grad', False)):
                    continue
                gradient_target_records.append((module_name, group_name, parameter_name, parameter))
                gradient_targets.append(
                    GradientTargetSpec(
                        parameter_name=f'{module_name}.{parameter_name}',
                        module_name=module_name,
                        optimizer_group_name=group_name,
                        shape=[int(dim) for dim in parameter.shape],
                        numel=int(parameter.numel()),
                        requires_grad=bool(parameter.requires_grad),
                        grad_is_none_pre_backward=getattr(parameter, 'grad', None) is None,
                    )
                )

        total_loss = self._materialize_total_loss_anchor(gradient_target_records)
        total_loss_shape = [int(dim) for dim in total_loss.shape] if total_loss is not None else None
        total_loss_requires_grad = bool(getattr(total_loss, 'requires_grad', False)) if total_loss is not None else False

        optimizer_param_names = {
            f'{group.module_name}.{parameter_name}'
            for group in optimizer_contract.groups
            for parameter_name in group.parameter_names
        }
        gradient_param_names = {target.parameter_name for target in gradient_targets}
        update_scope_trainable = set(update_scope_contract.metadata.get('trainable_module_names', []))
        execution_deferred = list(execution_contract.metadata.get('deferred_phase_names', []))
        total_gradient_elements = sum(target.numel for target in gradient_targets)

        invariants = {
            'gradient_targets_present': len(gradient_targets) > 0,
            'gradient_targets_match_optimizer_parameters': gradient_param_names == optimizer_param_names,
            'gradient_targets_only_cover_trainable_modules': all(target.module_name in update_scope_trainable for target in gradient_targets),
            'gradient_targets_require_grad': all(target.requires_grad for target in gradient_targets),
            'gradient_buffers_empty_pre_backward': all(target.grad_is_none_pre_backward for target in gradient_targets),
            'total_loss_scalar': total_loss is not None and total_loss.ndim == 0,
            'total_loss_requires_grad': total_loss_requires_grad,
            'total_loss_shape_empty': total_loss_shape == [],
            'gradient_group_names_match_optimizer_groups': sorted({target.optimizer_group_name for target in gradient_targets}) == sorted(
                optimizer_contract.metadata.get('group_names', [])
            ),
            'gradient_target_count_matches_optimizer': len(gradient_targets) == optimizer_contract.metadata.get('total_optimizer_parameter_count'),
            'gradient_target_elements_match_optimizer': total_gradient_elements == optimizer_contract.metadata.get('total_optimizer_element_count'),
            'zero_grad_strategy_expected': self.zero_grad_strategy == optimizer_contract.metadata.get('zero_grad_strategy'),
            'backward_phase_still_deferred': execution_deferred == ['backward_and_optimizer'],
            'optimizer_step_still_deferred': execution_deferred == ['backward_and_optimizer'],
            'backward_ready_once_enabled': (
                len(gradient_targets) > 0
                and total_loss is not None
                and total_loss_requires_grad
                and execution_deferred == ['backward_and_optimizer']
            ),
        }
        metadata = {
            'gradient_group_names': sorted({target.optimizer_group_name for target in gradient_targets}),
            'gradient_target_count': len(gradient_targets),
            'gradient_target_elements': total_gradient_elements,
            'total_loss_shape': total_loss_shape,
            'total_loss_requires_grad': total_loss_requires_grad,
            'zero_grad_strategy': self.zero_grad_strategy,
            'backward_deferred': execution_deferred == ['backward_and_optimizer'],
            'loss_anchor_note': (
                'A scalar total-loss anchor is materialized directly from trainable parameters so the backward lane can be verified '
                'without invoking real model forward/backward.'
            ),
        }
        return TrainingBackwardContract(
            gradient_targets=gradient_targets,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from step_readiness_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class TrainStepReadinessOperationSpec:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    readiness_checks: dict[str, bool]
    deferred: bool = True
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingStepReadinessContract:
    operations: list[TrainStepReadinessOperationSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'operations': [
                {
                    **asdict(operation),
                    'inputs': list(operation.inputs),
                    'outputs': list(operation.outputs),
                    'notes': list(operation.notes),
                }
                for operation in self.operations
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingStepReadinessContractBuilder:
    def build(
        self,
        *,
        optimizer_contract: TrainingOptimizerContract,
        backward_contract: TrainingBackwardContract,
    ) -> TrainingStepReadinessContract:
        optimizer_group_names = list(optimizer_contract.metadata.get('group_names', []))
        backward_group_names = list(backward_contract.metadata.get('gradient_group_names', []))
        grad_clip_norm = float(optimizer_contract.metadata.get('grad_clip_norm', 0.0))
        zero_grad_strategy = str(optimizer_contract.metadata.get('zero_grad_strategy'))

        operations = [
            TrainStepReadinessOperationSpec(
                name='zero_grad',
                inputs=('optimizer_groups',),
                outputs=('cleared_gradient_buffers',),
                readiness_checks={
                    'optimizer_groups_present': optimizer_contract.invariants.get('optimizer_groups_present', False),
                    'gradient_buffers_empty_pre_backward': backward_contract.invariants.get('gradient_buffers_empty_pre_backward', False),
                    'zero_grad_strategy_matches_backward': zero_grad_strategy == backward_contract.metadata.get('zero_grad_strategy'),
                },
                notes=(
                    'The real training loop should begin by zeroing the optimizer-managed gradient buffers.',
                    'Current MVP keeps this operation deferred while still proving the expected strategy and target groups.',
                ),
            ),
            TrainStepReadinessOperationSpec(
                name='backward',
                inputs=('total_loss', 'trainable_parameters'),
                outputs=('parameter_gradients',),
                readiness_checks={
                    'total_loss_scalar_ready': backward_contract.invariants.get('total_loss_scalar', False),
                    'total_loss_requires_grad': backward_contract.invariants.get('total_loss_requires_grad', False),
                    'gradient_targets_match_optimizer': backward_contract.invariants.get('gradient_targets_match_optimizer_parameters', False),
                },
                notes=(
                    'The train-side proof surface already exposes a scalar, grad-bearing total loss anchor.',
                    'Real backward execution remains deferred until full training is intentionally launched.',
                ),
            ),
            TrainStepReadinessOperationSpec(
                name='clip_grad_norm',
                inputs=('parameter_gradients',),
                outputs=('clipped_parameter_gradients',),
                readiness_checks={
                    'grad_clip_norm_positive': grad_clip_norm > 0.0,
                    'gradient_groups_match_optimizer_groups': sorted(backward_group_names) == sorted(optimizer_group_names),
                    'optimizer_ready_post_backward': optimizer_contract.invariants.get('optimizer_ready_post_backward', False),
                },
                notes=(
                    'Gradient clipping is locked as part of the first real optimizer lane.',
                    'The clip target groups are required to match the optimizer param-group surface exactly.',
                ),
            ),
            TrainStepReadinessOperationSpec(
                name='optimizer_step',
                inputs=('clipped_parameter_gradients', 'optimizer_state'),
                outputs=('updated_trainable_parameters', 'next_optimizer_state'),
                readiness_checks={
                    'optimizer_step_deferred': backward_contract.invariants.get('optimizer_step_still_deferred', False),
                    'optimizer_state_empty_pre_backward': optimizer_contract.invariants.get('optimizer_state_empty_before_backward', False),
                    'optimizer_groups_match_gradients': sorted(backward_group_names) == sorted(optimizer_group_names),
                },
                notes=(
                    'This operation remains intentionally deferred in the current scope.',
                    'The contract exists to prove that the first step can be turned on without changing tensor/module ownership semantics.',
                ),
            ),
        ]

        operation_names = [operation.name for operation in operations]
        deferred_operation_names = [operation.name for operation in operations if operation.deferred]
        all_operation_checks_true = all(all(operation.readiness_checks.values()) for operation in operations)
        invariants = {
            'operation_order_complete': operation_names == ['zero_grad', 'backward', 'clip_grad_norm', 'optimizer_step'],
            'all_operation_checks_true': all_operation_checks_true,
            'zero_grad_strategy_matches_backward': operations[0].readiness_checks['zero_grad_strategy_matches_backward'],
            'zero_grad_targets_match_optimizer_groups': operations[2].readiness_checks['gradient_groups_match_optimizer_groups'],
            'backward_inputs_match_backward_contract': (
                operations[1].readiness_checks['total_loss_scalar_ready']
                and operations[1].readiness_checks['total_loss_requires_grad']
                and operations[1].readiness_checks['gradient_targets_match_optimizer']
            ),
            'gradient_groups_match_optimizer_groups': operations[2].readiness_checks['gradient_groups_match_optimizer_groups'],
            'grad_clip_norm_positive': operations[2].readiness_checks['grad_clip_norm_positive'],
            'optimizer_step_targets_match_optimizer_groups': operations[3].readiness_checks['optimizer_groups_match_gradients'],
            'deferred_operations_expected': deferred_operation_names == operation_names,
            'scheduler_step_omitted_in_mvp': True,
            'step_ready_once_backward_enabled': all_operation_checks_true and deferred_operation_names == operation_names,
        }
        metadata = {
            'operation_names': operation_names,
            'deferred_operation_names': deferred_operation_names,
            'zero_grad_strategy': zero_grad_strategy,
            'grad_clip_norm': grad_clip_norm,
            'optimizer_group_names': optimizer_group_names,
            'gradient_group_names': backward_group_names,
            'scheduler_step_enabled': False,
            'step_scope_note': (
                'WanCanvas now locks the exact zero_grad -> backward -> clip_grad_norm -> optimizer_step execution lane, '
                'while still deferring real autograd and weight updates.'
            ),
        }
        return TrainingStepReadinessContract(
            operations=operations,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from enable_gate_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class TrainingEnableGateSpec:
    name: str
    structural_checks: dict[str, bool]
    manual_enable_flag: str
    currently_enabled: bool
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingEnableGateContract:
    gates: list[TrainingEnableGateSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'gates': [
                {
                    **asdict(gate),
                    'notes': list(gate.notes),
                }
                for gate in self.gates
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingEnableGateContractBuilder:
    def build(
        self,
        *,
        execution_contract: TrainingExecutionContract,
        update_scope_contract: TrainingUpdateScopeContract,
        optimizer_contract: TrainingOptimizerContract,
        backward_contract: TrainingBackwardContract,
        step_readiness_contract: TrainingStepReadinessContract,
    ) -> TrainingEnableGateContract:
        gates = [
            TrainingEnableGateSpec(
                name='real_train_forward_enable',
                manual_enable_flag='real_train_forward_enabled',
                currently_enabled=False,
                structural_checks={
                    'execution_ready_pre_optimizer': execution_contract.invariants.get('execution_ready_pre_optimizer', False),
                    'wrapper_scope_deferred_by_policy': update_scope_contract.invariants.get('wrapper_scope_deferred_until_real_train_forward', False),
                    'step_lane_ready_once_enabled': step_readiness_contract.invariants.get('step_ready_once_backward_enabled', False),
                },
                notes=(
                    'The only remaining blocker is the explicit policy choice to replace dry-run forward with real train forward.',
                ),
            ),
            TrainingEnableGateSpec(
                name='autograd_backward_enable',
                manual_enable_flag='autograd_backward_enabled',
                currently_enabled=False,
                structural_checks={
                    'backward_contract_all_true': all(backward_contract.invariants.values()),
                    'total_loss_scalar_ready': backward_contract.invariants.get('total_loss_scalar', False),
                    'step_backward_inputs_expected': step_readiness_contract.invariants.get('backward_inputs_match_backward_contract', False),
                },
                notes=(
                    'Autograd can be turned on once the explicit training execution flag is granted.',
                ),
            ),
            TrainingEnableGateSpec(
                name='optimizer_step_enable',
                manual_enable_flag='optimizer_step_enabled',
                currently_enabled=False,
                structural_checks={
                    'optimizer_contract_all_true': all(optimizer_contract.invariants.values()),
                    'step_optimizer_targets_expected': step_readiness_contract.invariants.get('optimizer_step_targets_match_optimizer_groups', False),
                    'grad_clip_lane_expected': step_readiness_contract.invariants.get('grad_clip_norm_positive', False),
                },
                notes=(
                    'Optimizer stepping remains intentionally disabled, but its target groups and clip policy are already locked.',
                ),
            ),
        ]

        gate_names = [gate.name for gate in gates]
        manual_enable_flags = [gate.manual_enable_flag for gate in gates]
        currently_enabled_flags = [gate.currently_enabled for gate in gates]
        all_structural_true = all(all(gate.structural_checks.values()) for gate in gates)
        invariants = {
            'gate_order_complete': gate_names == [
                'real_train_forward_enable',
                'autograd_backward_enable',
                'optimizer_step_enable',
            ],
            'all_gate_structural_checks_true': all_structural_true,
            'all_gates_manual_enable_required': manual_enable_flags == [
                'real_train_forward_enabled',
                'autograd_backward_enabled',
                'optimizer_step_enabled',
            ],
            'all_gates_currently_disabled': currently_enabled_flags == [False, False, False],
            'only_policy_blockers_remain': all_structural_true and currently_enabled_flags == [False, False, False],
            'ready_for_explicit_train_enable': all_structural_true and currently_enabled_flags == [False, False, False],
            'scheduler_step_omitted_in_mvp': step_readiness_contract.metadata.get('scheduler_step_enabled') is False,
            'wrapper_forward_scope_matches_gate': update_scope_contract.invariants.get('wrapper_scope_deferred_until_real_train_forward', False),
        }
        metadata = {
            'gate_names': gate_names,
            'manual_enable_flags': manual_enable_flags,
            'currently_enabled_flags': currently_enabled_flags,
            'scheduler_step_enabled': False,
            'enable_scope_note': (
                'All structural preconditions for real train execution are green; only explicit policy toggles keep the real forward/backward/step lane disabled.'
            ),
        }
        return TrainingEnableGateContract(
            gates=gates,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from autograd_preflight_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from ..models.wan_outpaint_wrapper import WanForwardRequest


@dataclass(slots=True)
class PreflightModuleSpec:
    module_name: str
    role: str
    parameter_device_types: tuple[str, ...]
    parameter_dtypes: tuple[str, ...]
    requires_grad_parameter_count: int
    current_mode: str
    participates_in_real_forward: bool
    manual_enable_flag: str | None
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class PreflightRequestTensorSpec:
    name: str
    shape: list[int] | None
    dtype: str | None
    device_type: str | None
    requires_grad: bool
    semantic_role: str
    is_float_tensor: bool
    is_integer_tensor: bool
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingAutogradPreflightContract:
    modules: list[PreflightModuleSpec]
    request_tensors: list[PreflightRequestTensorSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'modules': [
                {
                    **asdict(module),
                    'parameter_device_types': list(module.parameter_device_types),
                    'parameter_dtypes': list(module.parameter_dtypes),
                    'notes': list(module.notes),
                }
                for module in self.modules
            ],
            'request_tensors': [
                {
                    **asdict(spec),
                    'notes': list(spec.notes),
                }
                for spec in self.request_tensors
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingAutogradPreflightContractBuilder:
    @staticmethod
    def _iter_named_parameters(module: Any) -> list[tuple[str, Any]]:
        if nn is None or not isinstance(module, nn.Module):
            return []
        return list(module.named_parameters())

    @staticmethod
    def _shape_of(value: Any) -> list[int] | None:
        shape = getattr(value, 'shape', None)
        if shape is None:
            return None
        return [int(dim) for dim in shape]

    @staticmethod
    def _dtype_of(value: Any) -> str | None:
        dtype = getattr(value, 'dtype', None)
        if dtype is None:
            return None
        return str(dtype)

    @staticmethod
    def _device_type_of(value: Any) -> str | None:
        device = getattr(value, 'device', None)
        if device is None:
            return None
        return str(device.type)

    @staticmethod
    def _requires_grad_of(value: Any) -> bool:
        return bool(getattr(value, 'requires_grad', False))

    @staticmethod
    def _current_mode(module: Any) -> str:
        if nn is None or not isinstance(module, nn.Module):
            return 'not_applicable'
        return 'train' if bool(module.training) else 'eval'

    def _module_spec(self, *, module_name: str, role: str, module: Any, manual_enable_flag: str | None, participates_in_real_forward: bool, requires_grad_parameter_count: int) -> PreflightModuleSpec:
        named_parameters = self._iter_named_parameters(module)
        device_types = tuple(sorted({self._device_type_of(parameter) or 'unknown' for _, parameter in named_parameters}))
        dtypes = tuple(sorted({self._dtype_of(parameter) or 'unknown' for _, parameter in named_parameters}))
        return PreflightModuleSpec(
            module_name=module_name,
            role=role,
            parameter_device_types=device_types,
            parameter_dtypes=dtypes,
            requires_grad_parameter_count=requires_grad_parameter_count,
            current_mode=self._current_mode(module),
            participates_in_real_forward=participates_in_real_forward,
            manual_enable_flag=manual_enable_flag,
            notes=(
                'Preflight checks the device/dtype/module-mode surface immediately before autograd-on execution would begin.',
            ),
        )

    def _request_tensor_specs(self, request: WanForwardRequest) -> list[PreflightRequestTensorSpec]:
        entries = [
            ('prompt_embeds', request.prompt_embeds, 'text_prompt_embeddings'),
            ('noisy_latents', request.noisy_latents, 'latent_noise_input'),
            ('timesteps', request.timesteps, 'scheduler_timesteps'),
            ('layout_tokens', request.layout_tokens, 'layout_condition_tokens'),
            ('geometry_tokens', request.geometry_tokens, 'relative_region_embedding_tokens'),
            ('mask_tokens', request.mask_tokens, 'known_region_mask_summary_tokens'),
        ]
        specs: list[PreflightRequestTensorSpec] = []
        for name, value, semantic_role in entries:
            dtype = self._dtype_of(value)
            specs.append(
                PreflightRequestTensorSpec(
                    name=name,
                    shape=self._shape_of(value),
                    dtype=dtype,
                    device_type=self._device_type_of(value),
                    requires_grad=self._requires_grad_of(value),
                    semantic_role=semantic_role,
                    is_float_tensor=bool(dtype and 'float' in dtype),
                    is_integer_tensor=bool(dtype and ('int' in dtype or 'long' in dtype)),
                    notes=(
                        'This request tensor must remain device/dtype-compatible with the trainable LE/RRE lane before real autograd execution is enabled.',
                    ),
                )
            )
        return specs

    def build(
        self,
        *,
        update_scope_contract: TrainingUpdateScopeContract,
        module_mode_contract: TrainingModuleModeContract,
        enable_gate_contract: TrainingEnableGateContract,
        module_registry: dict[str, Any],
        request: WanForwardRequest,
        wrapper_forward_enabled: bool,
        autograd_enabled: bool,
    ) -> TrainingAutogradPreflightContract:
        mode_specs = {module.module_name: module for module in module_mode_contract.modules}
        update_specs = {module.module_name: module for module in update_scope_contract.modules}
        modules = [
            self._module_spec(
                module_name=name,
                role=update_specs[name].role,
                module=module_registry[name],
                manual_enable_flag=mode_specs[name].manual_enable_flag,
                participates_in_real_forward=mode_specs[name].participates_in_real_forward,
                requires_grad_parameter_count=update_specs[name].requires_grad_parameter_count,
            )
            for name in update_scope_contract.metadata.get('module_names', [])
        ]
        request_tensors = self._request_tensor_specs(request)

        trainable_modules = [module for module in modules if module.role == 'trainable']
        float_request_tensors = [spec for spec in request_tensors if spec.is_float_tensor]
        timestep_spec = next(spec for spec in request_tensors if spec.name == 'timesteps')
        trainable_device_types = sorted({device for module in trainable_modules for device in module.parameter_device_types if device != 'unknown'})
        trainable_dtypes = sorted({dtype for module in trainable_modules for dtype in module.parameter_dtypes if dtype != 'unknown'})
        request_float_devices = sorted({spec.device_type for spec in float_request_tensors if spec.device_type is not None})
        request_float_dtypes = sorted({spec.dtype for spec in float_request_tensors if spec.dtype is not None})

        invariants = {
            'trainable_modules_share_single_device': len(trainable_device_types) == 1,
            'trainable_modules_share_single_float_dtype': trainable_dtypes == ['torch.float32'],
            'request_float_tensors_share_single_device': len(request_float_devices) == 1,
            'request_float_tensors_match_trainable_device': bool(trainable_device_types and request_float_devices and trainable_device_types == request_float_devices),
            'request_float_tensors_expected_dtype': request_float_dtypes == ['torch.float32'],
            'request_condition_tokens_present': all(
                spec.shape is not None for spec in request_tensors if spec.name in {'layout_tokens', 'geometry_tokens', 'mask_tokens'}
            ),
            'timesteps_integer_dtype_expected': timestep_spec.is_integer_tensor and timestep_spec.dtype in {'torch.int64', 'int64'},
            'module_modes_match_policy_surface': (
                module_mode_contract.metadata.get('train_mode_module_names', []) == ['layout_encoder', 'geometry_encoder']
                and module_mode_contract.metadata.get('eval_mode_module_names', []) == ['mask_summary_encoder']
                and module_mode_contract.metadata.get('structural_only_module_names', []) == ['condition_adapter']
                and module_mode_contract.metadata.get('deferred_module_names', []) == ['wan_outpaint_wrapper']
            ),
            'known_region_mode_overwrite_preflight': isinstance(request.known_region_state, dict) and request.known_region_state.get('mode') == 'overwrite',
            'wrapper_forward_disabled_pre_enable': wrapper_forward_enabled is False,
            'autograd_disabled_pre_enable': autograd_enabled is False,
            'enable_gate_still_policy_only': enable_gate_contract.invariants.get('only_policy_blockers_remain', False),
            'only_explicit_enable_flip_and_grad_context_remain': all(
                [
                    len(trainable_device_types) == 1,
                    trainable_dtypes == ['torch.float32'],
                    len(request_float_devices) == 1,
                    bool(trainable_device_types and request_float_devices and trainable_device_types == request_float_devices),
                    request_float_dtypes == ['torch.float32'],
                    timestep_spec.is_integer_tensor and timestep_spec.dtype in {'torch.int64', 'int64'},
                    wrapper_forward_enabled is False,
                    autograd_enabled is False,
                    enable_gate_contract.invariants.get('only_policy_blockers_remain', False),
                ]
            ),
        }
        metadata = {
            'trainable_module_names': [module.module_name for module in trainable_modules],
            'trainable_module_device_types': trainable_device_types,
            'trainable_module_dtypes': trainable_dtypes,
            'request_float_tensor_names': [spec.name for spec in float_request_tensors],
            'request_float_tensor_device_types': request_float_devices,
            'request_float_tensor_dtypes': request_float_dtypes,
            'request_integer_tensor_names': [spec.name for spec in request_tensors if spec.is_integer_tensor],
            'request_timestep_dtype': timestep_spec.dtype,
            'wrapper_forward_enabled': wrapper_forward_enabled,
            'autograd_enabled': autograd_enabled,
            'grad_context_required_when_enabled': True,
            'mixed_precision_policy': 'fp32_mvp',
            'preflight_note': (
                'The WanCanvas MVP now locks the autograd-on preflight lane: trainable LE/RRE parameters and request tensors are device/dtype aligned, '
                'timesteps stay integer-typed, and only the explicit wrapper-forward/autograd policy flip remains before real execution.'
            ),
        }
        return TrainingAutogradPreflightContract(
            modules=modules,
            request_tensors=request_tensors,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from real_execution_preflight_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class RealExecutionPreflightSurfaceSpec:
    name: str
    structural_checks: dict[str, bool]
    manual_enable_flag: str
    currently_enabled: bool
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingRealExecutionPreflightContract:
    surfaces: list[RealExecutionPreflightSurfaceSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'surfaces': [
                {
                    **asdict(surface),
                    'notes': list(surface.notes),
                }
                for surface in self.surfaces
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingRealExecutionPreflightContractBuilder:
    def build(
        self,
        *,
        execution_contract: TrainingExecutionContract,
        update_scope_contract: TrainingUpdateScopeContract,
        optimizer_contract: TrainingOptimizerContract,
        backward_contract: TrainingBackwardContract,
        step_readiness_contract: TrainingStepReadinessContract,
        enable_gate_contract: TrainingEnableGateContract,
        module_mode_contract: TrainingModuleModeContract,
        autograd_preflight_contract: TrainingAutogradPreflightContract,
        wrapper_forward_enabled: bool,
        autograd_enabled: bool,
        optimizer_step_enabled: bool,
    ) -> TrainingRealExecutionPreflightContract:
        trainable_module_names = list(update_scope_contract.metadata.get('trainable_module_names', []))
        optimizer_group_names = list(optimizer_contract.metadata.get('group_names', []))
        optimizer_group_module_names = list(optimizer_contract.metadata.get('group_module_names', []))
        gradient_group_names = list(backward_contract.metadata.get('gradient_group_names', []))
        step_operation_names = list(step_readiness_contract.metadata.get('operation_names', []))
        gate_names = list(enable_gate_contract.metadata.get('gate_names', []))
        manual_enable_flags = list(enable_gate_contract.metadata.get('manual_enable_flags', []))
        currently_enabled_flags = [wrapper_forward_enabled, autograd_enabled, optimizer_step_enabled]
        autograd_trainable_module_names = list(autograd_preflight_contract.metadata.get('trainable_module_names', []))

        surfaces = [
            RealExecutionPreflightSurfaceSpec(
                name='real_train_forward_lane',
                manual_enable_flag='real_train_forward_enabled',
                currently_enabled=wrapper_forward_enabled,
                structural_checks={
                    'execution_contract_all_true': all(execution_contract.invariants.values()),
                    'module_mode_contract_all_true': all(module_mode_contract.invariants.values()),
                    'autograd_preflight_contract_all_true': all(autograd_preflight_contract.invariants.values()),
                    'wrapper_scope_matches_enable_gate': enable_gate_contract.invariants.get('wrapper_forward_scope_matches_gate', False),
                },
                notes=(
                    'The real train forward lane is structurally ready; only the explicit wrapper-forward enable flag remains off.',
                ),
            ),
            RealExecutionPreflightSurfaceSpec(
                name='autograd_backward_lane',
                manual_enable_flag='autograd_backward_enabled',
                currently_enabled=autograd_enabled,
                structural_checks={
                    'backward_contract_all_true': all(backward_contract.invariants.values()),
                    'step_readiness_contract_all_true': all(step_readiness_contract.invariants.values()),
                    'enable_gate_policy_only': enable_gate_contract.invariants.get('only_policy_blockers_remain', False),
                    'autograd_preflight_contract_all_true': all(autograd_preflight_contract.invariants.values()),
                },
                notes=(
                    'The backward lane is held back only by the explicit autograd enable flag; scalar loss and gradient targets are already locked.',
                ),
            ),
            RealExecutionPreflightSurfaceSpec(
                name='optimizer_step_lane',
                manual_enable_flag='optimizer_step_enabled',
                currently_enabled=optimizer_step_enabled,
                structural_checks={
                    'optimizer_contract_all_true': all(optimizer_contract.invariants.values()),
                    'backward_contract_all_true': all(backward_contract.invariants.values()),
                    'step_readiness_contract_all_true': all(step_readiness_contract.invariants.values()),
                    'scheduler_step_omitted_in_mvp': step_readiness_contract.metadata.get('scheduler_step_enabled') is False,
                },
                notes=(
                    'The optimizer step lane has stable parameter groups and clip policy; only the explicit optimizer-step enable flag remains off.',
                ),
            ),
        ]

        invariants = {
            'surface_names_expected': [surface.name for surface in surfaces] == [
                'real_train_forward_lane',
                'autograd_backward_lane',
                'optimizer_step_lane',
            ],
            'surface_manual_enable_flags_expected': [surface.manual_enable_flag for surface in surfaces] == [
                'real_train_forward_enabled',
                'autograd_backward_enabled',
                'optimizer_step_enabled',
            ],
            'all_surface_structural_checks_true': all(all(surface.structural_checks.values()) for surface in surfaces),
            'all_surfaces_currently_disabled': currently_enabled_flags == [False, False, False],
            'all_upstream_contracts_green': all(
                [
                    all(execution_contract.invariants.values()),
                    all(update_scope_contract.invariants.values()),
                    all(optimizer_contract.invariants.values()),
                    all(backward_contract.invariants.values()),
                    all(step_readiness_contract.invariants.values()),
                    all(enable_gate_contract.invariants.values()),
                    all(module_mode_contract.invariants.values()),
                    all(autograd_preflight_contract.invariants.values()),
                ]
            ),
            'trainable_modules_consistent_across_scope_optimizer_preflight': (
                trainable_module_names == optimizer_group_module_names == autograd_trainable_module_names
            ),
            'optimizer_and_backward_groups_consistent': (
                optimizer_group_names == ['layout_encoder_group', 'geometry_encoder_group']
                and gradient_group_names == ['geometry_encoder_group', 'layout_encoder_group']
            ),
            'step_order_matches_expected_execution_lane': step_operation_names == [
                'zero_grad',
                'backward',
                'clip_grad_norm',
                'optimizer_step',
            ],
            'device_dtype_surface_ready': all(
                [
                    autograd_preflight_contract.invariants.get('trainable_modules_share_single_device', False),
                    autograd_preflight_contract.invariants.get('trainable_modules_share_single_float_dtype', False),
                    autograd_preflight_contract.invariants.get('request_float_tensors_match_trainable_device', False),
                    autograd_preflight_contract.invariants.get('request_float_tensors_expected_dtype', False),
                    autograd_preflight_contract.invariants.get('timesteps_integer_dtype_expected', False),
                ]
            ),
            'module_mode_surface_ready': all(
                [
                    module_mode_contract.invariants.get('trainable_modules_in_train_mode_pre_enable', False),
                    module_mode_contract.invariants.get('frozen_modules_in_eval_mode_pre_enable', False),
                    module_mode_contract.invariants.get('wrapper_forward_disabled_pre_enable', False),
                ]
            ),
            'remaining_blockers_are_policy_toggles_only': (
                all(all(surface.structural_checks.values()) for surface in surfaces)
                and currently_enabled_flags == [False, False, False]
                and enable_gate_contract.invariants.get('only_policy_blockers_remain', False)
                and module_mode_contract.invariants.get('only_mode_and_policy_toggles_remain', False)
                and autograd_preflight_contract.invariants.get('only_explicit_enable_flip_and_grad_context_remain', False)
            ),
            'scheduler_step_omitted_by_mvp_policy': (
                step_readiness_contract.metadata.get('scheduler_step_enabled') is False
                and enable_gate_contract.metadata.get('scheduler_step_enabled') is False
            ),
            'ready_to_flip_real_execution_flags': all(
                [
                    all(all(surface.structural_checks.values()) for surface in surfaces),
                    currently_enabled_flags == [False, False, False],
                    enable_gate_contract.invariants.get('ready_for_explicit_train_enable', False),
                ]
            ),
        }
        metadata = {
            'surface_names': [surface.name for surface in surfaces],
            'manual_enable_flags': manual_enable_flags,
            'currently_enabled_flags': currently_enabled_flags,
            'trainable_module_names': trainable_module_names,
            'optimizer_group_names': optimizer_group_names,
            'gradient_group_names': gradient_group_names,
            'step_operation_names': step_operation_names,
            'scheduler_step_enabled': False,
            'mixed_precision_policy': autograd_preflight_contract.metadata.get('mixed_precision_policy', 'fp32_mvp'),
            'remaining_policy_toggles': [
                flag for flag, enabled in zip(manual_enable_flags, currently_enabled_flags, strict=True) if not enabled
            ],
            'preflight_scope_note': (
                'WanCanvas now locks the real-execution preflight lane: request/batch/forward/update/optimizer/backward/step/module-mode/autograd-preflight surfaces are green, '
                'and only the explicit real_train_forward/autograd_backward/optimizer_step policy toggles remain disabled.'
            ),
        }
        return TrainingRealExecutionPreflightContract(
            surfaces=surfaces,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from dry_execution_lane_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class DryExecutionLaneSurfaceSpec:
    name: str
    readiness_checks: dict[str, bool]
    current_state: str
    next_state_if_enabled: str
    mutates_weights: bool
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingDryExecutionLaneContract:
    surfaces: list[DryExecutionLaneSurfaceSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'surfaces': [
                {
                    **asdict(surface),
                    'notes': list(surface.notes),
                }
                for surface in self.surfaces
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingDryExecutionLaneContractBuilder:
    def build(
        self,
        *,
        forward_contract: TrainingForwardContract,
        execution_contract: TrainingExecutionContract,
        backward_contract: TrainingBackwardContract,
        step_readiness_contract: TrainingStepReadinessContract,
        enable_gate_contract: TrainingEnableGateContract,
        autograd_preflight_contract: TrainingAutogradPreflightContract,
        real_execution_preflight_contract: TrainingRealExecutionPreflightContract,
        wrapper_forward_enabled: bool,
        autograd_enabled: bool,
        optimizer_step_enabled: bool,
    ) -> TrainingDryExecutionLaneContract:
        surfaces = [
            DryExecutionLaneSurfaceSpec(
                name='forward_graph_materialization',
                readiness_checks={
                    'real_execution_preflight_ready': real_execution_preflight_contract.invariants.get('ready_to_flip_real_execution_flags', False),
                    'forward_contract_all_true': all(forward_contract.invariants.values()),
                    'execution_ready_pre_optimizer': execution_contract.invariants.get('execution_ready_pre_optimizer', False),
                    'autograd_preflight_ready': autograd_preflight_contract.invariants.get('only_explicit_enable_flip_and_grad_context_remain', False),
                },
                current_state='deferred_dry_run_payload_only',
                next_state_if_enabled='real_forward_graph_materialized',
                mutates_weights=False,
                notes=(
                    'Current scope stops before any real Wan wrapper forward call; only dry-run payload preparation is executed.',
                    'Once explicitly enabled, this lane would materialize the autograd graph that downstream backward would consume.',
                ),
            ),
            DryExecutionLaneSurfaceSpec(
                name='backward_graph_consumption',
                readiness_checks={
                    'backward_contract_all_true': all(backward_contract.invariants.values()),
                    'scalar_loss_ready': backward_contract.invariants.get('total_loss_scalar', False),
                    'loss_requires_grad': backward_contract.invariants.get('total_loss_requires_grad', False),
                    'step_readiness_matches_backward': step_readiness_contract.invariants.get('backward_inputs_match_backward_contract', False),
                },
                current_state='deferred_no_graph_consumption',
                next_state_if_enabled='backward_consumes_real_forward_graph',
                mutates_weights=False,
                notes=(
                    'The loss anchor exists, but no real autograd graph is consumed while training remains architecture-only.',
                ),
            ),
            DryExecutionLaneSurfaceSpec(
                name='optimizer_weight_mutation',
                readiness_checks={
                    'step_readiness_all_true': all(step_readiness_contract.invariants.values()),
                    'enable_gate_policy_ready': enable_gate_contract.invariants.get('ready_for_explicit_train_enable', False),
                    'scheduler_step_omitted_in_mvp': step_readiness_contract.invariants.get('scheduler_step_omitted_in_mvp', False),
                },
                current_state='deferred_no_weight_mutation',
                next_state_if_enabled='optimizer_updates_trainable_modules',
                mutates_weights=True,
                notes=(
                    'Optimizer stepping is still intentionally out of scope for the current branch, even though param-group and clip policy surfaces are ready.',
                ),
            ),
        ]

        surface_names = [surface.name for surface in surfaces]
        currently_enabled_flags = [wrapper_forward_enabled, autograd_enabled, optimizer_step_enabled]
        invariants = {
            'surface_names_expected': surface_names == [
                'forward_graph_materialization',
                'backward_graph_consumption',
                'optimizer_weight_mutation',
            ],
            'all_surface_checks_true': all(all(surface.readiness_checks.values()) for surface in surfaces),
            'upstream_execution_preflight_green': all(real_execution_preflight_contract.invariants.values()),
            'forward_graph_materialization_deferred_by_scope': (
                wrapper_forward_enabled is False
                and autograd_enabled is False
                and surfaces[0].current_state == 'deferred_dry_run_payload_only'
            ),
            'backward_graph_consumption_deferred_by_scope': (
                autograd_enabled is False
                and surfaces[1].current_state == 'deferred_no_graph_consumption'
            ),
            'weight_mutation_deferred_by_scope': (
                optimizer_step_enabled is False
                and surfaces[2].current_state == 'deferred_no_weight_mutation'
            ),
            'non_mutating_dry_execution_possible': (
                all(all(surface.readiness_checks.values()) for surface in surfaces)
                and currently_enabled_flags == [False, False, False]
            ),
            'remaining_runtime_dependency_is_real_forward_graph_materialization': (
                all(all(surface.readiness_checks.values()) for surface in surfaces)
                and wrapper_forward_enabled is False
                and autograd_enabled is False
            ),
            'no_weight_mutation_performed': optimizer_step_enabled is False,
            'execution_enable_flags_still_disabled': currently_enabled_flags == [False, False, False],
        }
        metadata = {
            'surface_names': surface_names,
            'currently_enabled_flags': currently_enabled_flags,
            'remaining_runtime_dependency': 'real_forward_graph_materialization',
            'weight_mutation_performed': False,
            'autograd_graph_materialized': False,
            'dry_execution_mode': 'non_mutating',
            'deferred_mutation_surfaces': [surface.name for surface in surfaces if surface.mutates_weights],
            'preflight_scope_note': (
                'WanCanvas now distinguishes structural readiness from graph/materialization execution: all upstream train-side contracts are green, '
                'but the branch still intentionally stops before real forward-graph materialization and any weight mutation.'
            ),
        }
        return TrainingDryExecutionLaneContract(
            surfaces=surfaces,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from graph_materialization_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class GraphMaterializationSurfaceSpec:
    name: str
    readiness_checks: dict[str, bool]
    current_state: str
    next_state_if_enabled: str
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingGraphMaterializationContract:
    surfaces: list[GraphMaterializationSurfaceSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'surfaces': [
                {
                    **asdict(surface),
                    'notes': list(surface.notes),
                }
                for surface in self.surfaces
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingGraphMaterializationContractBuilder:
    def build(
        self,
        *,
        forward_contract: TrainingForwardContract,
        execution_contract: TrainingExecutionContract,
        autograd_preflight_contract: TrainingAutogradPreflightContract,
        real_execution_preflight_contract: TrainingRealExecutionPreflightContract,
        dry_execution_lane_contract: TrainingDryExecutionLaneContract,
        runtime_payload: dict[str, Any],
        wrapper_forward_enabled: bool,
        autograd_enabled: bool,
    ) -> TrainingGraphMaterializationContract:
        request_contract = runtime_payload['request_contract']
        request_checks = request_contract['checks']
        request_shapes = request_contract['shapes']
        request_dtypes = request_contract['dtypes']
        condition_bundle = runtime_payload['condition_bundle']
        extras = runtime_payload.get('extras', {})

        latent_hw = list(forward_contract.metadata.get('latent_hw', []))
        frame_count = int(forward_contract.metadata.get('frame_count', 0))
        latent_channels = int(forward_contract.metadata.get('latent_channels', 0))
        noisy_latents_shape = request_shapes.get('noisy_latents') or []
        prompt_embeds_shape = request_shapes.get('prompt_embeds') or []
        concat_shape = condition_bundle.get('concat_shape') or []
        token_shapes = condition_bundle.get('token_shapes', {})

        surfaces = [
            GraphMaterializationSurfaceSpec(
                name='request_tensor_surface',
                readiness_checks={
                    'request_contract_all_true': all(request_checks.values()),
                    'noisy_latents_shape_matches_forward_metadata': noisy_latents_shape == [
                        1,
                        frame_count,
                        latent_channels,
                        *latent_hw,
                    ],
                    'timesteps_dtype_expected': request_dtypes.get('timesteps') == 'torch.int64',
                    'known_region_state_present': runtime_payload.get('has_known_region_state') is True,
                    'known_region_mode_overwrite': request_checks.get('known_region_mode_valid', False),
                },
                current_state='payload_ready_no_forward_call',
                next_state_if_enabled='real_forward_graph_materialized',
                notes=(
                    'The exact Wan wrapper request tensors are already materialized as dry-run payloads.',
                    'No real wrapper forward call is executed in this branch.',
                ),
            ),
            GraphMaterializationSurfaceSpec(
                name='condition_bundle_surface',
                readiness_checks={
                    'bundle_order_expected': condition_bundle.get('order') == ['text', 'layout', 'geometry', 'mask'],
                    'semantic_roles_expected': condition_bundle.get('semantic_roles') == [
                        'text',
                        'layout_encoder',
                        'relative_region_embedding',
                        'known_region_mask_summary',
                    ],
                    'token_keys_match_order': condition_bundle.get('token_keys') == ['text', 'layout', 'geometry', 'mask'],
                    'concat_shape_rank_3': bool(concat_shape and len(concat_shape) == 3),
                    'token_dim_matches_prompt_embeds': bool(
                        concat_shape and prompt_embeds_shape and concat_shape[-1] == prompt_embeds_shape[-1] == 1024
                    ),
                    'strict_dense_mvp': runtime_payload.get('strict_dense_mvp') is True,
                    'all_token_shapes_rank_3': all(bool(shape and len(shape) == 3) for shape in token_shapes.values()),
                },
                current_state='bundle_ready_no_transformer_call',
                next_state_if_enabled='bundle_consumed_by_wan_forward',
                notes=(
                    'The DiT-facing condition bundle is already assembled in text/layout/geometry/mask order.',
                ),
            ),
            GraphMaterializationSurfaceSpec(
                name='wrapper_forward_callsite_surface',
                readiness_checks={
                    'execution_contract_all_true': all(execution_contract.invariants.values()),
                    'autograd_preflight_all_true': all(autograd_preflight_contract.invariants.values()),
                    'real_execution_preflight_ready': real_execution_preflight_contract.invariants.get('ready_to_flip_real_execution_flags', False),
                    'dry_execution_lane_dependency_honest': dry_execution_lane_contract.invariants.get('remaining_runtime_dependency_is_real_forward_graph_materialization', False),
                    'base_pipeline_class_expected': runtime_payload.get('base_pipeline_class') == 'WanPipeline',
                    'base_model_id_expected': runtime_payload.get('base_model_id') == 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
                    'frame_count_matches_request_surface': extras.get('frame_count') == frame_count,
                },
                current_state='callsite_ready_policy_disabled',
                next_state_if_enabled='wan_wrapper_forward_invoked',
                notes=(
                    'The wrapper callsite is structurally ready, but the real forward call remains disabled by policy in architecture-only mode.',
                ),
            ),
        ]

        surface_names = [surface.name for surface in surfaces]
        invariants = {
            'surface_names_expected': surface_names == [
                'request_tensor_surface',
                'condition_bundle_surface',
                'wrapper_forward_callsite_surface',
            ],
            'all_surface_checks_true': all(all(surface.readiness_checks.values()) for surface in surfaces),
            'request_tensor_surface_ready': all(surfaces[0].readiness_checks.values()),
            'condition_bundle_surface_ready': all(surfaces[1].readiness_checks.values()),
            'wrapper_forward_callsite_surface_ready': all(surfaces[2].readiness_checks.values()),
            'upstream_preflight_green': (
                all(autograd_preflight_contract.invariants.values())
                and all(real_execution_preflight_contract.invariants.values())
                and all(dry_execution_lane_contract.invariants.values())
            ),
            'remaining_dependency_matches_dry_execution_lane': (
                dry_execution_lane_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization'
            ),
            'wrapper_forward_still_disabled_by_policy': wrapper_forward_enabled is False,
            'autograd_still_disabled_by_policy': autograd_enabled is False,
            'payload_ready_for_real_forward_graph_materialization': (
                all(all(surface.readiness_checks.values()) for surface in surfaces)
                and wrapper_forward_enabled is False
                and autograd_enabled is False
            ),
            'no_real_forward_call_performed': wrapper_forward_enabled is False and autograd_enabled is False,
        }
        metadata = {
            'surface_names': surface_names,
            'request_tensor_shapes': dict(request_shapes),
            'request_tensor_dtypes': dict(request_dtypes),
            'condition_bundle_order': list(condition_bundle.get('order', [])),
            'condition_bundle_semantic_roles': list(condition_bundle.get('semantic_roles', [])),
            'condition_bundle_token_keys': list(condition_bundle.get('token_keys', [])),
            'condition_bundle_concat_shape': list(concat_shape),
            'base_model_id': runtime_payload.get('base_model_id'),
            'base_pipeline_class': runtime_payload.get('base_pipeline_class'),
            'strict_dense_mvp': runtime_payload.get('strict_dense_mvp'),
            'wrapper_forward_enabled': wrapper_forward_enabled,
            'autograd_enabled': autograd_enabled,
            'remaining_runtime_dependency': 'real_forward_graph_materialization',
            'graph_materialization_mode': 'payload_ready_no_forward_call',
            'callsite_phase_name': 'wan_forward',
            'preflight_scope_note': (
                'WanCanvas now locks the exact payload/callsite surface immediately before real forward-graph materialization. '
                'The request tensors, condition bundle, and wrapper callsite are ready, but the branch still intentionally stops before invoking the real wrapper forward.'
            ),
        }
        return TrainingGraphMaterializationContract(
            surfaces=surfaces,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from pre_forward_dry_call_contract.py ===

from dataclasses import asdict, dataclass
import inspect
from typing import Any

from ..models.wan_outpaint_wrapper import WanForwardRequest, WanOutpaintWrapper


@dataclass(slots=True)
class PreForwardDryCallSurfaceSpec:
    name: str
    readiness_checks: dict[str, bool]
    current_state: str
    next_state_if_enabled: str
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingPreForwardDryCallContract:
    surfaces: list[PreForwardDryCallSurfaceSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'surfaces': [
                {
                    **asdict(surface),
                    'notes': list(surface.notes),
                }
                for surface in self.surfaces
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingPreForwardDryCallContractBuilder:
    def build(
        self,
        *,
        wrapper: WanOutpaintWrapper,
        request: WanForwardRequest,
        graph_materialization_contract: TrainingGraphMaterializationContract,
        dry_execution_lane_contract: TrainingDryExecutionLaneContract,
        wrapper_forward_enabled: bool,
        autograd_enabled: bool,
    ) -> TrainingPreForwardDryCallContract:
        signature = inspect.signature(WanOutpaintWrapper.forward)
        raised_exception: BaseException | None = None
        try:
            wrapper.forward(request)
        except BaseException as exc:  # pragma: no branch - expected architecture-only guard
            raised_exception = exc

        error_text = '' if raised_exception is None else str(raised_exception)
        surface_names = [
            'forward_method_surface',
            'dry_call_rehearsal_surface',
            'mutation_guard_surface',
        ]

        surfaces = [
            PreForwardDryCallSurfaceSpec(
                name='forward_method_surface',
                readiness_checks={
                    'graph_materialization_contract_all_true': all(graph_materialization_contract.invariants.values()),
                    'wrapper_is_expected_type': isinstance(wrapper, WanOutpaintWrapper),
                    'request_is_expected_type': isinstance(request, WanForwardRequest),
                    'forward_signature_has_single_request_param': list(signature.parameters.keys()) == ['self', 'request'],
                    'dry_execution_dependency_honest': dry_execution_lane_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization',
                },
                current_state='callsite_rehearsal_ready',
                next_state_if_enabled='wrapper_forward_rehearsed',
                notes=(
                    'The exact bound forward callsite exists and is ready to be rehearsed without materializing the real graph.',
                ),
            ),
            PreForwardDryCallSurfaceSpec(
                name='dry_call_rehearsal_surface',
                readiness_checks={
                    'dry_call_raised_runtime_error': isinstance(raised_exception, RuntimeError),
                    'error_mentions_architecture_only_scope': 'architecture-only bring-up' in error_text,
                    'error_mentions_use_dry_run_first': 'Use dry_run() first.' in error_text,
                    'error_mentions_prepared_payload': 'Prepared payload:' in error_text,
                    'wrapper_forward_still_disabled_by_policy': wrapper_forward_enabled is False,
                },
                current_state='rehearsed_runtime_error_no_graph_materialization',
                next_state_if_enabled='real_wrapper_forward_invoked',
                notes=(
                    'The dry call intentionally exercises the guardrail path and proves that the branch still stops before real forward graph materialization.',
                ),
            ),
            PreForwardDryCallSurfaceSpec(
                name='mutation_guard_surface',
                readiness_checks={
                    'autograd_disabled_by_policy': autograd_enabled is False,
                    'remaining_runtime_dependency_is_real_forward_graph_materialization': (
                        graph_materialization_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization'
                    ),
                    'graph_materialization_still_payload_only': (
                        graph_materialization_contract.metadata.get('graph_materialization_mode') == 'payload_ready_no_forward_call'
                    ),
                    'dry_execution_mode_non_mutating': dry_execution_lane_contract.metadata.get('dry_execution_mode') == 'non_mutating',
                    'weight_mutation_not_performed': dry_execution_lane_contract.metadata.get('weight_mutation_performed') is False,
                },
                current_state='non_mutating_pre_forward_dry_call',
                next_state_if_enabled='forward_graph_materialized_without_weight_mutation',
                notes=(
                    'Even after rehearsing the disabled callsite, autograd and weight mutation remain intentionally inactive.',
                ),
            ),
        ]

        invariants = {
            'surface_names_expected': [surface.name for surface in surfaces] == surface_names,
            'all_surface_checks_true': all(all(surface.readiness_checks.values()) for surface in surfaces),
            'forward_method_surface_ready': all(surfaces[0].readiness_checks.values()),
            'dry_call_rehearsal_surface_ready': all(surfaces[1].readiness_checks.values()),
            'mutation_guard_surface_ready': all(surfaces[2].readiness_checks.values()),
            'upstream_graph_materialization_green': all(graph_materialization_contract.invariants.values()),
            'dry_call_raised_expected_runtime_error': isinstance(raised_exception, RuntimeError),
            'dry_call_still_pre_forward_only': (
                isinstance(raised_exception, RuntimeError)
                and wrapper_forward_enabled is False
                and autograd_enabled is False
            ),
            'remaining_dependency_is_real_forward_graph_materialization': (
                graph_materialization_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization'
                and dry_execution_lane_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization'
            ),
            'no_real_forward_graph_materialized': graph_materialization_contract.invariants.get('no_real_forward_call_performed', False),
            'no_weight_mutation_performed': dry_execution_lane_contract.invariants.get('no_weight_mutation_performed', False),
        }
        metadata = {
            'surface_names': surface_names,
            'call_target': 'WanOutpaintWrapper.forward',
            'forward_signature': str(signature),
            'request_type_name': type(request).__name__,
            'wrapper_type_name': type(wrapper).__name__,
            'raised_exception_type': type(raised_exception).__name__ if raised_exception is not None else None,
            'raised_exception_message_excerpt': error_text[:240],
            'wrapper_forward_enabled': wrapper_forward_enabled,
            'autograd_enabled': autograd_enabled,
            'remaining_runtime_dependency': 'real_forward_graph_materialization',
            'dry_call_mode': 'rehearsed_runtime_error_no_graph_materialization',
            'callsite_phase_name': 'wan_forward',
            'preflight_scope_note': (
                'WanCanvas now rehearses the disabled Wan wrapper forward callsite itself: the exact request can be passed into '
                'WanOutpaintWrapper.forward, but the branch intentionally observes the architecture-only RuntimeError guard '
                'instead of materializing the real forward graph.'
            ),
        }
        return TrainingPreForwardDryCallContract(
            surfaces=surfaces,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from forward_invocation_gate_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any

from ..models.wan_outpaint_wrapper import WanForwardRequest, WanOutpaintWrapper


@dataclass(slots=True)
class ForwardInvocationGateSurfaceSpec:
    name: str
    readiness_checks: dict[str, bool]
    current_state: str
    next_state_if_enabled: str
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingForwardInvocationGateContract:
    surfaces: list[ForwardInvocationGateSurfaceSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'surfaces': [
                {
                    **asdict(surface),
                    'notes': list(surface.notes),
                }
                for surface in self.surfaces
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingForwardInvocationGateContractBuilder:
    def build(
        self,
        *,
        wrapper: WanOutpaintWrapper,
        request: WanForwardRequest,
        graph_materialization_contract: TrainingGraphMaterializationContract,
        pre_forward_dry_call_contract: TrainingPreForwardDryCallContract,
        dry_execution_lane_contract: TrainingDryExecutionLaneContract,
        wrapper_forward_enabled: bool,
        autograd_enabled: bool,
    ) -> TrainingForwardInvocationGateContract:
        request_summary = wrapper.describe_request(request)
        expected_gate_flag_names = ['wrapper_forward_enabled', 'autograd_enabled']
        surface_names = [
            'invocation_policy_surface',
            'request_summary_surface',
            'remaining_dependency_surface',
        ]

        surfaces = [
            ForwardInvocationGateSurfaceSpec(
                name='invocation_policy_surface',
                readiness_checks={
                    'pre_forward_dry_call_contract_all_true': all(pre_forward_dry_call_contract.invariants.values()),
                    'graph_materialization_contract_all_true': all(graph_materialization_contract.invariants.values()),
                    'gate_flag_names_expected': expected_gate_flag_names == ['wrapper_forward_enabled', 'autograd_enabled'],
                    'wrapper_forward_still_disabled_by_policy': wrapper_forward_enabled is False,
                    'autograd_still_disabled_by_policy': autograd_enabled is False,
                },
                current_state='policy_gated_after_dry_call_rehearsal',
                next_state_if_enabled='wrapper_forward_invoked_without_weight_mutation',
                notes=(
                    'The exact wrapper-forward callsite has already been rehearsed, and the only thing still preventing real invocation is the explicit policy gate.',
                ),
            ),
            ForwardInvocationGateSurfaceSpec(
                name='request_summary_surface',
                readiness_checks={
                    'request_summary_has_prompt_embeds': request_summary.get('prompt_embeds_shape') is not None,
                    'request_summary_has_noisy_latents': request_summary.get('noisy_latents_shape') is not None,
                    'request_summary_has_layout_tokens': request_summary.get('layout_tokens_shape') is not None,
                    'request_summary_has_geometry_tokens': request_summary.get('geometry_tokens_shape') is not None,
                    'request_summary_has_mask_tokens': request_summary.get('mask_tokens_shape') is not None,
                    'request_summary_has_timesteps': request_summary.get('timesteps_shape') is not None,
                    'request_summary_known_region_mode_overwrite': request_summary.get('known_region_mode') == 'overwrite',
                },
                current_state='prepared_request_summary_ready',
                next_state_if_enabled='wrapper_forward_consumes_prepared_request',
                notes=(
                    'The wrapper can already describe the exact request payload that would be passed into the real forward call.',
                ),
            ),
            ForwardInvocationGateSurfaceSpec(
                name='remaining_dependency_surface',
                readiness_checks={
                    'dry_call_was_rehearsed': pre_forward_dry_call_contract.invariants.get('dry_call_raised_expected_runtime_error', False),
                    'dry_call_still_pre_forward_only': pre_forward_dry_call_contract.invariants.get('dry_call_still_pre_forward_only', False),
                    'graph_materialization_still_payload_only': graph_materialization_contract.metadata.get('graph_materialization_mode') == 'payload_ready_no_forward_call',
                    'remaining_runtime_dependency_is_real_forward_graph_materialization': (
                        pre_forward_dry_call_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization'
                        and dry_execution_lane_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization'
                    ),
                    'no_weight_mutation_performed': dry_execution_lane_contract.invariants.get('no_weight_mutation_performed', False),
                },
                current_state='post_rehearsal_pre_graph_materialization',
                next_state_if_enabled='real_forward_graph_materialized_without_weight_mutation',
                notes=(
                    'Even after the dry-call rehearsal, the branch is still intentionally pre-graph-materialization and non-mutating.',
                ),
            ),
        ]

        invariants = {
            'surface_names_expected': [surface.name for surface in surfaces] == surface_names,
            'all_surface_checks_true': all(all(surface.readiness_checks.values()) for surface in surfaces),
            'invocation_policy_surface_ready': all(surfaces[0].readiness_checks.values()),
            'request_summary_surface_ready': all(surfaces[1].readiness_checks.values()),
            'remaining_dependency_surface_ready': all(surfaces[2].readiness_checks.values()),
            'upstream_pre_forward_dry_call_green': all(pre_forward_dry_call_contract.invariants.values()),
            'upstream_graph_materialization_green': all(graph_materialization_contract.invariants.values()),
            'only_policy_gate_flip_remains_for_real_forward_invocation': (
                all(surfaces[0].readiness_checks.values())
                and wrapper_forward_enabled is False
                and autograd_enabled is False
            ),
            'remaining_dependency_is_real_forward_graph_materialization': (
                pre_forward_dry_call_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization'
                and graph_materialization_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization'
                and dry_execution_lane_contract.metadata.get('remaining_runtime_dependency') == 'real_forward_graph_materialization'
            ),
            'no_real_forward_invocation_performed': (
                pre_forward_dry_call_contract.invariants.get('dry_call_still_pre_forward_only', False)
                and graph_materialization_contract.invariants.get('no_real_forward_call_performed', False)
            ),
            'no_weight_mutation_performed': dry_execution_lane_contract.invariants.get('no_weight_mutation_performed', False),
            'still_payload_ready_but_forward_disabled': (
                graph_materialization_contract.invariants.get('payload_ready_for_real_forward_graph_materialization', False)
                and wrapper_forward_enabled is False
                and autograd_enabled is False
            ),
        }
        metadata = {
            'surface_names': surface_names,
            'call_target': 'WanOutpaintWrapper.forward',
            'gate_flag_names': expected_gate_flag_names,
            'current_gate_flags': [wrapper_forward_enabled, autograd_enabled],
            'request_summary_keys': sorted(request_summary.keys()),
            'request_summary_prompt_shape': request_summary.get('prompt_embeds_shape'),
            'request_summary_noisy_latents_shape': request_summary.get('noisy_latents_shape'),
            'request_summary_layout_shape': request_summary.get('layout_tokens_shape'),
            'request_summary_geometry_shape': request_summary.get('geometry_tokens_shape'),
            'request_summary_mask_shape': request_summary.get('mask_tokens_shape'),
            'request_summary_timesteps_shape': request_summary.get('timesteps_shape'),
            'request_summary_known_region_mode': request_summary.get('known_region_mode'),
            'rehearsed_exception_type': pre_forward_dry_call_contract.metadata.get('raised_exception_type'),
            'rehearsed_dry_call_mode': pre_forward_dry_call_contract.metadata.get('dry_call_mode'),
            'remaining_runtime_dependency': 'real_forward_graph_materialization',
            'invocation_gate_mode': 'policy_gated_post_rehearsal',
            'callsite_phase_name': 'wan_forward',
            'preflight_scope_note': (
                'WanCanvas now proves the exact post-rehearsal invocation gate for WanOutpaintWrapper.forward: '
                'the request payload is summarized and ready, the disabled callsite has already been rehearsed, '
                'and only the explicit policy gate remains before real forward graph materialization would begin.'
            ),
        }
        return TrainingForwardInvocationGateContract(
            surfaces=surfaces,
            invariants=invariants,
            metadata=metadata,
        )

# === merged from pre_training_boundary_closure_contract.py ===

from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class PreTrainingBoundaryClosureSurfaceSpec:
    name: str
    readiness_checks: dict[str, bool]
    current_state: str
    next_state_if_enabled: str
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingPreTrainingBoundaryClosureContract:
    surfaces: list[PreTrainingBoundaryClosureSurfaceSpec]
    invariants: dict[str, bool]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'surfaces': [
                {
                    **asdict(surface),
                    'notes': list(surface.notes),
                }
                for surface in self.surfaces
            ],
            'invariants': dict(self.invariants),
            'metadata': dict(self.metadata),
        }


class TrainingPreTrainingBoundaryClosureContractBuilder:
    def build(
        self,
        *,
        real_execution_preflight_contract: TrainingRealExecutionPreflightContract,
        dry_execution_lane_contract: TrainingDryExecutionLaneContract,
        graph_materialization_contract: TrainingGraphMaterializationContract,
        pre_forward_dry_call_contract: TrainingPreForwardDryCallContract,
        forward_invocation_gate_contract: TrainingForwardInvocationGateContract,
        wrapper_forward_enabled: bool,
        autograd_enabled: bool,
        optimizer_step_enabled: bool,
    ) -> TrainingPreTrainingBoundaryClosureContract:
        current_gate_flags = [wrapper_forward_enabled, autograd_enabled, optimizer_step_enabled]
        simulated_next_phase_gate_flags = [True, False, False]
        blocked_flags = ['autograd_enabled', 'optimizer_step_enabled']
        remaining_runtime_dependency = 'real_forward_graph_materialization'
        surface_names = [
            'aggregate_non_mutating_boundary_surface',
            'forward_materialization_terminal_boundary_surface',
            'remaining_runtime_dependency_surface',
        ]

        forward_invocation_gate_metadata = forward_invocation_gate_contract.metadata
        forward_invocation_gate_invariants = forward_invocation_gate_contract.invariants
        graph_materialization_metadata = graph_materialization_contract.metadata
        graph_materialization_invariants = graph_materialization_contract.invariants
        pre_forward_dry_call_metadata = pre_forward_dry_call_contract.metadata
        pre_forward_dry_call_invariants = pre_forward_dry_call_contract.invariants
        dry_execution_lane_metadata = dry_execution_lane_contract.metadata
        dry_execution_lane_invariants = dry_execution_lane_contract.invariants
        real_execution_preflight_invariants = real_execution_preflight_contract.invariants

        surfaces = [
            PreTrainingBoundaryClosureSurfaceSpec(
                name='aggregate_non_mutating_boundary_surface',
                readiness_checks={
                    'real_execution_preflight_contract_all_true': all(real_execution_preflight_invariants.values()),
                    'dry_execution_lane_contract_all_true': all(dry_execution_lane_invariants.values()),
                    'graph_materialization_contract_all_true': all(graph_materialization_invariants.values()),
                    'pre_forward_dry_call_contract_all_true': all(pre_forward_dry_call_invariants.values()),
                    'forward_invocation_gate_contract_all_true': all(forward_invocation_gate_invariants.values()),
                    'current_gate_flags_expected': current_gate_flags == [False, False, False],
                    'simulated_next_phase_gate_flags_expected': simulated_next_phase_gate_flags == [True, False, False],
                    'blocked_flags_expected': blocked_flags == ['autograd_enabled', 'optimizer_step_enabled'],
                    'ready_to_flip_real_execution_flags': real_execution_preflight_invariants.get(
                        'ready_to_flip_real_execution_flags', False
                    ),
                },
                current_state='non_mutating_pre_training_boundary_closed',
                next_state_if_enabled='real_forward_graph_materialization',
                notes=(
                    'This aggregate surface replaces the old forward-only/materialization micro-state ladder with one closure-level handoff.',
                    'Only wrapper_forward_enabled would change in the immediate next phase; autograd and optimizer-step remain deliberately disabled.',
                ),
            ),
            PreTrainingBoundaryClosureSurfaceSpec(
                name='forward_materialization_terminal_boundary_surface',
                readiness_checks={
                    'graph_materialization_contract_all_true': all(graph_materialization_invariants.values()),
                    'pre_forward_dry_call_contract_all_true': all(pre_forward_dry_call_invariants.values()),
                    'forward_invocation_gate_contract_all_true': all(forward_invocation_gate_invariants.values()),
                    'call_target_expected': forward_invocation_gate_metadata.get('call_target') == 'WanOutpaintWrapper.forward',
                    'graph_materialization_mode_expected': (
                        graph_materialization_metadata.get('graph_materialization_mode')
                        == 'payload_ready_no_forward_call'
                    ),
                    'dry_call_mode_expected': (
                        pre_forward_dry_call_metadata.get('dry_call_mode')
                        == 'rehearsed_runtime_error_no_graph_materialization'
                    ),
                    'invocation_gate_mode_expected': (
                        forward_invocation_gate_metadata.get('invocation_gate_mode')
                        == 'policy_gated_post_rehearsal'
                    ),
                    'callsite_phase_name_expected': (
                        graph_materialization_metadata.get('callsite_phase_name')
                        == pre_forward_dry_call_metadata.get('callsite_phase_name')
                        == forward_invocation_gate_metadata.get('callsite_phase_name')
                        == 'wan_forward'
                    ),
                },
                current_state='exact_forward_materialization_terminal_boundary_rehearsed',
                next_state_if_enabled='real_forward_graph_materialization_without_backward_or_step',
                notes=(
                    'The graph payload, dry-call guard path, and invocation gate are all aligned to the exact WanOutpaintWrapper.forward callsite immediately before real graph materialization.',
                ),
            ),
            PreTrainingBoundaryClosureSurfaceSpec(
                name='remaining_runtime_dependency_surface',
                readiness_checks={
                    'remaining_runtime_dependency_matches_across_surfaces': (
                        dry_execution_lane_metadata.get('remaining_runtime_dependency')
                        == graph_materialization_metadata.get('remaining_runtime_dependency')
                        == pre_forward_dry_call_metadata.get('remaining_runtime_dependency')
                        == forward_invocation_gate_metadata.get('remaining_runtime_dependency')
                        == remaining_runtime_dependency
                    ),
                    'no_real_forward_invocation_performed': forward_invocation_gate_invariants.get(
                        'no_real_forward_invocation_performed', False
                    ),
                    'no_real_graph_materialization_performed': graph_materialization_invariants.get(
                        'no_real_forward_call_performed', False
                    ),
                    'no_weight_mutation_performed': dry_execution_lane_invariants.get(
                        'no_weight_mutation_performed', False
                    ),
                    'dry_execution_lane_non_mutating_possible': dry_execution_lane_invariants.get(
                        'non_mutating_dry_execution_possible', False
                    ),
                    'invocation_gate_still_payload_ready': forward_invocation_gate_invariants.get(
                        'still_payload_ready_but_forward_disabled', False
                    ),
                },
                current_state='still_pre_real_forward_graph_materialization',
                next_state_if_enabled='real_forward_graph_materialized_without_backward_or_step',
                notes=(
                    'The only honest remaining runtime dependency is real forward graph materialization; backward and weight mutation remain outside this closure surface.',
                ),
            ),
        ]

        invariants = {
            'surface_names_expected': [surface.name for surface in surfaces] == surface_names,
            'all_surface_checks_true': all(all(surface.readiness_checks.values()) for surface in surfaces),
            'aggregate_non_mutating_boundary_surface_ready': all(surfaces[0].readiness_checks.values()),
            'forward_materialization_terminal_boundary_surface_ready': all(surfaces[1].readiness_checks.values()),
            'remaining_runtime_dependency_surface_ready': all(surfaces[2].readiness_checks.values()),
            'upstream_non_mutating_boundary_green': all(
                [
                    all(real_execution_preflight_invariants.values()),
                    all(dry_execution_lane_invariants.values()),
                    all(graph_materialization_invariants.values()),
                    all(pre_forward_dry_call_invariants.values()),
                    all(forward_invocation_gate_invariants.values()),
                ]
            ),
            'current_gate_flags_expected': current_gate_flags == [False, False, False],
            'simulated_next_phase_gate_flags_expected': simulated_next_phase_gate_flags == [True, False, False],
            'remaining_runtime_dependency_is_real_forward_graph_materialization': (
                dry_execution_lane_metadata.get('remaining_runtime_dependency')
                == graph_materialization_metadata.get('remaining_runtime_dependency')
                == pre_forward_dry_call_metadata.get('remaining_runtime_dependency')
                == forward_invocation_gate_metadata.get('remaining_runtime_dependency')
                == remaining_runtime_dependency
            ),
            'terminal_non_mutating_boundary_closed': (
                all(all(surface.readiness_checks.values()) for surface in surfaces)
                and current_gate_flags == [False, False, False]
                and simulated_next_phase_gate_flags == [True, False, False]
            ),
            'next_phase_is_real_forward_graph_materialization': True,
            'real_forward_still_deferred': (
                forward_invocation_gate_invariants.get('no_real_forward_invocation_performed', False)
                and graph_materialization_invariants.get('no_real_forward_call_performed', False)
                and dry_execution_lane_invariants.get('no_weight_mutation_performed', False)
            ),
        }
        metadata = {
            'surface_names': surface_names,
            'call_target': 'WanOutpaintWrapper.forward',
            'current_gate_flags': current_gate_flags,
            'simulated_next_phase_gate_flags': simulated_next_phase_gate_flags,
            'blocked_flags': blocked_flags,
            'remaining_runtime_dependency': remaining_runtime_dependency,
            'deepest_upstream_contract_name': 'forward_invocation_gate',
            'closure_mode': 'non_mutating_pre_training_boundary_closed',
            'next_phase': 'real_forward_graph_materialization',
            'next_non_mutating_lane': 'real_forward_graph_materialization_without_backward_or_step',
            'closure_scope_note': (
                'WanCanvas now exposes one convergence surface for the non-mutating pre-training boundary: all meaningful upstream proof surfaces are green, '
                'the exact remaining dependency is real forward graph materialization, and no real forward/backward/step has been executed.'
            ),
        }
        return TrainingPreTrainingBoundaryClosureContract(
            surfaces=surfaces,
            invariants=invariants,
            metadata=metadata,
        )
