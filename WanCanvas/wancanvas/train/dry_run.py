"""Merged dry-run trainer surface for WanCanvas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..data.contracts import FYCOutpaintSample
from ..models.fyc_sample_bridge import FYCSampleToWanBridge
from ..models.wan_outpaint_wrapper import WanOutpaintWrapper
from .contracts import (
    DenoisingLoopContractBuilder,
    LossTargets,
    TrainingAutogradPreflightContractBuilder,
    TrainingBackwardContractBuilder,
    TrainingBatchContractBuilder,
    TrainingDryExecutionLaneContractBuilder,
    TrainingEnableGateContractBuilder,
    TrainingExecutionContractBuilder,
    TrainingForwardContractBuilder,
    TrainingForwardInvocationGateContractBuilder,
    TrainingGraphMaterializationContractBuilder,
    TrainingModuleModeContractBuilder,
    TrainingOptimizerContractBuilder,
    TrainingPreForwardDryCallContractBuilder,
    TrainingPreTrainingBoundaryClosureContractBuilder,
    TrainingRealExecutionPreflightContractBuilder,
    TrainingStepReadinessContractBuilder,
    TrainingUpdateScopeContractBuilder,
    describe_loss_targets,
)

# === merged from step_builder.py ===

from dataclasses import dataclass
from typing import Any



@dataclass(slots=True)
class TrainStepPlan:
    prompt: str
    fps: int
    frame_count: int
    relative_position_raw: tuple[int, int, int, int, int, int]
    known_mask_summary: dict[str, int]
    trainable_modules: tuple[str, ...]
    metadata: dict[str, Any]


class TrainStepBuilder:
    def __init__(self, trainable_modules: tuple[str, ...]) -> None:
        self.trainable_modules = trainable_modules

    def build(self, sample: FYCOutpaintSample) -> TrainStepPlan:
        flat = [value for row in sample.known_mask for value in row]
        return TrainStepPlan(
            prompt=sample.prompt,
            fps=sample.fps,
            frame_count=sample.frame_count,
            relative_position_raw=sample.relative_position_raw,
            known_mask_summary={"known": flat.count(0), "generate": flat.count(1)},
            trainable_modules=self.trainable_modules,
            metadata={
                "canvas_height": sample.canvas_meta.canvas_height,
                "canvas_width": sample.canvas_meta.canvas_width,
                "source_id": sample.canvas_meta.source_id,
            },
        )

# === merged from trainer.py ===

from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class TrainingDryRunReport:
    step_plan: dict[str, Any]
    bridge: dict[str, Any]
    batch_contract: dict[str, Any]
    forward_contract: dict[str, Any]
    module_mode_contract: dict[str, Any]
    autograd_preflight_contract: dict[str, Any]
    real_execution_preflight_contract: dict[str, Any]
    dry_execution_lane_contract: dict[str, Any]
    graph_materialization_contract: dict[str, Any]
    pre_forward_dry_call_contract: dict[str, Any]
    forward_invocation_gate_contract: dict[str, Any]
    pre_training_boundary_closure_contract: dict[str, Any]
    execution_contract: dict[str, Any]
    update_scope_contract: dict[str, Any]
    optimizer_contract: dict[str, Any]
    backward_contract: dict[str, Any]
    step_readiness_contract: dict[str, Any]
    enable_gate_contract: dict[str, Any]
    wrapper_payload: dict[str, Any]
    losses: dict[str, Any]
    consistency_checks: dict[str, bool]


class DryRunTrainer:
    def __init__(
        self,
        wrapper: WanOutpaintWrapper,
        step_builder: TrainStepBuilder,
        losses: LossTargets | None = None,
        sample_bridge: FYCSampleToWanBridge | None = None,
        batch_contract_builder: TrainingBatchContractBuilder | None = None,
        forward_contract_builder: TrainingForwardContractBuilder | None = None,
        denoising_contract_builder: DenoisingLoopContractBuilder | None = None,
        execution_contract_builder: TrainingExecutionContractBuilder | None = None,
        module_mode_contract_builder: TrainingModuleModeContractBuilder | None = None,
        autograd_preflight_contract_builder: TrainingAutogradPreflightContractBuilder | None = None,
        real_execution_preflight_contract_builder: TrainingRealExecutionPreflightContractBuilder | None = None,
        dry_execution_lane_contract_builder: TrainingDryExecutionLaneContractBuilder | None = None,
        graph_materialization_contract_builder: TrainingGraphMaterializationContractBuilder | None = None,
        pre_forward_dry_call_contract_builder: TrainingPreForwardDryCallContractBuilder | None = None,
        forward_invocation_gate_contract_builder: TrainingForwardInvocationGateContractBuilder | None = None,
        pre_training_boundary_closure_contract_builder: TrainingPreTrainingBoundaryClosureContractBuilder | None = None,
        update_scope_contract_builder: TrainingUpdateScopeContractBuilder | None = None,
        optimizer_contract_builder: TrainingOptimizerContractBuilder | None = None,
        backward_contract_builder: TrainingBackwardContractBuilder | None = None,
        step_readiness_contract_builder: TrainingStepReadinessContractBuilder | None = None,
        enable_gate_contract_builder: TrainingEnableGateContractBuilder | None = None,
        num_denoising_steps: int = 6,
    ) -> None:
        self.wrapper = wrapper
        self.step_builder = step_builder
        self.losses = losses or LossTargets()
        self.sample_bridge = sample_bridge or FYCSampleToWanBridge(wrapper=wrapper)
        self.batch_contract_builder = batch_contract_builder or TrainingBatchContractBuilder()
        self.forward_contract_builder = forward_contract_builder or TrainingForwardContractBuilder()
        self.denoising_contract_builder = denoising_contract_builder or DenoisingLoopContractBuilder()
        self.execution_contract_builder = execution_contract_builder or TrainingExecutionContractBuilder()
        self.module_mode_contract_builder = module_mode_contract_builder or TrainingModuleModeContractBuilder()
        self.autograd_preflight_contract_builder = autograd_preflight_contract_builder or TrainingAutogradPreflightContractBuilder()
        self.real_execution_preflight_contract_builder = (
            real_execution_preflight_contract_builder or TrainingRealExecutionPreflightContractBuilder()
        )
        self.dry_execution_lane_contract_builder = (
            dry_execution_lane_contract_builder or TrainingDryExecutionLaneContractBuilder()
        )
        self.graph_materialization_contract_builder = (
            graph_materialization_contract_builder or TrainingGraphMaterializationContractBuilder()
        )
        self.pre_forward_dry_call_contract_builder = (
            pre_forward_dry_call_contract_builder or TrainingPreForwardDryCallContractBuilder()
        )
        self.forward_invocation_gate_contract_builder = (
            forward_invocation_gate_contract_builder or TrainingForwardInvocationGateContractBuilder()
        )
        self.pre_training_boundary_closure_contract_builder = (
            pre_training_boundary_closure_contract_builder or TrainingPreTrainingBoundaryClosureContractBuilder()
        )
        self.update_scope_contract_builder = update_scope_contract_builder or TrainingUpdateScopeContractBuilder()
        self.optimizer_contract_builder = optimizer_contract_builder or TrainingOptimizerContractBuilder()
        self.backward_contract_builder = backward_contract_builder or TrainingBackwardContractBuilder()
        self.step_readiness_contract_builder = step_readiness_contract_builder or TrainingStepReadinessContractBuilder()
        self.enable_gate_contract_builder = enable_gate_contract_builder or TrainingEnableGateContractBuilder()
        self.num_denoising_steps = num_denoising_steps

    @staticmethod
    def _mask_2d(sample: FYCOutpaintSample) -> list[list[int]]:
        mask = sample.known_mask
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None
        if torch is not None and isinstance(mask, torch.Tensor):
            tensor = mask
            while tensor.ndim > 2:
                tensor = tensor[0]
            return tensor.detach().cpu().to(dtype=torch.int32).tolist()
        return [list(row) for row in mask]

    @staticmethod
    def _all_true(contract: dict[str, Any]) -> bool:
        return all(contract['invariants'].values())

    def _build_consistency_checks(
        self,
        *,
        bridge_payload: dict[str, Any],
        batch_contract: dict[str, Any],
        forward_contract: dict[str, Any],
        module_mode_contract: dict[str, Any],
        autograd_preflight_contract: dict[str, Any],
        real_execution_preflight_contract: dict[str, Any],
        dry_execution_lane_contract: dict[str, Any],
        graph_materialization_contract: dict[str, Any],
        pre_forward_dry_call_contract: dict[str, Any],
        forward_invocation_gate_contract: dict[str, Any],
        pre_training_boundary_closure_contract: dict[str, Any],
        execution_contract: dict[str, Any],
        update_scope_contract: dict[str, Any],
        optimizer_contract: dict[str, Any],
        backward_contract: dict[str, Any],
        step_readiness_contract: dict[str, Any],
        enable_gate_contract: dict[str, Any],
        runtime_payload: dict[str, Any],
    ) -> dict[str, bool]:
        bridge_bundle = bridge_payload['condition_bundle']
        runtime_bundle = runtime_payload['condition_bundle']
        request_contract = runtime_payload['request_contract']
        loss_targets = {
            'diffusion': float(self.losses.diffusion_weight),
            'known_region': float(self.losses.known_region_weight),
            'seam': float(self.losses.seam_weight),
        }

        graph_metadata = graph_materialization_contract['metadata']
        invocation_metadata = forward_invocation_gate_contract['metadata']
        closure_metadata = pre_training_boundary_closure_contract['metadata']

        expected_declared_modules = ['layout_encoder', 'geometry_encoder', 'condition_adapter', 'wan_outpaint_wrapper']
        expected_trainable_modules = ['layout_encoder', 'geometry_encoder']
        expected_optimizer_groups = ['layout_encoder_group', 'geometry_encoder_group']
        expected_step_ops = ['zero_grad', 'backward', 'clip_grad_norm', 'optimizer_step']
        expected_enable_flags = ['real_train_forward_enabled', 'autograd_backward_enabled', 'optimizer_step_enabled']
        expected_execution_phases = [
            'validate_request',
            'materialize_batch_targets',
            'prepare_denoising_schedule',
            'wan_forward',
            'assemble_losses',
            'backward_and_optimizer',
        ]
        expected_closure_surface_names = [
            'aggregate_non_mutating_boundary_surface',
            'forward_materialization_terminal_boundary_surface',
            'remaining_runtime_dependency_surface',
        ]

        return {
            'bridge_bundle_order_matches_runtime_bundle_order': bridge_bundle['order'] == runtime_bundle['order'],
            'bridge_bundle_semantic_roles_match_runtime_bundle': (
                bridge_bundle['semantic_roles'] == runtime_bundle['semantic_roles']
            ),
            'bridge_concat_shape_matches_runtime_bundle': bridge_bundle['concat_shape'] == runtime_bundle['concat_shape'],
            'batch_contract_all_true': self._all_true(batch_contract),
            'forward_contract_all_true': self._all_true(forward_contract),
            'forward_loss_weights_match_targets': forward_contract['metadata']['loss_weights'] == loss_targets,
            'forward_known_region_loss_target_matches_batch': (
                forward_contract['loss_components']['known_region']['target_spec']['shape']
                == batch_contract['target_specs']['known_latents']['shape']
            ),
            'forward_seam_loss_target_matches_batch': (
                forward_contract['loss_components']['seam']['target_spec']['shape']
                == batch_contract['target_specs']['known_latents']['shape']
            ),
            'update_scope_contract_all_true': self._all_true(update_scope_contract),
            'update_scope_has_trainable_targets': (
                update_scope_contract['metadata']['trainable_module_names'] == expected_trainable_modules
                and update_scope_contract['metadata']['total_trainable_parameter_count'] > 0
                and update_scope_contract['metadata']['total_trainable_element_count'] > 0
            ),
            'update_scope_mask_summary_frozen': (
                update_scope_contract['metadata']['frozen_module_names'] == ['mask_summary_encoder']
            ),
            'update_scope_wrapper_deferred': (
                update_scope_contract['metadata']['deferred_module_names'] == ['wan_outpaint_wrapper']
            ),
            'update_scope_declared_modules_expected': (
                update_scope_contract['metadata']['declared_trainable_modules'] == expected_declared_modules
            ),
            'module_mode_contract_all_true': self._all_true(module_mode_contract),
            'module_mode_trainable_modes_expected': (
                module_mode_contract['metadata']['train_mode_module_names'] == expected_trainable_modules
                and module_mode_contract['metadata']['eval_mode_module_names'] == ['mask_summary_encoder']
            ),
            'module_mode_wrapper_disabled': (
                module_mode_contract['metadata']['wrapper_forward_enabled'] is False
                and module_mode_contract['metadata']['autograd_enabled'] is False
                and module_mode_contract['metadata']['optimizer_step_enabled'] is False
            ),
            'module_mode_policy_dependency_expected': module_mode_contract['invariants'][
                'enable_gate_dependency_matches_policy'
            ],
            'autograd_preflight_contract_all_true': self._all_true(autograd_preflight_contract),
            'autograd_preflight_wrapper_disabled': (
                autograd_preflight_contract['metadata']['wrapper_forward_enabled'] is False
                and autograd_preflight_contract['metadata']['autograd_enabled'] is False
            ),
            'autograd_preflight_request_tensor_types_expected': (
                autograd_preflight_contract['metadata']['request_float_tensor_dtypes'] == ['torch.float32']
                and autograd_preflight_contract['metadata']['request_timestep_dtype'] == 'torch.int64'
            ),
            'autograd_preflight_trainable_modules_track_update_scope': (
                autograd_preflight_contract['metadata']['trainable_module_names']
                == update_scope_contract['metadata']['trainable_module_names']
            ),
            'execution_contract_all_true': self._all_true(execution_contract),
            'execution_ready_pre_optimizer': execution_contract['invariants']['execution_ready_pre_optimizer'],
            'execution_backward_only_phase_deferred': (
                execution_contract['metadata']['deferred_phase_names'] == ['backward_and_optimizer']
            ),
            'execution_loss_outputs_present': execution_contract['invariants']['loss_outputs_present'],
            'execution_phase_names_expected': (
                execution_contract['metadata']['phase_names'] == expected_execution_phases
            ),
            'execution_scheduler_is_verified_mvp': (
                execution_contract['metadata']['scheduler_name'] == 'FlowMatchEulerDiscreteScheduler'
                and execution_contract['metadata']['denoising_step_count'] == self.num_denoising_steps
            ),
            'real_execution_preflight_contract_all_true': self._all_true(real_execution_preflight_contract),
            'real_execution_preflight_flags_expected': (
                real_execution_preflight_contract['metadata']['manual_enable_flags'] == expected_enable_flags
                and real_execution_preflight_contract['metadata']['currently_enabled_flags'] == [False, False, False]
            ),
            'real_execution_preflight_ready_to_flip': real_execution_preflight_contract['invariants'][
                'ready_to_flip_real_execution_flags'
            ],
            'dry_execution_lane_contract_all_true': self._all_true(dry_execution_lane_contract),
            'dry_execution_lane_non_mutating': (
                dry_execution_lane_contract['metadata']['dry_execution_mode'] == 'non_mutating'
                and dry_execution_lane_contract['metadata']['weight_mutation_performed'] is False
                and dry_execution_lane_contract['metadata']['autograd_graph_materialized'] is False
            ),
            'dry_execution_lane_remaining_dependency_honest': (
                dry_execution_lane_contract['metadata']['remaining_runtime_dependency']
                == 'real_forward_graph_materialization'
            ),
            'graph_materialization_contract_all_true': self._all_true(graph_materialization_contract),
            'graph_materialization_payload_matches_request': (
                graph_metadata['request_tensor_shapes'] == request_contract['shapes']
                and graph_metadata['request_tensor_dtypes'] == request_contract['dtypes']
            ),
            'graph_materialization_bundle_matches_bridge': (
                graph_metadata['condition_bundle_order'] == bridge_bundle['order']
                and graph_metadata['condition_bundle_semantic_roles'] == bridge_bundle['semantic_roles']
                and graph_metadata['condition_bundle_concat_shape'] == bridge_bundle['concat_shape']
            ),
            'pre_forward_dry_call_contract_all_true': self._all_true(pre_forward_dry_call_contract),
            'pre_forward_dry_call_raised_expected_guard': (
                pre_forward_dry_call_contract['metadata']['call_target'] == 'WanOutpaintWrapper.forward'
                and pre_forward_dry_call_contract['metadata']['raised_exception_type'] == 'RuntimeError'
                and pre_forward_dry_call_contract['metadata']['remaining_runtime_dependency']
                == 'real_forward_graph_materialization'
            ),
            'forward_invocation_gate_contract_all_true': self._all_true(forward_invocation_gate_contract),
            'forward_invocation_gate_flags_expected': (
                invocation_metadata['call_target'] == 'WanOutpaintWrapper.forward'
                and invocation_metadata['gate_flag_names'] == ['wrapper_forward_enabled', 'autograd_enabled']
                and invocation_metadata['current_gate_flags'] == [False, False]
            ),
            'forward_invocation_gate_request_summary_matches_graph': (
                invocation_metadata['request_summary_prompt_shape'] == graph_metadata['request_tensor_shapes']['prompt_embeds']
                and invocation_metadata['request_summary_noisy_latents_shape'] == graph_metadata['request_tensor_shapes']['noisy_latents']
                and invocation_metadata['request_summary_layout_shape'] == graph_metadata['request_tensor_shapes']['layout_tokens']
                and invocation_metadata['request_summary_geometry_shape'] == graph_metadata['request_tensor_shapes']['geometry_tokens']
                and invocation_metadata['request_summary_mask_shape'] == graph_metadata['request_tensor_shapes']['mask_tokens']
                and invocation_metadata['request_summary_timesteps_shape'] == graph_metadata['request_tensor_shapes']['timesteps']
                and invocation_metadata['request_summary_known_region_mode'] == batch_contract['preserve_state']['mode']
            ),
            'forward_invocation_gate_remaining_dependency_honest': (
                invocation_metadata['remaining_runtime_dependency'] == 'real_forward_graph_materialization'
            ),
            'pre_training_boundary_closure_contract_all_true': self._all_true(pre_training_boundary_closure_contract),
            'pre_training_boundary_closure_flags_expected': (
                closure_metadata['current_gate_flags'] == [False, False, False]
                and closure_metadata['simulated_next_phase_gate_flags'] == [True, False, False]
                and closure_metadata['blocked_flags'] == ['autograd_enabled', 'optimizer_step_enabled']
            ),
            'pre_training_boundary_closure_scope_ready': (
                closure_metadata['surface_names'] == expected_closure_surface_names
                and closure_metadata['call_target'] == 'WanOutpaintWrapper.forward'
                and closure_metadata['next_phase'] == 'real_forward_graph_materialization'
                and closure_metadata['next_non_mutating_lane']
                == 'real_forward_graph_materialization_without_backward_or_step'
            ),
            'pre_training_boundary_closure_remaining_dependency_honest': (
                closure_metadata['remaining_runtime_dependency'] == 'real_forward_graph_materialization'
                and closure_metadata['deepest_upstream_contract_name'] == 'forward_invocation_gate'
            ),
            'optimizer_contract_all_true': self._all_true(optimizer_contract),
            'optimizer_group_names_expected': (
                optimizer_contract['metadata']['group_names'] == expected_optimizer_groups
            ),
            'optimizer_groups_match_trainable_modules': (
                optimizer_contract['metadata']['group_module_names'] == expected_trainable_modules
            ),
            'optimizer_state_empty_before_backward': optimizer_contract['invariants'][
                'optimizer_state_empty_before_backward'
            ],
            'optimizer_backward_deferred': optimizer_contract['metadata']['backward_deferred'],
            'optimizer_ready_post_backward': optimizer_contract['invariants']['optimizer_ready_post_backward'],
            'backward_contract_all_true': self._all_true(backward_contract),
            'backward_loss_scalar_ready': (
                backward_contract['invariants']['total_loss_scalar']
                and backward_contract['invariants']['total_loss_requires_grad']
                and backward_contract['metadata']['total_loss_shape'] == []
            ),
            'backward_grad_targets_match_optimizer': backward_contract['invariants'][
                'gradient_targets_match_optimizer_parameters'
            ],
            'backward_grads_empty_pre_backward': backward_contract['invariants']['gradient_buffers_empty_pre_backward'],
            'backward_zero_grad_strategy_expected': (
                backward_contract['metadata']['zero_grad_strategy']
                == optimizer_contract['metadata']['zero_grad_strategy']
                == 'set_to_none'
            ),
            'backward_group_names_expected': (
                sorted(backward_contract['metadata']['gradient_group_names'])
                == sorted(expected_optimizer_groups)
            ),
            'step_readiness_contract_all_true': self._all_true(step_readiness_contract),
            'step_readiness_order_expected': (
                step_readiness_contract['metadata']['operation_names'] == expected_step_ops
            ),
            'step_readiness_zero_grad_expected': (
                step_readiness_contract['metadata']['zero_grad_strategy'] == 'set_to_none'
            ),
            'step_readiness_backward_inputs_expected': step_readiness_contract['invariants'][
                'backward_inputs_match_backward_contract'
            ],
            'step_readiness_grad_clip_expected': (
                step_readiness_contract['metadata']['grad_clip_norm'] == 1.0
                and step_readiness_contract['invariants']['grad_clip_norm_positive']
            ),
            'step_readiness_optimizer_step_deferred': (
                step_readiness_contract['metadata']['deferred_operation_names'] == expected_step_ops
            ),
            'enable_gate_contract_all_true': self._all_true(enable_gate_contract),
            'enable_gate_order_expected': enable_gate_contract['metadata']['gate_names'] == [
                'real_train_forward_enable',
                'autograd_backward_enable',
                'optimizer_step_enable',
            ],
            'enable_gate_manual_flags_expected': (
                enable_gate_contract['metadata']['manual_enable_flags'] == expected_enable_flags
            ),
            'enable_gate_currently_disabled_expected': (
                enable_gate_contract['metadata']['currently_enabled_flags'] == [False, False, False]
            ),
            'enable_gate_only_policy_blockers': enable_gate_contract['invariants']['only_policy_blockers_remain'],
        }

    def _module_registry(self) -> dict[str, Any]:
        conditioning_builder = self.sample_bridge.conditioning_builder
        return {
            'layout_encoder': conditioning_builder.layout_encoder,
            'geometry_encoder': conditioning_builder.geometry_encoder,
            'mask_summary_encoder': conditioning_builder.mask_encoder,
            'condition_adapter': self.wrapper.condition_adapter,
            'wan_outpaint_wrapper': self.wrapper,
        }

    def run_once(self, sample: FYCOutpaintSample) -> TrainingDryRunReport:
        step_plan = self.step_builder.build(sample)
        bridged = self.sample_bridge.build(sample)
        module_registry = self._module_registry()

        update_scope_contract_obj = self.update_scope_contract_builder.build(
            declared_trainable_modules=step_plan.trainable_modules,
            module_registry=module_registry,
            wrapper_forward_enabled=False,
        )
        update_scope_contract = update_scope_contract_obj.to_dict()

        batch_contract_obj = self.batch_contract_builder.build(sample, bridged)
        batch_contract = batch_contract_obj.to_dict()

        forward_contract_obj = self.forward_contract_builder.build(batch_contract_obj, self.losses)
        forward_contract = forward_contract_obj.to_dict()

        wrapper_payload = self.wrapper.dry_run(bridged.request)
        timesteps = self.denoising_contract_builder.generate_timesteps(
            scheduler_name='FlowMatchEulerDiscreteScheduler',
            num_inference_steps=self.num_denoising_steps,
        )
        denoising_contract = self.denoising_contract_builder.build(
            mask_2d=self._mask_2d(sample),
            timesteps=timesteps,
            mode='overwrite',
            scheduler_name='FlowMatchEulerDiscreteScheduler',
            flow_shift=5.0,
            source='train-dry-run',
        )

        execution_contract_obj = self.execution_contract_builder.build(
            batch_contract=batch_contract_obj,
            forward_contract=forward_contract_obj,
            denoising_contract=denoising_contract,
            request_contract=wrapper_payload['request_contract'],
        )
        execution_contract = execution_contract_obj.to_dict()

        optimizer_contract_obj = self.optimizer_contract_builder.build(
            update_scope_contract=update_scope_contract_obj,
            execution_contract=execution_contract_obj,
            module_registry=module_registry,
        )
        optimizer_contract = optimizer_contract_obj.to_dict()

        backward_contract_obj = self.backward_contract_builder.build(
            update_scope_contract=update_scope_contract_obj,
            optimizer_contract=optimizer_contract_obj,
            execution_contract=execution_contract_obj,
            module_registry=module_registry,
        )
        backward_contract = backward_contract_obj.to_dict()

        step_readiness_contract_obj = self.step_readiness_contract_builder.build(
            optimizer_contract=optimizer_contract_obj,
            backward_contract=backward_contract_obj,
        )
        step_readiness_contract = step_readiness_contract_obj.to_dict()

        enable_gate_contract_obj = self.enable_gate_contract_builder.build(
            execution_contract=execution_contract_obj,
            update_scope_contract=update_scope_contract_obj,
            optimizer_contract=optimizer_contract_obj,
            backward_contract=backward_contract_obj,
            step_readiness_contract=step_readiness_contract_obj,
        )
        enable_gate_contract = enable_gate_contract_obj.to_dict()

        module_mode_contract_obj = self.module_mode_contract_builder.build(
            update_scope_contract=update_scope_contract_obj,
            enable_gate_contract=enable_gate_contract_obj,
            module_registry=module_registry,
            wrapper_forward_enabled=False,
            autograd_enabled=False,
            optimizer_step_enabled=False,
        )
        module_mode_contract = module_mode_contract_obj.to_dict()

        autograd_preflight_contract_obj = self.autograd_preflight_contract_builder.build(
            update_scope_contract=update_scope_contract_obj,
            module_mode_contract=module_mode_contract_obj,
            enable_gate_contract=enable_gate_contract_obj,
            module_registry=module_registry,
            request=bridged.request,
            wrapper_forward_enabled=False,
            autograd_enabled=False,
        )
        autograd_preflight_contract = autograd_preflight_contract_obj.to_dict()

        real_execution_preflight_contract_obj = self.real_execution_preflight_contract_builder.build(
            execution_contract=execution_contract_obj,
            update_scope_contract=update_scope_contract_obj,
            optimizer_contract=optimizer_contract_obj,
            backward_contract=backward_contract_obj,
            step_readiness_contract=step_readiness_contract_obj,
            enable_gate_contract=enable_gate_contract_obj,
            module_mode_contract=module_mode_contract_obj,
            autograd_preflight_contract=autograd_preflight_contract_obj,
            wrapper_forward_enabled=False,
            autograd_enabled=False,
            optimizer_step_enabled=False,
        )
        real_execution_preflight_contract = real_execution_preflight_contract_obj.to_dict()

        dry_execution_lane_contract_obj = self.dry_execution_lane_contract_builder.build(
            forward_contract=forward_contract_obj,
            execution_contract=execution_contract_obj,
            backward_contract=backward_contract_obj,
            step_readiness_contract=step_readiness_contract_obj,
            enable_gate_contract=enable_gate_contract_obj,
            autograd_preflight_contract=autograd_preflight_contract_obj,
            real_execution_preflight_contract=real_execution_preflight_contract_obj,
            wrapper_forward_enabled=False,
            autograd_enabled=False,
            optimizer_step_enabled=False,
        )
        dry_execution_lane_contract = dry_execution_lane_contract_obj.to_dict()

        graph_materialization_contract_obj = self.graph_materialization_contract_builder.build(
            forward_contract=forward_contract_obj,
            execution_contract=execution_contract_obj,
            autograd_preflight_contract=autograd_preflight_contract_obj,
            real_execution_preflight_contract=real_execution_preflight_contract_obj,
            dry_execution_lane_contract=dry_execution_lane_contract_obj,
            runtime_payload=wrapper_payload,
            wrapper_forward_enabled=False,
            autograd_enabled=False,
        )
        graph_materialization_contract = graph_materialization_contract_obj.to_dict()

        pre_forward_dry_call_contract_obj = self.pre_forward_dry_call_contract_builder.build(
            wrapper=self.wrapper,
            request=bridged.request,
            graph_materialization_contract=graph_materialization_contract_obj,
            dry_execution_lane_contract=dry_execution_lane_contract_obj,
            wrapper_forward_enabled=False,
            autograd_enabled=False,
        )
        pre_forward_dry_call_contract = pre_forward_dry_call_contract_obj.to_dict()

        forward_invocation_gate_contract_obj = self.forward_invocation_gate_contract_builder.build(
            wrapper=self.wrapper,
            request=bridged.request,
            graph_materialization_contract=graph_materialization_contract_obj,
            pre_forward_dry_call_contract=pre_forward_dry_call_contract_obj,
            dry_execution_lane_contract=dry_execution_lane_contract_obj,
            wrapper_forward_enabled=False,
            autograd_enabled=False,
        )
        forward_invocation_gate_contract = forward_invocation_gate_contract_obj.to_dict()

        pre_training_boundary_closure_contract_obj = self.pre_training_boundary_closure_contract_builder.build(
            real_execution_preflight_contract=real_execution_preflight_contract_obj,
            dry_execution_lane_contract=dry_execution_lane_contract_obj,
            graph_materialization_contract=graph_materialization_contract_obj,
            pre_forward_dry_call_contract=pre_forward_dry_call_contract_obj,
            forward_invocation_gate_contract=forward_invocation_gate_contract_obj,
            wrapper_forward_enabled=False,
            autograd_enabled=False,
            optimizer_step_enabled=False,
        )
        pre_training_boundary_closure_contract = pre_training_boundary_closure_contract_obj.to_dict()

        consistency_checks = self._build_consistency_checks(
            bridge_payload=bridged.wrapper_payload,
            batch_contract=batch_contract,
            forward_contract=forward_contract,
            module_mode_contract=module_mode_contract,
            autograd_preflight_contract=autograd_preflight_contract,
            real_execution_preflight_contract=real_execution_preflight_contract,
            dry_execution_lane_contract=dry_execution_lane_contract,
            graph_materialization_contract=graph_materialization_contract,
            pre_forward_dry_call_contract=pre_forward_dry_call_contract,
            forward_invocation_gate_contract=forward_invocation_gate_contract,
            pre_training_boundary_closure_contract=pre_training_boundary_closure_contract,
            execution_contract=execution_contract,
            update_scope_contract=update_scope_contract,
            optimizer_contract=optimizer_contract,
            backward_contract=backward_contract,
            step_readiness_contract=step_readiness_contract,
            enable_gate_contract=enable_gate_contract,
            runtime_payload=wrapper_payload,
        )

        return TrainingDryRunReport(
            step_plan=asdict(step_plan),
            bridge=bridged.to_dict(),
            batch_contract=batch_contract,
            forward_contract=forward_contract,
            module_mode_contract=module_mode_contract,
            autograd_preflight_contract=autograd_preflight_contract,
            real_execution_preflight_contract=real_execution_preflight_contract,
            dry_execution_lane_contract=dry_execution_lane_contract,
            graph_materialization_contract=graph_materialization_contract,
            pre_forward_dry_call_contract=pre_forward_dry_call_contract,
            forward_invocation_gate_contract=forward_invocation_gate_contract,
            pre_training_boundary_closure_contract=pre_training_boundary_closure_contract,
            execution_contract=execution_contract,
            update_scope_contract=update_scope_contract,
            optimizer_contract=optimizer_contract,
            backward_contract=backward_contract,
            step_readiness_contract=step_readiness_contract,
            enable_gate_contract=enable_gate_contract,
            wrapper_payload=wrapper_payload,
            losses=describe_loss_targets(self.losses),
            consistency_checks=consistency_checks,
        )
