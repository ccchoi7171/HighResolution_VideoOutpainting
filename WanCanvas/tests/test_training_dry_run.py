from __future__ import annotations

import unittest

from wancanvas.backbones.wan_loader import WanLoader
from wancanvas.data.outpaint_dataset import DatasetRecord, WanCanvasDataset
from wancanvas.data.samplers import AnchorTargetSamplingConfig
from wancanvas.models.wan_outpaint_wrapper import WanOutpaintWrapper
from wancanvas.train import DryRunTrainer, TrainStepBuilder


class DryRunTrainerTest(unittest.TestCase):
    def _report(self):
        dataset = WanCanvasDataset(
            records=[
                DatasetRecord(
                    source_id='train-demo',
                    prompt='expand the frame',
                    frame_height=720,
                    frame_width=1280,
                )
            ],
            sampling_config=AnchorTargetSamplingConfig(target_size=(512, 512), anchor_size=(384, 384), seed=11),
        )
        sample = dataset[0]
        trainer = DryRunTrainer(
            wrapper=WanOutpaintWrapper(WanLoader()),
            step_builder=TrainStepBuilder(
                ('layout_encoder', 'geometry_encoder', 'condition_adapter', 'wan_outpaint_wrapper')
            ),
        )
        return trainer.run_once(sample)

    def test_dry_run_converges_on_core_non_mutating_training_surfaces(self) -> None:
        report = self._report()

        self.assertEqual(report.batch_contract['latent_hw'], [64, 64])
        self.assertEqual(
            report.forward_contract['metadata']['loss_weights'],
            {'diffusion': 1.0, 'known_region': 1.0, 'seam': 0.25},
        )
        self.assertEqual(report.module_mode_contract['metadata']['train_mode_module_names'], ['layout_encoder', 'geometry_encoder'])
        self.assertEqual(report.module_mode_contract['metadata']['eval_mode_module_names'], ['mask_summary_encoder'])
        self.assertEqual(
            report.real_execution_preflight_contract['metadata']['manual_enable_flags'],
            ['real_train_forward_enabled', 'autograd_backward_enabled', 'optimizer_step_enabled'],
        )
        self.assertEqual(
            report.dry_execution_lane_contract['metadata']['remaining_runtime_dependency'],
            'real_forward_graph_materialization',
        )
        self.assertEqual(
            report.graph_materialization_contract['metadata']['graph_materialization_mode'],
            'payload_ready_no_forward_call',
        )
        self.assertEqual(
            report.pre_forward_dry_call_contract['metadata']['raised_exception_type'],
            'RuntimeError',
        )
        self.assertEqual(
            report.forward_invocation_gate_contract['metadata']['current_gate_flags'],
            [False, False],
        )
        self.assertEqual(
            report.pre_training_boundary_closure_contract['metadata']['deepest_upstream_contract_name'],
            'forward_invocation_gate',
        )
        self.assertEqual(
            report.optimizer_contract['metadata']['group_names'],
            ['layout_encoder_group', 'geometry_encoder_group'],
        )
        self.assertEqual(
            report.step_readiness_contract['metadata']['operation_names'],
            ['zero_grad', 'backward', 'clip_grad_norm', 'optimizer_step'],
        )
        self.assertEqual(
            report.enable_gate_contract['metadata']['gate_names'],
            ['real_train_forward_enable', 'autograd_backward_enable', 'optimizer_step_enable'],
        )
        self.assertTrue(all(report.consistency_checks.values()))

    def test_dry_run_report_exposes_only_consolidated_training_boundary_surfaces(self) -> None:
        report = self._report()
        report_keys = set(report.__dataclass_fields__.keys())

        self.assertIn('pre_training_boundary_closure_contract', report_keys)
        self.assertIn('forward_invocation_gate_contract', report_keys)
        self.assertNotIn('forward_only_enable_contract', report_keys)
        self.assertNotIn('forward_materialization_preflight_contract', report_keys)
        self.assertNotIn(
            'forward_materialization_enabled_state_invocation_preflight_handoff_preflight_handoff_lane_handoff_lane_handoff_lane_handoff_contract',
            report_keys,
        )


if __name__ == '__main__':
    unittest.main()
