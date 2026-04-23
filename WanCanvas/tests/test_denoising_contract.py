from __future__ import annotations

import unittest

from wancanvas.train import DenoisingLoopContractBuilder


class DenoisingLoopContractTest(unittest.TestCase):
    def test_generated_flowmatch_schedule_has_valid_invariants(self) -> None:
        builder = DenoisingLoopContractBuilder()
        timesteps = builder.generate_timesteps(num_inference_steps=6)
        contract = builder.build(
            mask_2d=[[0, 0, 1, 1], [0, 1, 1, 1]],
            timesteps=timesteps,
            mode='overwrite',
            scheduler_name='FlowMatchEulerDiscreteScheduler',
            flow_shift=5.0,
            source='generated',
        )
        payload = contract.to_dict()
        self.assertEqual(len(payload['steps']), 6)
        self.assertTrue(all(payload['invariants'].values()))
        self.assertTrue(all(step['blend_alpha'] == 1.0 for step in payload['steps']))

    def test_blend_mode_alpha_decreases_monotonically(self) -> None:
        builder = DenoisingLoopContractBuilder()
        contract = builder.build(
            mask_2d=[[0, 1], [1, 1]],
            timesteps=[1000.0, 750.0, 500.0, 250.0, 1.0],
            mode='blend',
            source='explicit',
        )
        payload = contract.to_dict()
        alphas = [step['blend_alpha'] for step in payload['steps']]
        self.assertTrue(all(alphas[idx] >= alphas[idx + 1] for idx in range(len(alphas) - 1)))
        self.assertTrue(payload['invariants']['blend_alpha_non_increasing'])
        self.assertTrue(payload['invariants']['timesteps_strictly_descending'])


if __name__ == '__main__':
    unittest.main()
