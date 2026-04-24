from __future__ import annotations

import unittest

from wancanvas.backbones.wan_loader import WanLoader
from wancanvas.data.contracts import Rect
from wancanvas.models.wan_outpaint_wrapper import WanOutpaintWrapper
from wancanvas.pipelines.size_alignment import SizeAlignmentRule
from wancanvas.pipelines.wan_outpaint_pipeline import MultiRoundOutpaintRequest, OutpaintRequest, WanOutpaintPipeline


class CanvasExpansionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = WanOutpaintPipeline(WanOutpaintWrapper(WanLoader()), size_rule=SizeAlignmentRule())

    def test_multi_round_contract_expands_monotonically_and_matches_final_canvas(self) -> None:
        plan = self.pipeline.plan_multi_round_request(
            MultiRoundOutpaintRequest(
                prompt='expand the frame',
                frame_count=16,
                fps=16,
                final_canvas_height=720,
                final_canvas_width=1280,
                anchor_region=Rect(top=168, left=448, height=384, width=384),
                tile_height=720,
                tile_width=720,
                overlap_height=176,
                overlap_width=176,
                rounds=3,
            )
        )
        self.assertEqual(len(plan['rounds']), 3)
        self.assertTrue(all(plan['invariants'].values()))
        self.assertEqual(plan['rounds'][-1]['canvas_size'], [720, 1280])
        heights = [round_plan['canvas_size'][0] for round_plan in plan['rounds']]
        widths = [round_plan['canvas_size'][1] for round_plan in plan['rounds']]
        self.assertEqual(heights, sorted(heights))
        self.assertEqual(widths, sorted(widths))

    def test_dry_run_multi_round_reports_round_counts(self) -> None:
        report = self.pipeline.dry_run_multi_round(
            MultiRoundOutpaintRequest(
                prompt='expand the frame',
                frame_count=16,
                fps=16,
                final_canvas_height=720,
                final_canvas_width=1280,
                anchor_region=Rect(top=168, left=448, height=384, width=384),
                tile_height=720,
                tile_width=720,
                overlap_height=176,
                overlap_width=176,
                rounds=2,
            )
        )
        self.assertEqual(report['wrapper']['extras']['round_count'], 2)
        tile_counts = report['wrapper']['extras']['tile_counts_by_round']
        self.assertEqual(tile_counts, [2, 2])
        self.assertTrue(all(report['plan']['invariants'].values()))
        self.assertTrue(report['wrapper']['has_known_region_state'])
        self.assertEqual(report['wrapper']['extras']['dry_run_token_dim'], 1024)
        self.assertTrue(all(report['wrapper']['request_contract']['checks'].values()))
        self.assertEqual(report['wrapper']['request_contract']['shapes']['prompt_embeds'], [1, 1, 1024])

    def test_single_round_dry_run_now_uses_overwrite_known_region_request_contract(self) -> None:
        report = self.pipeline.dry_run(
            OutpaintRequest(
                prompt='expand the frame',
                frame_count=16,
                fps=16,
                canvas_height=720,
                canvas_width=1280,
                anchor_region=Rect(top=168, left=448, height=384, width=384),
                tile_height=720,
                tile_width=720,
                overlap_height=176,
                overlap_width=176,
            )
        )
        self.assertEqual(report['wrapper']['extras']['dry_run_token_dim'], 1024)
        self.assertTrue(report['wrapper']['has_known_region_state'])
        self.assertTrue(all(report['wrapper']['request_contract']['checks'].values()))
        self.assertEqual(report['wrapper']['request_contract']['shapes']['prompt_embeds'], [1, 1, 1024])
        self.assertEqual(report['wrapper']['condition_bundle']['concat_shape'], None)


if __name__ == '__main__':
    unittest.main()
