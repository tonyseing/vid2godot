"""The real-clip regression fixture (tests/real_fixture.py): its metrics on synthetic data, and, when
VID2GODOT_REAL_FIXTURE names a fixture folder and VID2GODOT_BLENDER is set, the whole clip fitted and held to the
fixture's targets. The fixture's video is not in this repository; the metrics are written beside the run (to
VID2GODOT_REAL_FIXTURE_OUT when set) for the report."""
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests import real_fixture as rf
from tests.blender_case import BLENDER

REAL = os.environ.get("VID2GODOT_REAL_FIXTURE")


def wolf_mask():
    """A side silhouette: a body block with two legs whose paws stand on the ground at u = 50 and u = 130."""
    mask = np.zeros((120, 200), bool)
    mask[30:60, 30:170] = True
    mask[60:100, 45:55] = True
    mask[60:100, 125:135] = True
    return mask


class MetricTests(unittest.TestCase):
    def test_iou_metrics_spans_and_minimum(self):
        ious = {f: 0.9 for f in range(121)}
        ious[80], ious[84] = 0.5, 0.7
        m = rf.iou_metrics(ious, frames=[0, 40, 80, 84, 120], spans=((0, 25), (76, 100)))
        self.assertAlmostEqual(m["meanIoU"], round((0.9 * 3 + 0.5 + 0.7) / 5, 4))
        self.assertEqual(m["spans"], {"0-25": 0.9, "76-100": 0.6})
        self.assertEqual((m["minIoU"], m["minFrame"]), (0.5, 80))

    def test_find_paws_finds_the_ends_of_the_legs(self):
        points = np.array(rf.find_paws(wolf_mask()))
        for u in (50, 130):
            self.assertLess(float(np.min(np.hypot(points[:, 0] - u, points[:, 1] - 97))), 4.0)

    def test_forepaw_miss_share(self):
        masks = {0: wolf_mask()}

        def poses(paw):
            joints = {"front_L": [[0, 0], [paw[0], paw[1] - 2], [paw[0], paw[1] + 2]],
                      "front_R": [[0, 0], [52, 95], [52, 99]]}
            return {"frames": [{"index": 0, "joints": joints}]}

        self.assertEqual(rf.forepaw_miss_share(poses((128, 97)), masks, frames=[0])[0], 0.0)
        share, far = rf.forepaw_miss_share(poses((90, 97)), masks, frames=[0])
        self.assertEqual(share, 0.5)
        self.assertIn("0:front_L", far)

    def test_leg_swaps_count_identity_jumps_not_crossings(self):
        names = ["dx", "dz", "a1", "a2", "b1", "b2"]
        chains = {"L": ["a1", "a2"], "R": ["b1", "b2"]}
        frames = []
        for f in range(12):
            a = [0.6 * math.sin(f / 2), 0.4 * math.sin(f / 2 + 1)]
            b = [-v for v in a]                       # the legs cross twice a cycle: not a swap
            if 6 <= f < 9:
                a, b = b, a                           # an identity jump at 6 and back at 9
            frames.append({"index": f, "values": [0.0, 0.0] + a + b})
        swaps = rf.leg_swaps({"names": names, "frames": frames}, [["L", "R"]], chains)
        self.assertEqual(swaps, {"L/R": [6, 9]})

    def test_false_fill_counts_background_inside_the_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.full((240, 400, 3), 200, np.uint8)
            img[60:120, 60:340] = 40
            img[120:200, 90:110] = 40
            img[120:200, 250:270] = 40
            img[190:200, 90:270] = 40
            Image.fromarray(img).save(Path(tmp) / "00000.png")
            mask = np.zeros((120, 200), bool)
            mask[30:100, 30:170] = True               # a mask that filled the pocket between the legs
            counts = rf.false_fill(np.array([mask]), Path(tmp))
            self.assertEqual(counts["counts"][0], 70 * 140 - int((img[::2, ::2] == 40).all(axis=2).sum()))
            self.assertEqual(counts["frames"], [0])

    def metrics(self, **change):
        base = {"meanIoU": 0.86, "spans": {"0-25": 0.95, "76-100": 0.81}, "frames": {"80": 0.76},
                "forepawMissShare": 0.1, "legSwaps": {"front_L/front_R": 0, "hind_L/hind_R": 1},
                "period": {"found": False, "periodFrames": None, "score": 0.7},
                "facts": {"ok": True, "start": 45.1, "period": 15.04, "averagedCycleIoU": 0.87}}
        base.update(change)
        return base

    def test_failures_name_each_missed_target_and_regressed_span(self):
        fixture = {"targets": rf.TARGETS, "baseline": {"spans": {"0-25": 0.9, "76-100": 0.7}}}
        metrics = self.metrics(spans={"0-25": 0.89, "76-100": 0.81})
        self.assertEqual(rf.failures(fixture, metrics), ["span 0-25 IoU 0.89 is worse than the baseline 0.9"])
        metrics = self.metrics(meanIoU=0.8, spans={"0-25": 0.95, "76-100": 0.7}, frames={"80": 0.6})
        self.assertEqual(len(rf.failures(fixture, metrics)), 3)

    def test_failures_require_facts_visible_forepaws_and_no_more_swaps(self):
        """Temporal coherence: the facts stage itself must accept the fit (Task 17b fix round 2: its period comes
        from the video and its averaged cycle must match the masks; the all-angle period score is only reported);
        forepaws must be where the video shows paws; and no leg pair may swap more often than in the baseline."""
        fixture = {"targets": rf.TARGETS,
                   "baseline": {"spans": {}, "legSwaps": {"front_L/front_R": 0, "hind_L/hind_R": 1}}}
        self.assertEqual(rf.failures(fixture, self.metrics()), [], "the angle period score alone does not fail")
        problems = rf.failures(fixture, self.metrics(
            facts={"ok": False, "error": "the averaged cycle matches the video at mean IoU 0.75"},
            forepawMissShare=0.48, legSwaps={"front_L/front_R": 2, "hind_L/hind_R": 1}))
        self.assertEqual(len(problems), 3, problems)
        self.assertIn("facts refuses the fit", problems[0])
        self.assertIn("0.75", problems[0])
        self.assertIn("forepaw", problems[1])
        self.assertIn("front_L/front_R", problems[2])

    def test_period_metrics_use_the_facts_search(self):
        names = ["dx", "dz", "a", "b"]
        periodic = [{"index": f, "values": [0.0, 0.0, math.sin(2 * math.pi * f / 15.2),
                                            math.cos(2 * math.pi * f / 15.2)]} for f in range(121)]
        found = rf.period_metrics({"names": names, "fps": 24.0, "frames": periodic})
        self.assertTrue(found["found"])
        self.assertAlmostEqual(found["periodFrames"], 15.2, delta=0.2)
        rng = np.random.default_rng(1)
        noise = [{"index": f, "values": [0.0, 0.0, *rng.normal(size=2)]} for f in range(121)]
        missing = rf.period_metrics({"names": names, "fps": 24.0, "frames": noise})
        self.assertFalse(missing["found"])
        self.assertIsNone(missing["periodFrames"])
        self.assertGreater(missing["score"], 0.2)

    def test_baseline_block_is_written_from_a_measured_run(self):
        metrics = {**self.metrics(), "minIoU": 0.67, "minFrame": 80, "meanIoUAll": 0.79, "minutes": 19.0,
                   "forepawMisses": {"0:front_L": 90.0}, "frames": {"0": 0.9}}
        block = rf.baseline_block(metrics, "vid2godot 71fc632")
        self.assertEqual(block["fitter"], "vid2godot 71fc632")
        for key in ("meanIoU", "spans", "minIoU", "minFrame", "forepawMissShare", "legSwaps", "period",
                    "meanIoUAll"):
            self.assertEqual(block[key], metrics[key])
        self.assertNotIn("forepawMisses", block)

    def test_legacy_silhouette_fills_the_pocket_the_current_one_keeps(self):
        """The old ingest filled every enclosed hole; the fixture counts its false fill beside the new one's."""
        from anim import ingest
        img = np.full((180, 320, 3), 200, np.uint8)
        img[60:100, 100:220] = 40
        img[100:150, 110:122] = 40
        img[100:150, 190:202] = 40
        img[140:150, 110:202] = 40
        bg = ingest.background_colour(img)
        self.assertTrue(rf.legacy_silhouette(img, bg)[120, 150])
        self.assertFalse(ingest.silhouette(img, bg)[120, 150])


@unittest.skipUnless(REAL and BLENDER, "set VID2GODOT_REAL_FIXTURE and VID2GODOT_BLENDER to fit the real clip")
class RealClipFitTests(unittest.TestCase):
    def test_real_clip_meets_the_fixture_targets(self):
        fixture, _ = rf.load(REAL)
        out_root = os.environ.get("VID2GODOT_REAL_FIXTURE_OUT")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(out_root) if out_root else Path(tmp)
            metrics = rf.run(REAL, out, BLENDER)
        print(f"REALFIXTURE {json.dumps({k: v for k, v in metrics.items() if k != 'frames'})}", flush=True)
        self.assertEqual(rf.failures(fixture, metrics), [])
