"""The video's cycle (anim/cycle.py): the period from the masks' own self-similarity, the steady cycles after the
warm-up, the phase-aligned average and the helpers facts and the fitter's periodic pass share."""
import math
import unittest

import numpy as np

from anim import cycle
from tests.toy_gait import toy_mask

PHASES = (0.0, 0.25, 0.5, 0.75)
WOLF = {"chains": {"spine": ["pelvis", "chest"], "neck": ["neck_01", "head"], "tail": ["tail_01", "tail_02"],
                   "front_L": ["scapula_L", "upperarm_L", "forearm_L", "wrist_L", "paw_front_L"],
                   "front_R": ["scapula_R", "upperarm_R", "forearm_R", "wrist_R", "paw_front_R"],
                   "hind_L": ["thigh_L", "shin_L", "hock_L", "paw_hind_L"],
                   "hind_R": ["thigh_R", "shin_R", "hock_R", "paw_hind_R"]},
        "legPairs": [["front_L", "front_R"], ["hind_L", "hind_R"]]}


def gait(t, period, gain=1.0, phases=PHASES):
    return [0.0, 0.0] + [gain * math.sin(2 * math.pi * (t / period - p)) for p in phases]


def masks_of(times, period, gain=lambda t: 1.0, **kw):
    return np.array([toy_mask(gait(t, period, gain(t)), **kw) for t in times])


class MaskPeriodTests(unittest.TestCase):
    def test_finds_the_video_period_at_sub_frame_precision(self):
        found = cycle.mask_period(masks_of(range(121), 15.2), 24.0)
        self.assertAlmostEqual(found["period"], 15.2, delta=0.15)
        self.assertLessEqual(found["score"], cycle.MASK_PERIOD_MAX)
        self.assertGreaterEqual(found["margin"], cycle.MASK_PERIOD_MARGIN)
        self.assertEqual((found["gate"], found["marginGate"]), (cycle.MASK_PERIOD_MAX, cycle.MASK_PERIOD_MARGIN))

    def test_a_minority_of_time_displaced_frames_does_not_move_the_period(self):
        """The real Seedance clip repeats every 15 frames exactly, except every fourth frame, which sits about half a
        frame late; a mean over frame pairs drags the period to ~15.2, the median keeps 15."""
        times = [t + (0.5 if t % 4 == 0 else 0.0) for t in range(121)]
        found = cycle.mask_period(masks_of(times, 15.0), 24.0)
        self.assertAlmostEqual(found["period"], 15.0, delta=0.08)

    def test_one_shot_clips_are_not_cyclic(self):
        swing = np.array([toy_mask([0, 0] + [-1.0 + 2.0 * f / 59] * 4) for f in range(60)])
        self.assertIsNone(cycle.mask_period(swing, 24.0)["period"])
        still = np.array([toy_mask([0, 0, 0.3, -0.3, 0.3, -0.3])] * 60)
        found = cycle.mask_period(still, 24.0)
        self.assertIsNone(found["period"], "a still clip matches itself at every lag: no lag is clearly the period")
        self.assertIsNone(cycle.mask_period(masks_of(range(50), 30.0), 24.0)["period"])

    def test_alignment_ignores_the_animal_drifting_across_the_frame(self):
        masks = np.array([toy_mask(gait(t, 12.0), shift=int(t // 6) - 5) for t in range(72)])
        self.assertAlmostEqual(cycle.mask_period(masks, 24.0)["period"], 12.0, delta=0.15)


class SteadyCycleTests(unittest.TestCase):
    def test_warm_up_cycles_are_left_out(self):
        """Small swings until frame 26, full ones from 34: only cycles in the steady part are kept."""
        masks = masks_of(range(96), 12.0, gain=lambda t: 0.3 + 0.7 * min(1.0, max(0.0, (t - 26) / 8)))
        steady = cycle.steady_cycles(masks, 12.0, 10)
        self.assertEqual(steady["starts"], [10.0, 22.0, 34.0, 46.0, 58.0, 70.0, 82.0])
        self.assertEqual(steady["kept"], [34.0, 46.0, 58.0, 70.0, 82.0])
        self.assertIn(steady["ref"], steady["kept"])

    def test_a_steady_clip_keeps_every_cycle(self):
        steady = cycle.steady_cycles(masks_of(range(121), 15.2), 15.2, 13)
        self.assertEqual(len(steady["kept"]), len(steady["starts"]))
        self.assertGreaterEqual(len(steady["kept"]), 6)


class AverageTests(unittest.TestCase):
    def test_circular_mean_of_angles_and_anchored_translation(self):
        n = 4
        samples = np.zeros((2, n + 1, 3))
        samples[0, :, 0] = 1.0 + 0.1 * np.arange(n + 1)       # root x: cycle 0 starts at 1.0, travels 0.4
        samples[1, :, 0] = 5.0 + 0.2 * np.arange(n + 1)       # cycle 1 starts at 5.0, travels 0.8
        samples[0, :, 2], samples[1, :, 2] = 3.1, -3.1         # an angle on either side of pi
        avg = cycle.phase_average(samples, ["x", "z", "angle"], ref=0)
        np.testing.assert_allclose(avg[:, 0], 1.0 + 0.15 * np.arange(n + 1))
        self.assertTrue(np.all(np.abs(avg[:, 2]) > 3.1), "the circular mean of 3.1 and -3.1 is pi, not 0")

    def test_identity_is_aligned_per_cycle_before_averaging(self):
        n = 6
        phase = 2 * math.pi * np.arange(n + 1) / n
        left, right = np.sin(phase), np.sin(phase + 1.0)
        samples = np.zeros((3, n + 1, 4))
        for c in range(3):
            samples[c, :, 2], samples[c, :, 3] = (right, left) if c == 1 else (left, right)
        aligned, exchanged = cycle.align_identity(samples, [([2], [3])], ref=0)
        self.assertEqual(exchanged, [[False], [True], [False]])
        for c in range(3):
            np.testing.assert_allclose(aligned[c, :, 2], left)

    def test_roll_continues_translation_past_the_seam(self):
        n = 4
        avg = np.zeros((n + 1, 3))
        avg[:, 0] = 0.25 * np.arange(n + 1)                  # travels 1.0 a cycle
        avg[:, 2] = np.sin(2 * math.pi * np.arange(n + 1) / n)
        rolled = cycle.roll(avg, 1, ["x", "z", "angle"])
        np.testing.assert_allclose(rolled[:, 0], 0.25 + 0.25 * np.arange(n + 1))
        np.testing.assert_allclose(rolled[:, 2], np.sin(2 * math.pi * (np.arange(n + 1) + 1) / n), atol=1e-12)

    def test_inconsistency_separates_repeating_from_random_joints(self):
        rng = np.random.default_rng(0)
        n, cycles = 12, 6
        phase = 2 * math.pi * np.arange(n + 1) / n
        samples = np.zeros((cycles, n + 1, 4))
        samples[:, :, 2] = np.sin(phase)
        samples[:, :, 3] = rng.normal(size=(cycles, n + 1))
        score = cycle.inconsistency(samples, [2, 3])
        self.assertLess(score[0], 0.01)
        self.assertGreater(score[1], 0.7)


class LegTests(unittest.TestCase):
    def test_leg_joints_by_chain_role(self):
        """The well-observed joints: a girdled (front) leg's two bones above the paw, a hind leg's first two."""
        names = ["dx", "dz"] + [b for chain in WOLF["chains"].values() for b in chain]
        self.assertEqual(cycle.leg_joints(names, WOLF),
                         ["forearm_L", "wrist_L", "forearm_R", "wrist_R", "thigh_L", "shin_L", "thigh_R", "shin_R"])
        equal = {"chains": {"fl": ["upper_fl", "lower_fl", "paw_fl"], "fr": ["upper_fr", "lower_fr", "paw_fr"]},
                 "legPairs": [["fl", "fr"]]}
        self.assertEqual(cycle.leg_joints(["dx", "dz", "upper_fl", "lower_fl", "paw_fl", "upper_fr", "lower_fr",
                                           "paw_fr"], equal), ["upper_fl", "lower_fl", "upper_fr", "lower_fr"])

    def test_a_symmetric_silhouette_is_half_the_gait(self):
        """Left and right legs at one hip, half a cycle apart: the silhouette repeats every half gait period, but
        each leg matches its partner, not itself, half a gait period later."""
        project = {"chains": {"hl": ["a"], "fl": ["b"], "hr": ["c"], "fr": ["d"]},
                   "legPairs": [["fl", "fr"], ["hl", "hr"]]}
        names = ["dx", "dz", "a", "b", "c", "d"]
        pairs = cycle.leg_pairs(names, project)
        symmetric = np.array([gait(t, 24.0, phases=(0.0, 0.25, 0.5, 0.75)) for t in range(96)])
        self.assertTrue(cycle.half_period(symmetric, pairs, 12.0))
        self.assertFalse(cycle.half_period(symmetric, pairs, 24.0))
        lope = np.array([gait(t, 12.0, phases=(0.0, 0.3, 0.1, 0.4)) for t in range(96)])
        self.assertFalse(cycle.half_period(lope, pairs, 12.0))


class PeriodicPriorTests(unittest.TestCase):
    """The fitter's second pass pulls each frame's joints toward the first pass's phase average, weighted by how
    little the silhouette pinned them (their inconsistency from cycle to cycle)."""

    def clip(self, frames=121, period=15.2, seed=0):
        rng = np.random.default_rng(seed)
        values = np.array([gait(t, period) + list(rng.uniform(-0.6, 0.6, 2)) for t in range(frames)])
        masks = np.array([toy_mask(v) for v in values])
        return values, masks

    def test_weights_follow_inconsistency_and_targets_are_the_phase_average(self):
        values, masks = self.clip()
        pairs = [([3], [5]), ([2], [4])]
        prior = cycle.periodic_prior(values, masks, 24.0, pairs, first=13)
        self.assertAlmostEqual(prior["period"], 15.2, delta=0.15)
        weights = prior["weights"]
        self.assertEqual(len(weights), values.shape[1])
        self.assertEqual(list(weights[:2]), [0.0, 0.0], "the root is not pulled")
        self.assertTrue(np.all(weights[2:6] < 0.05), weights)
        self.assertTrue(np.all(weights[6:] > 0.9), weights)
        targets = prior["targets"]
        self.assertEqual(targets.shape, values.shape)
        np.testing.assert_allclose(targets[60:100, 2:6], values[60:100, 2:6], atol=0.05)
        self.assertLess(float(np.std(targets[:, 6])), 0.3 * float(np.std(values[:, 6])),
                        "a random joint's target is its (near-flat) phase average")

    def test_no_period_no_prior(self):
        swing = np.array([[0, 0] + [-1.0 + 2.0 * f / 59] * 4 for f in range(60)])
        masks = np.array([toy_mask(v) for v in swing])
        self.assertIsNone(cycle.periodic_prior(swing, masks, 24.0, [], first=6))


if __name__ == "__main__":
    unittest.main()
