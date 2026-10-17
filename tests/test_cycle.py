"""The video's cycle (anim/cycle.py): the period from the masks' own self-similarity, the steady cycles after the
warm-up, the phase-aligned average and the helpers facts and the fitter's periodic pass share."""
import json
import math
import unittest
from pathlib import Path

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


class CyclePeriodTests(unittest.TestCase):
    """Fix round 1: each steady cycle's own period, read directly (the cadence's one-time-scale check)."""

    def test_each_cycle_reads_its_own_period(self):
        steady = masks_of(np.arange(121), 15.2)
        periods = cycle.cycle_periods(steady, [10.0, 25.2, 40.4, 55.6], 15.2)
        self.assertEqual(len(periods), 4)
        for p in periods:
            self.assertAlmostEqual(p, 15.2, delta=0.35)

    def test_a_cycle_that_runs_slower_reads_longer(self):
        # Frames 0..60 at period 12, then the motion slows to period 16 (phase continuous at frame 60).
        times = [t if t <= 60 else 60 + (t - 60) * 12 / 16 for t in range(121)]
        periods = cycle.cycle_periods(masks_of(times, 12.0), [12.0, 24.0, 72.0, 88.0], 14.0, spread=0.3)
        self.assertLess(abs(periods[0] - 12.0), 0.6)
        self.assertLess(abs(periods[-1] - 16.0), 0.8)
        self.assertGreater(periods[-1] - periods[0], 3.0)


# ------------------------------------------------------------------------- round 3: the vole's derived fixture
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vole_scurry_derived.json"


def derived():
    return json.loads(FIXTURE.read_text())


def curve(name):
    block = derived()["lagScores"][name]
    return {int(k): v for k, v in block["scores"].items()}, block["low"], block["high"]


class PeriodFromScoresTests(unittest.TestCase):
    """Vole round 3: the period is the dip that is clearly a minimum against its neighbourhood (prominence) and the
    curve's typical level (margin), not the first dip under the gate. The field vole's low, rigid body changes its
    outline by at most 17 %, so every lag scores under MASK_PERIOD_MAX and a 0.003-deep noise dip at 16 frames came
    first; its silhouette repeats at 30 frames (0.025) and 60 (0.011)."""

    def test_the_vole_period_is_the_deep_repeating_dip_not_the_first_dip_under_the_gate(self):
        scores, low, high = curve("vole_scurry_r2")
        found = cycle.period_from_scores(scores, low, high)
        self.assertAlmostEqual(found["fundamentalLag"], 30.0, delta=0.3)
        self.assertTrue(found["chosen"], found["reason"])
        self.assertGreaterEqual(found["margin"], 10.0)
        self.assertGreaterEqual(found["prominence"], cycle.MASK_PERIOD_PROMINENCE * found["level"])
        noise = [c for c in found["candidates"] if abs(c["lag"] - 16.0) < 0.5]
        self.assertEqual(len(noise), 1)
        self.assertLess(noise[0]["prominence"], cycle.MASK_PERIOD_PROMINENCE * found["level"],
                        "the 16-frame dip is noise: 0.003 deep against a curve at 0.14")

    def test_the_wolf_lope_keeps_its_period(self):
        scores, low, high = curve("wolf_lope")
        found = cycle.period_from_scores(scores, low, high)
        self.assertTrue(found["chosen"], found["reason"])
        self.assertAlmostEqual(found["fundamentalLag"], 15.0918, delta=1e-3)
        self.assertAlmostEqual(found["margin"], 2.66, delta=0.01)

    def test_the_aperiodic_vole_round_one_is_still_refused(self):
        """Round 1's stilt-legged shuffle (stride ~5-6 frames, undersampled): nothing repeats; its best dips (57
        and 60 frames, 0.044 and 0.034) are rivals of each other, not a period and its multiples."""
        scores, low, high = curve("vole_scurry_r1")
        found = cycle.period_from_scores(scores, low, high)
        self.assertFalse(found["chosen"])
        self.assertIn("no lag is clearly the period", found["reason"])

    def test_the_fundamental_is_preferred_when_its_multiple_dips_deeper(self):
        lags = range(5, 62)
        scores = {lag: 0.4 - 0.3 * max(0.0, 1 - abs(lag - 20) / 4) - 0.35 * max(0.0, 1 - abs(lag - 40) / 4)
                  - 0.3 * max(0.0, 1 - abs(lag - 60) / 4) for lag in lags}
        found = cycle.period_from_scores(scores, 6, 60)
        self.assertTrue(found["chosen"], found["reason"])
        self.assertAlmostEqual(found["fundamentalLag"], 20.0, delta=0.3)

    def test_a_deep_dip_with_an_equally_deep_rival_is_not_a_period(self):
        lags = range(5, 62)
        scores = {lag: 0.4 - 0.3 * max(0.0, 1 - abs(lag - 23) / 3) - 0.3 * max(0.0, 1 - abs(lag - 37) / 3)
                  for lag in lags}
        found = cycle.period_from_scores(scores, 6, 60)
        self.assertFalse(found["chosen"])

    def test_mask_period_reports_the_new_numbers(self):
        found = cycle.mask_period(masks_of(range(121), 15.2), 24.0)
        for key in ("prominence", "level", "rival", "rivalGate", "prominenceGate"):
            self.assertIn(key, found)


def trot_tracks(period=40, frames=160, amplitude=0.02):
    """In-place paw x of a leg pair half a cycle apart: each sweeps back (stance) and forward (swing). They meet
    twice a cycle, crossing with opposite velocities."""
    t = np.arange(frames)
    return (amplitude * np.sin(2 * np.pi * t / period), amplitude * np.sin(2 * np.pi * t / period + np.pi))


def bounced(xa, xb, boundaries):
    """The pair as a fit that mixes identity at the given frames: from each boundary on the labels are exchanged."""
    flip = np.zeros(len(xa), bool)
    for b in boundaries:
        flip[b:] = ~flip[b:]
    return np.where(flip, xb, xa), np.where(flip, xa, xb), flip


class LegIdentityTests(unittest.TestCase):
    """Vole round 3: a side silhouette cannot tell near legs from far ones, and where a pair's paws meet the fit may
    bounce them back instead of letting them cross. Identity is resolved across the clip from the paw tracks'
    continuity: at each meeting the paws keep their velocities (one swinging forward, one sweeping back), so the
    labelling whose velocities carry on through the meeting is the right one."""

    def test_a_clean_crossing_pair_is_left_alone(self):
        xa, xb = trot_tracks()
        found = cycle.resolve_identity(xa, xb)
        self.assertEqual(found["boundaries"], [])
        self.assertFalse(found["flip"].any())
        self.assertGreaterEqual(len(found["meetings"]), 6)

    def test_bounces_are_undone(self):
        xa, xb = trot_tracks()
        # The paws meet at multiples of half a period (frames 20, 40, ...): bounce every other meeting.
        fa, fb, truth = bounced(xa, xb, [41, 81, 121])
        found = cycle.resolve_identity(fa, fb)
        self.assertEqual(len(found["boundaries"]), 3, found["meetings"])
        for got, want in zip(found["boundaries"], [41, 81, 121]):
            self.assertLessEqual(abs(got - want), 1)
        relabelled_a = np.where(found["flip"], fb, fa)
        np.testing.assert_allclose(relabelled_a[45:], xa[45:], atol=1e-3)

    def test_the_vole_fronts_are_relabelled_and_its_hinds_kept(self):
        tracks = derived()["voleR2Paws"]["tracks"]
        x = {c: np.array([p[0] for p in v]) for c, v in tracks.items()}
        fronts = cycle.resolve_identity(x["front_L"], x["front_R"])
        hinds = cycle.resolve_identity(x["hind_L"], x["hind_R"])
        self.assertEqual(hinds["boundaries"], [], "the hind pair crosses cleanly at 32, 62 and 92")
        self.assertEqual(len(fronts["boundaries"]), 2, fronts["meetings"])
        for got, want in zip(fronts["boundaries"], [32, 92]):
            self.assertLessEqual(abs(got - want), 1, "the fronts bounce at 31 and 91 and cross at 61")

    def test_after_relabelling_both_vole_pairs_say_the_masks_show_half_a_stride(self):
        tracks = derived()["voleR2Paws"]["tracks"]
        chains = ["front_L", "front_R", "hind_L", "hind_R"]
        x = np.stack([[p[0] for p in tracks[c]] for c in chains], axis=1)
        pairs = [([0], [1]), ([2], [3])]
        before = cycle.pair_verdicts(x, pairs, 30.05)
        self.assertEqual(before[0]["verdict"], "mixed", before)
        flips = [cycle.resolve_identity(x[:, a[0]], x[:, b[0]])["flip"] for a, b in pairs]
        fixed = cycle.relabel(x, pairs, flips)
        after = cycle.pair_verdicts(fixed, pairs, 30.05)
        self.assertEqual([v["verdict"] for v in after], ["half", "half"], after)


if __name__ == "__main__":
    unittest.main()
