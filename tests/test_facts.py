import math
import unittest

import numpy as np

from anim import facts
from tests.toy_gait import toy_mask, toy_video

# Changed with Task 17b fix round 2: facts reads the leg chains (the leg pairs' joints), so the project names them.
PROJECT = {"paws": {"hind_L": "paw_hl", "front_L": "paw_fl", "hind_R": "paw_hr", "front_R": "paw_fr"},
           "chains": {"hind_L": ["a"], "front_L": ["b"], "hind_R": ["c"], "front_R": ["d"]},
           "legPairs": [["front_L", "front_R"], ["hind_L", "hind_R"]], "phaseOrigin": "hind_L"}
PHASES = {"hind_L": 0.0, "front_L": 0.25, "hind_R": 0.5, "front_R": 0.75}
LOPE = {"clip": "lope", "loop": True, "targetSpeedMps": 3.0}


def synthetic(frames=60, period=12, fps=24.0, speed=3.0, phases=PHASES, envelope=0.0):
    """A four-beat gait. `envelope` > 0 makes the joint amplitude settle over the clip, so the cleanest seam is the
    last one; paw heights stay clean either way."""
    entries = []
    for f in range(frames):
        t = f / period
        amplitude = 1.0 + envelope * ((frames - f) / frames) ** 2
        values = [0.0, 0.0] + [amplitude * math.sin(2 * math.pi * (t - p)) for p in phases.values()]
        paws = {chain: [-speed * f / fps, max(0.0, math.sin(2 * math.pi * (t - p)))]
                for chain, p in phases.items()}
        entries.append({"index": f, "values": values, "paws": paws, "flags": []})
    return {"fps": fps, "names": ["dx", "dz", "a", "b", "c", "d"], "frames": entries}


def joint_values(poses):
    return np.array([f["values"][2:] for f in poses["frames"]])


def gallop_values(period, frames=120, powers=(0.6, 0.3, 0.1)):
    """Joint angles of a short gait with harmonic content: harmonic h has power powers[h - 1] (amplitude its root)."""
    t = np.arange(frames)
    return np.stack([sum(math.sqrt(w) * np.sin(2 * math.pi * h * (t / period - p)) for h, w in enumerate(powers, 1))
                     for p in PHASES.values()], axis=1)


def analytic_values(f, period=9.6, amplitude=0.015):
    """Every value column as a known function of (real) frame time: the root offsets are linear, so linear
    interpolation is exact on them and they expose any error in the sample times; the joints are sines whose
    amplitude keeps the linear-interpolation error, amplitude * (2 pi / period)^2 / 8 ~ 8.0e-4, under 1e-3."""
    return [-0.125 * f, 0.02 * f] + [amplitude * math.sin(2 * math.pi * (f / period - p)) for p in PHASES.values()]


def run(poses, project=PROJECT, brief=LOPE, gain=1.0, masks_from=None, render=None, hips=None):
    """facts on a toy video of the poses (tests/toy_gait.py): its masks, and its renderer for the cycle check."""
    kwargs = {"hips": hips} if hips else {}
    masks, toy_render = toy_video(poses, gain, masks_from=masks_from, **kwargs)
    return facts.clip_facts(poses, project, brief, masks=masks, render=render or toy_render)


def expected_cycle(result, function):
    """What facts' phase average should key for a clip whose values are `function(frame time)`: the circular mean,
    over the cycles it averaged, of the joints at each cycle's phase samples, and the root of the first averaged
    cycle (start, the phase origin's touchdown) at each phase, each cycle moved to it."""
    cycle = result["cycle"]
    period, n, start = cycle["periodFrames"], len(cycle["frames"]) - 1, cycle["start"]
    shift = start - result["cycleStarts"][0]
    rows = []
    for k in range(n):
        per = np.array([function(s + shift + k * period / n) for s in result["cycleStarts"]])
        joints = np.arctan2(np.sin(per[:, 2:]).mean(axis=0), np.cos(per[:, 2:]).mean(axis=0))
        root = np.array(function(start + k * period / n))[:2]
        rows.append(np.concatenate([root, joints]))
    return np.array(rows)


class FactsTests(unittest.TestCase):
    # Changed with Task 17b fix round 2: facts finds the period in the video's masks, not in the fitted angles, and
    # keys the phase-aligned average of the steady cycles, checked against the masks. `period_search` stays (the
    # leg-joint report); `period_frames`, `best_start`, `steadiness` and `steadiest_start` are gone with the single
    # cycle they chose.
    def test_angle_period_search(self):
        self.assertAlmostEqual(facts.period_search(joint_values(synthetic()), 24.0)["period"], 12, places=6)

    def test_contact_window_wraps(self):
        mask = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], bool)
        window, duty = facts.contact_window(mask)
        self.assertEqual(window, [0.75, 0.25])
        self.assertAlmostEqual(duty, 0.5)

    def test_two_touchdowns_refused(self):
        with self.assertRaises(facts.FactsError):
            facts.contact_window(np.array([1, 0, 1, 0, 0, 0], bool))

    def test_clip_facts(self):
        result = run(synthetic())
        entry = result["entry"]
        self.assertAlmostEqual(result["period"], 12, delta=0.05)
        self.assertAlmostEqual(entry["authored_speed_mps"], 3.0, places=2)
        self.assertAlmostEqual(entry["stride_m"], 1.5, delta=0.01)
        self.assertAlmostEqual(entry["duration_s"], result["period"] / 24.0)
        self.assertEqual(entry["footfall_phase"]["paw_hl"], 0.0)
        self.assertAlmostEqual(entry["footfall_phase"]["paw_fl"], 0.25, delta=1 / 12)
        self.assertAlmostEqual(entry["duty_factor"]["paw_hr"], 0.5, delta=1 / 12)
        self.assertEqual(list(entry)[:5], ["duration_s", "loop", "authored_speed_mps", "stride_m", "contacts"])
        self.assertEqual(result["warnings"], [])
        cycle = result["cycle"]
        self.assertEqual(len(cycle["frames"]), 13)
        self.assertEqual(cycle["frames"][-1], cycle["frames"][0])
        self.assertEqual(len(cycle["frames"][0]), 6)
        self.assertAlmostEqual(cycle["fps"], 12 * 24.0 / result["period"], places=6)
        self.assertEqual(cycle["start"], result["start"])
        self.assertEqual(cycle["periodFrames"], result["period"])
        self.assertGreaterEqual(result["start"], 6, "the cycle starts after the lead-in")

    def test_fractional_period(self):
        for truth, n in ((9.6, 10), (12.4, 12)):
            with self.subTest(period=truth):
                result = run(synthetic(frames=120, period=truth))
                entry, cycle = result["entry"], result["cycle"]
                period = cycle["periodFrames"]
                self.assertAlmostEqual(period, truth, delta=0.1)
                self.assertEqual(result["period"], period)
                self.assertEqual(entry["frames"], n)
                self.assertAlmostEqual(entry["duration_s"], period / 24.0, delta=1e-6)
                self.assertAlmostEqual(entry["fps"], n * 24.0 / period, delta=1e-9)
                self.assertAlmostEqual(cycle["fps"], entry["fps"], delta=1e-9)
                self.assertEqual(len(cycle["frames"]), n + 1)
                self.assertEqual(cycle["frames"][-1], cycle["frames"][0])
                self.assertGreaterEqual(cycle["start"], 12)
                self.assertLessEqual(cycle["start"] + period, 119)
                self.assertEqual(set(entry["contacts"]), {"paw_hl", "paw_fl", "paw_hr", "paw_fr"})
                for bone, window in entry["contacts"].items():
                    self.assertEqual(len(window), 2, bone)
                    self.assertAlmostEqual(entry["duty_factor"][bone], 0.5, delta=1.5 / n)
                self.assertEqual(entry["footfall_phase"]["paw_hl"], 0.0)
                self.assertAlmostEqual(entry["authored_speed_mps"], 3.0, delta=0.02 * 3.0)
                self.assertAlmostEqual(entry["stride_m"], 3.0 * truth / 24.0, delta=0.03 * 3.0 * truth / 24.0)

    def test_short_harmonic_period_is_the_fundamental(self):
        # Fix round 2: the gate is on the parabola-refined score, not on the whole-frame lag's raw score. 7.2 frames
        # is the brief's 0.3 s gallop; 7.5 frames (0.3125 s) puts the true period halfway between whole-frame lags,
        # where both neighbours score over PERIOD_MAX_SCORE and the old gate returned the double period (~15).
        for truth in (7.2, 7.5):
            with self.subTest(period=truth):
                self.assertAlmostEqual(facts.period_search(gallop_values(truth), 24.0)["period"], truth, delta=0.15)

    def test_cycle_samples_the_clip_at_its_period(self):
        # Fix round 2: the bake (Task 11) keys cycle.frames, so check them against the analytic clip (now: the
        # average, over the cycles facts averaged, of the analytic clip at each cycle's phase samples). The toy video
        # draws the small analytic angles at a visible gain.
        poses = synthetic(frames=120, period=9.6)
        for entry in poses["frames"]:
            entry["values"] = analytic_values(entry["index"])
        result = run(poses, gain=1 / 0.015)
        cycle = result["cycle"]
        period, keys = cycle["periodFrames"], cycle["frames"]
        n = len(keys) - 1
        self.assertAlmostEqual(period, 9.6, delta=0.1)
        self.assertEqual(n, 10)
        self.assertEqual(n, result["entry"]["frames"])
        np.testing.assert_allclose(np.array(keys[:n]), expected_cycle(result, analytic_values), rtol=0, atol=1e-3)
        self.assertEqual(keys[n], keys[0])
        self.assertAlmostEqual(cycle["fps"], n * 24.0 / period, delta=1e-9)
        self.assertAlmostEqual(result["entry"]["fps"], cycle["fps"], delta=1e-9)

    def drift_expected(self, result, root):
        """The mean, over the averaged cycles, of the root's travel over one period."""
        period = result["period"]
        travel = [np.array(root(s + period)) - np.array(root(s)) for s in result["cycleStarts"]]
        return np.mean(travel, axis=0)

    def test_cycle_drift_is_one_period_of_root_travel(self):
        # Task 11 fix round 2: the bake removes a loop's linear drift exactly, so facts reports dx and dz one period
        # on minus at the start (now averaged over the cycles), before the closing key is overwritten with key 0.
        def root(f):
            return [0.002 * f + 0.001 * math.sin(2 * math.pi * f / 12), -0.0005 * f + 0.001 * math.cos(2 * math.pi * f / 12)]
        poses = synthetic()
        for entry in poses["frames"]:
            entry["values"][:2] = root(entry["index"])
        result = run(poses)
        drift = result["cycle"]["drift"]
        self.assertEqual(set(drift), {"dx", "dz"})
        want = self.drift_expected(result, root)
        self.assertAlmostEqual(drift["dx"], want[0], delta=1e-6)
        self.assertAlmostEqual(drift["dz"], want[1], delta=1e-6)
        self.assertAlmostEqual(drift["dx"], 0.002 * result["period"], delta=1e-4)
        self.assertEqual(result["cycle"]["frames"][-1], result["cycle"]["frames"][0])

    def test_purely_periodic_root_has_no_drift(self):
        def root(f):
            return [0.001 * math.sin(2 * math.pi * f / 12), 0.001 * math.cos(2 * math.pi * f / 12)]
        poses = synthetic()
        for entry in poses["frames"]:
            entry["values"][:2] = root(entry["index"])
        drift = run(poses)["cycle"]["drift"]
        self.assertAlmostEqual(drift["dx"], 0.0, delta=1e-4)
        self.assertAlmostEqual(drift["dz"], 0.0, delta=1e-4)

    def test_drift_at_a_sub_frame_period(self):
        # analytic_values' root offsets are linear in frame time, so the resample at s + P is exact on them.
        poses = synthetic(frames=120, period=9.6)
        for entry in poses["frames"]:
            entry["values"] = analytic_values(entry["index"])
        result = run(poses, gain=1 / 0.015)
        period, drift = result["period"], result["cycle"]["drift"]
        self.assertNotAlmostEqual(period, round(period), places=3)
        self.assertAlmostEqual(drift["dx"], -0.125 * period, delta=1e-6)
        self.assertAlmostEqual(drift["dz"], 0.02 * period, delta=1e-6)

    def test_no_full_cycle_in_range(self):
        found = facts.period_search(joint_values(synthetic(frames=50, period=30)), 24.0)
        self.assertIsNone(found["period"])
        with self.assertRaisesRegex(facts.FactsError, "no full cycle found in the video between 0.25 s and 2.5 s"):
            run(synthetic(frames=50, period=30))

    def test_phase_origin_shift(self):
        project = {**PROJECT, "phaseOrigin": "front_L"}
        result = run(synthetic(), project=project)
        entry, period = result["entry"], result["period"]
        self.assertEqual(entry["footfall_phase"]["paw_fl"], 0.0)
        self.assertAlmostEqual(entry["footfall_phase"]["paw_hl"], 0.75, delta=1 / 12)
        self.assertGreaterEqual(result["start"], 6)
        self.assertLessEqual(result["start"] + period, 59)

    def test_touchdown_shift_rolls_the_average(self):
        """The averaged cycle is rolled to the phase origin's touchdown, so no real stretch of the clip has to start
        there: a 26-frame clip with one full cycle after the lead-in still gives a cycle that starts at the touchdown
        (before fix round 2 this clip was refused)."""
        phases = {chain: (p + 0.625) % 1.0 for chain, p in PHASES.items()}
        result = run(synthetic(frames=26, phases=phases, envelope=0.3))
        self.assertEqual(result["entry"]["footfall_phase"]["paw_hl"], 0.0)
        self.assertEqual(result["cyclesAveraged"], 1)
        keys = np.array(result["cycle"]["frames"])
        self.assertLess(float(np.abs(np.diff(keys[:, 2:], axis=0)).max()), 0.8, "the rolled keys stay continuous")

    def test_one_shot(self):
        planted = {"hind_L": {0, 1, 2}, "front_L": {4, 5, 8, 9, 10}, "hind_R": set(range(11)), "front_R": {7, 8, 9, 10}}
        frames = [{"index": f, "values": [0.0, 0.0, 0.1 * f, 0.0, 0.0, 0.0], "flags": [],
                   "paws": {c: [-3.0 * f / 24.0, 0.0 if f in planted[c] else 0.2] for c in planted}}
                  for f in range(11)]
        poses = {"fps": 24.0, "names": ["dx", "dz", "a", "b", "c", "d"], "frames": frames}
        result = facts.clip_facts(poses, PROJECT, {"clip": "pounce", "loop": False, "targetSpeedMps": None})
        entry = result["entry"]
        self.assertIs(entry["loop"], False)
        self.assertEqual(entry["frames"], 10)
        self.assertAlmostEqual(entry["duration_s"], 10 / 24.0)
        self.assertEqual(entry["stride_m"], 0.0)
        self.assertAlmostEqual(entry["authored_speed_mps"], 3.0, places=3)
        self.assertEqual(entry["contact_windows"], {"paw_hl": [[0.0, 0.3]], "paw_fl": [[0.4, 0.6], [0.8, 1.0]],
                                                    "paw_hr": [[0.0, 1.0]], "paw_fr": [[0.7, 1.0]]})
        self.assertEqual(entry["contacts"], {"paw_hl": [0.0, 0.3], "paw_fl": [0.4, 0.6], "paw_hr": [0.0, 1.0],
                                             "paw_fr": [0.7, 1.0]})
        self.assertEqual((result["start"], result["period"]), (0, 10))
        self.assertEqual(result["cycle"]["drift"], {"dx": 0.0, "dz": 0.0})     # a one-shot keeps its travel

    def test_windows(self):
        self.assertEqual(facts.windows(np.array([], bool)), [])
        self.assertEqual(facts.windows(np.zeros(5, bool)), [])
        self.assertEqual(facts.windows(np.array([0, 0, 0, 1, 1], bool)), [[0.75, 1.0]])
        self.assertEqual(facts.windows(np.array([1, 1, 0, 0, 1, 1, 0, 0, 0], bool)), [[0.0, 0.25], [0.5, 0.75]])

    def test_speed_mismatch_warns(self):
        result = run(synthetic(speed=5.0))
        self.assertTrue(result["warnings"])

    def test_too_short(self):
        with self.assertRaises(facts.FactsError):
            run(synthetic(frames=10), brief={"clip": "lope", "loop": True, "targetSpeedMps": None})

    def test_steady_cycles_skip_the_warm_up(self):
        """An AI video starts from the reference pose and settles over a couple of cycles. Here the joints swing at
        a third of their amplitude until frame 26 and reach full amplitude by frame 34: only cycles in the steady
        part are averaged."""
        poses = synthetic(frames=96)
        for entry in poses["frames"]:
            f = entry["index"]
            gain = 0.3 + 0.7 * min(1.0, max(0.0, (f - 26) / 8))
            entry["values"] = entry["values"][:2] + [gain * v for v in entry["values"][2:]]
        result = run(poses)
        self.assertAlmostEqual(result["period"], 12, delta=0.1)
        self.assertGreaterEqual(min(result["cycleStarts"]), 34)
        self.assertGreaterEqual(result["cyclesAveraged"], 4)
        self.assertGreaterEqual(result["start"], 34)
        self.assertEqual(result["entry"]["footfall_phase"]["paw_hl"], 0.0)

    def test_late_touchdown_never_averages_the_warm_up(self):
        """The hind_R touchdown falls late in the steadiest cycle; the average is rolled, never moved back into the
        warm-up (frames before 28)."""
        poses = synthetic(frames=48)
        for entry in poses["frames"]:
            f = entry["index"]
            gain = 0.3 + 0.7 * min(1.0, max(0.0, (f - 20) / 8))
            entry["values"] = entry["values"][:2] + [gain * v for v in entry["values"][2:]]
        result = run(poses, project={**PROJECT, "phaseOrigin": "hind_R"})
        self.assertGreaterEqual(min(result["cycleStarts"]), 28)
        self.assertEqual(result["entry"]["footfall_phase"]["paw_hr"], 0.0)

    def test_period_search_reports_the_score(self):
        """The period search returns the period and its refined score, and, when no minimum in range is under the
        gate, no period and the best score it saw (for reports that must say how far from periodic a fit is)."""
        values = joint_values(synthetic())
        found = facts.period_search(values, 24.0)
        self.assertAlmostEqual(found["period"], 12, delta=0.1)
        self.assertLess(found["score"], 0.01)
        noise = np.random.default_rng(0).normal(size=(60, 4))
        missing = facts.period_search(noise, 24.0)
        self.assertIsNone(missing["period"])
        self.assertGreater(missing["score"], facts.PERIOD_MAX_SCORE)


class VideoCycleTests(unittest.TestCase):
    """Task 17b fix round 2: the period comes from the video; the cycle is the phase-aligned average of the steady
    cycles; the average is rendered and checked against the masks."""

    def loose(self, frames=121, period=15.2, seed=0):
        """A clip whose leg joints follow the video but whose six extra joints (e to j: drawn by nothing, like the
        paw, neck and tail joints a silhouette barely constrains) are noise."""
        poses = synthetic(frames=frames, period=period)
        rng = np.random.default_rng(seed)
        poses["names"] = poses["names"] + list("efghij")
        for entry in poses["frames"]:
            entry["values"] = entry["values"] + list(rng.uniform(-1.5, 1.5, 6))
        return poses

    def test_loop_needs_the_video(self):
        with self.assertRaisesRegex(facts.FactsError, "masks"):
            facts.clip_facts(synthetic(), PROJECT, LOPE)

    def test_period_comes_from_the_video_not_the_angles(self):
        poses = self.loose()
        self.assertIsNone(facts.period_search(np.array([f["values"][2:] for f in poses["frames"]]), 24.0)["period"],
                          "the fitted angles alone hold no cycle")
        result = run(poses)
        self.assertAlmostEqual(result["period"], 15.2, delta=0.15)
        self.assertEqual(result["periodSource"], "video masks")
        mask = result["maskPeriod"]
        self.assertLessEqual(mask["score"], 0.25)
        self.assertGreaterEqual(mask["margin"], 2.0)
        legs = result["legPeriod"]
        self.assertEqual(legs["joints"], ["b", "d", "a", "c"])
        self.assertLess(legs["scoreAtPeriod"], 0.05)
        self.assertIn("search", legs)

    def test_facts_records_the_averaged_cycle_check(self):
        result = run(self.loose())
        n = result["entry"]["frames"]
        self.assertEqual(result["cyclesAveraged"], len(result["cycleStarts"]))
        self.assertGreaterEqual(result["cyclesAveraged"], 5)
        check = result["cycleCheck"]
        self.assertGreaterEqual(result["averagedCycleIoU"], facts.CYCLE_MIN_IOU)
        self.assertEqual(check["meanIoU"], result["averagedCycleIoU"])
        self.assertEqual(len(check["phaseIoU"]), n)
        self.assertEqual(check["gate"], 0.85)
        self.assertGreater(check["frames"], 4 * n)

    def test_an_averaged_cycle_that_misses_the_video_is_refused(self):
        def wrong(values):
            return [toy_mask(v, shift=30) for v in values]
        with self.assertRaisesRegex(facts.FactsError, r"averaged cycle .* IoU 0\.\d+ .* 0\.85") as caught:
            run(self.loose(), render=wrong)
        self.assertIn("per phase", str(caught.exception))

    def test_a_symmetric_silhouette_doubles_the_period(self):
        """Each leg pair at one hip, half a cycle apart: the video repeats every 12 frames, the gait every 24."""
        result = run(synthetic(frames=120, period=24), hips=(30, 70, 30, 70))
        self.assertAlmostEqual(result["maskPeriod"]["period"], 12, delta=0.15)
        self.assertAlmostEqual(result["period"], 24, delta=0.3)
        self.assertTrue(result["maskPeriod"]["doubled"])
