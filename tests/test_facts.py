import math
import unittest
import unittest.mock

import numpy as np

from anim import facts, motion
from tests.toy_gait import toy_mask, toy_video

# Changed with Task 17b fix round 2: facts reads the leg chains (the leg pairs' joints), so the project names them.
PROJECT = {"paws": {"hind_L": "paw_hl", "front_L": "paw_fl", "hind_R": "paw_hr", "front_R": "paw_fr"},
           "chains": {"hind_L": ["a"], "front_L": ["b"], "hind_R": ["c"], "front_R": ["d"]},
           "legPairs": [["front_L", "front_R"], ["hind_L", "hind_R"]], "phaseOrigin": "hind_L"}
PHASES = {"hind_L": 0.0, "front_L": 0.25, "hind_R": 0.5, "front_R": 0.75}
LOPE = {"clip": "lope", "loop": True, "targetSpeedMps": 3.0}
# Task 17b fix round 4: each paw's height is measured from its own rest-pose tail height (the contract's rig.json).
# The synthetic rigs' paw tails all rest at 0.
REST = {c: 0.0 for c in PROJECT["paws"]}


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


TOY_GAIT = "test: the toy brief names no gait"


def run(poses, project=PROJECT, brief=LOPE, gain=1.0, masks_from=None, render=None, hips=None, rest=REST, **extra):
    """facts on a toy video of the poses (tests/toy_gait.py): its masks, and its renderer for the cycle check.
    Changed with fix round 1: a loop brief that names no gait is refused unless accepted, so the toy briefs without
    one pass TOY_GAIT."""
    if brief.get("loop") and not brief.get("gait"):
        extra.setdefault("accept_gait", TOY_GAIT)
    kwargs = {"hips": hips} if hips else {}
    masks, toy_render = toy_video(poses, gain, masks_from=masks_from, **kwargs)
    return facts.clip_facts(poses, project, brief, rest, masks=masks, render=render or toy_render, **extra)


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
        # Changed with fix round 1: the toy brief names no gait, which is accepted and warned about.
        self.assertEqual([w for w in result["warnings"] if not w.startswith("Gait not checked")], [])
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
        result = facts.clip_facts(poses, PROJECT, {"clip": "pounce", "loop": False, "targetSpeedMps": None}, REST)
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
            facts.clip_facts(synthetic(), PROJECT, LOPE, REST)

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


def lope(frames=120, period=20, windows=None, lift=0.1):
    """A three-beat lope with a lead and one suspension: each paw is on the ground (height 0) over its window (phase
    fractions [touchdown, liftoff)) and lifts off steeply, arcing up to `lift` between; the joints swing at the touchdown phases."""
    windows = windows or {"hind_L": (0.0, 0.3), "hind_R": (0.3, 0.55), "front_L": (0.3, 0.55),
                          "front_R": (0.55, 0.8)}
    poses = synthetic(frames=frames, period=period, phases={c: w[0] for c, w in windows.items()})
    for entry in poses["frames"]:
        t = entry["index"] / period
        for chain, (on, off) in windows.items():
            phase = (t - on) % 1.0
            stance = (off - on) % 1.0
            swing = 0.0 if phase < stance else math.sin(math.pi * (phase - stance) / (1.0 - stance)) ** 0.25
            entry["paws"][chain][1] = lift * swing
    return poses


class GroundContactTests(unittest.TestCase):
    """Task 17b fix round 3: contacts are measured against one ground shared by every paw, not each paw's own
    lowest point, so a paw that never comes down to it has no contact (`plants: false`), and the beats, lead,
    suspension and ground speed come from the planted paws only."""

    def test_ground_is_shared_by_all_paws(self):
        heights = {"hind_L": np.array([0.0, 0.0, 0.1, 0.2]), "front_L": np.array([0.05, 0.06, 0.1, 0.2])}
        ground = facts.ground_height(heights)
        self.assertAlmostEqual(ground, 0.0, delta=1e-9)
        masks = facts.contact_masks(heights, ground)
        self.assertEqual(masks["hind_L"].tolist(), [True, True, False, False])
        self.assertFalse(masks["front_L"].any(), "a paw 5 cm over the shared ground never touches it")

    def test_lope_beats_lead_and_suspension(self):
        result = run(lope())
        self.assertAlmostEqual(result["period"], 20, delta=0.1)
        gait = result["gait"]
        self.assertEqual(gait["beats"], 3)
        self.assertEqual([sorted(b) for b in gait["beatPaws"]], [["hind_L"], ["front_L", "hind_R"], ["front_R"]])
        np.testing.assert_allclose(gait["beatPhases"], [0.0, 0.3, 0.55], atol=0.051)
        self.assertEqual(gait["lead"], {"front": "front_R", "hind": "hind_R"})
        self.assertEqual(len(gait["suspensions"]), 1)
        np.testing.assert_allclose(gait["suspensions"][0], [0.8, 1.0], atol=0.051)
        self.assertEqual(gait["neverPlants"], [])
        self.assertTrue(all(p["plants"] for p in result["paws"].values()))
        self.assertAlmostEqual(result["speedMps"], 3.0, delta=0.05)

    def test_a_paw_that_never_reaches_the_ground_never_plants(self):
        poses = lope()
        for entry in poses["frames"]:
            entry["paws"]["front_L"][1] += 0.06          # front_L's lowest point is 6 cm over the ground
        result = run(poses)
        self.assertIs(result["paws"]["front_L"]["plants"], False)
        self.assertAlmostEqual(result["paws"]["front_L"]["lowestAboveGround"], 0.06, delta=0.005)
        self.assertTrue(all(result["paws"][c]["plants"] for c in ("hind_L", "hind_R", "front_R")))
        gait = result["gait"]
        self.assertEqual(gait["neverPlants"], ["front_L"])
        self.assertEqual(gait["beats"], 3)
        self.assertEqual(gait["beatPaws"], [["hind_L"], ["hind_R"], ["front_R"]])
        self.assertEqual(gait["lead"], {"front": None, "hind": "hind_R"})
        entry = result["entry"]
        self.assertNotIn("paw_fl", entry["contacts"])
        self.assertNotIn("paw_fl", entry["footfall_phase"])
        self.assertEqual(entry["duty_factor"]["paw_fl"], 0.0)
        self.assertAlmostEqual(result["speedMps"], 3.0, delta=0.05, msg="speed from the three planted paws")

    def test_under_two_planted_paws_have_no_speed(self):
        poses = lope()
        for entry in poses["frames"]:
            for chain in ("front_L", "front_R", "hind_R"):
                entry["paws"][chain][1] += 0.06
        result = run(poses)
        self.assertIsNone(result["speedMps"])
        self.assertIsNone(result["strideM"])
        self.assertIsNone(result["entry"]["authored_speed_mps"])
        self.assertIsNone(result["entry"]["stride_m"])
        self.assertRegex(result["speedReason"], "1 paw plants")
        self.assertEqual(result["gait"]["neverPlants"], ["front_L", "hind_R", "front_R"])
        self.assertEqual(result["gait"]["beats"], 1)

    def test_one_shot_uses_the_shared_ground(self):
        frames = [{"index": f, "values": [0.0, 0.0, 0.1 * f, 0.0, 0.0, 0.0], "flags": [],
                   "paws": {"hind_L": [-3.0 * f / 24.0, 0.0], "hind_R": [-3.0 * f / 24.0, 0.0],
                            "front_L": [-3.0 * f / 24.0, 0.1 + 0.01 * (f % 2)], "front_R": [0.0, 0.2]}}
                  for f in range(11)]
        poses = {"fps": 24.0, "names": ["dx", "dz", "a", "b", "c", "d"], "frames": frames}
        result = facts.clip_facts(poses, PROJECT, {"clip": "pounce", "loop": False, "targetSpeedMps": None}, REST)
        self.assertEqual({c: p["plants"] for c, p in result["paws"].items()},
                         {"hind_L": True, "front_L": False, "hind_R": True, "front_R": False})
        self.assertEqual(result["entry"]["contact_windows"]["paw_fl"], [])
        self.assertNotIn("paw_fl", result["entry"]["contacts"], "no [0, 0] window that would read as a footfall")
        self.assertAlmostEqual(result["speedMps"], 3.0, places=3)

    def test_leg_period_says_whether_it_passes(self):
        """legPeriod's score is lower-is-better against its gate; `passes` says so explicitly."""
        poses = synthetic(frames=120)
        values = np.array([f["values"] for f in poses["frames"]])
        good = facts.leg_period(values, poses["names"], PROJECT, 24.0, 12.0)
        self.assertLess(good["scoreAtPeriod"], good["gate"])
        self.assertIs(good["passes"], True)
        noise = values.copy()
        noise[:, 2:] = np.random.default_rng(1).uniform(-1.5, 1.5, (120, 4))
        bad = facts.leg_period(noise, poses["names"], PROJECT, 24.0, 12.0)
        self.assertGreater(bad["scoreAtPeriod"], bad["gate"])
        self.assertIs(bad["passes"], False)
        self.assertIs(run(synthetic())["legPeriod"]["passes"], True)

    def test_a_one_sample_lift_inside_a_stance_is_one_contact(self):
        """Fit noise can lift a planted paw a few millimetres over the tolerance for one sample (the real lope's
        hind_L does); one free sample between contact samples is bridged, two or more are a real lift."""
        heights = {"hind_L": np.array([0.1, 0.0, 0.03, 0.0, 0.1, 0.1, 0.1, 0.1]),
                   "hind_R": np.array([0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0])}
        masks = facts.contact_masks(heights, 0.0, circular=True)
        self.assertEqual(masks["hind_L"].tolist(), [False, True, True, True, False, False, False, False])
        self.assertEqual(facts.contact_window(masks["hind_L"])[0], [0.125, 0.5])
        two = facts.contact_masks({"p": np.array([0.0, 0.1, 0.1, 0.0, 0.1, 0.1])}, 0.0, circular=True)["p"]
        self.assertEqual(two.tolist(), [True, False, False, True, False, False])
        wrap = facts.contact_masks({"p": np.array([0.03, 0.0, 0.1, 0.1, 0.1, 0.0])}, 0.0, circular=True)["p"]
        self.assertEqual(wrap.tolist(), [True, True, False, False, False, True], "the gap at the seam is bridged")
        edge = facts.contact_masks({"p": np.array([0.03, 0.0, 0.1, 0.1, 0.1, 0.0])}, 0.0)["p"]
        self.assertEqual(edge.tolist(), [False, True, False, False, False, True], "a clip's ends are not joined")


# The chase3d wolf's paw tails at rest (contract/rig.json): the fronts 6.4 mm up, the hinds 20.7 mm down.
WOLF_REST = {"front_L": 0.0064, "front_R": 0.0064, "hind_L": -0.0207, "hind_R": -0.0207}


def rested(poses, rest=WOLF_REST):
    """The poses of a rig whose paw tails rest at `rest`: every paw's height carries its own rest offset."""
    for entry in poses["frames"]:
        for chain, z in rest.items():
            entry["paws"][chain][1] += z
    return poses


def rig_of(rest, project=PROJECT):
    """A contract rig.json stand-in with each paw bone's rest tail at height `rest[chain]`."""
    return {"bones": [{"name": project["paws"][c], "head": [0.0, 0.0, z + 0.05], "tail": [0.1, 0.0, z]}
                      for c, z in rest.items()] + [{"name": "root", "head": [0, 0, 0], "tail": [0, 0, 0.5]}]}


class RestRelativeHeightTests(unittest.TestCase):
    """Task 17b fix round 4: a rig's paw tails need not rest at one height (the wolf's fronts rest 2.7 cm over its
    hinds), so each paw's height is measured from its own rest-pose tail height before the shared ground and the
    tolerance apply; the rest offsets come from the contract's rig.json and are required."""

    def test_rest_heights_from_the_rig(self):
        rest = facts.rest_heights(rig_of(WOLF_REST), PROJECT)
        self.assertEqual(rest, WOLF_REST)

    def test_missing_rest_data_is_refused(self):
        rig = rig_of({c: z for c, z in WOLF_REST.items() if c != "front_L"})
        with self.assertRaisesRegex(facts.FactsError, r"rest.*front_L.*paw_fl"):
            facts.rest_heights(rig, PROJECT)
        with self.assertRaisesRegex(facts.FactsError, "rest"):
            facts.rest_heights({}, PROJECT)
        broken = rig_of(WOLF_REST)
        del broken["bones"][0]["tail"]
        with self.assertRaisesRegex(facts.FactsError, "rest"):
            facts.rest_heights(broken, PROJECT)
        with self.assertRaisesRegex(facts.FactsError, "rest"):
            run(lope(), rest=None)
        with self.assertRaisesRegex(facts.FactsError, r"rest.*hind_R"):
            run(lope(), rest={c: 0.0 for c in ("hind_L", "front_L", "front_R")})

    def test_paws_resting_at_different_heights_all_plant(self):
        poses = rested(lope())
        flat = run(poses, rest=REST)
        self.assertEqual(sorted(flat["gait"]["neverPlants"]), ["front_L", "front_R"],
                         "measured from one absolute ground, the fronts read 2.7 cm high and never plant")
        result = run(poses, rest=WOLF_REST)
        self.assertEqual(result["gait"]["neverPlants"], [])
        for chain, paw in result["paws"].items():
            self.assertIs(paw["plants"], True, chain)
            self.assertEqual(paw["restZ"], WOLF_REST[chain])
            self.assertAlmostEqual(paw["lowestAboveGround"], 0.0, delta=0.003, msg=chain)
        gait = result["gait"]
        self.assertEqual(gait["beats"], 3)
        self.assertEqual([sorted(b) for b in gait["beatPaws"]], [["hind_L"], ["front_L", "hind_R"], ["front_R"]])
        self.assertEqual(gait["lead"], {"front": "front_R", "hind": "hind_R"})
        self.assertEqual(len(gait["suspensions"]), 1)
        self.assertAlmostEqual(result["speedMps"], 3.0, delta=0.05)

    def test_a_paw_raised_above_its_own_rest_ground_never_plants(self):
        poses = rested(lope())
        for entry in poses["frames"]:
            entry["paws"]["front_L"][1] += 0.06
        result = run(poses, rest=WOLF_REST)
        self.assertIs(result["paws"]["front_L"]["plants"], False)
        self.assertAlmostEqual(result["paws"]["front_L"]["lowestAboveGround"], 0.06, delta=0.005)
        self.assertEqual(result["gait"]["neverPlants"], ["front_L"])

    def test_one_shot_heights_are_rest_relative(self):
        frames = [{"index": f, "values": [0.0, 0.0, 0.1 * f, 0.0, 0.0, 0.0], "flags": [],
                   "paws": {c: [-3.0 * f / 24.0, WOLF_REST[c]] for c in PROJECT["paws"]}} for f in range(11)]
        poses = {"fps": 24.0, "names": ["dx", "dz", "a", "b", "c", "d"], "frames": frames}
        brief = {"clip": "pounce", "loop": False, "targetSpeedMps": None}
        flat = facts.clip_facts(poses, PROJECT, brief, REST)
        self.assertFalse(flat["paws"]["front_L"]["plants"])
        result = facts.clip_facts(poses, PROJECT, brief, WOLF_REST)
        self.assertTrue(all(p["plants"] for p in result["paws"].values()))
        self.assertEqual(result["paws"]["hind_L"]["restZ"], -0.0207)


# ------------------------------------------------------------------------------------------------------ Task 19
class MotionContactTests(unittest.TestCase):
    """Task 19 step 2: a paw is on the ground while it is near the shared ground AND world-still (its in-place
    backward speed within SLIP_SHARE of the body's ground speed); a stance seeded within CONTACT_TOLERANCE carries on
    while the paw stays still within STANCE_TOLERANCE (a fit holds a planted paw a few cm up)."""

    def test_a_low_paw_sliding_with_the_body_is_not_planted(self):
        z = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
        still = np.full(8, -3.0)                       # in place, a planted paw moves back at the ground speed
        sliding = np.full(8, 0.0)                      # moving with the body: sliding forward over the ground
        masks = facts.stance_masks({"a": z, "b": z}, {"a": still, "b": sliding}, 0.0, 3.0, circular=True)
        self.assertEqual(masks["a"].tolist(), [True] * 4 + [False] * 4)
        self.assertFalse(masks["b"].any())

    def test_a_still_paw_hovering_next_to_its_touch_is_one_stance(self):
        z = np.array([0.2, 0.05, 0.04, 0.01, 0.0, 0.05, 0.2, 0.2])
        vx = np.array([3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 3.0, 3.0])
        mask = facts.stance_masks({"a": z}, {"a": vx}, 0.0, 3.0, circular=True)["a"]
        self.assertEqual(mask.tolist(), [False, True, True, True, True, True, False, False])
        hover = facts.stance_masks({"a": z + 0.03}, {"a": vx}, 0.0, 3.0, circular=True)["a"]
        self.assertFalse(hover.any(), "still but never within the contact tolerance: no stance")

    def test_short_gaps_are_bridged(self):
        mask = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], bool)
        self.assertEqual(facts.bridge(mask, True, gap=2).tolist(), [True] * 6 + [False] * 6)
        self.assertEqual(facts.bridge(mask, True, gap=1).tolist(), mask.tolist())

    def test_motion_contacts_measure_the_speed_from_the_planted_paws(self):
        t = np.arange(24) / 24
        z = {"a": np.where(t < 0.5, 0.0, 0.1), "b": np.where(t >= 0.5, 0.0, 0.1)}
        vx = {"a": np.where(t < 0.5, -2.0, 2.0), "b": np.where(t >= 0.5, -2.2, 2.0)}
        ground, speed, reason, masks = facts.motion_contacts(z, vx, circular=True)
        self.assertAlmostEqual(ground, 0.0)
        self.assertAlmostEqual(speed, 2.1, delta=0.11)
        self.assertIsNone(reason)
        self.assertEqual(int(masks["a"].sum()), 12)


def gallop(frames=120, period=20):
    """A transverse gallop: hinds in sequence, then fronts, the same side leading both, one gathered suspension."""
    return lope(frames, period, windows={"hind_L": (0.0, 0.3), "hind_R": (0.12, 0.42), "front_L": (0.45, 0.65),
                                         "front_R": (0.57, 0.77)})


class GaitGateTests(unittest.TestCase):
    """Task 19 step 2: the measured footfall pattern is compared with the brief's `gait`; a mismatch refuses, naming
    the measured pattern, unless it is accepted with a recorded reason."""

    def test_the_vocabulary(self):
        self.assertEqual(facts.expected_gait("canid lope: a 3-beat canter with a lead"), "canter")
        self.assertEqual(facts.expected_gait("4-beat lateral walk"), "walk")
        self.assertEqual(facts.expected_gait("a working trot"), "trot")
        self.assertEqual(facts.expected_gait("a pace"), "pace")
        # Fix round 1, ruling 6: the inflections read too.
        self.assertEqual(facts.expected_gait("pacing camel"), "pace")
        self.assertEqual(facts.expected_gait("a wolf loping to the right"), "canter")
        self.assertEqual(facts.expected_gait("cantering"), "canter")
        self.assertEqual(facts.expected_gait("trotting steadily"), "trot")
        self.assertEqual(facts.expected_gait("walking slowly"), "walk")
        self.assertEqual(facts.expected_gait("Galloping hard"), "gallop")
        self.assertEqual(facts.expected_gait("a rotary galloping cheetah"), "rotary gallop")
        self.assertIsNone(facts.expected_gait("a walkway pacer"), "whole words only")
        self.assertEqual(facts.expected_gait("transverse gallop, gathered suspension"), "transverse gallop")
        self.assertEqual(facts.expected_gait("rotary gallop"), "rotary gallop")
        self.assertEqual(facts.expected_gait("galloping hard"), "gallop")
        self.assertIsNone(facts.expected_gait("pounce"))
        self.assertIsNone(facts.expected_gait(None))

    def test_a_lope_reads_as_a_canter(self):
        result = run(lope(), brief={**LOPE, "gait": "canid lope: a 3-beat canter with a lead, one suspension"})
        check = result["gaitCheck"]
        self.assertEqual(check["expected"], "canter")
        self.assertIn("canter", check["measured"]["reads"])
        self.assertIs(check["matches"], True)
        self.assertIsNone(check["accepted"])

    def test_a_mismatch_refuses_with_the_measured_pattern(self):
        brief = {**LOPE, "gait": "a working trot"}
        with self.assertRaisesRegex(facts.FactsError, r"trot.*3 beats.*hind_L.*suspension.*canter.*--accept-gait"):
            run(lope(), brief=brief)
        result = facts.clip_facts(lope(), PROJECT, brief, REST, *toy_video(lope()),
                                  accept_gait="the owner wants this lope as the trot placeholder")
        self.assertIs(result["gaitCheck"]["matches"], False)
        self.assertEqual(result["gaitCheck"]["accepted"], "the owner wants this lope as the trot placeholder")

    def test_a_transverse_gallop(self):
        result = run(gallop(), brief={**LOPE, "gait": "gallop"})
        reads = result["gaitCheck"]["measured"]["reads"]
        self.assertIn("transverse gallop", reads)
        self.assertIn("gallop", reads)
        self.assertEqual(result["gait"]["beats"], 4)
        with self.assertRaisesRegex(facts.FactsError, "rotary gallop"):
            run(gallop(), brief={**LOPE, "gait": "rotary gallop"})
        with self.assertRaisesRegex(facts.FactsError, "canter"):
            run(gallop(), brief={**LOPE, "gait": "canter"})

    def test_no_gait_word_is_refused_unless_accepted_and_one_shots_are_skipped(self):
        """Fix round 1, ruling 6: a loop whose brief names no gait the vocabulary knows is refused, unless accepted
        with a reason; the acceptance is a warning in facts.json (which the ledger and the lab carry)."""
        for brief in (LOPE, {**LOPE, "gait": "a bounding hop"}):
            with self.assertRaisesRegex(facts.FactsError, r"names no gait.*walk, trot.*--accept-gait"):
                run(lope(), brief=brief, accept_gait=None)
        result = run(lope(), brief={**LOPE, "gait": "a bounding hop"}, accept_gait="the owner names it later")
        self.assertIs(result["gaitCheck"]["checked"], False)
        self.assertEqual(result["gaitCheck"]["accepted"], "the owner names it later")
        self.assertTrue(any("a bounding hop" in w and "the owner names it later" in w for w in result["warnings"]),
                        result["warnings"])
        poses = rested(synthetic(frames=11), {c: 0.0 for c in PROJECT["paws"]})
        result = facts.clip_facts(poses, PROJECT, {"clip": "pounce", "loop": False, "gait": "trot"}, REST)
        self.assertNotIn("gaitCheck", result)


def footfalls(beats, suspensions=(), lead=None):
    """A measured footfall pattern as facts.gait_pattern reports it, every paw planted."""
    return {"beatPaws": [list(b) for b in beats], "suspensions": [list(x) for x in suspensions],
            "lead": lead or {"front": None, "hind": None}, "beats": len(beats),
            "plantedPaws": ["hind_L", "hind_R", "front_L", "front_R"]}


class GaitReadsTests(unittest.TestCase):
    """Fix round 1, ruling 6: each gait of the vocabulary is read from its footfall pattern, accepted against the
    brief's own gait and refused against a wrong one."""
    WALK = footfalls([["hind_L"], ["front_L"], ["hind_R"], ["front_R"]])
    TROT = footfalls([["hind_L", "front_R"], ["hind_R", "front_L"]], [[0.45, 0.5], [0.95, 1.0]])
    PACE = footfalls([["hind_L", "front_L"], ["hind_R", "front_R"]])
    ROTARY = footfalls([["hind_L"], ["hind_R"], ["front_R"], ["front_L"]], [[0.8, 1.0]],
                       {"front": "front_L", "hind": "hind_R"})

    def check(self, pattern, reads, right, wrong):
        self.assertIn(reads, facts.gait_reads(pattern))
        accepted, warnings = facts.gait_check(pattern, {"gait": right})
        self.assertIs(accepted["matches"], True)
        self.assertEqual(warnings, [])
        self.assertIsNone(accepted["accepted"])
        with self.assertRaisesRegex(facts.FactsError, "--accept-gait"):
            facts.gait_check(pattern, {"gait": wrong})

    def test_a_lateral_sequence_walk(self):
        self.check(self.WALK, "walk", "4-beat lateral walking", "a working trot")
        self.assertNotIn("gallop", facts.gait_reads(self.WALK))

    def test_a_trot_is_two_diagonal_pairs(self):
        self.check(self.TROT, "trot", "trotting", "pacing")
        self.assertNotIn("pace", facts.gait_reads(self.TROT))

    def test_a_pace_is_two_lateral_pairs(self):
        self.check(self.PACE, "pace", "a pacing camel", "trot")
        self.assertNotIn("trot", facts.gait_reads(self.PACE))

    def test_a_rotary_gallop_leads_on_opposite_sides(self):
        self.check(self.ROTARY, "rotary gallop", "rotary gallop", "transverse gallop")
        self.assertIs(facts.gait_check(self.ROTARY, {"gait": "galloping"})[0]["matches"], True)
        with self.assertRaisesRegex(facts.FactsError, "walk"):
            facts.gait_check(self.ROTARY, {"gait": "walking"})


class CadenceTests(unittest.TestCase):
    """Task 19 step 3: the stride frequency against the Heglund-Taylor band for the body mass (the brief's massKg, or
    a mass given to facts); a slow video with one consistent period is retimed to the band centre, never silently.
    Changed with fix round 1: the toy lope (1.2 Hz) is a 200 kg subject, so its retime (1.62) is inside the bound."""
    BRIEF = {**LOPE, "gait": "canid lope: a 3-beat canter", "massKg": 200, "targetSpeedMps": None}

    def test_the_band_for_a_mass(self):
        self.assertEqual([round(x, 2) for x in facts.cadence_band([30, 50])], [2.41, 2.61])
        self.assertEqual(facts.mass_range(40), [30.0, 50.0])
        self.assertEqual(facts.mass_range([20, 25]), [20.0, 25.0])

    def test_a_slow_video_is_retimed_to_the_band_centre(self):
        result = run(lope(), brief=self.BRIEF)
        cadence = result["cadence"]
        self.assertAlmostEqual(cadence["videoHz"], 24.0 / result["period"], places=4)
        self.assertEqual(cadence["massKg"], [150.0, 250.0])
        self.assertEqual(cadence["massSource"], "brief massKg")
        centre = sum(cadence["bandHz"]) / 2
        self.assertAlmostEqual(cadence["retime"], centre / cadence["videoHz"], places=3)
        self.assertTrue(cadence["consistent"])
        self.assertAlmostEqual(cadence["hz"], centre, places=3)
        entry = result["entry"]
        self.assertAlmostEqual(entry["duration_s"], result["period"] / 24.0 / cadence["retime"], places=6)
        self.assertAlmostEqual(entry["authored_speed_mps"], 3.0 * cadence["retime"], delta=0.05 * 3.0 * cadence["retime"])
        self.assertAlmostEqual(entry["stride_m"], 3.0 * result["period"] / 24.0, delta=0.05)
        self.assertAlmostEqual(result["cycle"]["fps"], entry["frames"] / entry["duration_s"], places=6)
        self.assertTrue(any("retimed" in w for w in result["warnings"]))

    def test_a_mass_given_to_facts_wins_and_without_one_nothing_is_checked(self):
        result = run(lope(), brief=self.BRIEF, mass={"kg": 100, "source": "facts --mass-kg"})
        self.assertEqual(result["cadence"]["massSource"], "facts --mass-kg")
        self.assertEqual(result["cadence"]["massKg"], [75.0, 125.0])
        result = run(lope(), brief={**self.BRIEF, "massKg": None})
        self.assertIs(result["cadence"]["checked"], False)
        self.assertEqual(result["cadence"]["retime"], 1.0)
        self.assertTrue(any("mass" in w for w in result["warnings"]))

    def test_a_fast_video_warns_and_is_not_retimed(self):
        result = run(lope(period=8, frames=96), brief=self.BRIEF)
        self.assertEqual(result["cadence"]["retime"], 1.0)
        self.assertTrue(any("above" in w for w in result["warnings"]))

    def test_inconsistent_cycles_are_not_retimed(self):
        from unittest import mock
        with mock.patch.object(facts.cycle, "cycle_periods", return_value=[20.0, 20.1, 24.5, 19.8]):
            result = run(lope(), brief=self.BRIEF)
        self.assertIs(result["cadence"]["consistent"], False)
        self.assertEqual(result["cadence"]["retime"], 1.0)
        self.assertTrue(any("not retimed" in w for w in result["warnings"]))

    def test_a_retime_outside_the_bound_is_refused_unless_accepted(self):
        """Fix round 1, ruling 2: a retime factor outside 0.5-2.0 is refused unless `facts --accept-retime <reason>`
        records why (the toy lope at 40 kg would play 2.09 times faster)."""
        brief = {**self.BRIEF, "massKg": 40}
        with self.assertRaisesRegex(facts.FactsError, r"retime.*2\.09.*0\.5-2\.0.*--accept-retime"):
            run(lope(), brief=brief)
        result = run(lope(), brief=brief, accept_retime="the owner wants a 40 kg wolf from this video")
        cadence = result["cadence"]
        self.assertAlmostEqual(cadence["retime"], 2.09, delta=0.01)
        self.assertEqual(cadence["retimeAccepted"], "the owner wants a 40 kg wolf from this video")
        self.assertTrue(any("accepted" in w and "2.09" in w for w in result["warnings"]), result["warnings"])

    def test_an_open_rig_mesh_mass_is_checked_but_never_retimes(self):
        """Fix round 1, ruling 2: the rig's mesh volume is a body mass only for a closed mesh. With open edges the
        cadence is checked and warned about, the warning naming the source and the open-edge count, and nothing is
        retimed; a closed mesh's volume retimes as a brief mass would."""
        source = "rig mesh volume x 1000 kg/m3 (rough)"
        result = run(lope(), brief={**self.BRIEF, "massKg": None},
                     mass={"kg": 200, "source": source, "openEdges": 1590})
        cadence = result["cadence"]
        self.assertIs(cadence["checked"], True)
        self.assertEqual(cadence["retime"], 1.0)
        self.assertIn("1590", cadence["retimeBlocked"])
        self.assertTrue(any(source in w and "1590 open edges" in w and "not retimed" in w
                            for w in result["warnings"]), result["warnings"])
        closed = run(lope(), brief={**self.BRIEF, "massKg": None}, mass={"kg": 200, "source": source, "openEdges": 0})
        self.assertGreater(closed["cadence"]["retime"], 1.5)

    def test_trots_and_walks_have_no_band(self):
        result = run(lope(), brief={**self.BRIEF, "gait": "trot"}, accept_gait="test")
        self.assertIsNone(result["cadence"]["bandHz"])
        self.assertEqual(result["cadence"]["retime"], 1.0)


class RigFactsTests(unittest.TestCase):
    """Task 19 steps 2 and 4 with the contract rig: contacts from the forward kinematics of the smooth cycle, the
    60 Hz keys with every planted paw pinned, and the re-detection from those keys gating the declared contacts."""

    def facts(self, poses=None, brief=None, **kw):
        from tests import toy_rig
        poses = poses or toy_rig.toy_gait()
        masks, render = toy_video(poses, gain=1.0)
        brief = brief or {**LOPE, "targetSpeedMps": 1.0}
        if not brief.get("gait"):
            kw.setdefault("accept_gait", TOY_GAIT)
        return facts.clip_facts(poses, toy_rig.PROJECT, brief, toy_rig.REST,
                                masks=masks, render=render, rig=toy_rig.toy_rig(), **kw)

    def test_contacts_come_from_the_motion_and_the_keys_pin_them(self):
        from tests import toy_rig
        result = self.facts()
        entry, cycle = result["entry"], result["cycle"]
        n = entry["frames"]
        self.assertEqual(n, round(60 * result["period"] / 24.0))
        self.assertAlmostEqual(cycle["fps"], n / entry["duration_s"], places=6)
        self.assertEqual(len(cycle["frames"]), n + 1)
        self.assertEqual(cycle["frames"][-1], cycle["frames"][0])
        self.assertEqual(cycle["drift"], {"dx": 0.0, "dz": 0.0})
        # The footfall phases are the phase origin's (its footfall at 0, half a key before its first pinned key);
        # the contacts are the clip's, which starts startPhase later (round 4).
        self.assertAlmostEqual(entry["footfall_phase"]["paw_hl"], 1 - 0.5 / n, places=4)
        shift = cycle["startPhase"]
        for chain, (on, off) in toy_rig.LOPE_WINDOWS.items():
            bone = toy_rig.PROJECT["paws"][chain]
            self.assertAlmostEqual(entry["footfall_phase"][bone], round((entry["contacts"][bone][0] + shift) % 1.0, 4),
                                   places=4)
            window = [round((x + shift) % 1.0, 6) for x in entry["contacts"][bone]]
            window = [x - 1.0 if x > 0.9 and on < 0.1 else x for x in window]
            # The toy's paw paths turn sharply at touchdown and liftoff; the smooth cycle rounds those corners.
            self.assertAlmostEqual(window[0], on, delta=2.5 / n, msg=chain)
            self.assertAlmostEqual(window[1], off, delta=2.5 / n, msg=chain)
            for edge, side in ((window[0] * n + 0.5, "before"), (window[1] * n - 0.5, "after")):
                self.assertLess(abs(edge - round(edge)), 1e-3, f"{chain}: half a key {side} a pinned key")
        check = result["contactCheck"]
        self.assertGreaterEqual(min(check["agreement"].values()), facts.CONTACT_AGREEMENT)
        self.assertEqual(check["gate"], 0.95)
        self.assertGreaterEqual(min(check["plantedStillShare"].values()), 0.9)
        self.assertAlmostEqual(result["speedMps"], 1.0, delta=0.03)
        paws = np.array(cycle["paws"]["paw_fl"])
        self.assertEqual(paws.shape, (n + 1, 2))
        self.assertEqual(result["gait"]["beats"], 3)
        self.assertEqual(result["smooth"]["orders"]["upper_hl"], motion.max_order(20), "legs keep every harmonic")

    def test_a_stance_the_fit_holds_off_the_ground_is_pinned_down(self):
        from tests import toy_rig
        result = self.facts(toy_rig.toy_gait(float_paw=("front_L", 0.04)))
        n = result["entry"]["frames"]
        on, off = result["entry"]["contacts"]["paw_fl"]
        shift = result["cycle"]["startPhase"]
        on, off = (on + shift) % 1.0, (off + shift) % 1.0
        self.assertAlmostEqual(on, 0.3, delta=2.5 / n)
        self.assertAlmostEqual(off, 0.55, delta=2.5 / n)
        paws = np.array(result["cycle"]["paws"]["paw_fl"])
        # Changed with fix round 1: the pin holds the keys a key inside the declared contact's ends.
        inside = [(k - result["cycle"]["startKey"]) % n for k in range(n + 1)
                  if on + 1.0 / n - 1e-9 <= k / n <= off - 1.0 / n + 1e-9]
        np.testing.assert_allclose(paws[inside, 1], 0.0, atol=1e-6)

    def test_trunk_columns_are_smooth_and_keep_the_range(self):
        result = self.facts()
        kept = result["smooth"]["keptRange"]
        for name in ("pelvis", "tail"):
            self.assertGreaterEqual(kept[name], 0.95, name)
            self.assertLessEqual(result["smooth"]["orders"][name], 2, name)

    def test_keys_that_disagree_with_the_declared_contacts_are_refused(self):
        from unittest import mock
        real = facts.motion.plan_keys

        def lifted(rig, smooth, names, project, *args, **kw):
            keys, _, report = real(rig, smooth, names, project, *args, **kw)
            keys[:, names.index("dz")] += 0.1
            return keys, rig.paws(keys, names, project), report
        with mock.patch.object(facts.motion, "plan_keys", side_effect=lifted):
            with self.assertRaisesRegex(facts.FactsError, r"declared contacts.*baked keys.*agree.*0\.95"):
                self.facts()


class FixRoundOneFactsTests(unittest.TestCase):
    """Task 19 fix round 1: the edge-velocity gate and record, a retime on the rig path, the world pitch range of
    the trunk and tail, and a one-shot's predicted paws."""

    def facts(self, poses=None, brief=None, **kw):
        from tests import toy_rig
        poses = poses or toy_rig.toy_gait()
        masks, render = toy_video(poses, gain=1.0)
        brief = brief or {**LOPE, "targetSpeedMps": 1.0, "gait": "3-beat canter"}
        # The toy's smooth cycle reads three short suspensions, not a canter: the mismatch is accepted.
        kw.setdefault("accept_gait", TOY_GAIT)
        return facts.clip_facts(poses, toy_rig.PROJECT, brief, toy_rig.REST, masks=masks, render=render,
                                rig=toy_rig.toy_rig(), **kw)

    def test_the_default_pin_release_is_recorded_and_its_contacts_agree(self):
        """Task 19 round 3: facts records the pin release it keyed with, and the default's keys agree with the
        declared contacts at facts' own gate."""
        check = self.facts()["contactCheck"]
        n = check["keys"]
        self.assertEqual(check["pinRelease"], {"span_keys": 3.0, "touchdown_orders": [2, 1],
                                               "liftoff_orders": [2, 1], "inset_keys": 0.5,
                                               "min_span": motion.BLEND,
                                               "spanKeys": round(motion.PIN_RELEASE.span(n) * n, 3)})
        self.assertGreaterEqual(min(check["agreement"].values()), facts.CONTACT_AGREEMENT)
        self.assertEqual(facts.CONTACT_AGREEMENT, 0.95)

    def test_the_edge_velocity_jumps_are_recorded_and_within_the_swing(self):
        jumps = self.facts()["contactCheck"]["edgeJumps"]
        self.assertEqual(sorted(jumps), ["paw_fl", "paw_fr", "paw_hl", "paw_hr"])
        for bone, jump in jumps.items():
            # the enforced (round-3) reading at the first and last pinned key; the span-wide one is warned
            self.assertLessEqual(max(jump["touchdownPinnedMps"], jump["liftoffPinnedMps"]), jump["swingMedianMps"],
                                 bone)

    def test_the_edge_gate_across_the_span_is_measured_warned_and_enforced_when_switched_on(self):
        """Round 4 ruling: the edge gate read across each pin's approach and release span. The toy's free paw paths
        turn sharply at touchdown and liftoff, so it is over there: recorded and warned, and with EDGE_GATE_SPAN on,
        refused."""
        from unittest import mock
        result = self.facts()
        check = result["contactCheck"]
        self.assertFalse(check["edgeSpanGate"]["pass"])
        self.assertFalse(check["edgeSpanGate"]["enforced"])
        self.assertTrue(check["edgeSpanGate"]["over"])
        self.assertTrue(any("approach and release spans" in w for w in result["warnings"]), result["warnings"])
        for bone, j in check["edgeJumps"].items():
            self.assertGreaterEqual(j["touchdownMps"], j["touchdownPinnedMps"], bone)
            self.assertGreaterEqual(j["liftoffMps"], j["liftoffPinnedMps"], bone)
        self.assertFalse(facts.EDGE_GATE_SPAN)
        with mock.patch.object(facts, "EDGE_GATE_SPAN", True):
            with self.assertRaisesRegex(facts.FactsError, r"approach or release span.*swing.*median"):
                self.facts()

    def test_a_velocity_jump_at_a_contact_edge_is_refused(self):
        from unittest import mock
        real = facts.motion.plan_keys

        def jumpy(rig, smooth, names, project, windows, *args, **kw):
            keys, paws, report = real(rig, smooth, names, project, windows, *args, **kw)
            n = len(keys) - 1
            on, off = motion.release_windows(windows, n)["front_L"]
            k = (int(math.floor(off * n + 1e-9)) + 1) % n          # the key after the last pinned one
            paws["front_L"] = paws["front_L"].copy()
            paws["front_L"][k] += [0.05, 0.0]
            return keys, paws, report
        with mock.patch.object(facts.motion, "plan_keys", side_effect=jumpy):
            with self.assertRaisesRegex(facts.FactsError, r"paw_fl.*liftoff.*m/s.*swing.*median"):
                self.facts()

    def test_the_loop_starts_outside_every_contact_ease(self):
        """Round 4 ruling: the toy's phase-origin touchdown key lies inside hind_L's approach; facts starts the
        loop at a key outside every paw's approach and release span instead. The loop is cyclic, so the keys are the
        same motion, rolled: the seam's second difference (the bar's loop-seam row: the bone with the largest second
        difference at key 0 against its median elsewhere) is <= 3, the start frame moves by the same share of the
        period, and the footfall phases stay the phase origin's."""
        result = self.facts()
        cycle, entry, check = result["cycle"], result["entry"], result["contactCheck"]
        n = entry["frames"]
        start = check["cycleStart"]
        self.assertLess(start["originClearanceKeys"], 0.0, "the old start was inside an ease")
        self.assertTrue(start["outsideEases"])
        self.assertGreater(start["clearanceKeys"], 0.0)
        self.assertEqual(cycle["startKey"], start["key"])
        self.assertAlmostEqual(cycle["startPhase"], start["key"] / n)
        self.assertNotEqual(start["key"], 0)
        self.assertAlmostEqual(cycle["start"], start["originStart"] + cycle["startPhase"] * result["period"], places=6)
        self.assertEqual(result["start"], cycle["start"])
        self.assertAlmostEqual(entry["footfall_phase"]["paw_hl"], 1 - 0.5 / n, places=4)
        ratios = {}
        for bone, track in cycle["paws"].items():
            p = np.asarray(track, float)[:n]
            sd = np.linalg.norm(np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0), axis=1)
            ratios[bone] = (sd[0], sd[0] / max(1e-12, float(np.median(sd[1:]))))
        worst = max(ratios, key=lambda b: ratios[b][0])
        self.assertLessEqual(ratios[worst][1], 3.0, ratios)
        # The same motion as a start at the phase origin, rolled.
        with unittest.mock.patch.object(facts.motion, "start_key", return_value=(0, {
                "key": 0, "outsideEases": False, "clearanceKeys": start["originClearanceKeys"],
                "originClearanceKeys": start["originClearanceKeys"]})):
            origin = self.facts()
        rolled = np.roll(np.asarray(origin["cycle"]["frames"], float)[:n], -start["key"], axis=0)
        np.testing.assert_allclose(np.asarray(cycle["frames"], float)[:n], rolled, atol=2e-3)
        for bone, (on, off) in origin["entry"]["contacts"].items():
            self.assertAlmostEqual((on - cycle["startPhase"]) % 1.0, entry["contacts"][bone][0] % 1.0, places=3)
        ratios0 = {}
        for bone, track in origin["cycle"]["paws"].items():
            p = np.asarray(track, float)[:n]
            sd = np.linalg.norm(np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0), axis=1)
            ratios0[bone] = (sd[0], sd[0] / max(1e-12, float(np.median(sd[1:]))))
        worst0 = max(ratios0, key=lambda b: ratios0[b][0])
        self.assertGreater(ratios0[worst0][1], 3.0, f"the toy's origin start is not a hard case: {ratios0}")

    def test_a_retime_on_the_rig_path(self):
        """A slow toy video at 200 kg is retimed by 1.62: 60 Hz keys over the shorter cycle, pins and checks on them."""
        result = self.facts(brief={**LOPE, "targetSpeedMps": None, "gait": "3-beat canter", "massKg": 200})
        cadence, entry, cycle = result["cadence"], result["entry"], result["cycle"]
        self.assertAlmostEqual(cadence["retime"], 1.62, delta=0.02)
        self.assertEqual(cycle["retime"], cadence["retime"])
        self.assertAlmostEqual(entry["duration_s"], result["period"] / 24.0 / cadence["retime"], places=6)
        self.assertEqual(entry["frames"], motion.key_count(entry["duration_s"]))
        self.assertEqual(len(cycle["paws"]["paw_hl"]), entry["frames"] + 1)
        self.assertAlmostEqual(entry["authored_speed_mps"], 1.0 * cadence["retime"], delta=0.05)
        self.assertGreaterEqual(min(result["contactCheck"]["agreement"].values()), facts.CONTACT_AGREEMENT)

    def test_the_trunk_and_tail_world_pitch_range_is_reported_and_warned(self):
        """Ruling 7: the world pitch range of each trunk and tail bone is reported against the video's; under
        RANGE_KEEP it is a warning (in facts.json, the ledger and the lab), not a refusal."""
        from unittest import mock
        result = self.facts()
        kept = result["smooth"]["worldPitchKept"]
        self.assertEqual(sorted(kept), ["chest", "pelvis", "tail"])
        self.assertTrue(all(v >= motion.RANGE_KEEP for v in kept.values()), kept)
        self.assertFalse(any("world pitch" in w for w in result["warnings"]))
        with mock.patch.object(facts.motion, "world_pitch_kept", return_value={"pelvis": 1.0, "tail": 0.69}):
            result = self.facts()
        self.assertTrue(any("world pitch" in w and "tail 0.69" in w for w in result["warnings"]), result["warnings"])

    def test_world_pitch_kept_compares_the_smooth_cycle_with_the_samples(self):
        from tests import toy_rig
        poses = toy_rig.toy_gait(frames=20, trunk=0.05)
        values = np.array([f["values"] for f in poses["frames"]])
        rig = motion.PlanarRig(toy_rig.toy_rig())
        smooth = motion.SmoothCycle(values, poses["names"], toy_rig.PROJECT)
        kept = motion.world_pitch_kept(rig, values, smooth, poses["names"], ["pelvis", "tail"])
        self.assertGreaterEqual(kept["tail"], 0.95)
        smooth.gain["tail"] = 0.5
        self.assertLess(motion.world_pitch_kept(rig, values, smooth, poses["names"], ["tail"])["tail"], 0.9)

    def test_a_one_shot_predicts_its_paws_for_the_bake(self):
        """Ruling 3: a one-shot's keys are its fitted frames; with the rig, facts predicts each paw at every key as
        the bake plays it (dx less its mean over keys 0..n-1), so `bake` can check a one-shot too."""
        from tests import toy_rig
        poses = toy_rig.toy_gait(frames=13)
        result = facts.clip_facts(poses, toy_rig.PROJECT, {"clip": "pounce", "loop": False, "targetSpeedMps": None},
                                  toy_rig.REST, rig=toy_rig.toy_rig())
        values = np.array([f["values"] for f in poses["frames"]])
        values[:, 0] -= values[:-1, 0].mean()
        rig = motion.PlanarRig(toy_rig.toy_rig())
        want = rig.paws(values, poses["names"], toy_rig.PROJECT)
        for chain, bone in toy_rig.PROJECT["paws"].items():
            np.testing.assert_allclose(result["cycle"]["paws"][bone], want[chain], atol=1e-6)
        self.assertEqual(len(result["cycle"]["frames"]), 13)
        self.assertNotIn("paws", facts.clip_facts(poses, toy_rig.PROJECT, {"clip": "pounce", "loop": False,
                                                                            "targetSpeedMps": None},
                                                  toy_rig.REST)["cycle"])
