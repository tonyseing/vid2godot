import math
import unittest

import numpy as np

from anim import facts

PROJECT = {"paws": {"hind_L": "paw_hl", "front_L": "paw_fl", "hind_R": "paw_hr", "front_R": "paw_fr"},
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


class FactsTests(unittest.TestCase):
    def test_period(self):
        # Changed with fix round 1: period_frames now returns the sub-frame period (a float), not an int.
        self.assertAlmostEqual(facts.period_frames(joint_values(synthetic()), 24.0), 12, places=6)

    def test_start_skips_the_lead_in(self):
        self.assertGreaterEqual(facts.best_start(joint_values(synthetic()), 12), 6)

    def test_contact_window_wraps(self):
        mask = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], bool)
        window, duty = facts.contact_window(mask)
        self.assertEqual(window, [0.75, 0.25])
        self.assertAlmostEqual(duty, 0.5)

    def test_two_touchdowns_refused(self):
        with self.assertRaises(facts.FactsError):
            facts.contact_window(np.array([1, 0, 1, 0, 0, 0], bool))

    def test_clip_facts(self):
        result = facts.clip_facts(synthetic(), PROJECT, LOPE)
        entry = result["entry"]
        # Changed with fix round 1: the period is the sub-frame float, so equality became a 6-place comparison.
        self.assertAlmostEqual(result["period"], 12, places=6)
        self.assertAlmostEqual(entry["authored_speed_mps"], 3.0, places=2)
        self.assertAlmostEqual(entry["stride_m"], 1.5, places=2)
        self.assertAlmostEqual(entry["duration_s"], 0.5)
        self.assertEqual(entry["footfall_phase"]["paw_hl"], 0.0)
        self.assertAlmostEqual(entry["footfall_phase"]["paw_fl"], 0.25, delta=1 / 12)
        self.assertAlmostEqual(entry["duty_factor"]["paw_hr"], 0.5, delta=1 / 12)
        self.assertEqual(list(entry)[:5], ["duration_s", "loop", "authored_speed_mps", "stride_m", "contacts"])
        self.assertEqual(result["warnings"], [])
        cycle = result["cycle"]
        self.assertEqual(len(cycle["frames"]), 13)
        self.assertEqual(cycle["frames"][-1], cycle["frames"][0])
        self.assertEqual(len(cycle["frames"][0]), 6)
        self.assertAlmostEqual(cycle["fps"], 24.0, places=6)
        self.assertEqual(cycle["start"], result["start"])
        self.assertEqual(cycle["periodFrames"], result["period"])

    def test_fractional_period(self):
        for truth, n in ((9.6, 10), (12.4, 12)):
            with self.subTest(period=truth):
                result = facts.clip_facts(synthetic(frames=120, period=truth), PROJECT, LOPE)
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

    def test_no_full_cycle_in_range(self):
        values = joint_values(synthetic(frames=50, period=30))
        with self.assertRaisesRegex(facts.FactsError, "no full cycle found between 0.25 s and 2.5 s"):
            facts.period_frames(values, 24.0)
        with self.assertRaises(facts.FactsError):
            facts.clip_facts(synthetic(frames=50, period=30), PROJECT, LOPE)

    def test_phase_origin_shift(self):
        project = {**PROJECT, "phaseOrigin": "front_L"}
        poses = synthetic()
        result = facts.clip_facts(poses, project, LOPE)
        entry, period = result["entry"], result["period"]
        unshifted = facts.best_start(joint_values(poses), period)
        self.assertNotAlmostEqual(result["start"], unshifted)
        self.assertEqual(entry["footfall_phase"]["paw_fl"], 0.0)
        self.assertAlmostEqual(entry["footfall_phase"]["paw_hl"], 0.75, delta=1 / 12)
        self.assertGreaterEqual(result["start"], 6)
        self.assertLessEqual(result["start"] + period, 59)

    def test_shift_near_the_end_moves_back_a_cycle(self):
        project = {**PROJECT, "phaseOrigin": "front_L"}
        poses = synthetic(frames=40, envelope=0.3)
        values = joint_values(poses)
        period = facts.period_frames(values, 24.0)
        unshifted = facts.best_start(values, period)
        self.assertEqual(unshifted, math.floor(39 - period))  # the cleanest seam is the last one
        result = facts.clip_facts(poses, project, LOPE)
        self.assertLess(result["start"], unshifted)
        self.assertGreaterEqual(result["start"], 4)
        self.assertLessEqual(result["start"] + result["period"], 39)
        self.assertEqual(result["entry"]["footfall_phase"]["paw_fl"], 0.0)

    def test_shift_never_enters_the_lead_in(self):
        phases = {chain: (p + 0.625) % 1.0 for chain, p in PHASES.items()}
        poses = synthetic(frames=26, phases=phases, envelope=0.3)
        values = joint_values(poses)
        period = facts.period_frames(values, 24.0)
        self.assertEqual(facts.best_start(values, period), math.floor(25 - period))
        with self.assertRaisesRegex(facts.FactsError, "touchdown fits after the lead-in"):
            facts.clip_facts(poses, PROJECT, LOPE)

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

    def test_windows(self):
        self.assertEqual(facts.windows(np.array([], bool)), [])
        self.assertEqual(facts.windows(np.zeros(5, bool)), [])
        self.assertEqual(facts.windows(np.array([0, 0, 0, 1, 1], bool)), [[0.75, 1.0]])
        self.assertEqual(facts.windows(np.array([1, 1, 0, 0, 1, 1, 0, 0, 0], bool)), [[0.0, 0.25], [0.5, 0.75]])

    def test_speed_mismatch_warns(self):
        result = facts.clip_facts(synthetic(speed=5.0), PROJECT, LOPE)
        self.assertTrue(result["warnings"])

    def test_too_short(self):
        poses = synthetic(frames=10)
        with self.assertRaises(facts.FactsError):
            facts.clip_facts(poses, PROJECT, {"clip": "lope", "loop": True, "targetSpeedMps": None})
