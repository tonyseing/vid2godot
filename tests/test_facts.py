import math
import unittest

import numpy as np

from anim import facts

PROJECT = {"paws": {"hind_L": "paw_hl", "front_L": "paw_fl", "hind_R": "paw_hr", "front_R": "paw_fr"},
           "legPairs": [["front_L", "front_R"], ["hind_L", "hind_R"]], "phaseOrigin": "hind_L"}
PHASES = {"hind_L": 0.0, "front_L": 0.25, "hind_R": 0.5, "front_R": 0.75}


def synthetic(frames=60, period=12, fps=24.0, speed=3.0):
    entries = []
    for f in range(frames):
        t = f / period
        values = [0.0, 0.0] + [math.sin(2 * math.pi * (t - p)) for p in PHASES.values()]
        paws = {chain: [-speed * f / fps, max(0.0, math.sin(2 * math.pi * (t - p)))]
                for chain, p in PHASES.items()}
        entries.append({"index": f, "values": values, "paws": paws, "flags": []})
    return {"fps": fps, "names": ["dx", "dz", "a", "b", "c", "d"], "frames": entries}


class FactsTests(unittest.TestCase):
    def test_period(self):
        values = np.array([f["values"][2:] for f in synthetic()["frames"]])
        self.assertEqual(facts.period_frames(values, 24.0), 12)

    def test_start_skips_the_lead_in(self):
        values = np.array([f["values"][2:] for f in synthetic()["frames"]])
        self.assertGreaterEqual(facts.best_start(values, 12), 6)

    def test_contact_window_wraps(self):
        mask = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], bool)
        window, duty = facts.contact_window(mask)
        self.assertEqual(window, [0.75, 0.25])
        self.assertAlmostEqual(duty, 0.5)

    def test_two_touchdowns_refused(self):
        with self.assertRaises(facts.FactsError):
            facts.contact_window(np.array([1, 0, 1, 0, 0, 0], bool))

    def test_clip_facts(self):
        brief = {"clip": "lope", "loop": True, "targetSpeedMps": 3.0}
        result = facts.clip_facts(synthetic(), PROJECT, brief)
        entry = result["entry"]
        self.assertEqual(result["period"], 12)
        self.assertAlmostEqual(entry["authored_speed_mps"], 3.0, places=2)
        self.assertAlmostEqual(entry["stride_m"], 1.5, places=2)
        self.assertAlmostEqual(entry["duration_s"], 0.5)
        self.assertEqual(entry["footfall_phase"]["paw_hl"], 0.0)
        self.assertAlmostEqual(entry["footfall_phase"]["paw_fl"], 0.25, delta=1 / 12)
        self.assertAlmostEqual(entry["duty_factor"]["paw_hr"], 0.5, delta=1 / 12)
        self.assertEqual(list(entry)[:5], ["duration_s", "loop", "authored_speed_mps", "stride_m", "contacts"])
        self.assertEqual(result["warnings"], [])

    def test_speed_mismatch_warns(self):
        result = facts.clip_facts(synthetic(speed=5.0), PROJECT, {"clip": "lope", "loop": True, "targetSpeedMps": 3.0})
        self.assertTrue(result["warnings"])

    def test_too_short(self):
        poses = synthetic(frames=10)
        with self.assertRaises(facts.FactsError):
            facts.clip_facts(poses, PROJECT, {"clip": "lope", "loop": True, "targetSpeedMps": None})
