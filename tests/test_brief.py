import json
import tempfile
import unittest
from pathlib import Path

from anim.brief import TWO_VIEW, BriefError, load_brief

PROJECT = json.loads((Path(__file__).parent / "fixtures" / "test_contract.json").read_text())
GOOD = {"clip": "walk", "subject": "box quadruped", "motion": "walking", "gait": "4-beat lateral walk",
        "loop": True, "planar": True, "videoSeconds": 5}


class BriefTests(unittest.TestCase):
    def load(self, **changes):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "brief.json"
            path.write_text(json.dumps({**GOOD, **changes}))
            return load_brief(path, PROJECT, "walk")

    def test_defaults_filled(self):
        brief = self.load()
        self.assertEqual(brief["iouThreshold"], 0.90)
        self.assertEqual(brief["fitChains"], list(PROJECT["chains"]))
        self.assertEqual(brief["backgrounds"][0], "#c8c8c8")

    def test_out_of_plane_motion_refused(self):
        with self.assertRaises(BriefError) as caught:
            self.load(planar=False)
        self.assertEqual(str(caught.exception), TWO_VIEW)

    def test_wrong_clip(self):
        with self.assertRaises(BriefError):
            self.load(clip="trot")

    def test_bad_values(self):
        for change in ({"videoSeconds": 2}, {"videoSeconds": 31}, {"iouThreshold": 0}, {"fitChains": ["wings"]},
                       {"limitsDeg": {"upper_fl": [10, -10]}}, {"backgrounds": ["grey"]}, {"loop": "yes"},
                       {"resolution": "hd"}, {"resolution": 720}, {"aspectRatio": "wide"}, {"aspectRatio": 1.7},
                       {"targetSpeedMps": 0}, {"targetSpeedMps": -1}, {"targetSpeedMps": True},
                       {"targetSpeedMps": "fast"}, {"targetSpeedMps": float("nan")},
                       {"targetSpeedMps": float("inf")},
                       {"iouThreshold": "0.9"}, {"iouThreshold": True}, {"iouThreshold": None},
                       {"limitsDeg": [1]}, {"limitsDeg": {"a": [1]}}, {"limitsDeg": {"a": ["x", "y"]}},
                       {"limitsDeg": {"a": [True, 5]}}, {"limitsDeg": {"a": 5}},
                       {"backgrounds": [1]}, {"backgrounds": "#c8c8c8"},
                       {"fitChains": []}, {"fitChains": "spine"}, {"fitChains": [1]}):
            with self.subTest(change=change), self.assertRaises(BriefError):
                self.load(**change)

    def test_valid_optional_values(self):
        brief = self.load(resolution="1080p", aspectRatio="9:16", targetSpeedMps=1.5, iouThreshold=1,
                          limitsDeg={"upper_fl": [-30, 45.5]}, fitChains=["spine"])
        self.assertEqual(brief["fitChains"], ["spine"])
        self.assertEqual(self.load(fitChains=None)["fitChains"], list(PROJECT["chains"]))

    def test_top_level_must_be_an_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "brief.json"
            path.write_text(json.dumps([GOOD]))
            with self.assertRaises(BriefError):
                load_brief(path, PROJECT, "walk")

    def test_missing_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "brief.json"
            path.write_text(json.dumps({k: v for k, v in GOOD.items() if k != "gait"}))
            with self.assertRaises(BriefError):
                load_brief(path, PROJECT, "walk")

    def test_slow_motion(self):
        """Task 20: an optional slowMotion factor (the video asked for that many times slower than real time, for a
        small fast animal whose steps 24 fps would undersample) defaults to none."""
        self.assertIsNone(self.load()["slowMotion"])
        self.assertEqual(self.load(slowMotion=3)["slowMotion"], 3)
        self.assertEqual(self.load(slowMotion=2.5)["slowMotion"], 2.5)
        for bad in (1, 0.5, 0, -2, 9, True, "3", float("nan"), float("inf"), [3]):
            with self.subTest(slowMotion=bad), self.assertRaises(BriefError):
                self.load(slowMotion=bad)

    def test_body_mass(self):
        """Task 19 step 3: an optional massKg (a number, or [lightest, heaviest]) sets the cadence band."""
        self.assertIsNone(self.load()["massKg"])
        self.assertEqual(self.load(massKg=40)["massKg"], 40)
        self.assertEqual(self.load(massKg=[30, 50])["massKg"], [30, 50])
        for bad in (0, -3, True, "40", [50, 30], [30], [30, 50, 70], [0, 5], float("nan")):
            with self.subTest(massKg=bad), self.assertRaises(BriefError):
                self.load(massKg=bad)
