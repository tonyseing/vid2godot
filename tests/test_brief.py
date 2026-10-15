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
                       {"limitsDeg": {"upper_fl": [10, -10]}}, {"backgrounds": ["grey"]}, {"loop": "yes"}):
            with self.subTest(change=change), self.assertRaises(BriefError):
                self.load(**change)

    def test_missing_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "brief.json"
            path.write_text(json.dumps({k: v for k, v in GOOD.items() if k != "gait"}))
            with self.assertRaises(BriefError):
                load_brief(path, PROJECT, "walk")
