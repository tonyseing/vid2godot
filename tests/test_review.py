import json
import tempfile
import unittest
from pathlib import Path

from anim import review

REPORT = {"frames": [{"index": 0, "iou": 0.95, "flags": []}, {"index": 1, "iou": 0.81, "flags": []},
                     {"index": 2, "iou": 0.93, "flags": ["leg_crossing:front_L/front_R"]}],
          "meanIoU": 0.9, "lowFrames": [1, 2]}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bound = {}
        for name in ("poses", "facts", "glb", "sheet", "closeup", "lab"):
            self.bound[name] = self.tmp / f"{name}.bin"
            self.bound[name].write_text(name)
        self.template = review.make_template("lope", self.bound, REPORT, 1)

    def passing(self):
        filled = json.loads(json.dumps(self.template))
        filled.update({"reviewer": "agent", "verdict": "pass", "inspectedOverlays": True, "inspectedSheet": True,
                       "inspectedReference": True, "observations": "lope reads at game distance"})
        for item in filled["lowFrames"]:
            item.update({"action": "explained", "note": "tail tip outside mask; within tolerance"})
        return filled

    def test_template_lists_low_frames(self):
        self.assertEqual([f["frame"] for f in self.template["lowFrames"]], [1, 2])
        self.assertEqual(self.template["artisticRound"], 1)

    def test_complete_review_passes(self):
        self.assertEqual(review.validate(self.passing(), self.template), [])

    def test_unexplained_frame_fails(self):
        filled = self.passing()
        filled["lowFrames"][0]["note"] = ""
        self.assertTrue(review.validate(filled, self.template))

    def test_dropped_frame_fails(self):
        filled = self.passing()
        filled["lowFrames"] = filled["lowFrames"][1:]
        self.assertTrue(review.validate(filled, self.template))

    def test_uninspected_sheet_fails(self):
        filled = self.passing()
        filled["inspectedSheet"] = False
        self.assertTrue(review.validate(filled, self.template))

    def test_changed_file_makes_review_stale(self):
        filled = self.passing()
        self.bound["glb"].write_text("rebaked")
        self.assertTrue(any("changed" in p for p in review.validate(filled, self.template)))

    def test_lab_page(self):
        page = review.lab_page("lope", REPORT, 0.9, {"paw_hl": [True, False, True]}, {"glb": "ab" * 32})
        self.assertIn('"threshold": 0.9', page)
        self.assertIn("Copy notes", page)
        self.assertNotIn("__DATA__", page)
