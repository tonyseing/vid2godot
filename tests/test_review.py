import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image

from anim import review
from anim.ledger import Ledger, StageError
from anim.pipeline import main
from tests.blender_case import FIXTURE_CONTRACT

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

    def test_lab_page_shows_the_motion_facts(self):
        """Task 19: the lab states the cadence retime factor, the gait check and the contact check, never silently."""
        motion = review.motion_summary({"cadence": {"videoHz": 1.589, "hz": 2.51, "retime": 1.5797,
                                                    "bandHz": [2.41, 2.61], "massKg": [30, 50],
                                                    "massSource": "facts --mass-kg", "consistent": True},
                                        "gaitCheck": {"expected": "canter", "matches": False,
                                                      "accepted": "wolves gallop transversely",
                                                      "measured": {"text": "4 beats"}},
                                        "contactCheck": {"agreement": {"paw_hl": 0.97}, "gate": 0.95},
                                        "entry": {"contacts": {"paw_hl": [0.0, 0.3]}}})
        self.assertEqual(motion["cadence"]["retime"], 1.5797)
        page = review.lab_page("lope", REPORT, 0.9, {"paw_hl": [True]}, {"glb": "ab" * 32}, motion)
        self.assertIn('"retime": 1.5797', page)
        self.assertIn("wolves gallop transversely", page)
        self.assertIn('id="motion"', page)
        self.assertIsNone(review.motion_summary(None))

    def test_contact_timeline_is_rest_relative(self):
        """Task 17b fix round 4: the lab's contact strip reads each paw from its own rest-pose tail height, as facts
        does (a front paw resting 2.7 cm over the hinds is still on the ground)."""
        project = {"paws": {"hind_L": "paw_hl", "front_L": "paw_fl"}}
        frames = [{"paws": {"hind_L": [0.0, -0.0207 + (0.0 if f < 3 else 0.1)],
                            "front_L": [0.0, 0.0064 + (0.0 if f >= 3 else 0.1)]}} for f in range(6)]
        rest = {"hind_L": -0.0207, "front_L": 0.0064}
        timeline = review.contact_timeline({"frames": frames}, project, rest)
        self.assertEqual(timeline, {"paw_hl": [True] * 3 + [False] * 3, "paw_fl": [False] * 3 + [True] * 3})

    def failing(self, template=None):
        filled = json.loads(json.dumps(template or self.template))
        filled.update({"reviewer": "agent", "verdict": "fail", "observations": "legs jump back at frame 2"})
        return filled

    def test_review_for_an_earlier_round_is_refused(self):
        stale = self.failing()
        current = review.make_template("lope", self.bound, REPORT, 2)
        self.assertTrue(any("round" in p for p in review.validate(stale, current)))

    def test_review_for_another_clip_is_refused(self):
        filled = self.passing()
        filled["clip"] = "trot"
        self.assertTrue(any("clip" in p for p in review.validate(filled, self.template)))

    def test_duplicate_submission_is_refused(self):
        history, failed = review.record_round([], self.template, self.failing())
        self.assertEqual((len(history), failed), (1, 1))
        with self.assertRaises(StageError):
            review.record_round(history, self.template, self.failing())

    def test_two_distinct_failed_rounds_ask_the_owner(self):
        history, failed = review.record_round([], self.template, self.failing())
        self.assertLess(failed, review.ARTISTIC_ROUNDS)
        second = review.make_template("lope", self.bound, REPORT, 2)
        history, failed = review.record_round(history, second, self.failing(second))
        self.assertEqual([item["round"] for item in history], [1, 2])
        self.assertGreaterEqual(failed, review.ARTISTIC_ROUNDS)

    def test_lab_page_keeps_markup_out_of_the_data(self):
        nasty = "</script><b>x</b>"
        report = json.loads(json.dumps(REPORT))
        report["frames"][2]["flags"] = ["<!--<script>"]
        data = {"clip": nasty, "report": report, "threshold": 0.9, "timeline": {"paw_hl": [True]},
                "hashes": {"glb": "ab" * 32}}
        page = review.lab_page(nasty, report, 0.9, data["timeline"], data["hashes"])
        self.assertNotIn("</script><b>", page)
        self.assertNotIn("<!--<script", page)
        inlined = page.split("const DATA = ", 1)[1].split(";\nconst $", 1)[0]
        self.assertNotIn("<", inlined)
        self.assertEqual(json.loads(inlined), data)

    def test_malformed_reviews_are_problems_not_crashes(self):
        for filled in ([], "pass", None, {"lowFrames": [{"action": "explained"}]}, {"lowFrames": "1, 2"},
                       {"verdict": "pass", "lowFrames": [{"action": "explained"}, 7, {"frame": [1]}]}):
            with self.subTest(filled=filled):
                self.assertTrue(review.validate(filled, self.template))

    def test_malformed_templates_are_stage_errors(self):
        broken = []
        for key in ("bound", "artisticRound", "clip", "lowFrames"):
            template = dict(self.template)
            del template[key]
            broken.append(template)
        broken += [[], "template", dict(self.template, lowFrames=[{"iou": 0.8}]),
                   dict(self.template, bound={"glb": "abc"})]
        for template in broken:
            with self.subTest(template=template), self.assertRaises(StageError):
                review.validate(self.passing(), template)


def run(*argv):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, json.loads(out.getvalue() or "{}"), err.getvalue()


class ReviewCommandTests(unittest.TestCase):
    """The review flow through the CLI, on a ledger whose stages are recorded with small stand-in files."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "w"
        work = self.work

        def put(rel, data):
            path = work / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
            return path

        model = self.tmp / "model.glb"
        model.write_bytes(b"model")
        ledger = Ledger.init(work, model, "lope")
        project = json.loads(FIXTURE_CONTRACT.read_text())
        put("contract/project-contract.json", project)
        # The paws' rest tails (Task 17b fix round 4: the lab's contact strip is rest-relative).
        rig = {"bones": [{"name": bone, "head": [0.0, 0.0, 0.05], "tail": [0.1, 0.0, 0.0]}
                         for bone in project["paws"].values()]}
        put("contract/rig.json", rig)
        put("brief/brief.json", {"iouThreshold": 0.9})
        frames, height, width = 3, 30, 40
        for f in range(frames):
            path = work / "ingest" / "registered" / f"{f:05d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80, 60), (90, 90, 90)).save(path)
        masks = np.zeros((frames, height, width), bool)
        masks[:, 10:20, 10:30] = True
        np.savez(work / "ingest" / "masks.npz", masks=masks)
        uv = np.tile(np.array([[10, 10], [30, 10], [30, 20], [10, 20]], float), (frames, 1, 1))
        (work / "fit").mkdir()
        np.savez(work / "fit" / "projection.npz", uv=uv, faces=np.array([[0, 1, 2], [0, 2, 3]]))
        chains = list(project["paws"])
        poses = [{"index": f, "flags": [],
                  "paws": {c: [0.0, 0.0 if (f + i) % 2 else 0.1] for i, c in enumerate(chains)},
                  "joints": {c: [[15, 12], [16, 16], [16, 19]] for c in chains}} for f in range(frames)]
        put("fit/poses.json", {"frames": poses})
        put("fit/report.json", REPORT)
        outputs = {"contract": {"rig": put("contract/rig.json", rig)}, "brief": {"brief": work / "brief/brief.json"},
                   "reference": {"x": put("reference/x.json", {})}, "generate": {"x": put("generate/x.json", {})},
                   "accept": {"x": put("accept/x.json", {})},
                   "ingest": {"masks": work / "ingest/masks.npz", "meta": put("ingest/ingest.json", {})},
                   "fit": {"poses": work / "fit/poses.json", "projection": work / "fit/projection.npz",
                           "report": work / "fit/report.json"},
                   "facts": {"facts": put("facts/facts.json", {"period": 3.0})},
                   "bake": {"glb": put("bake/eval.glb", b"glb"), "sheet": put("bake/sheet.png", b"sheet"),
                            "closeup": put("bake/closeup.png", b"closeup")}}
        for stage, files in outputs.items():
            ledger.record(stage, {}, files)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def submit(self, verdict):
        template = json.loads((self.work / "review" / "template.json").read_text())
        template.update({"reviewer": "agent", "verdict": verdict, "inspectedReference": True,
                         "inspectedOverlays": True, "inspectedSheet": True, "observations": "reads at game distance"})
        for item in template["lowFrames"]:
            item.update({"action": "explained", "note": "within tolerance"})
        filled = self.tmp / f"filled-{template['artisticRound']}-{verdict}.json"
        filled.write_text(json.dumps(template))
        return run("review", "--work", str(self.work), "--file", str(filled))

    def test_file_without_a_template_is_a_clean_error(self):
        filled = self.tmp / "filled.json"
        filled.write_text("{}")
        code, _, err = run("review", "--work", str(self.work), "--file", str(filled))
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith("error: "))
        self.assertIn("review --template", err)
        self.assertNotIn("Traceback", err)

    def test_two_failed_rounds_then_a_pass(self):
        code, data, _ = run("review", "--work", str(self.work), "--template")
        self.assertEqual((code, data["lowFrames"]), (0, [1, 2]))
        code, data, _ = self.submit("fail")
        self.assertEqual((code, data["ok"], data["failedRounds"]), (0, False, 1))
        self.assertNotIn("owner", data["next"])
        code, _, err = self.submit("fail")
        self.assertEqual(code, 1)
        self.assertIn("already", err)
        self.assertEqual(len(json.loads((self.work / "review" / "history.json").read_text())), 1)
        self.assertEqual(run("review", "--work", str(self.work), "--template")[0], 0)
        code, data, _ = self.submit("fail")
        self.assertEqual((code, data["failedRounds"]), (0, 2))
        self.assertIn("Ask the owner", data["next"])
        self.assertEqual(run("review", "--work", str(self.work), "--template")[0], 0)
        code, data, _ = self.submit("pass")
        self.assertEqual((code, data["ok"], data["next"]), (0, True, "verify"))
        ledger = Ledger(self.work)
        self.assertEqual(ledger.state("review"), "ok")
        stage = ledger.data["stages"]["review"]
        self.assertEqual(set(stage["inputs"]), {"poses", "facts", "glb", "sheet", "closeup", "lab", "template"})
        self.assertEqual(stage["facts"], {"round": 3})
        history = json.loads((self.work / "review" / "history.json").read_text())
        self.assertEqual([(item["round"], item["verdict"]) for item in history],
                         [(1, "fail"), (2, "fail"), (3, "pass")])
        (self.work / "bake" / "eval.glb").write_bytes(b"rebaked")
        self.assertEqual(Ledger(self.work).state("review"), "stale")
