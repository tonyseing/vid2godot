"""Ground truth: render the fixture's known walk as the 'video', fit it, and recover the joint angles.

The reference and the 24 walk frames are rendered once; each scenario encodes its own video from them, ingests it,
fits it and prints one `ROUNDTRIP <name>: {json}` metrics line."""
import json
import math
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from anim.blender_run import run_blender
from anim.camera import SideCamera
from anim.contract import load_project_contract
from anim.ingest import run_ingest
from anim.raster import score_fit
from tests.blender_case import BLENDER, FIXTURE_CONTRACT, fixture_glb, requires_blender

LEGS = ["upper_fl", "lower_fl", "upper_fr", "lower_fr", "upper_hl", "lower_hl", "upper_hr", "lower_hr"]
PAWS = ["paw_fl", "paw_fr", "paw_hl", "paw_hr"]
PAIRS = (("fl", "fr"), ("hl", "hr"))


def leg_identity(poses, truth, shift):
    """Leg error with each leg pair's identity chosen once for the whole clip.

    A side silhouette cannot tell left from right, so a pair may come out exchanged; that is only acceptable for the
    whole clip at once. For each pair the summed error over every walk frame is taken as labelled ("same") and with
    the pair exchanged ("cross"); the smaller is the pair's clip-wide error. Also returned, per pair, which walk frames
    are closer to the truth exchanged than as labelled (a consistent fit has none, or all of them)."""
    names = poses["names"]
    walk = [frame for frame in poses["frames"] if frame["index"] >= shift]
    totals, swapped = {}, {}
    for a, b in PAIRS:
        same_total = cross_total = 0.0
        flags = []
        for frame in walk:
            want = truth[str(frame["index"] - shift)]

            def value(bone, frame=frame):
                return frame["values"][names.index(bone)]

            same = sum(abs(value(f"{k}_{a}") - want[f"{k}_{a}"]) + abs(value(f"{k}_{b}") - want[f"{k}_{b}"])
                       for k in ("upper", "lower"))
            cross = sum(abs(value(f"{k}_{a}") - want[f"{k}_{b}"]) + abs(value(f"{k}_{b}") - want[f"{k}_{a}"])
                        for k in ("upper", "lower"))
            same_total, cross_total = same_total + same, cross_total + cross
            flags.append(bool(cross < same))
        totals[f"{a}/{b}"] = min(same_total, cross_total)
        swapped[f"{a}/{b}"] = flags
    error = sum(totals.values()) / (4 * len(PAIRS) * max(len(walk), 1))
    return {"legErrClipIdentityDeg": round(math.degrees(error), 3),
            "swappedFrames": {pair: int(sum(flags)) for pair, flags in swapped.items()},
            "walkFrames": len(walk)}, swapped


def assert_consistent_identity(case, metrics):
    for pair, count in metrics["swappedFrames"].items():
        case.assertIn(count, (0, metrics["walkFrames"]),
                      f"{pair} is exchanged on {count} of {metrics['walkFrames']} walk frames: identity flips "
                      f"within the clip")


class LegIdentityMetricTests(unittest.TestCase):
    """The clip-wide identity metric, on synthetic poses (no Blender): a whole-clip exchange passes, per-frame flips
    do not."""

    def setUp(self):
        self.truth = {str(f): {f"{k}_{s}": (0.4 if s.endswith("l") else -0.4) * math.sin(f / 3 + (k == "lower"))
                               for k in ("upper", "lower") for s in ("fl", "fr", "hl", "hr")} for f in range(12)}
        self.names = ["dx", "dz"] + LEGS

    def poses(self, exchange):
        frames = []
        for f in range(12):
            t = dict(self.truth[str(f)])
            if exchange(f):
                for k in ("upper", "lower"):
                    t[f"{k}_fl"], t[f"{k}_fr"] = t[f"{k}_fr"], t[f"{k}_fl"]
            frames.append({"index": f, "values": [0.0, 0.0] + [t[b] for b in LEGS]})
        return {"names": self.names, "frames": frames}

    def test_labelled_fit_has_no_error_and_no_swaps(self):
        metrics, _ = leg_identity(self.poses(lambda f: False), self.truth, 0)
        self.assertEqual(metrics["legErrClipIdentityDeg"], 0.0)
        self.assertEqual(metrics["swappedFrames"], {"fl/fr": 0, "hl/hr": 0})

    def test_whole_clip_exchange_is_consistent(self):
        metrics, _ = leg_identity(self.poses(lambda f: True), self.truth, 0)
        self.assertEqual(metrics["legErrClipIdentityDeg"], 0.0)
        self.assertEqual(metrics["swappedFrames"]["fl/fr"], 12)
        assert_consistent_identity(self, metrics)

    def test_per_frame_flips_fail(self):
        metrics, _ = leg_identity(self.poses(lambda f: f % 2 == 1), self.truth, 0)
        self.assertGreater(metrics["legErrClipIdentityDeg"], 6.0)
        with self.assertRaises(AssertionError):
            assert_consistent_identity(self, metrics)


@requires_blender
class RoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.glb, truth_path = fixture_glb()
        cls.truth = json.loads(truth_path.read_text())
        cls.project = load_project_contract(FIXTURE_CONTRACT)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.ref = run_blender("render_reference", {"model": str(cls.glb), "outDir": str(cls.tmp / "ref"),
                                                   "backgrounds": ["#c8c8c8"]}, cls.tmp / "ref", BLENDER)
        cls.walk = run_blender("render_frames", {"model": str(cls.glb), "clip": "walk", "camera": cls.ref["camera"],
                                                 "background": "#c8c8c8", "frames": list(range(24)),
                                                 "outDir": str(cls.tmp / "walk"), "bones": PAWS},
                               cls.tmp / "walk", BLENDER)
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def ingest(self, name, rest_first):
        """Encode the walk (after one rest-pose frame when rest_first) and ingest it."""
        work = self.tmp / name
        (work / "video").mkdir(parents=True)
        frames = ([self.ref["renders"]["#c8c8c8"]] if rest_first else []) + list(self.walk["frames"])
        for i, frame in enumerate(frames):
            shutil.copyfile(frame, work / "video" / f"{i:05d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-framerate", "24", "-i", str(work / "video" / "%05d.png"),
                        "-c:v", "mpeg4", "-q:v", "2", str(work / "walk.mp4")], check=True)
        meta = run_ingest(work / "walk.mp4", self.ref["mask"], SideCamera.from_json(self.ref["camera"]),
                          work / "ingest")
        self.assertEqual(meta["frames"], 24 + (1 if rest_first else 0))
        return work, meta

    def marks_for(self, meta, rest_first, mark_frames):
        """Marks are read off the registered frames: video px -> reference px."""
        shift, reg = (1 if rest_first else 0), meta["registration"]

        def registered(point):
            return [(point[i] - reg["src"][i]) * reg["scale"] + reg["dst"][i] for i in (0, 1)]

        return {str(f): {b: registered(self.walk["points"][str(f - shift)][b]) for b in PAWS} for f in mark_frames}

    def run_fit(self, name, work, meta, marks, out_dir, shift, extra=None):
        brief = {"fitChains": list(self.project["chains"]), "limitsDeg": {}, "iouThreshold": 0.9}
        start = time.time()
        run_blender("fit_clip", {"model": str(self.glb), "project": self.project, "brief": brief,
                                 "camera": meta["camera"], "masks": str(work / "ingest" / "masks.npz"),
                                 "marks": marks, "workScale": 0.5, "fps": meta["fps"],
                                 "out": str(out_dir / "poses.json"), "projection": str(out_dir / "projection.npz"),
                                 **(extra or {})},
                    out_dir / "fit", BLENDER)
        seconds = time.time() - start
        poses = json.loads((out_dir / "poses.json").read_text())
        report = score_fit(poses, out_dir / "projection.npz", work / "ingest" / "masks.npz", 0.9, {})
        names = poses["names"]
        ious = {item["index"]: item["iou"] for item in report["frames"]}
        walk = [frame for frame in poses["frames"] if frame["index"] >= shift]
        errors = [abs(frame["values"][names.index(b)] - self.truth[str(frame["index"] - shift)][b])
                  for frame in walk for b in LEGS]
        identity, swapped = leg_identity(poses, self.truth, shift)
        metrics = {"meanIoU": round(float(np.mean([ious[frame["index"]] for frame in walk])), 4),
                   "meanIoUAll": report["meanIoU"],
                   "legErrDeg": round(math.degrees(float(np.mean(errors))), 3), **identity,
                   "cameraScale": poses.get("cameraScale"), "deformModel": poses.get("deformModel"),
                   "secPerFrame": round(seconds / len(poses["frames"]), 2)}
        print(f"ROUNDTRIP {name}: {json.dumps(metrics)}", flush=True)
        return {"metrics": metrics, "report": report, "poses": poses, "swapped": swapped, "ious": ious}

    def fit(self, name, rest_first, mark_frames):
        """Ingest, mark the given video frames, fit and score (once per scenario; later tests reuse the run)."""
        if name not in self.runs:
            work, meta = self.ingest(name, rest_first)
            marks = self.marks_for(meta, rest_first, mark_frames)
            run = self.run_fit(name, work, meta, marks, work, 1 if rest_first else 0)
            self.runs[name] = {**run, "work": work, "meta": meta, "marks": marks}
        return self.runs[name]

    def test_fit_recovers_known_walk(self):
        run = self.fit("s1", False, (0, 6, 12, 18))
        self.assertGreaterEqual(run["report"]["meanIoU"], 0.92)
        self.assertLessEqual(run["metrics"]["legErrDeg"], 6.0)

    def test_fit_recovers_walk_after_rest_first_frame(self):
        """Seedance's first_frame reference makes the video's frame 0 the rest pose; the 24 walk frames follow it."""
        metrics = self.fit("s2", True, (1, 7, 13, 19))["metrics"]
        self.assertGreaterEqual(metrics["meanIoU"], 0.92)
        self.assertLessEqual(metrics["legErrDeg"], 6.0)

    def test_fit_without_marks_keeps_each_pair_consistent_for_the_whole_clip(self):
        """No marks: a side silhouette cannot tell a left leg from a right one, so a pair may come out exchanged, but
        only for the whole clip. Identity is chosen once per pair over the whole clip; per-frame flips fail."""
        run = self.fit("s3", False, ())
        self.assertGreaterEqual(run["report"]["meanIoU"], 0.92)
        self.assertLessEqual(run["metrics"]["legErrClipIdentityDeg"], 6.0)
        assert_consistent_identity(self, run["metrics"])

    def test_span_refit_without_marks_keeps_the_neighbours_leg_identity(self):
        """Refit spans of the full S1 fit with no marks inside them: the fixed frames on either side tie the span's
        leg identity to theirs, and every frame outside the span is left exactly as it was. 0:8 has fixed frames
        after it only, and its first frame is solved from scratch (which can come out exchanged); 8:16 has both."""
        full = self.fit("s1", False, (0, 6, 12, 18))
        for span in ((0, 8), (8, 16)):
            with self.subTest(span=span):
                refit = self.tmp / f"s1_span_{span[0]}_{span[1]}"
                refit.mkdir()
                shutil.copyfile(full["work"] / "poses.json", refit / "poses.json")
                shutil.copyfile(full["work"] / "projection.npz", refit / "projection.npz")
                marks = {f: m for f, m in full["marks"].items() if not span[0] <= int(f) < span[1]}
                run = self.run_fit(f"s1_span_{span[0]}_{span[1]}", full["work"], full["meta"], marks, refit, 0,
                                   {"span": list(span), "init": str(refit / "poses.json")})
                before, after = full["poses"]["frames"], run["poses"]["frames"]
                for f in range(len(before)):
                    if not span[0] <= f < span[1]:
                        self.assertEqual(after[f]["values"], before[f]["values"], f"frame {f} is outside the span")
                for pair in full["swapped"]:
                    self.assertEqual(run["swapped"][pair][span[0]:span[1]], full["swapped"][pair][span[0]:span[1]],
                                     f"{pair} changed identity inside the span")
                self.assertGreaterEqual(min(run["ious"][f] for f in range(*span)), 0.9)
