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

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def fit(self, name, rest_first, mark_frames):
        """Encode the walk (after one rest-pose frame when rest_first), ingest it, mark the given video frames, fit
        and score. Returns the metrics over the walk frames and score_fit's report over every frame."""
        shift = 1 if rest_first else 0
        work = self.tmp / name
        (work / "video").mkdir(parents=True)
        frames = ([self.ref["renders"]["#c8c8c8"]] if rest_first else []) + list(self.walk["frames"])
        for i, frame in enumerate(frames):
            shutil.copyfile(frame, work / "video" / f"{i:05d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-framerate", "24", "-i", str(work / "video" / "%05d.png"),
                        "-c:v", "mpeg4", "-q:v", "2", str(work / "walk.mp4")], check=True)
        meta = run_ingest(work / "walk.mp4", self.ref["mask"], SideCamera.from_json(self.ref["camera"]),
                          work / "ingest")
        self.assertEqual(meta["frames"], 24 + shift)
        reg = meta["registration"]   # marks are read off the registered frames: video px -> reference px

        def registered(point):
            return [(point[i] - reg["src"][i]) * reg["scale"] + reg["dst"][i] for i in (0, 1)]

        marks = {str(f): {b: registered(self.walk["points"][str(f - shift)][b]) for b in PAWS} for f in mark_frames}
        brief = {"fitChains": list(self.project["chains"]), "limitsDeg": {}, "iouThreshold": 0.9}
        start = time.time()
        run_blender("fit_clip", {"model": str(self.glb), "project": self.project, "brief": brief,
                                 "camera": meta["camera"], "masks": str(work / "ingest" / "masks.npz"),
                                 "marks": marks, "workScale": 0.5, "fps": meta["fps"],
                                 "out": str(work / "poses.json"), "projection": str(work / "projection.npz")},
                    work / "fit", BLENDER)
        seconds = time.time() - start
        poses = json.loads((work / "poses.json").read_text())
        report = score_fit(poses, work / "projection.npz", work / "ingest" / "masks.npz", 0.9, {})
        names = poses["names"]
        ious = {item["index"]: item["iou"] for item in report["frames"]}
        walk = [frame for frame in poses["frames"] if frame["index"] >= shift]
        errors, free_errors, swaps = [], [], 0
        for frame in walk:
            truth = self.truth[str(frame["index"] - shift)]

            def value(bone, frame=frame):
                return frame["values"][names.index(bone)]

            errors += [abs(value(b) - truth[b]) for b in LEGS]
            for a, b in PAIRS:   # identity-free: the better of the pair as labelled and the pair exchanged
                same = sum(abs(value(f"{k}_{a}") - truth[f"{k}_{a}"]) + abs(value(f"{k}_{b}") - truth[f"{k}_{b}"])
                           for k in ("upper", "lower"))
                cross = sum(abs(value(f"{k}_{a}") - truth[f"{k}_{b}"]) + abs(value(f"{k}_{b}") - truth[f"{k}_{a}"])
                            for k in ("upper", "lower"))
                swaps += cross < same
                free_errors.append(min(same, cross) / 4)
        metrics = {"meanIoU": round(float(np.mean([ious[frame["index"]] for frame in walk])), 4),
                   "meanIoUAll": report["meanIoU"],
                   "legErrDeg": round(math.degrees(float(np.mean(errors))), 3),
                   "legErrIdentityFreeDeg": round(math.degrees(float(np.mean(free_errors))), 3),
                   "swappedPairFrames": int(swaps), "cameraScale": poses.get("cameraScale"),
                   "deformModel": poses.get("deformModel"),
                   "secPerFrame": round(seconds / len(poses["frames"]), 2)}
        print(f"ROUNDTRIP {name}: {json.dumps(metrics)}", flush=True)
        return metrics, report

    def test_fit_recovers_known_walk(self):
        metrics, report = self.fit("s1", False, (0, 6, 12, 18))
        self.assertGreaterEqual(report["meanIoU"], 0.92)
        self.assertLessEqual(metrics["legErrDeg"], 6.0)

    def test_fit_recovers_walk_after_rest_first_frame(self):
        """Seedance's first_frame reference makes the video's frame 0 the rest pose; the 24 walk frames follow it."""
        metrics, _ = self.fit("s2", True, (1, 7, 13, 19))
        self.assertGreaterEqual(metrics["meanIoU"], 0.92)
        self.assertLessEqual(metrics["legErrDeg"], 6.0)

    def test_fit_without_marks_keeps_each_leg_its_own(self):
        """No marks: a side silhouette cannot tell a left leg from a right one, so a pair may come out exchanged for
        the whole clip. The gate is the fit and the identity-free leg error; the swap count is reported."""
        metrics, report = self.fit("s3", False, ())
        self.assertGreaterEqual(report["meanIoU"], 0.92)
        self.assertLessEqual(metrics["legErrIdentityFreeDeg"], 6.0)
