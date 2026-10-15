"""Ground truth: render the fixture's known walk as the 'video', fit it, and recover the joint angles."""
import json
import math
import subprocess
import tempfile
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


@requires_blender
class RoundTripTests(unittest.TestCase):
    def test_fit_recovers_known_walk(self):
        glb, truth_path = fixture_glb()
        truth = json.loads(truth_path.read_text())
        project = load_project_contract(FIXTURE_CONTRACT)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ref = run_blender("render_reference", {"model": str(glb), "outDir": str(tmp / "ref"),
                                                   "backgrounds": ["#c8c8c8"]}, tmp / "ref", BLENDER)
            camera = SideCamera.from_json(ref["camera"])
            video = run_blender("render_frames", {"model": str(glb), "clip": "walk", "camera": ref["camera"],
                                                  "background": "#c8c8c8", "frames": list(range(24)),
                                                  "outDir": str(tmp / "video"), "bones": PAWS}, tmp / "video", BLENDER)
            subprocess.run(["ffmpeg", "-v", "error", "-framerate", "24", "-i", str(tmp / "video" / "%05d.png"),
                            "-c:v", "mpeg4", "-q:v", "2", str(tmp / "walk.mp4")], check=True)
            meta = run_ingest(tmp / "walk.mp4", ref["mask"], camera, tmp / "ingest")
            self.assertEqual(meta["frames"], 24)
            reg = meta["registration"]   # marks are read off the registered frames: video px -> reference px

            def registered(point):
                return [(point[i] - reg["src"][i]) * reg["scale"] + reg["dst"][i] for i in (0, 1)]

            marks = {str(f): {b: registered(video["points"][str(f)][b]) for b in PAWS} for f in (0, 6, 12, 18)}
            brief = {"fitChains": list(project["chains"]), "limitsDeg": {}, "iouThreshold": 0.9}
            run_blender("fit_clip", {"model": str(glb), "project": project, "brief": brief,
                                     "camera": meta["camera"], "masks": str(tmp / "ingest" / "masks.npz"),
                                     "marks": marks, "workScale": 0.5, "fps": meta["fps"],
                                     "out": str(tmp / "poses.json"), "projection": str(tmp / "projection.npz")},
                        tmp / "fit", BLENDER)
            poses = json.loads((tmp / "poses.json").read_text())
            report = score_fit(poses, tmp / "projection.npz", tmp / "ingest" / "masks.npz", 0.9, {})
            self.assertGreaterEqual(report["meanIoU"], 0.92)
            names = poses["names"]
            errors = [abs(frame["values"][names.index(bone)] - truth[str(frame["index"])][bone])
                      for frame in poses["frames"] for bone in LEGS]
            self.assertLessEqual(math.degrees(float(np.mean(errors))), 6.0)
