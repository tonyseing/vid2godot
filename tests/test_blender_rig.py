import json
import tempfile
import unittest
from pathlib import Path

from anim.blender_run import run_blender
from anim.contract import check_rig, load_project_contract
from tests.blender_case import BLENDER, FIXTURE_CONTRACT, bad_fixture_glb, fixture_glb, requires_blender


@requires_blender
class RigInspectionTests(unittest.TestCase):
    def test_fixture_rig_meets_its_contract(self):
        glb, _ = fixture_glb()
        project = load_project_contract(FIXTURE_CONTRACT)
        with tempfile.TemporaryDirectory() as tmp:
            rig = run_blender("inspect_rig", {"model": str(glb), "chains": project["chains"], "headBone": "head",
                                              "pelvisBone": "pelvis"}, Path(tmp), BLENDER)
        self.assertEqual(len(rig["bones"]), 16)
        self.assertEqual(rig["forwardSign"], 1)
        self.assertTrue(all(item["edges"] == 0 for item in rig["collapse"]))
        self.assertEqual(check_rig(rig, project), ([], []))

    def test_joint_collapse_is_detected(self):
        glb = bad_fixture_glb()
        project = load_project_contract(FIXTURE_CONTRACT)
        with tempfile.TemporaryDirectory() as tmp:
            rig = run_blender("inspect_rig", {"model": str(glb), "chains": project["chains"], "headBone": "head",
                                              "pelvisBone": "pelvis"}, Path(tmp), BLENDER)
        collapse = {item["bone"]: item["edges"] for item in rig["collapse"]}
        self.assertGreater(collapse["lower_fl"], 0)
        problems, _ = check_rig(rig, project)
        self.assertTrue(any("Joint collapse at lower_fl" in problem for problem in problems))


@requires_blender
class PoseSilhouetteTests(unittest.TestCase):
    """facts' renderer for the averaged cycle (anim/blender/pose_silhouettes.py through
    pipeline.silhouette_renderer): given value vectors, the rig's silhouettes as the fit scores them."""

    def test_rest_pose_matches_the_reference_mask_and_a_bent_leg_differs(self):
        import numpy as np
        from anim.pipeline import silhouette_renderer
        from anim.raster import iou
        from anim.reference import model_mask
        glb, _ = fixture_glb()
        project = load_project_contract(FIXTURE_CONTRACT)
        bones = [b for chain in project["chains"].values() for b in chain]
        names = ["dx", "dz"] + bones
        with tempfile.TemporaryDirectory() as tmp:
            ref = run_blender("render_reference", {"model": str(glb), "outDir": str(Path(tmp) / "ref"),
                                                   "backgrounds": ["#c8c8c8"]}, Path(tmp) / "ref", BLENDER)
            mask = model_mask(ref["mask"])
            poses = {"bones": bones, "names": names, "camera": ref["camera"]}
            render = silhouette_renderer(glb, poses, mask.shape, Path(tmp) / "render", BLENDER)
            bent = [0.0] * len(names)
            bent[names.index("upper_fl")] = 0.6
            rest, moved = render([[0.0] * len(names), bent])
        self.assertEqual(rest.shape, mask.shape)
        self.assertGreaterEqual(iou(rest, mask), 0.97)
        self.assertLess(iou(moved, rest), 0.99)
        self.assertGreater(iou(moved, rest), 0.8)
