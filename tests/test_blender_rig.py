import json
import tempfile
import unittest
from pathlib import Path

from anim.blender_run import run_blender
from anim.contract import check_rig, load_project_contract
from tests.blender_case import BLENDER, FIXTURE_CONTRACT, fixture_glb, requires_blender


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
