"""Manual landmarks constrain articulated bone heads, without changing paw-tail marks."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from anim import marks as marks_mod
from anim.blender_run import run_blender
from anim.camera import SideCamera
from anim.contract import load_project_contract
from anim.ledger import Ledger
from tests.blender_case import BLENDER, FIXTURE_CONTRACT, FIXTURES, fixture_glb, requires_blender
from tests.test_pipeline import run


class MarkValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        model = self.tmp / "model.glb"
        model.write_bytes(b"not needed: invalid marks must fail before Blender")
        self.work = self.tmp / "work"
        ledger = Ledger.init(self.work, model, "walk")
        contract = self.work / "contract"
        contract.mkdir()
        shutil.copyfile(FIXTURE_CONTRACT, contract / "project-contract.json")
        stub = contract / "rig.json"
        stub.write_text("{}")
        ledger.record("contract", {}, {"rig": stub})
        brief = self.work / "brief.json"
        brief.write_text(json.dumps({"fitChains": ["front_L", "front_R"], "limitsDeg": {}, "iouThreshold": 0.9}))
        ledger.record("brief", {}, {"brief": brief})
        for stage in ("reference", "generate", "accept"):
            ledger.record(stage, {}, {"stub": stub})
        meta = self.work / "ingest.json"
        meta.write_text(json.dumps({"frames": 3, "camera": SideCamera(128, 128, 100, 0, 0.3).to_json(),
                                    "workScale": 0.5, "fps": 24, "flags": {}}))
        masks = self.work / "masks.npz"
        np.savez(masks, masks=np.ones((3, 128, 128), bool), outside=np.zeros((3, 128, 128)))
        ledger.record("ingest", {}, {"meta": meta, "masks": masks})

    def rejected(self, document, diagnostic):
        marks = self.tmp / "marks.json"
        marks.write_text(json.dumps(document))
        code, _, err = run("fit", "--work", str(self.work), "--marks", str(marks),
                           "--blender", str(self.tmp / "never-launch-Blender"))
        self.assertEqual(code, 1)
        self.assertRegex(err, diagnostic)
        self.assertNotIn("Traceback", err)

    def test_joint_frame_indices_must_identify_ingested_frames(self):
        for frame in ("-1", "3", "1.5", "nan"):
            with self.subTest(frame=frame):
                self.rejected({"frames": {}, "jointFrames": {frame: {"lower_fl": [10, 20]}}},
                              "jointFrames.*frame")

    def test_frame_aliases_cannot_silently_replace_landmarks(self):
        for field, bone in (("frames", "paw_fl"), ("jointFrames", "lower_fl")):
            with self.subTest(field=field):
                self.rejected({"frames": {}, field: {"1": {bone: [10, 20]}, "00001": {bone: [30, 40]}}},
                              field + ".*duplicate.*frame")

    def test_padded_frame_numbers_target_the_same_ingested_frame_without_rewriting_evidence(self):
        document = {"frames": {"00001": {"paw_fl": [10, 20]}},
                    "jointFrames": {"00002": {"lower_fl": [30, 40]}}}
        project = load_project_contract(FIXTURE_CONTRACT)
        brief = {"fitChains": ["front_L", "front_R"]}
        paw_frames, joint_frames = marks_mod.validate(document, project, brief, 3)
        lab_marks = marks_mod.normalize(document)
        self.assertEqual(paw_frames[str(1)]["paw_fl"], [10, 20])
        self.assertEqual(joint_frames[str(2)]["lower_fl"], [30, 40])
        self.assertEqual(lab_marks["frames"][str(1)]["paw_fl"], [10, 20])
        self.assertEqual(lab_marks["jointFrames"][str(2)]["lower_fl"], [30, 40])
        self.assertIn("00001", document["frames"])
        self.assertIn("00002", document["jointFrames"])

    def test_joint_bones_must_be_fitted_nonroot_chain_members(self):
        for bone in ("missing", "root", "lower_hl"):
            with self.subTest(bone=bone):
                self.rejected({"frames": {}, "jointFrames": {"0": {bone: [10, 20]}}},
                              "jointFrames.*bone")

    def test_joint_coordinates_must_be_finite_numeric_pairs(self):
        for point in ([float("nan"), 1], [1, float("inf")], [True, 1], ["1", 2], [1], [1, 2, 3], None):
            with self.subTest(point=point):
                self.rejected({"frames": {}, "jointFrames": {"0": {"lower_fl": point}}},
                              "jointFrames.*(finite|coordinate|pair)")

    def test_paw_validation_uses_the_same_frame_and_coordinate_rules(self):
        for frame, bone, point in (("3", "paw_fl", [1, 2]), ("0", "lower_fl", [1, 2]),
                                   ("0", "paw_fl", [1, float("nan")])):
            with self.subTest(frame=frame, bone=bone, point=point):
                self.rejected({"frames": {frame: {bone: point}}}, "frames.*(frame|bone|finite|coordinate|pair)")


@requires_blender
class JointGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        glb, _ = fixture_glb()
        cls.metrics = run_blender(FIXTURES / "probe_marks.py", {"model": str(glb),
                                  "project": load_project_contract(FIXTURE_CONTRACT), "outDir": cls.tmp.name},
                                  Path(cls.tmp.name) / "probe", BLENDER)

    def test_elbow_and_wrist_landmarks_use_heads_not_toes(self):
        self.assertGreater(self.metrics["headToToePx"], 10)
        self.assertAlmostEqual(self.metrics["jointMatchingPenalty"], 0.0, places=5)
        self.assertAlmostEqual(self.metrics["jointSwappedPenalty"], self.metrics["expectedJointSwappedPenalty"], places=5)
        self.assertGreater(self.metrics["jointSwappedPenalty"], 1)

    def test_numpy_and_depsgraph_heads_agree_with_the_blender_landmarks(self):
        self.assertEqual(self.metrics["deformModel"], "numpy")
        self.assertLess(self.metrics["numpyHeadErrorPx"], 1e-4)
        self.assertLess(self.metrics["depsgraphHeadErrorPx"], 1e-4)

    def test_joint_only_input_disambiguates_identity_with_coincident_toes(self):
        self.assertAlmostEqual(self.metrics["toeSeparationPx"], 0.0, places=4)
        self.assertEqual(self.metrics["jointRelabelled"], 3)
        self.assertLess(self.metrics["identityError"], 1e-8)

    def test_paw_only_input_retains_existing_distance_behavior(self):
        self.assertAlmostEqual(self.metrics["pawMatchingPenalty"], 0.0, places=5)
        self.assertAlmostEqual(self.metrics["pawOffsetPenalty"], 2.5, places=5)
        self.assertEqual(self.metrics["emptyJointPawPenalty"], self.metrics["pawOffsetPenalty"])

    def test_mixed_marks_share_one_existing_manual_distance_average(self):
        self.assertAlmostEqual(self.metrics["mixedPenalty"], 2.5 / 3, places=5)
