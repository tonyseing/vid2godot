"""Recover a rendered known fold with the actual fitter, checked by Blender bone heads.

The opposing front folds exchange without changing their combined silhouette or paw
positions. Only source-visible elbow evidence can distinguish the labelled solution.
All constructed poses live in temporary test fixtures, never in shipped animation.
"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from anim.blender_run import run_blender
from anim.camera import SideCamera
from anim.contract import load_project_contract
from anim.ingest import background_colour, boundary_points, silhouette
from tests.blender_case import BLENDER, FIXTURE_CONTRACT, FIXTURES, fixture_glb, requires_blender


@requires_blender
class FoldedRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls._tmp.name)
        cls.model, _ = fixture_glb()
        cls.project = load_project_contract(FIXTURE_CONTRACT)
        cls.camera = SideCamera(384, 288, 320.0, 0.08, 0.35)
        cls.work_scale = 0.5
        cls.oracle = run_blender(
            FIXTURES / "articulation_fixture.py",
            {"model": str(cls.model), "camera": cls.camera.to_json(), "outDir": str(cls.work / "source")},
            cls.work / "source", BLENDER)
        image = np.array(Image.open(cls.oracle["sourceImage"]).convert("RGB"))
        full_mask = silhouette(image, background_colour(image))
        mask = full_mask[::2, ::2]
        boundary = boundary_points(mask)
        cls.masks = cls.work / "masks.npz"
        np.savez_compressed(cls.masks, masks=mask[None],
                            outside=ndimage.distance_transform_edt(~mask)[None].astype(np.float32),
                            boundary=boundary, offsets=np.array([0, len(boundary)]))
        cls.brief = {"fitChains": list(cls.project["chains"]),
                     "limitsDeg": {bone: [-100, 100] for chain in cls.project["chains"].values() for bone in chain},
                     "iouThreshold": 0.9}
        points = cls.oracle["source"]["landmarks"]
        cls.document = {"schemaVersion": 1,
                        "source": {"videoSha256": hashlib.sha256(Path(cls.oracle["sourceImage"]).read_bytes()).hexdigest(),
                                   "coordinateSpace": "registered_source_px"},
                        "frames": {"0": {"phase": "front recovery fold", "observations": [
                            {"id": "visible-elbow", "bone": "lower_fr", "end": "head",
                             "point": points["lower_fr"]["head"], "uncertaintyPx": 3.0,
                             "visibility": "visible", "correspondence": "confirmed"},
                            {"id": "visible-paw", "bone": "paw_fr", "end": "tail",
                             "point": points["paw_fr"]["tail"], "uncertaintyPx": 3.0,
                             "visibility": "visible", "correspondence": "confirmed"}]}}}
        cls.fitted = None

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def observations(self, document=None):
        from anim.articulation import validate
        return validate(document or self.document, self.project, self.brief, 1)

    def fit(self):
        if self.fitted is None:
            out = self.work / "fit"
            out.mkdir()
            run_blender("fit_clip", {
                "model": str(self.model), "project": self.project, "brief": self.brief,
                "camera": self.camera.scaled(self.work_scale).to_json(), "masks": str(self.masks),
                "marks": {}, "sourceObservations": self.observations(), "frameIndices": [0],
                "workScale": self.work_scale, "fps": 24, "periodic": False,
                "out": str(out / "poses.json"), "projection": str(out / "projection.npz")}, out, BLENDER)
            poses = json.loads((out / "poses.json").read_text())
            oracle = run_blender(FIXTURES / "articulation_fixture.py", {
                "model": str(self.model), "camera": poses["camera"], "poses": str(out / "poses.json"),
                "outDir": str(self.work / "independent")}, self.work / "independent", BLENDER)
            type(self).fitted = poses, oracle
        return self.fitted

    def test_production_fit_recovers_visible_elbow_not_merely_paw_or_overlap(self):
        from anim.articulation import assess
        poses, oracle = self.fit()
        self.assertEqual([frame["index"] for frame in poses["frames"]], [0])
        independent = oracle["frames"]["0"]["landmarks"]
        for observation in self.document["frames"]["0"]["observations"]:
            bone, end = observation["bone"], observation["end"]
            recovered = np.array(independent[bone][end]) / self.work_scale
            residual = float(np.linalg.norm(recovered - observation["point"]))
            self.assertLessEqual(residual, observation["uncertaintyPx"],
                                 f"{bone}.{end}: independent Blender residual {residual:.3f} source px")
            np.testing.assert_allclose(poses["frames"][0]["landmarks"][bone][end],
                                       independent[bone][end], atol=1e-3)
        self.assertEqual(assess(self.observations(), poses, work_scale=self.work_scale)["status"], "pass")

    def test_wrong_fold_has_matching_paw_and_high_overlap_but_fails_elbow(self):
        from anim.articulation import assess
        source, wrong = self.oracle["source"], self.oracle["wrong"]
        source_mask = silhouette(np.array(Image.open(source["image"]).convert("RGB")), np.array([200] * 3))
        wrong_mask = silhouette(np.array(Image.open(wrong["image"]).convert("RGB")), np.array([200] * 3))
        iou = float((source_mask & wrong_mask).sum() / (source_mask | wrong_mask).sum())
        self.assertGreaterEqual(iou, 0.97, "adversarial pose must actually retain high rendered overlap")
        self.assertLessEqual(float(np.linalg.norm(np.array(source["landmarks"]["paw_fr"]["tail"])
                                                 - wrong["landmarks"]["paw_fr"]["tail"])), 0.1)
        self.assertGreater(float(np.linalg.norm(np.array(source["landmarks"]["lower_fr"]["head"])
                                               - wrong["landmarks"]["lower_fr"]["head"])), 20.0)
        report = assess(self.observations(), {"frames": [{"index": 0, "landmarks": wrong["landmarks"]}]})
        self.assertEqual(report["status"], "fail")
        by_id = {item["id"]: item for item in report["frames"][0]["observations"]}
        self.assertEqual(by_id["visible-paw"]["status"], "pass")
        self.assertEqual(by_id["visible-elbow"]["status"], "fail")

    def test_hidden_or_ambiguous_elbow_does_not_acquire_fixture_ground_truth(self):
        from anim.articulation import assess
        for visibility, correspondence in (("occluded", "confirmed"), ("visible", "ambiguous"),
                                            ("visible", "tentative")):
            with self.subTest(visibility=visibility, correspondence=correspondence):
                document = copy.deepcopy(self.document)
                elbow = document["frames"]["0"]["observations"][0]
                elbow.update(visibility=visibility, correspondence=correspondence)
                if visibility == "occluded":
                    del elbow["point"]
                for candidate in ("source", "wrong"):
                    poses = {"frames": [{"index": 0, "landmarks": self.oracle[candidate]["landmarks"]}]}
                    report = assess(self.observations(document), poses)
                    self.assertEqual(report["status"], "unresolved")
                    assessed = report["frames"][0]["observations"][0]
                    self.assertEqual(assessed["status"], "unresolved")
                    if visibility == "occluded":
                        self.assertIsNone(assessed.get("residualPx"))
