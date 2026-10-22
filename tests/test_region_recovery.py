"""Visible contour recovery is certified by a second, independent Blender render.

A wrong three-link pose has the same visible paw target and high global overlap,
but misses the cropped fold. The fitter receives no elbow/wrist identity or angle
truth: a mask, the visible paw disk, and a source-pixel contour rectangle suffice.
"""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from anim.articulation import validate as validate_observations
from anim.blender_run import run_blender
from anim.camera import SideCamera
from anim.contract import load_project_contract
from anim.ingest import background_colour, boundary_points, silhouette
from anim.silhouette_regions import validate as validate_regions
from tests.blender_case import BLENDER, FIXTURE_CONTRACT, FIXTURES, fixture_glb, requires_blender

ORACLE = FIXTURES / "region_recovery_fixture.py"


def image_mask(path):
    image = np.array(Image.open(path).convert("RGB"))
    return silhouette(image, background_colour(image))


def overlap(a, b):
    union = np.count_nonzero(a | b)
    if not union:
        raise AssertionError("an empty source/candidate crop cannot certify visible contour recovery")
    return float(np.count_nonzero(a & b) / union)


@requires_blender
class RegionRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls._tmp.name)
        cls.model, _ = fixture_glb()
        cls.project = load_project_contract(FIXTURE_CONTRACT)
        cls.registered = SideCamera(384, 288, 320.0, 0.08, 0.35)
        cls.work_scale = 0.5
        cls.truth = run_blender(ORACLE, {"model": str(cls.model), "camera": cls.registered.to_json(),
                                        "outDir": str(cls.work / "source")}, cls.work / "source", BLENDER)
        cls.source_mask = image_mask(cls.truth["source"]["image"])
        mask = cls.source_mask[::2, ::2]
        boundary = boundary_points(mask)
        cls.masks = cls.work / "masks.npz"
        np.savez_compressed(cls.masks, masks=mask[None],
                            outside=ndimage.distance_transform_edt(~mask)[None].astype(np.float32),
                            boundary=boundary, offsets=np.array([0, len(boundary)]))
        cls.brief = {"fitChains": list(cls.project["chains"]),
                     "limitsDeg": {bone: [-100, 100] for chain in cls.project["chains"].values() for bone in chain},
                     "iouThreshold": 0.9}
        cls.source_hash = hashlib.sha256(Path(cls.truth["source"]["image"]).read_bytes()).hexdigest()
        source = {"videoSha256": cls.source_hash, "coordinateSpace": "registered_source_px"}
        cls.regions = {"schemaVersion": 1, "source": source,
                       "frames": {"0": [{"id": "visible-fold", "bounds": cls.truth["bounds"],
                                          "visibility": "visible", "minIoU": 0.85, "weight": 1.0}]}}
        # Only the visible distal endpoint is labelled. The hidden chain identities
        # and the constructive source/wrong angles are never sent to fit_clip.
        cls.observations = {"schemaVersion": 1, "source": source,
                            "frames": {"0": {"phase": "visible fold", "observations": [
                                {"id": "visible-paw", "bone": "paw_fr", "end": "tail",
                                 "point": cls.truth["source"]["landmarks"]["paw_fr"]["tail"],
                                 "uncertaintyPx": 0.8, "visibility": "visible", "correspondence": "confirmed"}]}}}
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def independent(self, poses_path, label):
        out = self.work / label
        return run_blender(ORACLE, {"model": str(self.model), "poses": str(poses_path),
                                    "workScale": self.work_scale, "outDir": str(out)}, out, BLENDER)

    def fit(self, label, regions=None):
        if label not in self.runs:
            out = self.work / label
            out.mkdir()
            args = {"model": str(self.model), "project": self.project, "brief": self.brief,
                    "camera": self.registered.scaled(self.work_scale).to_json(), "masks": str(self.masks),
                    "marks": {}, "frameIndices": [0], "workScale": self.work_scale, "fps": 24,
                    "periodic": False, "out": str(out / "poses.json"), "projection": str(out / "projection.npz"),
                    "sourceObservations": validate_observations(self.observations, self.project, self.brief, 1)}
            if regions is not None:
                args["silhouetteRegions"] = validate_regions(regions, self.source_hash, 384, 288, 1)
            run_blender("fit_clip", args, out, BLENDER)
            type(self).runs[label] = {"poses": json.loads((out / "poses.json").read_text()), "out": out}
        return self.runs[label]

    def crop(self, mask):
        left, top, right, bottom = self.truth["bounds"]
        return mask[top:bottom, left:right]

    def assert_limits(self, values):
        for bone, angle in values.items():
            if bone in self.brief["limitsDeg"]:
                low, high = self.brief["limitsDeg"][bone]
                self.assertLessEqual(low - 1e-9, math.degrees(angle), bone)
                self.assertLessEqual(math.degrees(angle), high + 1e-9, bone)

    def test_constructive_wrong_fold_matches_paw_and_global_gate_but_fails_local_crop(self):
        source, wrong = self.truth["source"], self.truth["wrong"]
        self.assert_limits(self.truth["values"])
        self.assert_limits(self.truth["wrongValues"])
        np.testing.assert_allclose(source["landmarks"]["paw_fr"]["tail"],
                                   wrong["landmarks"]["paw_fr"]["tail"], atol=1e-3)
        self.assertGreater(np.linalg.norm(np.array(source["landmarks"]["paw_fr"]["head"])
                                          - wrong["landmarks"]["paw_fr"]["head"]), 5.0)
        wrong_mask = image_mask(wrong["image"])
        global_iou = overlap(self.source_mask, wrong_mask)
        local_iou = overlap(self.crop(self.source_mask), self.crop(wrong_mask))
        print("REGION_WRONG_WITNESS " + json.dumps({"wholeBodyIoU": global_iou, "localIoU": local_iou,
                                                     "bounds": self.truth["bounds"]}), flush=True)
        self.assertGreaterEqual(global_iou, 0.90, "global body acceptance alone must miss this local failure")
        self.assertLess(local_iou, 0.85, "matching a visible paw does not determine the adjacent folded contour")
        self.assertFalse(self.source_mask[[0, -1]].any())
        self.assertFalse(self.source_mask[:, [0, -1]].any())
        witness_path = self.work / "constructive-witness.json"
        witness_path.write_text(json.dumps({"camera": self.registered.scaled(self.work_scale).to_json(),
                                           "names": list(self.truth["values"]),
                                           "frames": [{"index": 0, "values": list(self.truth["values"].values())}]}))
        witness = self.independent(witness_path, "witness-oracle")["frames"]["0"]
        witness_mask = image_mask(witness["image"])
        self.assertGreaterEqual(overlap(self.source_mask, witness_mask), 0.995)
        self.assertGreaterEqual(overlap(self.crop(self.source_mask), self.crop(witness_mask)), 0.995)

    def test_guided_production_fit_recovers_visible_contour_without_hidden_joint_targets(self):
        run = self.fit("guided", self.regions)
        poses = run["poses"]
        frame = poses["frames"][0]
        self.assertEqual([row["index"] for row in poses["frames"]], [0])
        self.assertGreaterEqual(poses["cameraScale"], 0.8)
        self.assertLessEqual(poses["cameraScale"], 1.25)
        self.assert_limits(dict(zip(poses["names"], frame["values"])))
        self.assertIn("regions", frame["objective"], "the production objective must actually include local contour loss")
        oracle = self.independent(run["out"] / "poses.json", "guided-oracle")["frames"]["0"]
        candidate_mask = image_mask(oracle["image"])
        whole_iou = overlap(self.source_mask, candidate_mask)
        local_iou = overlap(self.crop(self.source_mask), self.crop(candidate_mask))
        paw_residual = float(np.linalg.norm(np.array(oracle["landmarks"]["paw_fr"]["tail"])
                                            - self.observations["frames"]["0"]["observations"][0]["point"]))
        print("REGION_RECOVERY " + json.dumps({"independentWholeBodyIoU": whole_iou,
                                               "independentLocalIoU": local_iou, "independentPawResidualPx": paw_residual,
                                               "bounds": self.truth["bounds"], "cameraScale": poses["cameraScale"]}), flush=True)
        self.assertGreaterEqual(local_iou, 0.85, "the independently rendered visible fold must recover, not just its paw")
        self.assertGreaterEqual(whole_iou, 0.90, "local contour recovery must not sacrifice the rest of the animal")
        self.assertLessEqual(paw_residual, 0.8 + 1e-6, "the original visible-paw disk must remain satisfied")
        np.testing.assert_allclose(np.array(frame["landmarks"]["paw_fr"]["tail"]) / self.work_scale,
                                   oracle["landmarks"]["paw_fr"]["tail"], atol=1e-3)
