"""Production-Blender recovery must satisfy joint disks without sacrificing the body.

A valid, rendered pose couples root translation, camera registration, trunk ancestors,
and a folded front chain. The source camera and pose are withheld from the fitter.
The oracle re-imports the unchanged GLB, applies returned parameters in Blender, reads
evaluated bone ends and renders the complete animal; reported fit landmarks alone
cannot make this regression pass.
"""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from anim.articulation import assess, validate
from anim.blender_run import run_blender
from anim.camera import SideCamera
from anim.contract import load_project_contract
from anim.ingest import background_colour, boundary_points, silhouette
from tests.blender_case import BLENDER, FIXTURE_CONTRACT, FIXTURES, fixture_glb, requires_blender

ORACLE = FIXTURES / "constrained_recovery_fixture.py"


def image_mask(path):
    image = np.array(Image.open(path).convert("RGB"))
    return silhouette(image, background_colour(image))


def overlap(a, b):
    return float(np.count_nonzero(a & b) / np.count_nonzero(a | b))


@requires_blender
class ConstrainedRecoveryTests(unittest.TestCase):
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
        points = cls.truth["source"]["landmarks"]
        targets = (("visible-elbow", "lower_fr", "head", 0.8),
                   ("visible-wrist", "paw_fr", "head", 0.8),
                   ("visible-paw", "paw_fr", "tail", 0.8),
                   ("body-anchor", "pelvis", "head", 2.0),
                   ("neck-anchor", "head", "head", 2.0))
        cls.document = {"schemaVersion": 1,
                        "source": {"videoSha256": hashlib.sha256(Path(cls.truth["source"]["image"]).read_bytes()).hexdigest(),
                                   "coordinateSpace": "registered_source_px"},
                        "frames": {"0": {"phase": "coupled recovery fold", "observations": [
                            {"id": name, "bone": bone, "end": end, "point": points[bone][end],
                             "uncertaintyPx": uncertainty, "visibility": "visible", "correspondence": "confirmed"}
                            for name, bone, end, uncertainty in targets]}}}
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def independent(self, poses_path, label):
        out = self.work / label
        return run_blender(ORACLE, {"model": str(self.model), "poses": str(poses_path),
                                    "workScale": self.work_scale, "outDir": str(out)}, out, BLENDER)

    def fit(self, label, observations=None):
        if label not in self.runs:
            out = self.work / label
            out.mkdir()
            args = {"model": str(self.model), "project": self.project, "brief": self.brief,
                    "camera": self.registered.scaled(self.work_scale).to_json(), "masks": str(self.masks),
                    "marks": {}, "frameIndices": [0], "workScale": self.work_scale, "fps": 24,
                    "periodic": False, "out": str(out / "poses.json"), "projection": str(out / "projection.npz")}
            if observations is not None:
                args["sourceObservations"] = validate(observations, self.project, self.brief, 1)
            run_blender("fit_clip", args, out, BLENDER)
            type(self).runs[label] = {"poses": json.loads((out / "poses.json").read_text()), "out": out}
        return self.runs[label]

    def test_source_is_recoverable_with_original_camera_family_and_limits(self):
        """Constructive witness: a second Blender import reproduces both body and disks."""
        self.assertGreaterEqual(self.truth["cameraScale"], 0.8)
        self.assertLessEqual(self.truth["cameraScale"], 1.25)
        self.assertLess(self.truth["cameraScale"], 0.94, "exercise recovery outside the first-frame start scales")
        values = self.truth["values"]
        self.assertGreater(abs(values["dx"]), 0.02)
        self.assertGreater(abs(values["dz"]), 0.02)
        for bone, angle in values.items():
            if bone not in ("dx", "dz"):
                low, high = self.brief["limitsDeg"][bone]
                self.assertLessEqual(low, math.degrees(angle))
                self.assertLessEqual(math.degrees(angle), high)
        self.assertFalse(self.source_mask[[0, -1]].any(), "source must not be vertically clipped")
        self.assertFalse(self.source_mask[:, [0, -1]].any(), "source must not be horizontally clipped")
        witness_path = self.work / "constructive-witness.json"
        witness_path.write_text(json.dumps({
            "camera": SideCamera.from_json(self.truth["camera"]).scaled(self.work_scale).to_json(),
            "names": list(values), "frames": [{"index": 0, "values": list(values.values())}]}))
        witness = self.independent(witness_path, "witness-oracle")["frames"]["0"]
        self.assertGreaterEqual(overlap(self.source_mask, image_mask(witness["image"])), 0.995)
        for item in self.document["frames"]["0"]["observations"]:
            np.testing.assert_allclose(witness["landmarks"][item["bone"]][item["end"]], item["point"], atol=1e-3)

    def test_constrained_fit_recovers_joint_disks_and_plausible_whole_body(self):
        run = self.fit("constrained", self.document)
        poses = run["poses"]
        self.assertEqual([frame["index"] for frame in poses["frames"]], [0])
        self.assertGreaterEqual(poses["cameraScale"], 0.8)
        self.assertLessEqual(poses["cameraScale"], 1.25)
        frame = poses["frames"][0]
        for bone, angle in zip(poses["names"], frame["values"]):
            if bone in self.brief["limitsDeg"]:
                low, high = self.brief["limitsDeg"][bone]
                self.assertLessEqual(low - 1e-9, math.degrees(angle))
                self.assertLessEqual(math.degrees(angle), high + 1e-9)
        oracle = self.independent(run["out"] / "poses.json", "constrained-oracle")["frames"]["0"]
        residuals = {item["id"]: float(np.linalg.norm(np.array(oracle["landmarks"][item["bone"]][item["end"]])
                                                      - item["point"]))
                     for item in self.document["frames"]["0"]["observations"]}
        iou = overlap(self.source_mask, image_mask(oracle["image"]))
        print("CONSTRAINED_RECOVERY " + json.dumps({"independentSourceResidualPx": residuals,
                                                   "independentWholeBodyIoU": iou,
                                                   "cameraScale": poses["cameraScale"]}), flush=True)
        # Evaluate the entire rendered silhouette as well as the labelled chain.
        # A joint-feasible limb pasted onto a poor body fit is not recovery.
        self.assertGreaterEqual(iou, 0.90, "independent full-body render must remain plausible, not merely joint-feasible")
        for item in self.document["frames"]["0"]["observations"]:
            self.assertLessEqual(residuals[item["id"]], item["uncertaintyPx"] + 1e-6,
                                 f"{item['id']}: independent Blender residual {residuals[item['id']]:.3f} source px")
            np.testing.assert_allclose(np.array(frame["landmarks"][item["bone"]][item["end"]]) / self.work_scale,
                                       oracle["landmarks"][item["bone"]][item["end"]], atol=1e-3)
        independent_poses = {"frames": [{"index": 0, "landmarks": oracle["landmarks"]}]}
        self.assertEqual(assess(self.document, independent_poses)["status"], "pass")

    def test_no_usable_observations_preserve_legacy_fit_exactly(self):
        omitted = self.fit("legacy-omitted")
        unresolved = copy.deepcopy(self.document)
        for item in unresolved["frames"]["0"]["observations"]:
            item["correspondence"] = "ambiguous"
        ignored = self.fit("legacy-unresolved", unresolved)
        for key in ("names", "camera", "cameraScale", "registrationCamera", "deformModel", "periodicPass"):
            self.assertEqual(omitted["poses"][key], ignored["poses"][key], key)
        for key in ("values", "loss", "objective", "landmarks", "joints", "paws", "flags"):
            self.assertEqual(omitted["poses"]["frames"][0][key], ignored["poses"]["frames"][0][key], key)
        with np.load(omitted["out"] / "projection.npz") as a, np.load(ignored["out"] / "projection.npz") as b:
            np.testing.assert_array_equal(a["uv"], b["uv"])
            np.testing.assert_array_equal(a["faces"], b["faces"])
