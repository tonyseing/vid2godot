"""Independent source articulation assessment, diagnosis, and hash-bound acceptance."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from anim import articulation
from anim.ledger import StageError
from anim import review


PROJECT = {"rootBone": "root", "chains": {"front_R": ["upper", "elbow", "paw"]},
           "paws": {"front_R": "paw"}}
BRIEF = {"fitChains": ["front_R"]}


def observation():
    return {"schemaVersion": 1,
            "source": {"videoSha256": "a" * 64, "coordinateSpace": "registered_source_px"},
            "frames": {"0": {"phase": "fold", "observations": [
                {"id": "elbow", "bone": "elbow", "end": "head", "point": [20, 30],
                 "uncertaintyPx": 5, "visibility": "visible", "correspondence": "confirmed"},
                {"id": "toe", "bone": "paw", "end": "tail", "point": [40, 50],
                 "uncertaintyPx": 2, "visibility": "visible", "correspondence": "confirmed"}]}}}


def poses(elbow=(10, 15), paw=(20, 25)):
    return {"frames": [{"index": 0, "landmarks": {"elbow": {"head": list(elbow)},
                                                               "paw": {"tail": list(paw)}}}]}


class ArticulationTests(unittest.TestCase):
    def validated(self, doc=None):
        return articulation.validate(doc or observation(), PROJECT, BRIEF, 2, "a" * 64)

    def test_positive_finite_uncertainty_and_source_provenance_are_required(self):
        for uncertainty in (None, 0, -1, True, float("nan"), float("inf")):
            doc = observation()
            doc["frames"]["0"]["observations"][0]["uncertaintyPx"] = uncertainty
            with self.subTest(uncertainty=uncertainty), self.assertRaises(StageError):
                self.validated(doc)
        with self.assertRaises(StageError):
            articulation.validate(observation(), PROJECT, BRIEF, 2, "b" * 64)

    def test_invalid_frames_phases_targets_and_duplicate_identity_are_rejected(self):
        for key, value in (("phase", ""), ("phase", None), ("observations", [])):
            doc = observation()
            doc["frames"]["0"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(StageError):
                self.validated(doc)
        for field, value in (("bone", "missing"), ("end", "tail"), ("point", [float("nan"), 0]),
                             ("visibility", "guessed"), ("correspondence", "certain")):
            doc = observation()
            doc["frames"]["0"]["observations"][0][field] = value
            with self.subTest(field=field), self.assertRaises(StageError):
                self.validated(doc)
        doc = observation()
        doc["frames"]["00"] = copy.deepcopy(doc["frames"]["0"])
        with self.assertRaises(StageError):
            self.validated(doc)
        doc = observation()
        doc["frames"]["0"]["observations"][1]["id"] = "elbow"
        with self.assertRaises(StageError):
            self.validated(doc)

    def test_good_paw_and_silhouette_do_not_excuse_wrong_elbow(self):
        fitted = poses(elbow=(30, 15))
        fitted["frames"][0]["iou"] = 0.99
        result = articulation.assess(self.validated(), fitted, work_scale=0.5)
        self.assertEqual(result["status"], "fail")
        elbow, toe = result["frames"][0]["observations"]
        self.assertEqual(elbow["status"], "fail")
        self.assertEqual(toe["status"], "pass")
        self.assertAlmostEqual(elbow["residualPx"], 40)
        self.assertAlmostEqual(elbow["normalizedResidual"], 8)
        self.assertEqual(result["coverage"]["phases"], ["fold"])
        self.assertFalse(result["coverage"]["fullMotionVerified"])

    def test_uncertainty_boundary_is_inclusive_in_source_pixels(self):
        doc = self.validated()
        self.assertEqual(articulation.assess(doc, poses(elbow=(12.5, 15)), .5)["status"], "pass")
        self.assertEqual(articulation.assess(doc, poses(elbow=(12.50001, 15)), .5)["status"], "fail")

    def test_missing_hidden_and_uncertain_correspondence_never_pass(self):
        for state in ("tentative", "ambiguous"):
            doc = observation()
            doc["frames"]["0"]["observations"][0]["correspondence"] = state
            self.assertEqual(articulation.assess(self.validated(doc), poses(), .5)["status"], "unresolved")
        doc = observation()
        item = doc["frames"]["0"]["observations"][0]
        item["visibility"] = "occluded"
        item.pop("point")
        item.pop("uncertaintyPx")
        self.assertEqual(articulation.assess(self.validated(doc), poses(), .5)["status"], "unresolved")
        self.assertEqual(articulation.assess(self.validated(), {"frames": []}, .5)["status"], "unresolved")
        self.assertEqual(articulation.assess(None, poses(), .5)["status"], "not_assessed")

    def test_diagnosis_distinguishes_search_from_objective_only_with_passing_witness(self):
        failed = articulation.assess(self.validated(), poses(elbow=(30, 15)), .5)
        witness = articulation.assess(self.validated(), poses(), .5)
        for got, want, cause in ((10, 5, "search_miss"), (5, 10, "objective_rejection"),
                                 (5, 5, "unresolved")):
            result = articulation.diagnose(failed, witness, [{"index": 0, "fittedObjective": got,
                                                             "witnessObjective": want, "validWitness": True}])
            self.assertEqual(result["frames"][0]["cause"], cause)
        self.assertEqual(articulation.diagnose(failed)["frames"][0]["cause"], "unresolved")
        bad = articulation.diagnose(failed, failed, [{"index": 0, "fittedObjective": 10,
                                                    "witnessObjective": 1, "validWitness": True}])
        self.assertEqual(bad["frames"][0]["cause"], "unresolved")

    def test_acceptance_rejects_changed_observations_camera_rig_source_and_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name in ("observations", "camera", "rig", "source", "poses", "projection"):
                paths[name] = Path(tmp) / name
                paths[name].write_text(name)
            report = articulation.bind(articulation.assess(self.validated(), poses(), .5), paths)
            path = Path(tmp) / "assessment.json"
            path.write_text(json.dumps(report))
            self.assertEqual(articulation.require_current(path, paths)["status"], "pass")
            for name, target in paths.items():
                with self.subTest(name=name):
                    target.write_text(name + " changed")
                    with self.assertRaises(StageError):
                        articulation.require_current(path, paths)
                    target.write_text(name)
            report["status"] = "unresolved"
            path.write_text(json.dumps(report))
            with self.assertRaises(StageError):
                articulation.require_current(path, paths)
            with self.assertRaises(StageError):
                articulation.require_current(path, {"poses": paths["poses"]})

    def test_review_explanation_cannot_override_failed_articulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "poses"
            target.write_text("poses")
            report = {"frames": [{"index": 0, "iou": .99, "flags": []}], "lowFrames": [],
                      "articulation": articulation.assess(self.validated(), poses(elbow=(30, 15)), .5)}
            template = review.make_template("walk", {"poses": target}, report, 1)
            filled = {**template, "reviewer": "reviewer", "verdict": "pass", "inspectedReference": True,
                      "inspectedOverlays": True, "inspectedSheet": True, "observations": "Looks fine; explain away"}
            self.assertTrue(any("articulation" in p.lower() for p in review.validate(filled, template)))


class ArticulationPipelineTests(unittest.TestCase):
    """Real ledger/files and CLI refusal; no mocked Blender or success echoes."""
    def setUp(self):
        import shutil
        import numpy as np
        from PIL import Image
        from anim.ledger import Ledger, sha256_file
        from tests.blender_case import FIXTURE_CONTRACT
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model = self.root / "model.glb"
        model.write_bytes(b"gate tests never load a model")
        self.work = self.root / "work"
        self.ledger = Ledger.init(self.work, model, "walk")
        folder = self.work / "contract"
        folder.mkdir()
        shutil.copyfile(FIXTURE_CONTRACT, folder / "project-contract.json")
        stub = folder / "rig.json"
        stub.write_text("{}")
        self.ledger.record("contract", {}, {"rig": stub})
        brief = self.work / "brief.json"
        brief.write_text(json.dumps({"fitChains": ["front_R"], "limitsDeg": {}, "iouThreshold": .9}))
        self.ledger.record("brief", {}, {"brief": brief})
        for stage in ("reference", "generate", "accept"):
            self.ledger.record(stage, {}, {"stub": stub})
        video = self.work / "source.mp4"
        video.write_bytes(b"source")
        ingest = self.work / "ingest"
        (ingest / "registered").mkdir(parents=True)
        Image.new("RGB", (128, 128), "white").save(ingest / "registered" / "00000.png")
        meta = ingest / "ingest.json"
        meta.write_text(json.dumps({"frames": 1, "workScale": .5, "camera": {}, "fps": 24, "flags": {}}))
        masks = ingest / "masks.npz"
        np.savez(masks, masks=np.ones((1, 64, 64), bool))
        self.ledger.record("ingest", {"video": video}, {"meta": meta, "masks": masks})
        observations = self.work / "articulation" / "observations.json"
        observations.parent.mkdir()
        doc = observation()
        doc["source"]["videoSha256"] = sha256_file(video)
        doc["frames"]["0"]["observations"][0]["bone"] = "lower_fr"
        doc["frames"]["0"]["observations"][1]["bone"] = "paw_fr"
        observations.write_text(json.dumps(doc))
        fit = self.work / "fit"
        fit.mkdir()
        self.pose_path = fit / "poses.json"
        fitted = poses()
        fitted["frames"][0]["landmarks"] = {"lower_fr": {"head": [10, 15]}, "paw_fr": {"tail": [20, 25]}}
        self.pose_path.write_text(json.dumps(fitted))
        projection = fit / "projection.npz"
        np.savez(projection, uv=np.zeros((1, 3, 2)), faces=[[0, 1, 2]])
        report = fit / "report.json"
        report.write_text("{}")
        marks = fit / "marks.json"
        marks.write_text('{"frames": {}}')
        self.ledger.record("fit", {"observations": observations, "marks": marks},
                           {"poses": self.pose_path, "projection": projection, "report": report})

    def test_current_assessment_blocks_all_downstream_commands_without_explanation_override(self):
        from anim.pipeline import assess_fit, require_articulation
        from tests.test_pipeline import run
        report = assess_fit(self.ledger)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(require_articulation(self.ledger)["status"], "pass")
        report_path = self.work / "articulation" / "assessment.json"
        report["status"] = "unresolved"
        report_path.write_text(json.dumps(report))
        commands = [("facts",), ("bake",), ("review", "--template"),
                    ("verify", "--godot47", "/never", "--godot45", "/never", "--launcher", "/never"),
                    ("export", "--destination", str(self.root / "never-export"))]
        # Complete the ordinary ledger so verify/export cannot merely fail on a missing earlier stage.
        stub = self.work / "contract" / "rig.json"
        for stage in ("facts", "bake", "review", "verify"):
            self.ledger.record(stage, {}, {"stub": stub})
        for command in commands:
            with self.subTest(command=command):
                code, _, error = run(*command, "--work", str(self.work))
                self.assertEqual(code, 1)
                self.assertIn("Articulation", error)
        self.assertFalse((self.root / "never-export").exists())

    def test_forged_pass_and_deleted_recorded_sidecar_cannot_become_legacy(self):
        from anim.pipeline import assess_fit, require_articulation
        assess_fit(self.ledger)
        doc = json.loads(self.pose_path.read_text())
        doc["frames"][0]["landmarks"]["lower_fr"]["head"] = [30, 15]
        self.pose_path.write_text(json.dumps(doc))
        report = assess_fit(self.ledger)
        self.assertEqual(report["status"], "fail")
        report["status"] = "pass"
        (self.work / "articulation" / "assessment.json").write_text(json.dumps(report))
        with self.assertRaises(StageError):
            require_articulation(self.ledger)
        (self.work / "articulation" / "observations.json").unlink()
        with self.assertRaises((StageError, OSError)):
            require_articulation(self.ledger)

    def test_legacy_workspace_is_usable_but_not_assessed(self):
        from anim.pipeline import require_articulation
        self.ledger.data["stages"]["fit"]["inputs"].pop("observations")
        (self.work / "articulation" / "observations.json").unlink()
        self.assertEqual(require_articulation(self.ledger)["status"], "not_assessed")

    def test_preflight_enrollment_survives_deleted_sidecar_before_first_observation_fit(self):
        from anim.pipeline import require_articulation
        from tests.test_pipeline import run
        self.ledger.data["stages"]["fit"]["inputs"].pop("observations")
        self.ledger.path.write_text(json.dumps(self.ledger.data))
        source = self.work / "articulation" / "observations.json"
        preflight = self.work / "articulation" / "preflight"
        preflight.mkdir()
        recorded = preflight / "observations.json"
        recorded.write_bytes(source.read_bytes())
        assessment = articulation.bind({"status": "unresolved", "frames": []}, {"observations": recorded})
        (preflight / "assessment.json").write_text(json.dumps(assessment))
        source.unlink()
        recorded.unlink()
        with self.assertRaises((StageError, OSError)):
            require_articulation(self.ledger)
        code, _, error = run("fit", "--work", str(self.work), "--blender", "/never-run-Blender")
        self.assertEqual(code, 1)
        self.assertNotIn("not_assessed", error)
        self.assertIn("observations.json", error)

    def test_preflight_rejects_witness_output_collision_before_overwrite(self):
        from tests.test_pipeline import run
        folder = self.work / "articulation" / "preflight"
        folder.mkdir()
        witness = folder / "poses.json"
        original = b'{"retainedWitness": "must not be overwritten or falsely rebound"}'
        witness.write_bytes(original)
        code, _, error = run("preflight", "--work", str(self.work), "--source-observations",
                             str(self.work / "articulation" / "observations.json"),
                             "--witness", str(witness), "--blender", "/never-run-Blender")
        self.assertEqual(code, 1)
        self.assertRegex(error.lower(), "witness.*(overwrite|collision|output)")
        self.assertEqual(witness.read_bytes(), original)

    def test_first_observation_fit_assessment_remains_current_after_canonical_promotion(self):
        from anim.pipeline import assess_fit, require_articulation
        canonical = self.work / "articulation" / "observations.json"
        preflight = self.work / "articulation" / "preflight"
        preflight.mkdir()
        sparse_observations = preflight / "observations.json"
        sparse_observations.write_bytes(canonical.read_bytes())
        (preflight.parent / "enrollment.json").write_text(json.dumps(
            {"observations": {"path": str(sparse_observations.resolve())}}))
        # The first full fit has copied its selected sidecar to canonical storage, but has not
        # recorded its fit input yet; this is the real ordering used by the command.
        previous = self.ledger.data["stages"]["fit"]
        previous["inputs"].pop("observations")
        self.ledger.path.write_text(json.dumps(self.ledger.data))
        report = assess_fit(self.ledger)
        self.assertEqual(report["status"], "pass")
        self.ledger.record("fit", {"observations": canonical, "marks": self.work / "fit" / "marks.json"},
                           {name: Path(item["path"]) for name, item in previous["outputs"].items()})
        current = require_articulation(self.ledger)
        self.assertEqual(current["status"], "pass")
        self.assertEqual(Path(current["bound"]["observations"]["path"]), canonical.resolve())
