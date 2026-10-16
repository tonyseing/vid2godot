import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from anim.pipeline import main
from tests.blender_case import FIXTURE_CONTRACT, fixture_glb, requires_blender


def run(*argv):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, json.loads(out.getvalue() or "{}"), err.getvalue()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_init_status_next(self):
        model = self.tmp / "m.glb"
        model.write_bytes(b"x")
        self.assertEqual(run("init", "--work", str(self.tmp / "w"), "--model", str(model), "--clip", "lope")[0], 0)
        code, data, _ = run("status", "--work", str(self.tmp / "w"))
        self.assertEqual((code, data["next"]), (0, "contract"))

    def test_status_without_ledger_is_an_error(self):
        code, _, err = run("status", "--work", str(self.tmp / "none"))
        self.assertEqual(code, 1)
        self.assertIn("No ledger", err)

    def test_contract_with_missing_file_is_an_error(self):
        model = self.tmp / "m.glb"
        model.write_bytes(b"x")
        work = str(self.tmp / "w")
        run("init", "--work", work, "--model", str(model), "--clip", "lope")
        code, _, err = run("contract", "--work", work, "--project-contract", str(self.tmp / "missing.json"))
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith("error: "))
        self.assertIn("missing.json", err)
        self.assertNotIn("Traceback", err)

    def test_facts_without_the_paws_rest_heights_is_refused(self):
        """Task 17b fix round 4: facts measures each paw from its rest-pose tail height in the contract's rig.json;
        a rig.json without them is a clean refusal, never a silent fallback to absolute heights."""
        from anim.ledger import Ledger
        model = self.tmp / "m.glb"
        model.write_bytes(b"x")
        work = self.tmp / "w"
        ledger = Ledger.init(work, model, "lope")
        (work / "contract").mkdir()
        shutil.copyfile(FIXTURE_CONTRACT, work / "contract" / "project-contract.json")
        rig = work / "contract" / "rig.json"
        rig.write_text(json.dumps({"bones": [{"name": "root", "head": [0, 0, 0], "tail": [0, 0, 1]}]}))
        ledger.record("contract", {}, {"rig": rig})
        code, _, err = run("facts", "--work", str(work))
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith("error: "))
        self.assertRegex(err, "rest.*paw_")
        self.assertNotIn("Traceback", err)

    @requires_blender
    def test_contract_on_fixture(self):
        glb, _ = fixture_glb()
        work = str(self.tmp / "w")
        run("init", "--work", work, "--model", str(glb), "--clip", "walk")
        code, data, _ = run("contract", "--work", work, "--project-contract", str(FIXTURE_CONTRACT))
        self.assertEqual((code, data["ok"]), (0, True))
        self.assertEqual(run("next", "--work", work)[1]["next"], "brief")

    @requires_blender
    def test_brief_and_reference_on_fixture(self):
        glb, _ = fixture_glb()
        work = str(self.tmp / "w")
        brief = self.tmp / "brief.json"
        brief.write_text(json.dumps({"clip": "walk", "subject": "box quadruped", "motion": "walking",
                                     "gait": "4-beat lateral walk", "loop": True, "planar": True,
                                     "videoSeconds": 5}))
        run("init", "--work", work, "--model", str(glb), "--clip", "walk")
        run("contract", "--work", work, "--project-contract", str(FIXTURE_CONTRACT))
        self.assertEqual(run("brief", "--work", work, "--file", str(brief))[0], 0)
        code, data, _ = run("reference", "--work", work)
        self.assertEqual((code, data["background"]), (0, "#c8c8c8"))
        self.assertTrue((Path(work) / "reference" / "reference.png").is_file())
        self.assertEqual(run("next", "--work", work)[1]["next"], "generate")


class FactsCommandTests(unittest.TestCase):
    """Task 19: `facts` hands clip_facts the contract's rig (contacts from the motion, the 60 Hz keys), the body
    mass (--mass-kg, the brief's massKg, or the rig's mesh volume) and an explicit --accept-gait reason, and records
    the mass and the acceptance in the ledger."""

    def setUp(self):
        import numpy as np
        from anim.ledger import Ledger
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        model = self.tmp / "m.glb"
        model.write_bytes(b"x")
        self.work = work = self.tmp / "w"
        ledger = Ledger.init(work, model, "lope")
        (work / "contract").mkdir()
        shutil.copyfile(FIXTURE_CONTRACT, work / "contract" / "project-contract.json")
        self.rig = {"bones": [{"name": b, "parent": None, "head": [0, 0, 0.1], "tail": [0.1, 0, 0.0]}
                              for b in ("paw_fl", "paw_fr", "paw_hl", "paw_hr")]}
        (work / "contract" / "rig.json").write_text(json.dumps(self.rig))
        ledger.record("contract", {}, {"rig": work / "contract" / "rig.json"})
        np.savez(work / "masks.npz", masks=np.zeros((4, 8, 8), bool))
        (work / "poses.json").write_text(json.dumps({"names": ["dx", "dz"], "frames": [], "fps": 24.0,
                                                     "camera": {}, "bones": []}))
        (work / "stub.json").write_text("{}")
        self.brief({"clip": "lope", "loop": True, "gait": "lope"})
        self.result = {"start": 1.0, "period": 15.0, "speedMps": 3.0, "strideM": 1.9, "warnings": [],
                       "entry": {"duration_s": 0.4}, "cycle": {"frames": []},
                       "gaitCheck": {"expected": "canter", "matches": False, "accepted": "transverse is right"}}

    def brief(self, data):
        """Write the brief and record it and the stages up to fit again (recording a stage drops the later ones)."""
        from anim.ledger import Ledger
        ledger = Ledger(self.work)
        (self.work / "brief.json").write_text(json.dumps(data))
        ledger.record("brief", {}, {"brief": self.work / "brief.json"})
        for stage in ("reference", "generate", "accept"):
            ledger.record(stage, {}, {"stub": self.work / "stub.json"})
        ledger.record("ingest", {}, {"masks": self.work / "masks.npz"})
        ledger.record("fit", {}, {"poses": self.work / "poses.json"})

    def facts(self, *extra):
        from unittest import mock
        from anim import pipeline
        with mock.patch.object(pipeline.facts_mod, "clip_facts", return_value=self.result) as clip, \
                mock.patch.object(pipeline, "rig_mass", return_value={"kg": 85.0, "source": "rig volume"}) as volume:
            code, data, err = run("facts", "--work", str(self.work), *extra)
        return code, err, clip, volume

    def test_the_rig_mass_and_acceptance_reach_facts_and_the_ledger(self):
        from anim.ledger import Ledger
        code, err, clip, volume = self.facts("--mass-kg", "40", "--accept-gait", "transverse is right")
        self.assertEqual(code, 0, err)
        kwargs = clip.call_args.kwargs
        self.assertEqual(kwargs["rig"], self.rig)
        self.assertEqual(kwargs["mass"], {"kg": 40.0, "source": "facts --mass-kg"})
        self.assertEqual(kwargs["accept_gait"], "transverse is right")
        volume.assert_not_called()
        stage = Ledger(self.work).require("facts")
        self.assertEqual(stage["facts"]["acceptGait"], "transverse is right")
        self.assertEqual(stage["facts"]["mass"], {"kg": 40.0, "source": "facts --mass-kg"})

    def test_accept_retime_reaches_facts_and_the_ledger(self):
        """Fix round 1, ruling 2: `facts --accept-retime <reason>` is passed to clip_facts and recorded."""
        from anim.ledger import Ledger
        code, err, clip, _ = self.facts("--mass-kg", "40", "--accept-retime", "the owner wants it at 40 kg")
        self.assertEqual(code, 0, err)
        self.assertEqual(clip.call_args.kwargs["accept_retime"], "the owner wants it at 40 kg")
        self.assertEqual(Ledger(self.work).require("facts")["facts"]["acceptRetime"], "the owner wants it at 40 kg")

    def test_rig_mass_carries_the_open_edge_count(self):
        """Fix round 1, ruling 2: the mesh-volume mass carries its non-manifold edge count, so facts never retimes
        on an open mesh's volume."""
        from unittest import mock
        from anim import pipeline
        from anim.ledger import Ledger
        found = {"volumeM3": 0.085, "meshes": [{"name": "body", "volumeM3": 0.08, "nonManifoldEdges": 1500},
                                                {"name": "eyes", "volumeM3": 0.005, "nonManifoldEdges": 90}]}
        with mock.patch.object(pipeline, "run_blender", return_value=found):
            mass = pipeline.rig_mass(Ledger(self.work), self.tmp / "mass")
        self.assertEqual(mass["kg"], 85.0)
        self.assertEqual(mass["openEdges"], 1590)
        self.assertIn("rig mesh volume", mass["source"])
        with mock.patch.object(pipeline, "run_blender", return_value={"volumeM3": 0.04, "meshes": [
                {"name": "body", "volumeM3": 0.04, "nonManifoldEdges": 0}]}):
            self.assertEqual(pipeline.rig_mass(Ledger(self.work), self.tmp / "mass")["openEdges"], 0)

    def test_the_brief_mass_then_the_rig_volume(self):
        self.brief({"clip": "lope", "loop": True, "gait": "lope", "massKg": [30, 50]})
        code, err, clip, volume = self.facts()
        self.assertEqual(code, 0, err)
        self.assertEqual(clip.call_args.kwargs["mass"], {"kg": [30, 50], "source": "brief massKg"})
        self.assertIsNone(clip.call_args.kwargs["accept_gait"])
        volume.assert_not_called()
        self.brief({"clip": "lope", "loop": True, "gait": "lope"})
        code, err, clip, volume = self.facts()
        self.assertEqual(code, 0, err)
        volume.assert_called_once()
        self.assertEqual(clip.call_args.kwargs["mass"], {"kg": 85.0, "source": "rig volume"})
