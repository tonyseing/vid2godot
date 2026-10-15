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
