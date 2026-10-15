import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from anim.pipeline import main
from tests.blender_case import FIXTURE_CONTRACT, fixture_glb, requires_blender


def run(*argv):
    buffer = StringIO()
    with redirect_stdout(buffer):
        code = main(list(argv))
    return code, json.loads(buffer.getvalue() or "{}")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_init_status_next(self):
        model = self.tmp / "m.glb"
        model.write_bytes(b"x")
        self.assertEqual(run("init", "--work", str(self.tmp / "w"), "--model", str(model), "--clip", "lope")[0], 0)
        code, data = run("status", "--work", str(self.tmp / "w"))
        self.assertEqual((code, data["next"]), (0, "contract"))

    def test_status_without_ledger_is_an_error(self):
        self.assertEqual(run("status", "--work", str(self.tmp / "none"))[0], 1)

    @requires_blender
    def test_contract_on_fixture(self):
        glb, _ = fixture_glb()
        work = str(self.tmp / "w")
        run("init", "--work", work, "--model", str(glb), "--clip", "walk")
        code, data = run("contract", "--work", work, "--project-contract", str(FIXTURE_CONTRACT))
        self.assertEqual((code, data["ok"]), (0, True))
        self.assertEqual(run("next", "--work", work)[1]["next"], "brief")
