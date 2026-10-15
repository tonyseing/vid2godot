import json
import sys
import tempfile
import unittest
from pathlib import Path

from anim import verify
from anim.ledger import StageError, sha256_file

FAKE = '''
import json, sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
out = Path(args["--out"]); out.mkdir(parents=True, exist_ok=True)
import hashlib
sha = hashlib.sha256(Path(args["--model"]).read_bytes()).hexdigest()
report = {"status": "passed", "modelSha256": sha, "engine": {"version": "4.7.2.stable.official"},
          "mode": "activation-suppressed capture" if "--launcher" in args else "headless",
          "facts": {"animations": [{"name": "lope", "lengthSeconds": 0.5, "tracks": 30, "loopMode": 0}]}}
(out / "godot-report.json").write_text(json.dumps(report))
'''


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.script = self.tmp / "verify.py"
        self.script.write_text(FAKE)
        self.model = self.tmp / "m.glb"
        self.model.write_bytes(b"glb")

    def test_run_and_check(self):
        report = verify.run_verify(self.script, self.model, self.tmp / "out", Path(sys.executable))
        sha = sha256_file(self.model)
        self.assertEqual(verify.check_report(report, sha, "lope", 0.5, 24.0, "4.7"), [])
        self.assertTrue(verify.check_report(report, sha, "trot", 0.5, 24.0, "4.7"))
        self.assertTrue(verify.check_report(report, sha, "lope", 1.0, 24.0, "4.7"))
        self.assertTrue(verify.check_report(report, sha, "lope", 0.5, 24.0, "4.5"))
        self.assertTrue(verify.check_report(report, "other", "lope", 0.5, 24.0, "4.7"))

    def test_missing_verifier(self):
        with self.assertRaises(StageError) as caught:
            verify.verifier(self.tmp / "nope.py")
        self.assertIn("game-asset-3d-pipeline", str(caught.exception))

    def test_export_bundle(self):
        files = {"model.glb": self.model}
        out = verify.export_bundle(files, self.tmp / "bundle")
        manifest = json.loads((out / "MANIFEST.json").read_text())
        self.assertEqual(manifest["model.glb"], sha256_file(self.model))
        with self.assertRaises(StageError):
            verify.export_bundle(files, self.tmp / "bundle")
        with self.assertRaises(StageError):
            verify.export_bundle(files, self.tmp / "game" / "assets" / "models" / "x")
