import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from anim import verify
from anim.ledger import Ledger, StageError, sha256_file
from anim.pipeline import main

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


FAILING = '''
import sys
sys.exit(1)
'''

CAPTURING = '''
import hashlib, json, sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
out = Path(args["--out"]); out.mkdir(parents=True, exist_ok=True)
sha = hashlib.sha256(Path(args["--model"]).read_bytes()).hexdigest()
version = "4.5.1.stable.official" if "4.5" in Path(args["--godot"]).name else "4.7.2.stable.official"
captures = []
if "--launcher" in args:
    folder = out / "run-1" / "project" / "captures"; folder.mkdir(parents=True, exist_ok=True)
    for name in ("phone-front.png", "tablet-side.png"):
        (folder / name).write_bytes(version.encode() + name.encode())
        captures.append({"profile": name.split("-")[0], "path": str(folder / name)})
report = {"status": "passed", "modelSha256": sha, "engine": {"version": version},
          "mode": "activation-suppressed capture" if "--launcher" in args else "headless", "captures": captures,
          "facts": {"animations": [{"name": "lope", "lengthSeconds": 0.5, "tracks": 30, "loopMode": 0}]}}
(out / "godot-report.json").write_text(json.dumps(report))
'''


def good_report(sha, **changes):
    report = {"status": "passed", "modelSha256": sha, "engine": {"version": "4.7.2.stable.official"},
              "mode": "headless", "captures": [],
              "facts": {"animations": [{"name": "lope", "lengthSeconds": 0.5, "tracks": 30, "loopMode": 0}]}}
    report.update(changes)
    return report


class VerifyHardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.model = self.tmp / "m.glb"
        self.model.write_bytes(b"glb")
        self.sha = sha256_file(self.model)
        self.files = {"model.glb": self.model}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    # Finding 1: the game/assets refusal.
    def assert_refused(self, destination):
        with self.assertRaises(StageError) as caught:
            verify.export_bundle(self.files, destination)
        self.assertIn("game assets", str(caught.exception))

    def test_game_assets_mixed_case_refused(self):
        (self.tmp / "repo" / "game" / "assets" / "models").mkdir(parents=True)
        self.assert_refused(self.tmp / "repo" / "Game" / "Assets" / "models" / "x")
        self.assert_refused(self.tmp / "other" / "GAME" / "assets" / "x")

    def test_game_assets_symlink_refused(self):
        real = self.tmp / "repo" / "game" / "assets"
        real.mkdir(parents=True)
        (self.tmp / "shortcut").symlink_to(real, target_is_directory=True)
        self.assert_refused(self.tmp / "shortcut" / "models" / "x")

    def test_game_assets_dotdot_refused(self):
        (self.tmp / "repo" / "game" / "assets").mkdir(parents=True)
        (self.tmp / "repo" / "asset-work").mkdir(parents=True)
        self.assert_refused(self.tmp / "repo" / "asset-work" / ".." / "game" / "assets" / "x")
        self.assert_refused(self.tmp / "repo" / "asset-work" / ".." / "Game" / "x" / ".." / "Assets" / "y")

    def test_game_assets_same_file_refused(self):
        (self.tmp / "repo" / "game" / "assets").mkdir(parents=True)
        destination = self.tmp / "repo" / "asset-work" / "fox" / "bundle"
        with mock.patch.object(verify.os.path, "samefile", lambda a, b: True):
            self.assert_refused(destination)
        self.assertFalse(destination.exists())

    def test_asset_work_destination_accepted(self):
        (self.tmp / "repo" / "game" / "assets").mkdir(parents=True)
        destination = self.tmp / "repo" / "asset-work" / "fox" / "bundle"
        out = verify.export_bundle(self.files, destination)
        self.assertEqual(out, destination.resolve())
        self.assertTrue((out / "model.glb").is_file())

    # Finding 2: a stale report never stands in for a failed run.
    def test_stale_report_and_failing_exit(self):
        out = self.tmp / "out"
        out.mkdir()
        (out / "godot-report.json").write_text(json.dumps(good_report(self.sha)))
        script = self.tmp / "failing.py"
        script.write_text(FAILING)
        with self.assertRaises(verify.VerifyError) as caught:
            verify.run_verify(script, self.model, out, Path(sys.executable))
        self.assertIn("verify.log", str(caught.exception))
        self.assertFalse((out / "godot-report.json").exists())
        self.assertTrue((out / "verify.log").is_file())

    # Finding 3: the launcher run must report activation-suppressed captures.
    def test_launcher_requires_captures(self):
        headless = good_report(self.sha)
        self.assertEqual(verify.check_report(headless, self.sha, "lope", 0.5, 24.0, "4.7"), [])
        self.assertTrue(verify.check_report(headless, self.sha, "lope", 0.5, 24.0, "4.7", launcher_used=True))
        png = self.tmp / "phone.png"
        png.write_bytes(b"png")
        captured = good_report(self.sha, mode="activation-suppressed capture", captures=[{"path": str(png)}])
        self.assertEqual(verify.check_report(captured, self.sha, "lope", 0.5, 24.0, "4.7", launcher_used=True), [])
        no_captures = good_report(self.sha, mode="activation-suppressed capture")
        self.assertTrue(verify.check_report(no_captures, self.sha, "lope", 0.5, 24.0, "4.7", launcher_used=True))
        for captures in ([{"path": str(self.tmp / "gone.png")}], [{}], ["x"], "x"):
            report = good_report(self.sha, mode="activation-suppressed capture", captures=captures)
            self.assertTrue(verify.check_report(report, self.sha, "lope", 0.5, 24.0, "4.7", launcher_used=True),
                            captures)

    # Finding 4a: malformed reports are problems, never tracebacks.
    def test_malformed_reports_are_problems(self):
        lope = {"name": "lope", "lengthSeconds": 0.5, "tracks": 30}
        cases = [good_report(self.sha, engine={"version": None}), good_report(self.sha, engine=None),
                 good_report(self.sha, engine="4.7"), good_report(self.sha, facts=None),
                 good_report(self.sha, facts={"animations": None}), good_report(self.sha, facts={"animations": {}}),
                 good_report(self.sha, facts={"animations": [{"lengthSeconds": 0.5, "tracks": 3}, lope]}),
                 good_report(self.sha, facts={"animations": ["lope", lope]}),
                 good_report(self.sha, facts={"animations": [{"name": "lope", "lengthSeconds": 0.5}]}),
                 good_report(self.sha, facts={"animations": [{"name": "lope", "tracks": 3}]}),
                 good_report(self.sha, facts={"animations": [{"name": "lope", "tracks": "3", "lengthSeconds": 0.5}]}),
                 good_report(self.sha, facts={"animations": [{"name": "lope", "tracks": 3, "lengthSeconds": None}]}),
                 good_report(self.sha, facts="x"), [], "x", None]
        for report in cases:
            self.assertTrue(verify.check_report(report, self.sha, "lope", 0.5, 24.0, "4.7"), report)

    def test_duplicate_clip_names_all_checked(self):
        good = {"name": "lope", "lengthSeconds": 0.5, "tracks": 30}
        for bad in ({"name": "lope", "lengthSeconds": 0.9, "tracks": 30},
                    {"name": "lope", "lengthSeconds": 0.5, "tracks": 0}):
            for order in ([good, bad], [bad, good]):
                report = good_report(self.sha, facts={"animations": order})
                self.assertTrue(verify.check_report(report, self.sha, "lope", 0.5, 24.0, "4.7"), order)
        report = good_report(self.sha, facts={"animations": [good, dict(good)]})
        self.assertEqual(verify.check_report(report, self.sha, "lope", 0.5, 24.0, "4.7"), [])

    # Finding 4b: no partial bundle.
    def test_export_failure_leaves_no_bundle(self):
        destination = self.tmp / "exports" / "bundle"
        files = {"model.glb": self.model, "missing.json": self.tmp / "missing.json"}
        with self.assertRaises((StageError, OSError)):
            verify.export_bundle(files, destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list((self.tmp / "exports").iterdir()) if (self.tmp / "exports").exists() else [], [])
        out = verify.export_bundle({"model.glb": self.model, "captures/4.7/a.png": self.model}, destination)
        manifest = json.loads((out / "MANIFEST.json").read_text())
        self.assertEqual(sorted(manifest), ["captures/4.7/a.png", "model.glb"])
        self.assertTrue((out / "captures" / "4.7" / "a.png").is_file())
        self.assertEqual(sorted(p.name for p in (self.tmp / "exports").iterdir()), ["bundle"])


def run_cli(*argv):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, json.loads(out.getvalue() or "{}"), err.getvalue()


class VerifyExportPipelineTests(unittest.TestCase):
    """`verify` then `export` on a synthetic ledger (stand-in files, a fake verifier, no Godot)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        model = self.tmp / "fox.glb"
        model.write_bytes(b"fox")
        self.work = self.tmp / "work"
        ledger = Ledger.init(self.work, model, "lope")
        made = {}

        def make(name, content=None):
            path = self.work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content if content is not None else name)
            made[name] = path
            return path

        facts = json.dumps({"fps": 24.0, "cycle": {"fps": 30.0}, "entry": {"duration_s": 0.5}})
        ledger.record("contract", {}, {"rig": make("rig.json")})
        ledger.record("brief", {}, {"brief": make("brief.json")})
        ledger.record("reference", {}, {"image": make("reference.png")})
        ledger.record("generate", {}, {"record": make("generate.json")})
        ledger.record("accept", {}, {"accepted": make("accepted.json")})
        ledger.record("ingest", {}, {"meta": make("ingest.json")})
        ledger.record("fit", {"marks": make("marks.json")}, {"report": make("fit-report.json")})
        ledger.record("facts", {}, {"facts": make("facts.json", facts), "entry": make("entry.json")})
        ledger.record("bake", {}, {"glb": make("bake/fox-lope.glb"), "sheet": make("bake/sheet.png"),
                                   "closeup": make("bake/closeup.png")})
        make("review/overlays/00000.png")
        make("review/overlays/00001.png")
        ledger.record("review", {"lab": make("review/lab.html")}, {"review": make("review/review.json")})
        self.script = self.tmp / "fake_verify.py"
        self.script.write_text(CAPTURING)
        self.engines = ["--godot47", str(self.tmp / "godot-4.7"), "--godot45", str(self.tmp / "godot-4.5")]

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_verify_requires_captures_and_export_bundles_them(self):
        code, _, err = run_cli("verify", "--work", str(self.work), *self.engines, "--launcher",
                               str(self.tmp / "launcher.mjs"), "--threed-verify", str(self.script))
        self.assertEqual(code, 0, err)
        destination = self.tmp / "asset-work" / "fox" / "lope-eval"
        code, data, err = run_cli("export", "--work", str(self.work), "--destination", str(destination))
        self.assertEqual(code, 0, err)
        manifest = json.loads((destination / "MANIFEST.json").read_text())
        for name in ("captures/4.7/phone-front.png", "captures/4.7/tablet-side.png", "captures/4.5/phone-front.png",
                     "captures/4.5/tablet-side.png", "lab.html", "overlays/00000.png", "overlays/00001.png"):
            self.assertIn(name, manifest)
            self.assertEqual(manifest[name], sha256_file(destination / name))
        self.assertEqual((destination / "captures" / "4.5" / "phone-front.png").read_bytes(),
                         b"4.5.1.stable.officialphone-front.png")

    def test_verify_refuses_a_headless_report(self):
        headless = self.tmp / "headless_verify.py"
        headless.write_text(CAPTURING.replace('if "--launcher" in args:', "if False:")
                            .replace('"activation-suppressed capture" if "--launcher" in args else ', ""))
        code, _, err = run_cli("verify", "--work", str(self.work), *self.engines, "--launcher",
                               str(self.tmp / "launcher.mjs"), "--threed-verify", str(headless))
        self.assertEqual(code, 1)
        self.assertIn("[4.7]", err)
        self.assertIn("capture", err)
