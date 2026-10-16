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
        self.work = self.ledger_to_review(self.tmp)
        # Changed with Task 19: verify patches the 3D skill's capture script for each clip phase, so the fake
        # verifier sits in a folder beside a capture script.
        self.script = self.fake_skill(self.tmp / "skill", PHASED)
        self.engines = ["--godot47", str(self.tmp / "godot-4.7"), "--godot45", str(self.tmp / "godot-4.5")]

    @staticmethod
    def fake_skill(folder, verifier):
        folder.mkdir(parents=True)
        (folder / "verify.py").write_text(verifier)
        (folder / "capture_asset.gd").write_text(CAPTURE_GD)
        return folder / "verify.py"

    @staticmethod
    def ledger_to_review(tmp):
        """A synthetic ledger through `review` (stand-in files); returns the work folder."""
        model = tmp / "fox.glb"
        model.write_bytes(b"fox")
        work = tmp / "work"
        ledger = Ledger.init(work, model, "lope")
        made = {}

        def make(name, content=None):
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content if content is not None else name)
            made[name] = path
            return path

        facts = json.dumps({"fps": 24.0, "cycle": {"fps": 30.0, "paws": {"paw_hl": [], "paw_fl": []}},
                            "entry": {"duration_s": 0.5}})
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
        return work

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
        for name in ("captures/4.7/phase000-phone-side.png", "captures/4.7/phase667-tablet-side.png",
                     "captures/4.5/phase333-phone-side.png", "captures/4.5/phase000-tablet-side.png", "lab.html",
                     "overlays/00000.png", "overlays/00001.png"):
            self.assertIn(name, manifest)
            self.assertEqual(manifest[name], sha256_file(destination / name))
        first = (destination / "captures" / "4.5" / "phase000-phone-side.png").read_bytes()
        self.assertTrue(first.startswith(b"\x89PNG"))
        self.assertNotEqual(first, (destination / "captures" / "4.5" / "phase333-phone-side.png").read_bytes())

    def test_verify_refuses_a_headless_report(self):
        headless = self.fake_skill(self.tmp / "headless", PHASED.replace(
            '"mode": "activation-suppressed capture", "captures": captures',
            '"mode": "headless", "captures": []'))
        code, _, err = run_cli("verify", "--work", str(self.work), *self.engines, "--launcher",
                               str(self.tmp / "launcher.mjs"), "--threed-verify", str(headless))
        self.assertEqual(code, 1)
        self.assertIn("[4.7]", err)
        self.assertIn("capture", err)


# Task 19 step 5: the 3D skill's capture script, as far as vid2godot's phase patch reads it.
CAPTURE_GD = '''extends SceneTree

func _capture() -> void:
	var model := packed.instantiate() as Node3D
	scene.add_child(model)
	for profile in PROFILES:
		for view in VIEWS:
			var filename: String = "captures/%s-%s.png" % [profile["name"], view["name"]]
			captures.append({"path": filename, "profile": profile["name"], "view": view["name"]})
'''

# Changed with fix round 1: the fake renders real PNGs in which the "clip" moves with the phase (a bar whose x
# follows the phase tag, unless STILL), and records the patched script's paw positions per capture (moving with the
# phase unless PAWS_STILL), since verify now measures that the phases differ. STATUS is the later phases' status.
PHASED = """
import hashlib, json, re, sys
from pathlib import Path
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
STILL, PAWS_STILL, STATUS = False, False, "passed"
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
out = Path(args["--out"]); out.mkdir(parents=True, exist_ok=True)
sha = hashlib.sha256(Path(args["--model"]).read_bytes()).hexdigest()
version = "4.5.1.stable.official" if "4.5" in Path(args["--godot"]).name else "4.7.2.stable.official"
script = (HERE / "capture_asset.gd").read_text()
phase = float(re.search(r"const CLIP_PHASE := ([0-9.]+)", script).group(1))
clip = re.search(r'const CLIP_NAME := "([^"]+)"', script).group(1)
tag = int(re.search(r"const PHASE_TAG := ([0-9]+)", script).group(1))
bones = json.loads(re.search(r"const PAW_BONES := (\\[.*\\])", script).group(1))
assert "clip_player.seek(clip_seconds, true)" in script
assert "get_bone_global_pose" in script
captures = []
folder = out / "run-1" / "project" / "captures"; folder.mkdir(parents=True, exist_ok=True)
for name in ("phone-side", "tablet-side"):
    path = folder / ("phase%03d-%s.png" % (tag, name))
    image = Image.new("RGB", (60, 30), (48, 51, 56))
    x = 5 if STILL else 5 + tag // 25
    ImageDraw.Draw(image).rectangle([x, 8, x + 12, 22], fill=(200, 200, 210))
    image.save(path)
    paws = {b: [0.1 * i + (0.0 if PAWS_STILL else 0.3 * phase), 0.0, 0.0] for i, b in enumerate(bones)}
    captures.append({"profile": name.split("-")[0], "view": "side", "path": str(path), "phase": phase, "clip": clip,
                     "clipSeconds": 0.5 * phase, "paws": paws})
report = {"status": STATUS if phase > 0.5 else "passed", "modelSha256": sha, "engine": {"version": version},
          "mode": "activation-suppressed capture", "captures": captures,
          "facts": {"animations": [{"name": "lope", "lengthSeconds": 0.5, "tracks": 30, "loopMode": 0}]}}
(out / "godot-report.json").write_text(json.dumps(report))
"""


class PhaseCaptureTests(unittest.TestCase):
    """Task 19 step 5: verify captures the clip playing, at PHASES of its length, in every view: the 3D skill's
    capture script is copied and patched to seek the clip before each capture, and the report records the phase."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.skill = self.tmp / "forge" / "godot"
        self.skill.mkdir(parents=True)
        (self.skill / "verify.py").write_text(PHASED)
        (self.skill / "capture_asset.gd").write_text(CAPTURE_GD)
        (self.skill / "inspect_asset.gd").write_text("extends SceneTree\n")

    def test_the_patch_seeks_the_clip_and_tags_each_capture(self):
        script = verify.phase_verifier(self.skill / "verify.py", "lope", 1 / 3, self.tmp / "patched", ["paw_fl", "paw_hl"])
        self.assertEqual(script, self.tmp / "patched" / "verify.py")
        self.assertEqual((self.tmp / "patched" / "inspect_asset.gd").read_text(), "extends SceneTree\n")
        patched = (self.tmp / "patched" / "capture_asset.gd").read_text()
        self.assertIn('const CLIP_NAME := "lope"', patched)
        self.assertIn("const CLIP_PHASE := 0.333333", patched)
        self.assertIn("clip_player.play(CLIP_NAME)", patched)
        self.assertLess(patched.index("scene.add_child(model)"), patched.index("clip_player.seek"))
        self.assertIn('"captures/phase%03d-%s-%s.png" % [PHASE_TAG,', patched)
        self.assertIn('captures.append({"phase": CLIP_PHASE, "clip": CLIP_NAME, "clipSeconds": clip_seconds,', patched)
        self.assertIn('const PAW_BONES := ["paw_fl", "paw_hl"]', patched)
        self.assertIn('"paws": paw_positions,', patched)
        self.assertIn("skeleton.get_bone_global_pose(bone_index)", patched)
        self.assertLess(patched.index("clip_player.pause()"), patched.index("get_bone_global_pose"))
        self.assertEqual((self.skill / "capture_asset.gd").read_text(), CAPTURE_GD, "the skill's copy is untouched")

    def test_a_changed_capture_script_is_refused(self):
        (self.skill / "capture_asset.gd").write_text(CAPTURE_GD.replace("scene.add_child(model)", "add(model)"))
        with self.assertRaisesRegex(verify.VerifyError, "capture script"):
            verify.phase_verifier(self.skill / "verify.py", "lope", 0.0, self.tmp / "patched", ["paw_fl"])

    def test_check_report_requires_the_phase_on_every_capture(self):
        model = self.tmp / "m.glb"
        model.write_bytes(b"glb")
        script = verify.phase_verifier(self.skill / "verify.py", "lope", 2 / 3, self.tmp / "patched", ["paw_fl"])
        report = verify.run_verify(script, model, self.tmp / "out", Path(self.tmp / "godot-4.7"), self.tmp / "l.mjs")
        sha = sha256_file(model)
        self.assertEqual(verify.check_report(report, sha, "lope", 0.5, 60.0, "4.7", launcher_used=True,
                                             phase=2 / 3), [])
        problems = verify.check_report(report, sha, "lope", 0.5, 60.0, "4.7", launcher_used=True, phase=0.0)
        self.assertTrue(any("phase" in p for p in problems), problems)
        del report["captures"][0]["phase"]
        problems = verify.check_report(report, sha, "lope", 0.5, 60.0, "4.7", launcher_used=True, phase=2 / 3)
        self.assertTrue(any("phase" in p for p in problems), problems)

    def test_verify_runs_every_phase_on_both_engines_and_export_bundles_them(self):
        work = VerifyExportPipelineTests.ledger_to_review(self.tmp)
        engines = ["--godot47", str(self.tmp / "godot-4.7"), "--godot45", str(self.tmp / "godot-4.5")]
        code, data, err = run_cli("verify", "--work", str(work), *engines, "--launcher", str(self.tmp / "l.mjs"),
                                  "--threed-verify", str(self.skill / "verify.py"))
        self.assertEqual(code, 0, err)
        for prefix in ("4.7", "4.5"):
            report = json.loads((work / "verify" / prefix / "godot-report.json").read_text())
            self.assertEqual(report["phases"], [0.0, 0.333333, 0.666667])
            self.assertEqual(sorted({c["phase"] for c in report["captures"]}), [0.0, 0.333333, 0.666667])
            self.assertEqual(len(report["captures"]), 6)
            self.assertEqual(len(report["runs"]), 3)
            self.assertEqual(report["verifier"]["path"], str(self.skill / "verify.py"))
        destination = self.tmp / "asset-work" / "fox" / "lope-eval"
        code, _, err = run_cli("export", "--work", str(work), "--destination", str(destination))
        self.assertEqual(code, 0, err)
        names = json.loads((destination / "MANIFEST.json").read_text())
        for tag in ("000", "333", "667"):
            self.assertIn(f"captures/4.5/phase{tag}-phone-side.png", names)


class PhaseEvidenceTests(unittest.TestCase):
    """Fix round 1, ruling 5: that the clip plays is measured. For each profile and view the captures at different
    phases must differ (image hash, and a pixel-difference share over PIXEL_FLOOR), and the paw bones' positions the
    patched script records must differ between phases; otherwise verify fails with the numbers. The combined report
    takes the worst run's status."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.work = VerifyExportPipelineTests.ledger_to_review(self.tmp)
        self.engines = ["--godot47", str(self.tmp / "godot-4.7"), "--godot45", str(self.tmp / "godot-4.5")]

    def verify(self, still=False, paws_still=False, status="passed"):
        source = PHASED.replace('STILL, PAWS_STILL, STATUS = False, False, "passed"',
                                f"STILL, PAWS_STILL, STATUS = {still!r}, {paws_still!r}, {status!r}")
        script = VerifyExportPipelineTests.fake_skill(self.tmp / f"skill-{still}-{paws_still}-{status}", source)
        return run_cli("verify", "--work", str(self.work), *self.engines, "--launcher", str(self.tmp / "l.mjs"),
                       "--threed-verify", str(script))

    def test_phases_that_render_the_same_image_are_refused_with_the_numbers(self):
        code, _, err = self.verify(still=True)
        self.assertEqual(code, 1)
        self.assertRegex(err, r"\[4\.7\] phone/side: phases 0\.00 and 0\.33 render the same image \(same hash; "
                              r"0\.0000 of the pixels differ, floor 0\.001\)")

    def test_phases_whose_paws_do_not_move_are_refused(self):
        code, _, err = self.verify(paws_still=True)
        self.assertEqual(code, 1)
        self.assertRegex(err, r"\[4\.5\] phases 0\.00 and 0\.33: no paw bone moved \(largest 0\.0000 m, floor "
                              r"0\.005 m; paw_fl, paw_hl\)")

    def test_moving_phases_pass_and_record_the_evidence(self):
        code, _, err = self.verify()
        self.assertEqual(code, 0, err)
        report = json.loads((self.work / "verify" / "4.7" / "godot-report.json").read_text())
        evidence = report["phaseEvidence"]
        self.assertEqual(evidence["pixelFloor"], verify.PIXEL_FLOOR)
        pair = evidence["views"]["phone/side"][0]
        self.assertEqual((pair["phases"], pair["sameHash"]), ([0.0, 0.333333], False))
        self.assertGreater(pair["pixelsDiffer"], verify.PIXEL_FLOOR)
        self.assertGreater(evidence["paws"][0]["maxMoveM"], verify.PAW_MOVE_MIN)
        self.assertEqual(report["status"], "passed")

    def test_the_combined_report_takes_the_worst_status(self):
        script = VerifyExportPipelineTests.fake_skill(self.tmp / "skill-combine", PHASED)
        reports = [(0.0, self.tmp / "a" / "godot-report.json", {"status": "passed", "captures": []}),
                   (0.5, self.tmp / "b" / "godot-report.json", {"status": "failed", "captures": []})]
        self.assertEqual(verify.combine(reports, script)["status"], "failed")
        self.assertEqual(verify.combine(reports[:1], script)["status"], "passed")
        code, _, err = self.verify(status="failed")
        self.assertEqual(code, 1)
        report = json.loads((self.work / "verify" / "4.7" / "godot-report.json").read_text())
        self.assertEqual(report["status"], "failed")
