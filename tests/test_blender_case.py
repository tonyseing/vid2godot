"""The Blender fixture caches rebuild when their build script or rigutil.py changes (no Blender needed)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import blender_case


class FixtureCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.fixtures, self.out = root / "fixtures", root / "fixtures" / "out"
        self.fixtures.mkdir()
        (self.fixtures / "build_test_rig.py").write_text("# good rig v1\n")
        (self.fixtures / "build_bad_rig.py").write_text("# bad rig v1\n")
        self.rigutil = root / "rigutil.py"
        self.rigutil.write_text("# rigutil v1\n")
        self.calls = []
        patches = [mock.patch.object(blender_case, "FIXTURES", self.fixtures),
                   mock.patch.object(blender_case, "OUT", self.out),
                   mock.patch.object(blender_case, "RIGUTIL", self.rigutil),
                   mock.patch.object(blender_case, "run_blender", self.fake_run_blender)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def fake_run_blender(self, script, args, log_dir, blender=None):
        self.calls.append(Path(script).name)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        for key in ("out", "truth"):
            if key in args:
                Path(args[key]).write_text(f"built from {Path(script).read_text()}")
        return {}

    def test_good_rig_rebuilds_only_when_its_digest_changes(self):
        glb, truth = blender_case.fixture_glb()
        self.assertEqual(self.calls, ["build_test_rig.py"])
        self.assertTrue(glb.is_file() and truth.is_file())
        blender_case.fixture_glb()
        self.assertEqual(len(self.calls), 1, "an equal digest must reuse the cached GLB")
        (self.fixtures / "build_test_rig.py").write_text("# good rig v2\n")
        blender_case.fixture_glb()
        self.assertEqual(len(self.calls), 2, "a changed build script must rebuild")
        self.assertIn("v2", glb.read_text())
        self.rigutil.write_text("# rigutil v2\n")
        blender_case.fixture_glb()
        self.assertEqual(len(self.calls), 3, "a changed rigutil.py must rebuild")
        blender_case.fixture_glb()
        self.assertEqual(len(self.calls), 3)

    def test_bad_rig_rebuilds_only_when_its_digest_changes(self):
        blender_case.bad_fixture_glb()
        blender_case.bad_fixture_glb()
        self.assertEqual(self.calls, ["build_bad_rig.py"])
        (self.fixtures / "build_bad_rig.py").write_text("# bad rig v2\n")
        blender_case.bad_fixture_glb()
        self.rigutil.write_text("# rigutil v2\n")
        blender_case.bad_fixture_glb()
        self.assertEqual(self.calls, ["build_bad_rig.py"] * 3)

    def test_a_glb_from_an_unknown_build_is_rebuilt(self):
        """A GLB left by an older checkout (no digest beside it) is not trusted."""
        self.out.mkdir(parents=True)
        (self.out / "test_rig.glb").write_text("old geometry")
        (self.out / "walk_truth.json").write_text("{}")
        (self.out / "bad_rig.glb").write_text("old geometry")
        blender_case.fixture_glb()
        blender_case.bad_fixture_glb()
        self.assertEqual(self.calls, ["build_test_rig.py", "build_bad_rig.py"])

    def test_a_missing_truth_file_is_rebuilt(self):
        glb, truth = blender_case.fixture_glb()
        truth.unlink()
        blender_case.fixture_glb()
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
