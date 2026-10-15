import stat
import tempfile
import unittest
from pathlib import Path

from anim.blender_run import BlenderError, run_blender


class RunnerFailureTests(unittest.TestCase):
    def fake(self, tmp, body):
        path = Path(tmp) / "fake-blender"
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return str(path)

    def test_timeout_is_a_blender_error_with_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = self.fake(tmp, "echo started\nsleep 5")
            with self.assertRaises(BlenderError) as caught:
                run_blender("inspect_rig", {}, Path(tmp) / "logs", fake, timeout=0.5)
            self.assertIn("timed out", str(caught.exception))
            self.assertTrue((Path(tmp) / "logs" / "inspect_rig.log").is_file())

    def test_unlaunchable_blender_is_a_blender_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake-blender"
            fake.write_text("not executable")
            with self.assertRaises(BlenderError):
                run_blender("inspect_rig", {}, Path(tmp) / "logs", str(fake))
