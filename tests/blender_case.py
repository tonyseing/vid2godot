import os
import unittest
from pathlib import Path

from anim.blender_run import run_blender

BLENDER = os.environ.get("VID2GODOT_BLENDER")
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OUT = FIXTURES / "out"
FIXTURE_CONTRACT = FIXTURES / "test_contract.json"
requires_blender = unittest.skipUnless(BLENDER, "set VID2GODOT_BLENDER to run Blender tests")


def fixture_glb():
    glb, truth = OUT / "test_rig.glb", OUT / "walk_truth.json"
    if not glb.is_file():
        run_blender(FIXTURES / "build_test_rig.py", {"out": str(glb), "truth": str(truth)}, OUT, BLENDER)
    return glb, truth
