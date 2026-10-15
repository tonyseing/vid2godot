import hashlib
import os
import unittest
from pathlib import Path

from anim.blender_run import run_blender

BLENDER = os.environ.get("VID2GODOT_BLENDER")
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OUT = FIXTURES / "out"
FIXTURE_CONTRACT = FIXTURES / "test_contract.json"
RIGUTIL = Path(__file__).resolve().parents[1] / "anim" / "blender" / "rigutil.py"
requires_blender = unittest.skipUnless(BLENDER, "set VID2GODOT_BLENDER to run Blender tests")


def build_digest(script):
    """sha256 over the build script and rigutil.py, the code that decides the fixture's geometry."""
    digest = hashlib.sha256()
    for path in (script, RIGUTIL):
        data = Path(path).read_bytes()
        digest.update(f"{Path(path).name}:{len(data)}:".encode())
        digest.update(data)
    return digest.hexdigest()


def cached_build(script, outputs, args):
    """Build the fixture with `script` unless every output exists and the sidecar digest beside the first output
    matches the current build code: a GLB left by older build code is rebuilt, never silently reused."""
    stamp = outputs[0].with_name(outputs[0].name + ".sha256")
    want = build_digest(script)
    have = stamp.read_text().strip() if stamp.is_file() else None
    if have != want or not all(path.is_file() for path in outputs):
        stamp.unlink(missing_ok=True)
        run_blender(script, args, OUT, BLENDER)
        stamp.write_text(want + "\n")
    return outputs


def fixture_glb():
    glb, truth = OUT / "test_rig.glb", OUT / "walk_truth.json"
    return tuple(cached_build(FIXTURES / "build_test_rig.py", [glb, truth], {"out": str(glb), "truth": str(truth)}))


def bad_fixture_glb():
    glb = OUT / "bad_rig.glb"
    return cached_build(FIXTURES / "build_bad_rig.py", [glb], {"out": str(glb)})[0]
