import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from anim import motion
from anim.blender_run import run_blender
from anim.contract import load_project_contract
from anim.ledger import Ledger
from PIL import Image
from tests.blender_case import BLENDER, FIXTURE_CONTRACT, FIXTURES, fixture_glb, requires_blender
from tests.test_pipeline import run

WIDTH = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}
PROBE = FIXTURES / "probe_pelvis.py"


def read_glb(path):
    """The glTF JSON and its binary chunk."""
    data = Path(path).read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + length])
    return doc, data[20 + length + 8:]


def accessor(doc, binary, index):
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    assert acc["componentType"] == 5126, "float accessors only"
    width = WIDTH[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    flat = np.frombuffer(binary, np.float32, acc["count"] * width, start)
    return flat.reshape(acc["count"], width) if width > 1 else flat


def channels(path):
    """{animation: {(node name, path): (times, values, interpolation)}} plus each node's rest translation."""
    doc, binary = read_glb(path)
    out = {}
    for anim in doc.get("animations", []):
        out[anim["name"]] = {}
        for channel in anim["channels"]:
            sampler = anim["samplers"][channel["sampler"]]
            node = doc["nodes"][channel["target"]["node"]]["name"]
            out[anim["name"]][(node, channel["target"]["path"])] = (
                accessor(doc, binary, sampler["input"]), accessor(doc, binary, sampler["output"]),
                sampler.get("interpolation", "LINEAR"))
    rest = {node["name"]: np.array(node.get("translation", [0.0, 0.0, 0.0])) for node in doc["nodes"]}
    return out, rest


def fixture_cycle(n=24, fps=24.0, period=23.6, dx=None, dz=None, loop=True):
    """A cycle block like facts.json's, keyed from the fixture walk's known angles: a sub-frame period, so the
    cycle's rate is not an integer. The pelvis travels `dx(k)` forward and `dz(k)` up at key k. As facts writes it,
    a loop's key n repeats key 0 and its drift is the root travel one period on, dx(n) - dx(0) and dz(n) - dz(0);
    a one-shot's key n is its own last frame and its drift is zero."""
    dx = dx or (lambda k: 0.01 * k + sway(k, n, 0.004))
    dz = dz or (lambda k: sway(k, n, 0.005, -np.pi / 2))          # 0.005 sin(2 pi k / n)
    _, truth_path = fixture_glb()
    truth = json.loads(truth_path.read_text())
    bones = [b for chain in load_project_contract(FIXTURE_CONTRACT)["chains"].values() for b in chain]
    keys = [[float(dx(k)), float(dz(k))] + [truth[str(k)].get(b, 0.0) for b in bones] for k in range(n)]
    if loop:
        keys.append(list(keys[0]))
        drift = {"dx": float(dx(n) - dx(0)), "dz": float(dz(n) - dz(0))}
    else:
        keys.append([float(dx(n)), float(dz(n))] + keys[0][2:])
        drift = {"dx": 0.0, "dz": 0.0}
    return bones, ["dx", "dz"] + bones, {"frames": keys, "fps": n * fps / period, "periodFrames": period,
                                         "start": 3.25, "drift": drift}


def sway(k, n, amplitude, phase=0.0):
    """A purely periodic root sway: one cycle over n keys, so sway(n) = sway(0)."""
    return amplitude * np.cos(2 * np.pi * np.asarray(k, float) / n + phase)


def minus_mean(values, n):
    """The analytic expectation for an in-place offset: `values` at keys 0..n less their mean over keys 0..n-1."""
    values = np.asarray(values, float)
    return values - values[:n].mean()


def bake(tmp, cycle, bones, names, loop=True):
    """Bake `cycle` onto the fixture rig; return the bake result, the eval GLB path and the pelvis head's world
    displacement from rest at each key 0..n, measured on a fresh re-import of the GLB."""
    glb, _ = fixture_glb()
    (tmp / "facts.json").write_text(json.dumps({"cycle": cycle}))
    out = tmp / "eval.glb"
    result = run_blender("bake_clip", {"model": str(glb), "out": str(out), "cycle": str(tmp / "facts.json"),
                                       "clip": "walk_fit", "loop": loop, "pelvisBone": "pelvis",
                                       "bones": bones, "names": names}, tmp, BLENDER)
    probe = run_blender(PROBE, {"model": str(out), "fps": cycle["fps"], "clip": "walk_fit", "bone": "pelvis",
                                "keys": len(cycle["frames"]) - 1}, tmp / "probe", BLENDER)
    return result, out, np.array(probe["displacement"])


@requires_blender
class BakeTests(unittest.TestCase):
    def test_bake_adds_clip_and_keeps_others(self):
        glb, _ = fixture_glb()
        n, fps, period = 24, 24.0, 23.6
        bones, names, cycle = fixture_cycle(n, fps, period)
        rate = cycle["fps"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result, out, moves = bake(tmp, cycle, bones, names)
            self.assertIn("walk", result["actions"])
            first, last = result["actions"]["walk_fit"]
            self.assertAlmostEqual(first, 0.0, places=4)
            self.assertAlmostEqual(last, float(n), places=4)
            baked, _ = channels(out)
            original, _ = channels(glb)
            fit = baked["walk_fit"]
            # Godot's clip length: key k at k / cycle fps, so n / cycle fps = period / fps.
            times, _, interpolation = fit[("upper_fl", "rotation")]
            np.testing.assert_allclose(times, np.arange(n + 1) / rate, atol=1e-5)
            self.assertAlmostEqual(float(times[-1]), period / fps, places=5)
            self.assertEqual(interpolation, "LINEAR")
            # Key k is cycle frame k: the fixture's walk was keyed with the same side-axis angles at frame k.
            for bone in ("upper_fl", "lower_fl", "upper_hr", "lower_hr"):
                want_times, want, _ = original["walk"][(bone, "rotation")]
                got = fit[(bone, "rotation")][1]
                np.testing.assert_allclose(np.abs(np.sum(got[:n] * want[:n], axis=1)), 1.0, atol=1e-5,
                                           err_msg=bone)
                np.testing.assert_allclose(got[n], got[0], atol=1e-6)            # the seam closes
            # In place, in world axes: the pelvis head moves forward (+X) by the in-place dx, up (+Z) by dz, and
            # never sideways (Y), measured on the re-imported GLB.
            # The fixture's dx is 0.01 k of drift plus a 4 mm sway; dz is a 5 mm sway with no drift.
            k = np.arange(n + 1)
            np.testing.assert_allclose(moves[:, 0], minus_mean(sway(k, n, 0.004), n), atol=1e-4, err_msg="forward")
            np.testing.assert_allclose(moves[:, 2], sway(k, n, 0.005, -np.pi / 2), atol=1e-4, err_msg="up")
            np.testing.assert_allclose(moves[:, 1], 0.0, atol=1e-4, err_msg="lateral")
            self.assertGreater(np.ptp(moves[:, 0]), 0.005)                  # the checks are not vacuous
            self.assertGreater(np.ptp(moves[:, 2]), 0.005)
            # The other clips are kept: same key times (the GLB's length) and the same values.
            for channel, (want_times, want, _) in original["walk"].items():
                got_times, got, _ = baked["walk"][channel]
                np.testing.assert_allclose(got_times, want_times, atol=1e-5, err_msg=str(channel))
                np.testing.assert_allclose(got, want, atol=1e-5, err_msg=str(channel))

    def test_periodic_sway_bakes_without_a_seam_pop(self):
        # Fix round 2: a purely periodic sine-phase sway (drift 0) must keep its own step across the seam; the old
        # least-squares detrend tilted it and added a 1.56 mm pop at n = 24.
        n = 24
        k = np.arange(n + 1)
        want = sway(k, n, 0.001, -np.pi / 2)                          # 0.001 sin(2 pi k / n)
        bones, names, cycle = fixture_cycle(n, dx=lambda i: sway(i, n, 0.001, -np.pi / 2), dz=lambda i: 0.0)
        self.assertAlmostEqual(cycle["drift"]["dx"], 0.0, places=12)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, moves = bake(Path(tmp), cycle, bones, names)
        forward = moves[:, 0]
        self.assertAlmostEqual(abs(forward[n - 1] - forward[0]), abs(want[n - 1] - want[0]), delta=1e-5,
                               msg="the seam step is the sway's own step: no added pop")
        self.assertAlmostEqual(forward[n], forward[0], places=5)
        np.testing.assert_allclose(forward, minus_mean(want, n), atol=1e-4)

    def test_loop_removes_linear_drift_exactly(self):
        # The camera tracked a little off the creature's speed: dx drifts 2 mm and dz -1 mm a key on top of their
        # periodic sways. The bake removes drift * k / n exactly, so only each sway is left (dx also loses its mean).
        n = 24
        k = np.arange(n + 1)
        sway_x, sway_z = sway(k, n, 0.001, -np.pi / 2), sway(k, n, 0.002)
        bones, names, cycle = fixture_cycle(n, dx=lambda i: 0.002 * i + sway(i, n, 0.001, -np.pi / 2),
                                            dz=lambda i: -0.001 * i + sway(i, n, 0.002))
        self.assertAlmostEqual(cycle["drift"]["dx"], 0.002 * n, places=9)
        self.assertAlmostEqual(cycle["drift"]["dz"], -0.001 * n, places=9)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, moves = bake(Path(tmp), cycle, bones, names)
        np.testing.assert_allclose(moves[:, 0], minus_mean(sway_x, n), atol=1e-4, err_msg="forward")
        np.testing.assert_allclose(moves[:, 2], sway_z, atol=1e-4, err_msg="up")
        self.assertAlmostEqual(moves[n, 0], moves[0, 0], places=5)
        self.assertAlmostEqual(moves[n, 2], moves[0, 2], places=5)

    def test_one_shot_removes_only_the_mean(self):
        n = 24
        bones, names, cycle = fixture_cycle(n, dx=lambda k: 0.002 * k, dz=lambda k: 0.0, loop=False)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, moves = bake(Path(tmp), cycle, bones, names, loop=False)
        np.testing.assert_allclose(moves[:, 0], 0.002 * (np.arange(n + 1) - (n - 1) / 2), atol=1e-4)
        self.assertAlmostEqual(moves[n, 0] - moves[0, 0], 0.002 * n, places=4)   # a one-shot keeps its travel shape

    def bake_command(self, offset=0.0):
        """`bake` through the CLI on the fixture rig. Changed with Task 19: facts hands the bake periodic keys (no
        drift, dx mean-free) and its own prediction of every paw at every key, which the bake checks on the
        re-imported GLB; `offset` moves the prediction to test that check."""
        glb, _ = fixture_glb()
        n = 24
        bones, names, cycle = fixture_cycle(n, dx=lambda k: sway(k, n, 0.004), dz=lambda k: sway(k, n, 0.005, 1.0))
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        work = tmp / "w"
        brief = tmp / "brief.json"
        brief.write_text(json.dumps({"clip": "walk_fit", "subject": "box quadruped", "motion": "walking",
                                     "gait": "4-beat lateral walk", "loop": True, "planar": True,
                                     "videoSeconds": 5}))
        run("init", "--work", str(work), "--model", str(glb), "--clip", "walk_fit")
        self.assertEqual(run("contract", "--work", str(work), "--project-contract", str(FIXTURE_CONTRACT))[0], 0)
        self.assertEqual(run("brief", "--work", str(work), "--file", str(brief))[0], 0)
        self.assertEqual(run("reference", "--work", str(work))[0], 0)
        # The video stages are stood in by files: bake reads only fit/poses.json and facts/facts.json.
        ledger = Ledger(work)
        stub = tmp / "stub.json"
        stub.write_text("{}")
        for stage in ("generate", "accept", "ingest"):
            ledger.record(stage, {}, {"stub": stub})
        (work / "fit").mkdir()
        (work / "fit" / "poses.json").write_text(json.dumps({"bones": bones, "names": names, "fps": 24.0}))
        ledger.record("fit", {}, {"poses": work / "fit" / "poses.json"})
        project = load_project_contract(FIXTURE_CONTRACT)
        rig = motion.PlanarRig(json.loads((work / "contract" / "rig.json").read_text()))
        predicted = rig.paws(np.array(cycle["frames"]), names, project)
        cycle["paws"] = {project["paws"][c]: (p + offset).tolist() for c, p in predicted.items()}
        (work / "facts").mkdir()
        (work / "facts" / "facts.json").write_text(json.dumps({"cycle": cycle}))
        ledger.record("facts", {}, {"facts": work / "facts" / "facts.json"})
        return work, run("bake", "--work", str(work))

    def test_a_bake_whose_paws_miss_the_facts_prediction_is_refused(self):
        _, (code, _, err) = self.bake_command(offset=0.005)
        self.assertEqual(code, 1)
        self.assertRegex(err, r"paw.*facts.*mm")

    def test_bake_command_writes_glb_sheet_and_closeup(self):
        work, (code, data, err) = self.bake_command()
        self.assertEqual(code, 0, err)
        self.assertLess(data["pawErrorMm"], 0.5)
        self.assertEqual(Path(data["glb"]).name, "test_rig-walk_fit-eval.glb")
        self.assertIn("walk", channels(data["glb"])[0])
        with Image.open(data["sheet"]) as image:
            sheet = np.array(image.convert("RGB"))
        offsets = [0, 3, 6, 9, 12, 15, 18, 21]          # 8 poses across cycle frames 0..n-1
        frames = sorted((work / "bake" / "sheet-frames").glob("*.png"))
        self.assertEqual([int(p.stem) for p in frames], offsets)
        tile = sheet.shape[1] // 8
        self.assertEqual(sheet.shape[1], 8 * tile)
        self.assertTrue(np.any(sheet[:, :tile] != sheet[:, 3 * tile:4 * tile]), "the poses differ across the cycle")
        self.assertTrue(np.all(sheet[0, 0] == [0x6b, 0x5b, 0x9f]), "the palette background")
        self.assertTrue(Path(data["closeup"]).is_file())
        self.assertEqual(run("next", "--work", str(work))[1]["next"], "review")
