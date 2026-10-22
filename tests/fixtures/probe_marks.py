"""Exercise the actual fitter objective/identity on the fixture rig, without optimizing any frames."""
import json
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anim" / "blender"))
import rigutil
from anim.camera import SideCamera

args = rigutil.load_args()
folder = Path(args["outDir"])
folder.mkdir(parents=True, exist_ok=True)
project = args["project"]
bones = list(dict.fromkeys(b for chain in project["chains"].values() for b in chain if b != project["rootBone"]))
names = ["dx", "dz"] + bones
camera = SideCamera(128, 128, 100, 0, 0.3)
scale = 0.5
angles = {"upper_fl": 0.6, "lower_fl": -1.2, "paw_fl": 0.6,
          "upper_fr": -0.6, "lower_fr": 1.2, "paw_fr": -0.6}
values = np.array([0.0, 0.0] + [angles.get(b, 0.0) for b in bones])
arm, _, _ = rigutil.load_glb(args["model"])
rigutil.rest_pose(arm)
for b, angle in angles.items():
    rigutil.set_angle(arm, b, rigutil.side_axis_local(arm, b), angle)
import bpy
bpy.context.view_layer.update()
heads = {b: camera.project(np.array(rigutil.bone_point(arm, b, "head")))
         for b in ("lower_fl", "lower_fr", "paw_fl")}
toes = {b: camera.project(np.array(rigutil.bone_point(arm, b))) for b in ("paw_fl", "paw_fr")}
joint_targets = {b: (heads[b] / scale).tolist() for b in ("lower_fl", "paw_fl")}
matching_paw = {"paw_fl": (toes["paw_fl"] / scale).tolist()}
offset_paw = {"paw_fl": ((toes["paw_fl"] + [3, 4]) / scale).tolist()}
masks = np.zeros((3, 128, 128), bool)
masks[:, 30:100, 20:110] = True
np.savez(folder / "masks.npz", masks=masks, outside=np.zeros_like(masks, float))
previous = {"cameraScale": 1.0, "frames": [{"index": f, "values": values.tolist(), "loss": 0} for f in range(3)]}
(folder / "init.json").write_text(json.dumps(previous))
np.savez(folder / "projection.npz", uv=np.zeros((3, 1, 2)))
base = {"model": args["model"], "project": project,
        "brief": {"fitChains": list(project["chains"]), "limitsDeg": {}, "iouThreshold": 0.9},
        "camera": camera.to_json(), "masks": str(folder / "masks.npz"), "workScale": scale,
        "fps": 24, "init": str(folder / "init.json"), "span": [0, 0], "periodic": False,
        "out": str(folder / "poses.json"), "projection": str(folder / "projection.npz"),
        "result": str(folder / "fit-result.json")}


def load(paw_marks, joint_marks=None):
    fit_args = {**base, "marks": {"1": paw_marks} if paw_marks else {}}
    if joint_marks is not None:
        fit_args["jointFrames"] = {"1": joint_marks} if joint_marks else {}
    path = folder / "fit-args.json"
    path.write_text(json.dumps(fit_args))
    sys.argv = [__file__, "--", str(path)]
    return runpy.run_path(str(Path(rigutil.__file__).with_name("fit_clip.py")))


def penalty(ns, x):
    return ns["score"](x, 1, None, 1.0) - ns["score"](x, 1, None, 1.0, 0.0)


joint = load({}, joint_targets)
x = np.r_[values, 1.0]
pair = next(p for p in joint["pairs"] if p[0] == "front_L")
wrong = joint["swapped"](x, pair)
xs = {f: wrong.copy() for f in range(3)}
joint_changed = joint["relabel"](xs, list(range(3)))
metrics = {"headToToePx": float(np.linalg.norm(heads["lower_fl"] - toes["paw_fl"])),
           "toeSeparationPx": float(np.linalg.norm(toes["paw_fl"] - toes["paw_fr"])),
           "jointMatchingPenalty": penalty(joint, x), "jointSwappedPenalty": penalty(joint, wrong),
           "expectedJointSwappedPenalty": 0.5 * float(np.linalg.norm(heads["lower_fl"] - heads["lower_fr"])) / 2,
           "jointRelabelled": joint_changed,
           "identityError": float(max(np.max(np.abs(y - x)) for y in xs.values()))}
metrics["deformModel"] = joint["DEFORM_MODEL"]
point_globals = joint["mark_points"].__globals__
original_fast = point_globals["FAST"]
for backend, fast in (("numpy", True), ("depsgraph", False)):
    point_globals["FAST"] = fast
    projected = joint["mark_points"](x, camera, list(joint_targets), "head")
    metrics[backend + "HeadErrorPx"] = float(max(
        np.linalg.norm(projected[b] - heads[b]) for b in joint_targets))
point_globals["FAST"] = original_fast
paw = load(matching_paw)
metrics["pawMatchingPenalty"] = penalty(paw, x)
xs = {f: wrong.copy() for f in range(3)}
metrics["pawRelabelled"] = paw["relabel"](xs, list(range(3)))
paw_offset = load(offset_paw)
metrics["pawOffsetPenalty"] = penalty(paw_offset, x)
empty_joint = load(offset_paw, {})
metrics["emptyJointPawPenalty"] = penalty(empty_joint, x)
mixed = load(offset_paw, joint_targets)
metrics["mixedPenalty"] = penalty(mixed, x)
rigutil.write_result(args, metrics)
