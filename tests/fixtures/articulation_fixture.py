"""Test-only known folded poses and independent Blender projection of fitted values.

Reuses the existing rigid-skinned quadruped; no animation is exported or shipped.
Equal upper/lower segment lengths let opposite folds retain identical paw positions.
Exchanging the two front folds retains the combined silhouette while changing the
labelled elbow. The fit never receives these pose values, only source observations.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anim" / "blender"))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import rigutil  # noqa: E402
from anim.camera import SideCamera  # noqa: E402

args = rigutil.load_args()
arm, _, others = rigutil.load_glb(args["model"])
for obj in others:
    obj.hide_render = True
camera = SideCamera.from_json(args["camera"])
out = Path(args["outDir"])
out.mkdir(parents=True, exist_ok=True)


def apply(values):
    rigutil.rest_pose(arm)
    arm.location = (float(values.get("dx", 0.0)), 0.0, float(values.get("dz", 0.0)))
    for bone, angle in values.items():
        if bone not in ("dx", "dz"):
            rigutil.set_angle(arm, bone, rigutil.side_axis_local(arm, bone), float(angle))
    bpy.context.view_layer.update()


def landmarks():
    # Read evaluated pose bones, not fit_clip's numpy FK or its reported landmarks.
    evaluated = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return {bone.name: {end: camera.project(np.array(rigutil.bone_point(evaluated, bone.name, end))).tolist()
                        for end in ("head", "tail")} for bone in evaluated.pose.bones}


if "poses" in args:
    poses = json.loads(Path(args["poses"]).read_text())
    frames = {}
    for frame in poses["frames"]:
        apply(dict(zip(poses["names"], frame["values"])))
        frames[str(frame["index"])] = {"landmarks": landmarks()}
    rigutil.write_result(args, {"frames": frames})
else:
    rigutil.render_setup(camera, "#c8c8c8")
    source = {f"{part}_{side}": math.radians(angle * direction)
              for side, direction in (("fr", 1), ("fl", -1))
              for part, angle in (("upper", 40), ("lower", -80), ("paw", 40))}
    wrong = {bone: -angle for bone, angle in source.items()}
    result = {}
    for label, values in (("source", source), ("wrong", wrong)):
        apply(values)
        points = landmarks()
        path = out / f"{label}.png"
        rigutil.render(path)
        result[label] = {"image": str(path), "landmarks": points}
    rigutil.write_result(args, {"sourceImage": result["source"]["image"], **result})
