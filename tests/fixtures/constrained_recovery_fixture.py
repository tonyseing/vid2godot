"""Render one recoverable coupled pose, or independently render fitted parameters.

This uses the existing rigid-skinned quadruped unchanged. Source pose values and its
calibrated camera never enter fit_clip: only a rendered mask and measured landmarks do.
All pose authoring here is test-only and no animation is exported.
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
arm, skinned, others = rigutil.load_glb(args["model"])
for obj in others:
    obj.hide_render = True
out = Path(args["outDir"])
out.mkdir(parents=True, exist_ok=True)


def apply(values):
    rigutil.rest_pose(arm)
    arm.location = (float(values.get("dx", 0.0)), 0.0, float(values.get("dz", 0.0)))
    for bone, angle in values.items():
        if bone not in ("dx", "dz"):
            rigutil.set_angle(arm, bone, rigutil.side_axis_local(arm, bone), float(angle))
    bpy.context.view_layer.update()


def measure(camera, label):
    # Deliberately read Blender's evaluated bones, never fit_clip's FK/projection NPZ.
    evaluated = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = {bone.name: {end: camera.project(np.array(rigutil.bone_point(evaluated, bone.name, end))).tolist()
                          for end in ("head", "tail")} for bone in evaluated.pose.bones}
    rigutil.render_setup(camera, "#c8c8c8")
    path = out / f"{label}.png"
    rigutil.render(path)
    return {"landmarks": points, "image": str(path)}


if "poses" in args:
    poses = json.loads(Path(args["poses"]).read_text())
    # Return source-resolution evidence, independently of the fitter's working-pixel report.
    camera = SideCamera.from_json(poses["camera"]).scaled(1.0 / args["workScale"])
    frames = {}
    for frame in poses["frames"]:
        apply(dict(zip(poses["names"], frame["values"])))
        frames[str(frame["index"])] = measure(camera, f"fit-{frame['index']}")
    rigutil.write_result(args, {"camera": camera.to_json(), "frames": frames})
else:
    registered = SideCamera.from_json(args["camera"])
    rigutil.rest_pose(arm)
    rest = registered.project(rigutil.evaluated_vertices(skinned)[:, [0, 2]])
    pivot_u = float((rest[:, 0].min() + rest[:, 0].max()) / 2)
    pivot_v = float(rest[:, 1].max())
    scale = 0.8731
    # A valid member of the fitter's existing bounded camera family, not a new
    # perspective/rotation degree of freedom or an incompatible synthetic camera.
    camera = SideCamera(registered.width, registered.height, registered.px_per_m * scale,
                        registered.center_x + (1 - scale) * (registered.width / 2 - pivot_u)
                        / (scale * registered.px_per_m),
                        registered.center_z - (1 - scale) * (registered.height / 2 - pivot_v)
                        / (scale * registered.px_per_m))
    angles = {"pelvis": 6.3, "chest": -9.7, "head": 8.6,
              "upper_fr": 32.7, "lower_fr": -73.6, "paw_fr": 40.9,
              "upper_fl": -24.6, "lower_fl": 48.2, "paw_fl": -23.6,
              "upper_hr": -18.4, "lower_hr": 33.8, "paw_hr": -15.4,
              "upper_hl": 21.3, "lower_hl": -39.6, "paw_hl": 18.3}
    values = {"dx": 0.037, "dz": -0.023, **{bone: math.radians(angle) for bone, angle in angles.items()}}
    apply(values)
    source = measure(camera, "source")
    rigutil.write_result(args, {"source": source, "camera": camera.to_json(),
                               "cameraScale": scale, "values": values})
