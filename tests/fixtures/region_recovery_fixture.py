"""Independently render a visible folded contour and paw-equivalent wrong pose.

The unchanged skinned fixture is imported afresh for every oracle invocation. Only
its rendered source mask, a visible contour rectangle, and a paw-tail disk are
provided to the production fitter; no elbow/wrist targets or source angles enter it.
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
    evaluated = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = {bone.name: {end: camera.project(np.array(rigutil.bone_point(evaluated, bone.name, end))).tolist()
                          for end in ("head", "tail")} for bone in evaluated.pose.bones}
    rigutil.render_setup(camera, "#c8c8c8")
    path = out / f"{label}.png"
    rigutil.render(path)
    return {"landmarks": points, "image": str(path)}


if "poses" in args:
    poses = json.loads(Path(args["poses"]).read_text())
    camera = SideCamera.from_json(poses["camera"]).scaled(1.0 / args["workScale"])
    frames = {}
    for frame in poses["frames"]:
        apply(dict(zip(poses["names"], frame["values"])))
        frames[str(frame["index"])] = measure(camera, f"fit-{frame['index']}")
    rigutil.write_result(args, {"camera": camera.to_json(), "frames": frames})
else:
    camera = SideCamera.from_json(args["camera"])
    angles = {"pelvis": 0, "chest": 0, "head": 0,
              "upper_fr": 35, "lower_fr": -70, "paw_fr": 35,
              "upper_fl": -20, "lower_fl": 40, "paw_fl": -20,
              "upper_hr": -12, "lower_hr": 24, "paw_hr": -12,
              "upper_hl": 18, "lower_hl": -36, "paw_hl": 18}
    values = {"dx": 0.0, "dz": 0.0, **{bone: math.radians(angle) for bone, angle in angles.items()}}
    apply(values)
    source = measure(camera, "source")
    # The crop names a visible lower fold, not an asserted anatomical joint.
    # Keep the paw tip outside it so endpoint agreement cannot certify this ROI.
    elbow = np.array(source["landmarks"]["lower_fr"]["head"])
    wrist = np.array(source["landmarks"]["paw_fr"]["head"])
    midpoint = 0.5 * (elbow + wrist)
    low = np.floor(np.minimum(midpoint, wrist) - 9).astype(int)
    high = np.ceil(np.maximum(midpoint, wrist) + 9).astype(int)
    bounds = [int(low[0]), int(low[1]), int(high[0]), int(high[1])]
    # Construct the adversary using evaluated imported-bone geometry, not an
    # assumed world-space two-link basis: GLB rest transforms are significant.
    # This fixture-only correction is never a fitting seed or an observed joint.
    target = np.array(source["landmarks"]["paw_fr"]["tail"])
    wrong_values = {**values, "paw_fr": values["paw_fr"] + math.radians(25)}

    def projected_tail(parameters):
        apply(parameters)
        evaluated = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
        return camera.project(np.array(rigutil.bone_point(evaluated, "paw_fr", "tail")))

    step = 1e-3
    for _ in range(30):
        point = projected_tail(wrong_values)
        residual = target - point
        if np.linalg.norm(residual) < 1e-5:
            break
        columns = []
        for bone in ("upper_fr", "lower_fr"):
            perturbed = {**wrong_values, bone: wrong_values[bone] + step}
            columns.append((projected_tail(perturbed) - point) / step)
        delta = np.linalg.solve(np.column_stack(columns), residual)
        delta *= min(1.0, math.radians(10) / max(float(np.max(np.abs(delta))), 1e-12))
        for bone, change in zip(("upper_fr", "lower_fr"), delta):
            wrong_values[bone] += float(change)
    if np.linalg.norm(projected_tail(wrong_values) - target) >= 1e-3:
        raise AssertionError("fixture could not construct the exact paw-equivalent wrong pose")
    apply(wrong_values)
    wrong = measure(camera, "paw-equivalent-wrong")
    rigutil.write_result(args, {"source": source, "wrong": wrong, "camera": camera.to_json(),
                               "values": values, "wrongValues": wrong_values, "bounds": bounds})
