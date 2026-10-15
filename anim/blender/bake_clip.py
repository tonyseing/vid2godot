"""Write the fitted cycle as an in-place clip on the rig and export an evaluation GLB (other clips kept).

`cycle` is facts.json's cycle block (or a path to facts.json): n + 1 value vectors in poses.json `names` order,
resampled at sub-frame precision, played at `cycle.fps`. Key k goes at frame k, so the clip lasts n / cycle.fps."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy  # noqa: E402
from bpy_extras.anim_utils import action_get_channelbag_for_slot  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
import rigutil  # noqa: E402

args = rigutil.load_args()
cycle = args["cycle"]
if isinstance(cycle, str):
    cycle = json.loads(Path(cycle).read_text())["cycle"]
fps = float(cycle["fps"])
arm, skinned, others = rigutil.load_glb(args["model"], fps=fps)   # the scene rate is the cycle's, before import
bones, names = args["bones"], args["names"]
index = {name: i for i, name in enumerate(names)}
axes = [rigutil.side_axis_local(arm, b) for b in bones]
existing = bpy.data.actions.get(args["clip"])
if existing:
    bpy.data.actions.remove(existing)
rigutil.rest_pose(arm)
action = bpy.data.actions.new(args["clip"])
action.use_fake_user = True
arm.animation_data_create()
arm.animation_data.action = action
frames = [list(values) for values in cycle["frames"]]
if args["loop"]:
    frames[-1] = frames[0]                       # the last key equals the first: a seamless loop
mean_dx = float(np.mean([values[0] for values in frames[:-1]]))
pelvis = args["pelvisBone"]
to_pelvis = rigutil.rest_rotation(arm, pelvis).inverted()
for key, values in enumerate(frames):
    arm.pose.bones[pelvis].location = to_pelvis @ Vector((values[0] - mean_dx, 0.0, values[1]))
    arm.pose.bones[pelvis].keyframe_insert("location", frame=key)
    for name, axis in zip(bones, axes):
        rigutil.set_angle(arm, name, axis, values[index[name]])
        arm.pose.bones[name].keyframe_insert("rotation_quaternion", frame=key)
for fcurve in action_get_channelbag_for_slot(action, arm.animation_data.action_slot).fcurves:
    for point in fcurve.keyframe_points:
        point.interpolation = "LINEAR"           # the keys are dense samples of the cycle already
# Unsampled export writes every action's own keys: forced sampling steps whole frames up to int(frame_range[1]),
# which drops the fractional tail of the other clips once the scene runs at a non-integer cycle rate.
bpy.ops.export_scene.gltf(filepath=args["out"], export_format="GLB", export_yup=True, export_animations=True,
                          export_animation_mode="ACTIONS", export_force_sampling=False, export_cameras=False,
                          export_lights=False)
rigutil.load_glb(args["out"], fps=fps)
rigutil.write_result(args, {"actions": {a.name: [float(a.frame_range[0]), float(a.frame_range[1])]
                                        for a in bpy.data.actions}})
