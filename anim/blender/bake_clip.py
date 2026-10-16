"""Write the fitted cycle as an in-place clip on the rig and export an evaluation GLB (other clips kept).

`cycle` is facts.json's cycle block (or a path to facts.json): n + 1 value vectors in poses.json `names` order,
resampled at sub-frame precision, played at `cycle.fps`, and a loop's `drift` (the root's dx, dz travel over one
period). Key k goes at frame k, so the clip lasts n / cycle.fps."""
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
n = len(frames) - 1
dx = np.array([values[0] for values in frames], float)
dz = np.array([values[1] for values in frames], float)
if args["loop"]:
    # In place without a seam pop: lose the root's drift over the period (facts' cycle.drift, the travel from key 0
    # to one period on) as drift * k / n at every key, so a periodic sway is untouched; then lose dx's mean over keys
    # 0..n-1; key n then repeats key 0 exactly.
    drift = cycle["drift"]
    ramp = np.arange(n + 1, dtype=float) / n
    dx = dx - float(drift["dx"]) * ramp
    dz = dz - float(drift["dz"]) * ramp
    dx = dx - dx[:n].mean()
    dx[n], dz[n] = dx[0], dz[0]
    frames[-1] = list(frames[0])                 # the last key's bones equal the first: a seamless loop
else:
    dx = dx - dx[:n].mean()                      # a one-shot loses only its mean dx
pelvis = args["pelvisBone"]
to_pelvis = rigutil.rest_rotation(arm, pelvis).inverted()      # world (Blender: +X forward, +Z up) to pelvis-local
for key, values in enumerate(frames):
    arm.pose.bones[pelvis].location = to_pelvis @ Vector((float(dx[key]), 0.0, float(dz[key])))
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
arm, _, _ = rigutil.load_glb(args["out"], fps=fps)
# Task 19: the re-imported clip's paw tails at every key (world x, z), for the bake's check against facts' IK pins.
paws = {}
if args.get("paws"):
    clip = bpy.data.actions[args["clip"]]
    arm.animation_data_create()
    arm.animation_data.action = clip
    if getattr(clip, "slots", None):
        arm.animation_data.action_slot = clip.slots[0]
    start = int(round(clip.frame_range[0]))
    for bone in args["paws"]:
        paws[bone] = []
    for key in range(n + 1):
        bpy.context.scene.frame_set(start + key)
        for bone in args["paws"]:
            paws[bone].append(rigutil.bone_point(arm, bone))
rigutil.write_result(args, {"actions": {a.name: [float(a.frame_range[0]), float(a.frame_range[1])]
                                        for a in bpy.data.actions}, "paws": paws})
