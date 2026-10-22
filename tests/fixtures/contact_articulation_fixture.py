"""Test-only redundant leg on the existing generated quadruped, never exported.

Import build_test_rig.py's GLB, insert a 20 cm scapula above its front-left
upper leg, and split its existing paw at the midpoint. The three original
joint heads stay put; only this temporary test scene gets the two extra bones.
Source and witness poses are constructive inputs, not fitted/shipping motion.
All oracle points come from Blender's evaluated pose bones, not solver FK.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anim" / "blender"))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import rigutil  # noqa: E402

args = rigutil.load_args()
arm, _, _ = rigutil.load_glb(args["model"])
rigutil.rest_pose(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bones = arm.data.edit_bones
upper, lower, paw = (bones[name] for name in ("upper_fl", "lower_fl", "paw_fl"))
upper.tail = lower.head
lower.tail = paw.head
scapula = bones.new("scapula_fl")
scapula.head = upper.head + arm.matrix_world.inverted().to_3x3() @ Vector((0.0, 0.0, 0.20))
scapula.tail = upper.head
scapula.parent = upper.parent
upper.parent = scapula
foot = bones.new("foot_fl")
foot.head = paw.head
foot.tail = (paw.head + paw.tail) / 2
foot.parent = lower
paw.head = foot.tail
paw.parent = foot
bpy.ops.object.mode_set(mode="OBJECT")
rigutil.rest_pose(arm)

chain = ["scapula_fl", "upper_fl", "lower_fl", "foot_fl", "paw_fl"]
rig = {"bones": [{"name": bone.name, "parent": bone.parent.name if bone.parent else None,
                   "head": list(arm.matrix_world @ bone.head_local),
                   "tail": list(arm.matrix_world @ bone.tail_local)} for bone in arm.data.bones]}
names = ["dx", "dz"] + [bone["name"] for bone in rig["bones"]]
source = {name: 0.0 for name in names}
source.update(dx=0.07, dz=0.04)
source.update({name: math.radians(angle) for name, angle in zip(chain, (0, 20, -40, 20, 0))})
witness = dict(source)
witness.update({name: math.radians(angle) for name, angle in zip(chain, (30, 20, -40, -10, 0))})


def measure(values):
    rigutil.rest_pose(arm)
    arm.location = (float(values["dx"]), 0.0, float(values["dz"]))
    for name in names[2:]:
        rigutil.set_angle(arm, name, rigutil.side_axis_local(arm, name), float(values[name]))
    bpy.context.view_layer.update()
    evaluated = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return {bone.name: {end: list(evaluated.matrix_world @ getattr(bone, end))
                        for end in ("head", "tail")} for bone in evaluated.pose.bones}


poses = {"source": source, "witness": witness}
poses.update(args.get("poses", {}))
rigutil.write_result(args, {"rig": rig, "names": names, "chain": chain,
                            "values": {label: [pose[name] for name in names] for label, pose in poses.items()},
                            "points": {label: measure(pose) for label, pose in poses.items()}})
