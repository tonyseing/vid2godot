"""Test probe: re-import a GLB and report, per key of one clip, the pelvis head's world displacement from rest.

Args: {model, fps, clip, bone, keys}. Result: {"displacement": [[x, y, z] per key 0..keys]} in Blender world axes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anim" / "blender"))
import bpy  # noqa: E402
import rigutil  # noqa: E402

args = rigutil.load_args()
arm, _, _ = rigutil.load_glb(args["model"], fps=args["fps"])
rigutil.rest_pose(arm)
rest = arm.matrix_world @ arm.pose.bones[args["bone"]].head
action = bpy.data.actions[args["clip"]]
arm.animation_data_create()
arm.animation_data.action = action
if getattr(action, "slots", None):
    arm.animation_data.action_slot = action.slots[0]
start = int(round(action.frame_range[0]))
moves = []
for key in range(args["keys"] + 1):
    bpy.context.scene.frame_set(start + key)
    head = arm.matrix_world @ arm.pose.bones[args["bone"]].head
    moves.append(list(head - rest))
rigutil.write_result(args, {"displacement": moves})
