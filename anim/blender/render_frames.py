"""Render chosen frames of one clip with a given side camera; also report projected bone points.

`fps` (optional) is the scene rate to import at: a clip baked at a non-integer cycle rate keys frame k at k / fps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import rigutil  # noqa: E402
from anim.camera import SideCamera  # noqa: E402

args = rigutil.load_args()
out = Path(args["outDir"])
out.mkdir(parents=True, exist_ok=True)
arm, skinned, others = rigutil.load_glb(args["model"], fps=args.get("fps"))
for obj in others:
    obj.hide_render = True
camera = SideCamera.from_json(args["camera"])
rigutil.rest_pose(arm)
rigutil.render_setup(camera, args["background"], args.get("colorType", "TEXTURE"))
action = bpy.data.actions[args["clip"]]
arm.animation_data_create()
arm.animation_data.action = action
if getattr(action, "slots", None):
    arm.animation_data.action_slot = action.slots[0]
start = int(round(action.frame_range[0]))
paths, points = [], {}
for offset in args["frames"]:
    bpy.context.scene.frame_set(start + offset)
    path = out / f"{offset:05d}.png"
    rigutil.render(path)
    paths.append(str(path))
    points[str(offset)] = {bone: camera.project(np.array(rigutil.bone_point(arm, bone))).tolist()
                           for bone in args.get("bones", [])}
rigutil.write_result(args, {"frames": paths, "points": points, "actionStart": start})
