"""Project given poses of the rig to the image: the projected vertices of each pose (in poses.json `names` order:
dx, dz, then one side-axis angle per bone), for facts' check of the averaged cycle against the video masks.

Args: model, camera (the fit's camera, working px), bones, names, values (a list of value vectors), out (npz:
uv (poses, vertices, 2) float32 and faces)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import rigutil  # noqa: E402
from anim.camera import SideCamera  # noqa: E402

args = rigutil.load_args()
arm, skinned, _ = rigutil.load_glb(args["model"])
rigutil.rest_pose(arm)
camera = SideCamera.from_json(args["camera"])
bones, names = args["bones"], args["names"]
index = {name: i for i, name in enumerate(names)}
axes = [rigutil.side_axis_local(arm, b) for b in bones]
uv = []
for values in args["values"]:
    arm.location = (float(values[index["dx"]]), 0.0, float(values[index["dz"]]))
    for name, axis in zip(bones, axes):
        rigutil.set_angle(arm, name, axis, float(values[index[name]]))
    bpy.context.view_layer.update()
    uv.append(camera.project(rigutil.evaluated_vertices(skinned)[:, [0, 2]]).astype(np.float32))
np.savez_compressed(args["out"], uv=np.array(uv, np.float32).reshape(len(uv), -1, 2),
                    faces=rigutil.triangles(skinned))
rigutil.write_result(args, {"poses": len(uv), "out": args["out"]})
