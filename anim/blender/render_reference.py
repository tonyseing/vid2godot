"""Render the rigged model at rest, exactly side-on: an exact mask and one textured render per background."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigutil  # noqa: E402
from anim.camera import SideCamera  # noqa: E402

args = rigutil.load_args()
out = Path(args["outDir"])
arm, skinned, others = rigutil.load_glb(args["model"])
for obj in others:
    obj.hide_render = True
rigutil.rest_pose(arm)
verts = rigutil.evaluated_vertices(skinned)
camera = SideCamera.framing(float(verts[:, 0].min()), float(verts[:, 0].max()), 0.0, float(verts[:, 2].max()))
rigutil.render_setup(camera, "#ffffff", "SINGLE")
rigutil.render(out / "mask.png")
renders = {}
for colour in args["backgrounds"]:
    rigutil.render_setup(camera, colour, "TEXTURE")
    renders[colour] = str(out / f"reference-{colour[1:]}.png")
    rigutil.render(renders[colour])
rigutil.write_result(args, {"camera": camera.to_json(), "mask": str(out / "mask.png"), "renders": renders})
