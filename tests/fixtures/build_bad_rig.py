"""Test fixture: the good rig's skeleton with one connected triangle across the lower_fl joint, weighted so a
45-degree side bend of lower_fl folds one edge flat (parent-weighted vertex vs child-weighted vertex)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anim" / "blender"))
import bpy  # noqa: E402
import rigutil  # noqa: E402

args = rigutil.load_args()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
LEGS = {"fl": (0.25, 0.08, "chest"), "fr": (0.25, -0.08, "chest"),
        "hl": (-0.25, 0.08, "pelvis"), "hr": (-0.25, -0.08, "pelvis")}

arm_data = bpy.data.armatures.new("rig")
arm = bpy.data.objects.new("rig", arm_data)
scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")


def bone(name, head, tail, parent=None):
    edit = arm_data.edit_bones.new(name)
    edit.head, edit.tail = head, tail
    if parent:
        edit.parent = arm_data.edit_bones[parent]


bone("root", (0, 0, 0), (0.1, 0, 0))
bone("pelvis", (-0.3, 0, 0.55), (-0.1, 0, 0.55), "root")
bone("chest", (-0.1, 0, 0.55), (0.3, 0, 0.55), "pelvis")
bone("head", (0.3, 0, 0.6), (0.5, 0, 0.65), "chest")
for key, (x, y, parent) in LEGS.items():
    bone(f"upper_{key}", (x, y, 0.5), (x, y, 0.28), parent)
    bone(f"lower_{key}", (x, y, 0.28), (x, y, 0.06), f"upper_{key}")
    bone(f"paw_{key}", (x, y, 0.06), (x + 0.06, y, 0.0), f"lower_{key}")
bpy.ops.object.mode_set(mode="OBJECT")

# Joint of lower_fl at (0.25, 0.08, 0.28). `below` hangs straight under it on the child; `above` sits 45 degrees
# off that line on the parent, so bending the child by 45 degrees lands `below` on top of `above`.
x, y, z = 0.25, 0.08, 0.28
r = 0.1
below = (x, y, z - r)
above = (x + r * math.sin(math.radians(45)), y, z - r * math.cos(math.radians(45)))
beside = (above[0], y + 0.03, above[2])
mesh = bpy.data.meshes.new("body")
mesh.from_pydata([above, below, beside], [], [(0, 1, 2)])
body = bpy.data.objects.new("body", mesh)
scene.collection.objects.link(body)
body.vertex_groups.new(name="upper_fl").add([0, 2], 1.0, "REPLACE")
body.vertex_groups.new(name="lower_fl").add([1], 1.0, "REPLACE")
body.parent = arm
body.modifiers.new("rig", "ARMATURE").object = arm
bpy.ops.export_scene.gltf(filepath=args["out"], export_format="GLB", export_yup=True)
rigutil.write_result(args, {"glb": args["out"]})
