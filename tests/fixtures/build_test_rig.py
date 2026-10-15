"""Test fixture: a box quadruped with rigid skinning and a known 4-beat walk (24 frames at 24 fps)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anim" / "blender"))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import rigutil  # noqa: E402

args = rigutil.load_args()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
LEGS = {"fl": (0.25, 0.08, "chest"), "fr": (0.25, -0.08, "chest"),
        "hl": (-0.25, 0.08, "pelvis"), "hr": (-0.25, -0.08, "pelvis")}
PHASES = {"hl": 0.0, "fl": 0.25, "hr": 0.5, "fr": 0.75}

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

mesh = bpy.data.meshes.new("body")
body = bpy.data.objects.new("body", mesh)
scene.collection.objects.link(body)
coords, faces, indices = [], [], []


def box(center, size, group, cuts=8):
    """A box with every edge cut `cuts` times, so its outline carries vertices (the fitter's chamfer reads vertices,
    as it would on a real, denser model)."""
    bm = bmesh.new()
    verts = bmesh.ops.create_cube(bm, size=1.0)["verts"]
    for vert in verts:
        vert.co = Vector(center) + Vector((vert.co.x * size[0], vert.co.y * size[1], vert.co.z * size[2]))
    bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=cuts, use_grid_fill=True)
    bm.verts.index_update()
    first = len(coords)
    coords.extend(tuple(v.co) for v in bm.verts)
    faces.extend([first + v.index for v in face.verts] for face in bm.faces)
    indices.append((group, list(range(first, len(coords)))))
    bm.free()


# The parts overlap so the side silhouette is one connected shape, like a real animal's: ingest keeps only the
# largest connected silhouette component, so a detached head or paw would vanish from the "video".
box((-0.2, 0, 0.55), (0.2, 0.16, 0.18), "pelvis")
box((0.1, 0, 0.55), (0.4, 0.18, 0.2), "chest")
box((0.39, 0, 0.62), (0.2, 0.1, 0.1), "head")
for key, (x, y, _) in LEGS.items():
    box((x, y, 0.38), (0.05, 0.04, 0.24), f"upper_{key}")
    box((x, y, 0.17), (0.04, 0.035, 0.22), f"lower_{key}")
    box((x + 0.03, y, 0.04), (0.08, 0.04, 0.06), f"paw_{key}")
mesh.from_pydata(coords, [], faces)
mesh.update()
for group, members in indices:
    body.vertex_groups.new(name=group).add(members, 1.0, "REPLACE")
material = bpy.data.materials.new("coat")
material.diffuse_color = (0.12, 0.08, 0.06, 1.0)
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.12, 0.08, 0.06, 1.0)
mesh.materials.append(material)
body.parent = arm
body.modifiers.new("rig", "ARMATURE").object = arm

scene.render.fps = 24
rigutil.rest_pose(arm)
arm.animation_data_create()
arm.animation_data.action = bpy.data.actions.new("walk")
truth = {}
for frame in range(25):
    t = frame / 24
    truth[frame] = {}
    for key, phase in PHASES.items():
        p = 2 * math.pi * (t + phase)
        for name, angle in ((f"upper_{key}", math.radians(25) * math.sin(p)),
                            (f"lower_{key}", math.radians(15) * math.sin(p + 0.9))):
            rigutil.set_angle(arm, name, rigutil.side_axis_local(arm, name), angle)
            arm.pose.bones[name].keyframe_insert("rotation_quaternion", frame=frame)
            truth[frame][name] = angle
bpy.ops.export_scene.gltf(filepath=args["out"], export_format="GLB", export_yup=True, export_animations=True,
                          export_animation_mode="ACTIONS", export_force_sampling=True)
Path(args["truth"]).write_text(json.dumps(truth))
rigutil.write_result(args, {"glb": args["out"]})
