"""Helpers shared by the Blender stage scripts. Runs inside Blender's Python (bpy, mathutils, numpy)."""
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # lets the stage scripts `from anim.camera import SideCamera`

SIDE = Vector((0.0, 1.0, 0.0))     # Blender +Y: the side camera's view axis for a +X-facing creature


def load_args():
    return json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())


def write_result(args, result):
    Path(args["result"]).write_text(json.dumps(result, indent=2) + "\n")


def set_fps(scene, fps):
    scene.render.fps = int(round(fps))
    scene.render.fps_base = scene.render.fps / fps


def load_glb(path, fps=None):
    """Import a GLB into an empty scene. `fps` sets the scene rate first, so import and export agree on time."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if fps:
        set_fps(bpy.context.scene, fps)
    bpy.ops.import_scene.gltf(filepath=str(path))
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    arm = arms[0] if arms else None
    for action in bpy.data.actions:
        action.use_fake_user = True          # keep every clip for re-export
    if arm and arm.animation_data:           # NLA strips would blend into every pose we set
        for track in list(arm.animation_data.nla_tracks):
            arm.animation_data.nla_tracks.remove(track)
    if arm:                                  # the importer adds an Icosphere as every bone's display shape
        shapes = {pb.custom_shape for pb in arm.pose.bones if pb.custom_shape}
        for pose_bone in arm.pose.bones:
            pose_bone.custom_shape = None
        for shape in shapes:
            bpy.data.objects.remove(shape, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    skinned = [m for m in meshes
               if arm and any(md.type == "ARMATURE" and md.object == arm for md in m.modifiers)]
    return arm, skinned, [m for m in meshes if m not in skinned]


def rest_rotation(arm, name):
    return arm.matrix_world.to_3x3() @ arm.data.bones[name].matrix_local.to_3x3()


def side_axis_local(arm, name):
    """The bone-local axis that is world +Y at rest: turning about it keeps the bone in the side plane."""
    return (rest_rotation(arm, name).inverted() @ SIDE).normalized()


def along_side(arm, name):
    return abs(rest_rotation(arm, name).col[1].normalized().dot(SIDE))


def rest_pose(arm):
    if arm.animation_data:
        arm.animation_data.action = None
    for pose_bone in arm.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
    arm.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()


def set_angle(arm, name, axis, angle):
    arm.pose.bones[name].rotation_quaternion = Quaternion(axis, angle)


def evaluated_vertices(objects):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chunks = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        matrix = np.array(evaluated.matrix_world, dtype=np.float64)
        chunks.append(co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3])
        evaluated.to_mesh_clear()
    return np.concatenate(chunks) if chunks else np.zeros((0, 3))


def triangles(objects):
    faces, offset = [], 0
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        tri = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri)
        faces.append(tri.reshape(-1, 3) + offset)
        offset += len(mesh.vertices)
    return np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int32)


def edges(objects):
    pairs, offset = [], 0
    for obj in objects:
        flat = np.empty(len(obj.data.edges) * 2, dtype=np.int32)
        obj.data.edges.foreach_get("vertices", flat)
        pairs.append(flat.reshape(-1, 2) + offset)
        offset += len(obj.data.vertices)
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int32)


def bone_point(arm, name, end="tail"):
    point = arm.matrix_world @ getattr(arm.pose.bones[name], end)
    return [point.x, point.z]


def srgb_to_linear(hex_colour):
    values = [int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values)


def render_setup(camera, background_hex, color_type="TEXTURE"):
    """Workbench render with an exact flat background and an orthographic camera matching `camera`."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO" if color_type == "TEXTURE" else "FLAT"
    scene.display.shading.color_type = color_type
    scene.display.shading.single_color = (0.0, 0.0, 0.0)
    scene.view_settings.view_transform = "Standard"
    scene.render.dither_intensity = 0.0
    scene.render.resolution_x, scene.render.resolution_y = camera.width, camera.height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.world = scene.world or bpy.data.worlds.new("world")
    scene.world.color = srgb_to_linear(background_hex)
    if scene.camera is not None:   # later calls only change the look
        return scene.camera
    data = bpy.data.cameras.new("side")
    data.type, data.ortho_scale, data.clip_end = "ORTHO", camera.ortho_scale, 100.0
    obj = bpy.data.objects.new("side", data)
    scene.collection.objects.link(obj)
    obj.location = (camera.center_x, -20.0, camera.center_z)
    obj.rotation_euler = (np.pi / 2, 0.0, 0.0)
    scene.camera = obj
    return obj


def render(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
