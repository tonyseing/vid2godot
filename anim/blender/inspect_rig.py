"""Record a rigged GLB's facts for the hand-off contract: bones, weights, a 45-degree bend test, forward axis."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import rigutil  # noqa: E402

BEND_DEGREES = 45       # side bend applied to every chain bone, both directions
COLLAPSE_RATIO = 0.1    # an edge shorter than this fraction of its rest length counts as collapsed

args = rigutil.load_args()
arm, skinned, others = rigutil.load_glb(args["model"])
result = {"armature": arm.name if arm else None, "bones": [], "meshes": [], "triangles": 0, "collapse": [],
          "forwardSign": 0, "bounds": None}
for obj in skinned + others:
    obj.data.calc_loop_triangles()
    result["triangles"] += len(obj.data.loop_triangles)
for obj in others:
    result["meshes"].append({"name": obj.name, "skinned": False, "vertices": len(obj.data.vertices),
                             "unweightedVertices": None})
if arm:
    rigutil.rest_pose(arm)
    deform = {bone.name for bone in arm.data.bones if bone.use_deform}
    for bone in arm.data.bones:
        result["bones"].append({"name": bone.name, "parent": bone.parent.name if bone.parent else None,
                                "head": list(arm.matrix_world @ bone.head_local),
                                "tail": list(arm.matrix_world @ bone.tail_local),
                                "alongSide": rigutil.along_side(arm, bone.name)})
    for obj in skinned:
        names = {group.index: group.name for group in obj.vertex_groups}
        unweighted = sum(1 for vertex in obj.data.vertices
                         if sum(g.weight for g in vertex.groups if names.get(g.group) in deform) <= 1e-6)
        result["meshes"].append({"name": obj.name, "skinned": True, "vertices": len(obj.data.vertices),
                                 "unweightedVertices": unweighted})
    rest = rigutil.evaluated_vertices(skinned)
    result["bounds"] = {"min": rest.min(0).tolist(), "max": rest.max(0).tolist()}
    pairs = rigutil.edges(skinned)
    rest_length = np.linalg.norm(rest[pairs[:, 0]] - rest[pairs[:, 1]], axis=1)
    valid = rest_length > 1e-6
    chain_bones = [b for chain in args["chains"].values() for b in chain if b in arm.data.bones]
    for name in dict.fromkeys(chain_bones):
        worst = 0
        for sign in (1, -1):
            rigutil.rest_pose(arm)
            rigutil.set_angle(arm, name, rigutil.side_axis_local(arm, name), sign * math.radians(BEND_DEGREES))
            bpy.context.view_layer.update()
            posed = rigutil.evaluated_vertices(skinned)
            length = np.linalg.norm(posed[pairs[:, 0]] - posed[pairs[:, 1]], axis=1)
            worst = max(worst, int(np.count_nonzero(length[valid] < COLLAPSE_RATIO * rest_length[valid])))
        result["collapse"].append({"bone": name, "edges": worst})
    rigutil.rest_pose(arm)
    bones = arm.data.bones
    if args["headBone"] in bones and args["pelvisBone"] in bones:
        head = (arm.matrix_world @ bones[args["headBone"]].head_local).x
        pelvis = (arm.matrix_world @ bones[args["pelvisBone"]].head_local).x
        result["forwardSign"] = 1 if head > pelvis else -1
rigutil.write_result(args, result)
