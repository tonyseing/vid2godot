"""The rig's skinned mesh volume at rest (m^3), for facts' rough body mass when the brief gives none.

bmesh's signed volume of each skinned mesh's closed shell; an open or self-overlapping mesh reads only roughly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bmesh  # noqa: E402
import bpy  # noqa: E402
import rigutil  # noqa: E402

args = rigutil.load_args()
arm, skinned, _ = rigutil.load_glb(args["model"])
if arm:
    rigutil.rest_pose(arm)
depsgraph = bpy.context.evaluated_depsgraph_get()
total, meshes = 0.0, []
for obj in skinned:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    shell = bmesh.new()
    shell.from_mesh(mesh)
    shell.transform(obj.matrix_world)
    volume = abs(shell.calc_volume(signed=True))
    open_edges = sum(1 for edge in shell.edges if not edge.is_manifold)
    shell.free()
    evaluated.to_mesh_clear()
    total += volume
    meshes.append({"name": obj.name, "volumeM3": volume, "nonManifoldEdges": open_edges})
rigutil.write_result(args, {"volumeM3": total, "meshes": meshes})
