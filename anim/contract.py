"""The hand-off from game-asset-3d-pipeline: load the project's contract and check a rigged GLB against it."""
import json
from pathlib import Path

from anim.ledger import StageError

REDIRECT = "Fix it in game-asset-3d-pipeline (its interaction stage owns the rig), then rerun `contract`."
KEYS = ("schemaVersion", "forward", "rootBone", "pelvisBone", "headBone", "requiredBones", "chains", "legPairs",
        "paws", "sheet")


class ContractError(StageError):
    pass


def load_project_contract(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ContractError(f"Cannot read project contract {path}: {error}") from error
    missing = [key for key in KEYS if key not in data]
    if missing:
        raise ContractError(f"Project contract {path} lacks {missing}")
    if data["schemaVersion"] != 1:
        raise ContractError("Project contract schemaVersion must be 1")
    if data["forward"] != "+X":
        raise ContractError("v1 supports +X-forward creatures only (the side camera looks along Blender +Y)")
    for pair in data["legPairs"]:
        for chain in pair:
            if chain not in data["chains"] or chain not in data["paws"]:
                raise ContractError(f"legPairs names unknown chain {chain!r}")
    if not {"background", "pxPerMetre"} <= set(data["sheet"]):
        raise ContractError("sheet needs background and pxPerMetre")
    return data


def check_rig(rig, project):
    problems, warnings = [], []
    if not rig.get("armature"):
        return [f"The model has no armature. {REDIRECT}"], warnings
    by_name = {bone["name"]: bone for bone in rig["bones"]}
    missing = [name for name in project["requiredBones"] if name not in by_name]
    if missing:
        problems.append(f"Missing required bones {missing}. {REDIRECT}")
    for chain, bones in project["chains"].items():
        for name in bones:
            if name in by_name and by_name[name]["alongSide"] > 0.94:
                problems.append(f"Bone {name} ({chain}) points sideways, so it has no side-plane hinge. {REDIRECT}")
    for mesh in rig["meshes"]:
        if not mesh["skinned"]:
            warnings.append(f"Mesh {mesh['name']} is not skinned to the armature; silhouettes and renders leave it out.")
        elif mesh["unweightedVertices"]:
            problems.append(f"Mesh {mesh['name']} has {mesh['unweightedVertices']} vertices with no deforming "
                            f"weight. {REDIRECT}")
    for item in rig["collapse"]:
        if item["edges"] > 0:
            problems.append(f"Joint collapse at {item['bone']}: {item['edges']} edges shrink below 10% under a "
                            f"45-degree side bend. {REDIRECT}")
    if rig["forwardSign"] != 1:
        problems.append(f"The head is not toward +X of the pelvis; the creature must face +X. {REDIRECT}")
    return problems, warnings
