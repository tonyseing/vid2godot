"""Source-pixel articulation observations, independent assessment and bounded diagnosis.

This is not a motion solver. Only visible, confirmed correspondences constrain the
existing fitter; every recorded observation still has to be resolved for acceptance.
"""
import copy
import math
import re
from pathlib import Path

from anim.ledger import StageError, sha256_file


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(v) for v in value)


def validate(document, project, brief, frame_count, source_sha256=None):
    if not isinstance(document, dict) or document.get("schemaVersion") != 1:
        raise StageError("Source observations need schemaVersion 1")
    source = document.get("source", {})
    if not isinstance(source, dict) or source.get("coordinateSpace") != "registered_source_px":
        raise StageError("Source observations must use registered_source_px coordinates")
    digest = source.get("videoSha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise StageError("Source observations need source videoSha256")
    if source_sha256 is not None and digest != source_sha256:
        raise StageError("Source observations do not match the current source video hash")
    frames = document.get("frames")
    if not isinstance(frames, dict) or not frames:
        raise StageError("Source observations need representative frames")
    fitted = {b for c in brief["fitChains"] for b in project["chains"][c]} - {project["rootBone"]}
    paws = set(project["paws"].values())
    result = copy.deepcopy(document)
    result["frames"] = {}
    for key, frame in frames.items():
        try:
            index = int(key)
        except (TypeError, ValueError, OverflowError):
            raise StageError(f"Invalid source observation frame {key!r}") from None
        if isinstance(key, bool) or str(index) in result["frames"] or not 0 <= index < frame_count:
            raise StageError(f"Invalid or duplicate source observation frame {key!r}")
        if not isinstance(frame, dict) or not isinstance(frame.get("phase"), str) or not frame["phase"].strip():
            raise StageError(f"Observation frame {index} needs a representative phase label")
        observations = frame.get("observations")
        if not isinstance(observations, list) or not observations:
            raise StageError(f"Observation frame {index} needs observations")
        ids = set()
        targets = set()
        for item in observations:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
                raise StageError("Every source observation needs a nonempty id")
            if item["id"] in ids:
                raise StageError(f"Duplicate observation id {item['id']!r} in frame {index}")
            ids.add(item["id"])
            bone, end = item.get("bone"), item.get("end")
            if not ((end == "head" and bone in fitted) or (end == "tail" and bone in paws)):
                raise StageError(f"Observation {item['id']}: unsupported bone/end target")
            if (bone, end) in targets:
                raise StageError(f"Duplicate source target {bone}/{end} in frame {index}")
            targets.add((bone, end))
            if item.get("visibility") not in ("visible", "occluded", "out_of_frame"):
                raise StageError("Observation visibility must be visible, occluded or out_of_frame")
            if item.get("correspondence") not in ("confirmed", "tentative", "ambiguous"):
                raise StageError("Observation correspondence must be confirmed, tentative or ambiguous")
            if item["visibility"] == "visible" or "point" in item:
                if not _point(item.get("point")):
                    raise StageError("Visible observations need a finite numeric coordinate pair")
                sigma = item.get("uncertaintyPx")
                if not _finite(sigma) or sigma <= 0:
                    raise StageError("Visible observations need positive finite uncertaintyPx")
            elif "uncertaintyPx" in item:
                sigma = item["uncertaintyPx"]
                if not _finite(sigma) or sigma <= 0:
                    raise StageError("Observation uncertaintyPx must be positive and finite")
        result["frames"][str(index)] = copy.deepcopy(frame)
    return result


def usable(item):
    return item["visibility"] == "visible" and item["correspondence"] == "confirmed"


def _status(items):
    states = [x["status"] for x in items]
    if not states or "unresolved" in states:
        return "unresolved"
    return "fail" if "fail" in states else "pass"


def assess(document, poses, work_scale=1.0):
    if document is None:
        return {"status": "not_assessed", "frames": [],
                "coverage": {"phases": [], "fullMotionVerified": False}}
    if not _finite(work_scale) or work_scale <= 0:
        raise StageError("Articulation assessment requires positive workScale")
    entries = {f["index"]: f for f in poses["frames"] if f is not None}
    frames = []
    for key, source in sorted(document["frames"].items(), key=lambda kv: int(kv[0])):
        index = int(key)
        landmarks = entries.get(index, {}).get("landmarks", {})
        observations = []
        for item in source["observations"]:
            row = {**copy.deepcopy(item), "status": "unresolved", "projected": None,
                   "residualPx": None, "normalizedResidual": None}
            projected = landmarks.get(item["bone"], {}).get(item["end"])
            if _point(projected):
                row["projected"] = [float(v) / work_scale for v in projected]
            if item.get("point") is not None and row["projected"] is not None:
                row["residualPx"] = math.dist(item["point"], row["projected"])
                row["normalizedResidual"] = row["residualPx"] / item["uncertaintyPx"]
            if not usable(item):
                row["reason"] = "Source visibility or correspondence is unresolved"
            elif row["normalizedResidual"] is None:
                row["reason"] = "Projected landmark is missing or nonfinite"
            else:
                row["status"] = "pass" if row["normalizedResidual"] <= 1.0 + 1e-12 else "fail"
            observations.append(row)
        frames.append({"index": index, "phase": source["phase"], "status": _status(observations),
                       "observations": observations})
    return {"status": _status(frames), "frames": frames,
            "coverage": {"phases": sorted({f["phase"] for f in frames}),
                         "frameIndices": [f["index"] for f in frames],
                         "fullMotionVerified": False,
                         "scope": "Only the recorded source observations; not unobserved phases or joints"}}


def diagnose(assessment, witness_assessment=None, comparisons=None):
    witnesses = {f["index"]: f for f in (witness_assessment or {}).get("frames", [])}
    compared = {f["index"]: f for f in comparisons or []}
    frames = []
    for frame in assessment["frames"]:
        cause = "unresolved"
        evidence = compared.get(frame["index"], {})
        if frame["status"] == "pass":
            cause = "observations_satisfied"
        elif any(not usable(o) for o in frame["observations"]):
            cause = "source_correspondence"
        elif (witnesses.get(frame["index"], {}).get("status") == "pass" and evidence.get("validWitness")
              and _finite(evidence.get("fittedObjective")) and _finite(evidence.get("witnessObjective"))):
            delta = evidence["fittedObjective"] - evidence["witnessObjective"]
            tolerance = 1e-9 * max(1, abs(evidence["fittedObjective"]), abs(evidence["witnessObjective"]))
            if delta > tolerance:
                cause = "search_miss"
            elif delta < -tolerance:
                cause = "objective_rejection"
        frames.append({"index": frame["index"], "cause": cause, "comparison": evidence})
    return {"frames": frames, "limitation": "Finite unsuccessful search is not proof of impossibility"}


def bind(assessment, paths):
    return {**assessment, "bound": {name: {"path": str(Path(p).resolve()), "sha256": sha256_file(p)}
                                   for name, p in paths.items()}}


def require_current(path, expected_paths):
    import json
    path = Path(path)
    if not path.is_file():
        raise StageError("Articulation assessment is missing; run `assess` or refit")
    report = json.loads(path.read_text())
    expected = bind({}, expected_paths)["bound"]
    if report.get("bound") != expected:
        raise StageError("Articulation assessment is stale; run `assess` against current inputs")
    if report.get("status") != "pass":
        raise StageError(f"Articulation assessment is {report.get('status')}; resolve observations and refit")
    return report
