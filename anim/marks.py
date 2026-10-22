"""Manual image landmarks: paw tails in frames, fitted bone heads in jointFrames."""
import math

from anim.ledger import StageError


def normalize(document):
    """Canonical frame keys for fitter/lab lookup, leaving the hash-bound source document untouched."""
    if not isinstance(document, dict):
        raise StageError("Marks must be a JSON object with frames and optional jointFrames")
    normalized = dict(document)
    for field in ("frames", "jointFrames"):
        frames = document.get(field, {})
        if not isinstance(frames, dict):
            raise StageError(f"{field} must map frame numbers to bone coordinates")
        normalized[field] = {}
        for frame, points in frames.items():
            try:
                index = int(frame)
            except (TypeError, ValueError, OverflowError):
                raise StageError(f"{field}: invalid frame {frame!r}") from None
            if isinstance(frame, bool):
                raise StageError(f"{field}: invalid frame {frame!r}")
            key = str(index)
            if key in normalized[field]:
                raise StageError(f"{field}: duplicate ingested frame {index} via frame key {frame!r}")
            normalized[field][key] = points
    return normalized


def validate(document, project, brief, frame_count):
    """Validate registered-image pixel coordinates before scaling them for the working masks."""
    normalized = normalize(document)
    fitted = {b for chain in brief["fitChains"] for b in project["chains"][chain]} - {project["rootBone"]}
    allowed = {"frames": set(project["paws"].values()), "jointFrames": fitted}
    for field, bones in allowed.items():
        frames = normalized[field]
        for frame, points in frames.items():
            index = int(frame)
            if not 0 <= index < frame_count:
                raise StageError(f"{field}: frame {frame!r} must be an ingested frame in 0:{frame_count}")
            if not isinstance(points, dict):
                raise StageError(f"{field}: frame {frame} must map bone names to coordinate pairs")
            for bone, point in points.items():
                if bone not in bones:
                    kind = "fitted non-root chain bone" if field == "jointFrames" else "contract paw bone"
                    raise StageError(f"{field}: bone {bone!r} is not a {kind}")
                if not isinstance(point, (list, tuple)) or len(point) != 2 or any(
                        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
                    raise StageError(f"{field}: frame {frame} bone {bone!r} needs a finite numeric coordinate pair")
    return normalized["frames"], normalized["jointFrames"]
