"""Source-pixel visible-contour constraints, without anatomical correspondence.

Bounds are [left, top, right, bottom], with exclusive right/bottom. At working
resolution the crop expands outward (floor left/top, ceil right/bottom), and the
identical crop is measured in both masks. Hidden or empty source regions cannot
certify a fit; unlike the global raster IoU, empty/empty is not a local success.
"""
import copy
import math
import re

import numpy as np

from anim.ledger import StageError


_VISIBILITIES = ("visible", "occluded", "unknown", "out_of_frame")


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _regions(regions, width=None, height=None):
    if not isinstance(regions, list):
        raise StageError("Silhouette regions must be a list")
    ids = set()
    for item in regions:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise StageError("Every silhouette region needs a nonempty id")
        if item["id"] in ids:
            raise StageError(f"Duplicate silhouette region id {item['id']!r}")
        ids.add(item["id"])
        if item.get("visibility") not in _VISIBILITIES:
            raise StageError("Silhouette region visibility must be visible, occluded, unknown or out_of_frame")
        threshold, weight = item.get("minIoU"), item.get("weight")
        if not _finite(threshold) or not 0 < threshold <= 1:
            raise StageError("Silhouette region minIoU must be finite and in (0, 1]")
        if not _finite(weight) or weight <= 0:
            raise StageError("Silhouette region weight must be positive and finite")
        bounds = item.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or not all(_finite(v) for v in bounds):
            raise StageError("Silhouette region bounds need four finite numeric source coordinates")
        left, top, right, bottom = bounds
        if not (0 <= left < right and 0 <= top < bottom):
            raise StageError("Silhouette region bounds must be nonnegative and strictly ordered")
        if (width is not None and right > width) or (height is not None and bottom > height):
            raise StageError("Silhouette region bounds extend outside the source image")


def validate(document, video_sha256, width, height, frame_count):
    """Validate provenance and source bounds; normalize frame keys to integers.

    A document must contain at least one region. Nonvisible regions may record
    unresolved evidence but cannot become acceptance evidence in ``evaluate``.
    """
    if not isinstance(document, dict) or type(document.get("schemaVersion")) is not int or document["schemaVersion"] != 1:
        raise StageError("Silhouette regions need schemaVersion 1")
    for name, value in (("width", width), ("height", height), ("frame_count", frame_count)):
        if type(value) is not int or value <= 0:
            raise StageError(f"Silhouette regions need positive integer {name}")
    if not isinstance(video_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", video_sha256):
        raise StageError("Silhouette regions need the current source video SHA-256")
    source = document.get("source")
    if not isinstance(source, dict) or source.get("coordinateSpace") != "registered_source_px":
        raise StageError("Silhouette regions must use registered_source_px coordinates")
    if source.get("videoSha256") != video_sha256:
        raise StageError("Silhouette regions do not match the current source video hash")
    frames = document.get("frames")
    if not isinstance(frames, dict) or not frames:
        raise StageError("Silhouette regions need observed frames")
    result = copy.deepcopy(document)
    result["frames"] = {}
    for key, items in frames.items():
        if type(key) is int:
            index = key
        elif isinstance(key, str) and re.fullmatch(r"[0-9]+", key):
            try:
                index = int(key)
            except ValueError:
                raise StageError(f"Invalid silhouette region frame {key!r}") from None
        else:
            raise StageError(f"Invalid silhouette region frame {key!r}")
        if not 0 <= index < frame_count or index in result["frames"]:
            raise StageError(f"Invalid or duplicate silhouette region frame {key!r}")
        _regions(items, width, height)
        if not items:
            raise StageError(f"Silhouette region frame {index} needs regions")
        result["frames"][index] = copy.deepcopy(items)
    return result


def _mask(value):
    try:
        array = np.asarray(value)
    except (ValueError, TypeError):
        raise StageError("Silhouette measurement needs rectangular binary masks") from None
    if array.ndim != 2 or not all(array.shape):
        raise StageError("Silhouette measurement needs nonempty two-dimensional masks")
    if array.dtype == np.bool_:
        return array
    if array.dtype.kind not in "iuf" or not np.all((array == 0) | (array == 1)):
        raise StageError("Silhouette measurement needs finite binary masks")
    return array.astype(bool)


def evaluate(source_mask, candidate_mask, regions, work_scale=1.0):
    """Measure local IoU and weighted mean ``100 * (1 - localIoU)`` loss.

    Unresolved regions have no IoU and a loss of 100, never a passing status.
    Empty candidate foreground against a nonempty source is a measurable failure.
    No regions means zero additional loss and no local acceptance obligations.
    """
    if not _finite(work_scale) or work_scale <= 0:
        raise StageError("Silhouette measurement needs positive finite work_scale")
    source, candidate = _mask(source_mask), _mask(candidate_mask)
    if source.shape != candidate.shape:
        raise StageError("Silhouette measurement masks must have identical working resolution")
    _regions(regions)
    if not regions:
        return {"loss": 0.0, "pass": True, "regions": []}
    height, width = source.shape
    rows, penalties = [], []
    for item in regions:
        scaled = [v * work_scale for v in item["bounds"]]
        if not all(_finite(v) for v in scaled):
            raise StageError("Silhouette region scaled bounds must be finite")
        left, top, right, bottom = scaled
        # Ingest rounds the source dimensions. Only that half-pixel rounding
        # margin may be clipped, never a materially out-of-bounds source region.
        if right > width + .5 or bottom > height + .5:
            raise StageError("Silhouette region bounds extend outside the working image")
        x0, y0 = math.floor(left), math.floor(top)
        x1, y1 = min(width, math.ceil(right)), min(height, math.ceil(bottom))
        reference, fitted = source[y0:y1, x0:x1], candidate[y0:y1, x0:x1]
        source_pixels, candidate_pixels = int(np.count_nonzero(reference)), int(np.count_nonzero(fitted))
        row = {"id": item["id"], "bounds": list(item["bounds"]), "visibility": item["visibility"],
               "minIoU": item["minIoU"], "weight": item["weight"], "workBounds": [x0, y0, x1, y1],
               "sourcePixels": source_pixels, "candidatePixels": candidate_pixels,
               "iou": None, "status": "unresolved"}
        penalty = 100.0
        if item["visibility"] != "visible":
            row["reason"] = "Source region is not visibly observed"
        elif not reference.size or not source_pixels:
            row["reason"] = "Source region has no measurable foreground"
        else:
            intersection = int(np.count_nonzero(reference & fitted))
            union = source_pixels + candidate_pixels - intersection
            local_iou = intersection / union
            row["iou"] = local_iou
            row["status"] = "pass" if local_iou >= item["minIoU"] else "fail"
            penalty = 100.0 * (1.0 - local_iou)
        rows.append(row)
        penalties.append(penalty)
    # Scaling the weights avoids overflow for otherwise-valid finite weights.
    max_weight = max(item["weight"] for item in regions)
    weights = [item["weight"] / max_weight for item in regions]
    loss = math.fsum(w * p for w, p in zip(weights, penalties)) / math.fsum(weights)
    return {"loss": loss, "pass": all(row["status"] == "pass" for row in rows), "regions": rows}
