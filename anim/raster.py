"""Rasterise the fitted model's projected triangles, score them against the video mask, draw review overlays."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

LEFT, RIGHT = (255, 140, 0), (40, 120, 255)


def rasterize(uv, faces, width, height):
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for tri in uv[faces]:
        draw.polygon([(float(x), float(y)) for x, y in tri], fill=1)
    return np.array(image, dtype=bool)


def iou(a, b):
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else np.count_nonzero(a & b) / union


def outline(mask):
    return mask & ~ndimage.binary_erosion(mask)


def overlay(frame_rgb, fitted, video, skeleton=None):
    out = frame_rgb.copy()
    out[outline(video)] = (0, 220, 220)
    out[outline(fitted)] = (230, 0, 160)
    image = Image.fromarray(out)
    draw = ImageDraw.Draw(image)
    for points, colour in skeleton or []:
        draw.line([tuple(map(float, p)) for p in points], fill=colour, width=2)
    return image


def score_fit(poses, projection_npz, masks_npz, threshold, ingest_flags):
    projection, masks = np.load(projection_npz), np.load(masks_npz)["masks"]
    uv, faces = projection["uv"], projection["faces"]
    frames = []
    for entry in poses["frames"]:
        f = entry["index"]
        fitted = rasterize(uv[f], faces, masks.shape[2], masks.shape[1])
        flags = list(entry.get("flags", [])) + list(ingest_flags.get(str(f), []))
        frames.append({"index": f, "iou": round(iou(fitted, masks[f]), 4), "flags": flags})
    low = [item["index"] for item in frames if item["iou"] < threshold or item["flags"]]
    return {"frames": frames, "meanIoU": round(float(np.mean([item["iou"] for item in frames])), 4),
            "lowFrames": low}
