"""Video in: frames, the real motion rate, the animal's silhouette and its registration to the reference camera."""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from anim.ledger import StageError
from anim.reference import model_mask

FRAME_RATES = (12, 15, 24, 25, 30, 48, 50, 60)


class IngestError(StageError):
    pass


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,avg_frame_rate:format=duration", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    num, den = stream["avg_frame_rate"].split("/")
    return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": float(num) / float(den),
            "duration": float(data["format"]["duration"]),
            "audio": any(s["codec_type"] == "audio" for s in data["streams"])}


def clear_pngs(folder):
    """A re-run writes into the same folder: stale frames from a longer earlier video must not survive."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()


def extract_frames(video, out_dir):
    clear_pngs(out_dir)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-fps_mode", "passthrough",
                    str(out_dir / "%05d.png")], check=True)
    return sorted(out_dir.glob("*.png"))


def frame_difference(a, b, block=16):
    """Largest block-mean difference: a repeated frame stays near zero everywhere, real motion does not."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    h, w = (d.shape[0] // block) * block, (d.shape[1] // block) * block
    return float(d[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3)).max())


def unique_indices(frames, tol=4.0):
    return [0] + [i for i in range(1, len(frames)) if frame_difference(frames[i], frames[i - 1]) > tol]


def snap_rate(rate):
    nearest = min(FRAME_RATES, key=lambda r: abs(r - rate))
    return float(nearest) if abs(nearest - rate) / rate <= 0.03 else rate


def true_rate(n_unique, n_total, container_fps):
    return snap_rate(container_fps * n_unique / n_total)


def cadence(unique, n_total, container_fps):
    """Which frames to keep, their rate and their times (s), from the runs of identical frames.

    A run starts at each unique index and lasts until the next one. Runs longer than twice the median run are holds
    (real stillness), left out of the rate estimate: content rate = container rate * runs / frames over the steady
    runs, snapped. If that equals the container rate (within 3%) the video is native: keep every frame at the container
    rate, held frames included. Otherwise it is pulldown: keep the first frame of each run at the content rate.
    """
    starts = list(unique)
    runs = np.diff(starts + [n_total])
    steady = runs[runs <= 2 * float(np.median(runs))]
    rate = snap_rate(container_fps * len(steady) / float(steady.sum()))
    if abs(rate - container_fps) / container_fps <= 0.03:
        keep, fps = list(range(n_total)), float(container_fps)
    else:
        keep, fps = starts, float(rate)
    return keep, fps, [i / container_fps for i in keep]


def background_colour(img, border=8):
    edge = np.concatenate([img[:border].reshape(-1, 3), img[-border:].reshape(-1, 3),
                           img[:, :border].reshape(-1, 3), img[:, -border:].reshape(-1, 3)])
    return np.median(edge, axis=0)


def silhouette(img, bg, threshold=40.0):
    mask = ndimage.binary_opening(np.linalg.norm(img.astype(float) - bg, axis=2) > threshold,
                                  structure=np.ones((3, 3)))
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return ndimage.binary_fill_holes(labels == int(np.argmax(sizes)) + 1)


def touches_border(mask, margin=2):
    return bool(mask[:margin].any() or mask[-margin:].any() or mask[:, :margin].any() or mask[:, -margin:].any())


def bbox(mask):
    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def registration(ref_mask, mask):
    """Scale by height and pin the ground line and the body centre: video pixel p -> (p - src) * scale + dst."""
    ru0, rv0, ru1, rv1 = bbox(ref_mask)
    mu0, mv0, mu1, mv1 = bbox(mask)
    return {"scale": (rv1 - rv0) / (mv1 - mv0), "src": [(mu0 + mu1) / 2, mv1], "dst": [(ru0 + ru1) / 2, rv1]}


def warp(array, reg, shape, extra=1.0, order=0, cval=0.0):
    s = reg["scale"] * extra
    offset = [reg["src"][1] - reg["dst"][1] / reg["scale"], reg["src"][0] - reg["dst"][0] / reg["scale"]]
    return ndimage.affine_transform(array.astype(float), [1 / s, 1 / s], offset=offset, output_shape=shape,
                                    order=order, cval=cval)


def boundary_points(mask, max_points=600):
    rows, cols = np.nonzero(mask & ~ndimage.binary_erosion(mask))
    pick = np.linspace(0, len(rows) - 1, min(max_points, len(rows))).astype(int)
    return np.stack([cols[pick], rows[pick]], axis=1).astype(np.float32)


def run_ingest(video, ref_mask_png, camera, out_dir, work_scale=0.5, threshold=40.0):
    out_dir = Path(out_dir)
    info = probe(video)
    arrays = [np.array(Image.open(p).convert("RGB")) for p in extract_frames(video, out_dir / "raw")]
    keep, fps, times = cadence(unique_indices(arrays), len(arrays), info["fps"])
    ref_mask = model_mask(ref_mask_png)
    work_shape = (round(ref_mask.shape[0] * work_scale), round(ref_mask.shape[1] * work_scale))
    clear_pngs(out_dir / "registered")
    reg, masks, outside, boundary, offsets, flags = None, [], [], [], [0], {}
    for n, index in enumerate(keep):
        img = arrays[index]
        bg = background_colour(img)
        mask = silhouette(img, bg, threshold)
        if not mask.any():
            raise IngestError(f"No animal found in video frame {index}")
        if reg is None:
            reg = registration(ref_mask, mask)
        if touches_border(mask):
            flags[str(n)] = ["border_touch"]
        full = np.stack([warp(img[..., c], reg, ref_mask.shape, order=1, cval=float(bg[c])) for c in range(3)], -1)
        Image.fromarray(full.clip(0, 255).astype(np.uint8)).save(out_dir / "registered" / f"{n:05d}.png")
        work = warp(mask, reg, work_shape, extra=work_scale) > 0.5
        masks.append(work)
        outside.append(ndimage.distance_transform_edt(~work).astype(np.float32))
        points = boundary_points(work)
        boundary.append(points)
        offsets.append(offsets[-1] + len(points))
    heights = [bbox(m)[3] - bbox(m)[1] for m in masks]
    warnings = []
    if max(heights) / max(min(heights), 1) > 1.3:
        warnings.append(f"Silhouette height varies {min(heights)}-{max(heights)} px: check the video for zoom")
    np.savez_compressed(out_dir / "masks.npz", masks=np.stack(masks), outside=np.stack(outside),
                        boundary=np.concatenate(boundary), offsets=np.array(offsets))
    audio = None
    if info["audio"]:
        audio = out_dir / "audio.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-c:a", "pcm_s16le", str(audio)],
                       check=True)
    meta = {"video": str(video), "container": info, "uniqueIndices": keep, "frames": len(keep), "fps": fps,
            "frameTimes": times,
            "registration": reg, "workScale": work_scale, "camera": camera.scaled(work_scale).to_json(),
            "fullCamera": camera.to_json(), "flags": flags, "warnings": warnings,
            "audio": str(audio) if audio else None}
    (out_dir / "ingest.json").write_text(json.dumps(meta, indent=2))
    return meta


def grid_image(png, out, step=50):
    """A registered frame with a labelled pixel grid, so marks can be read off in reference pixels."""
    image = Image.open(png).convert("RGB")
    draw = ImageDraw.Draw(image)
    for u in range(0, image.width, step):
        draw.line([(u, 0), (u, image.height)], fill=(255, 0, 255) if u % (step * 4) == 0 else (255, 160, 255))
        draw.text((u + 2, 2), str(u), fill=(0, 0, 0))
    for v in range(0, image.height, step):
        draw.line([(0, v), (image.width, v)], fill=(255, 0, 255) if v % (step * 4) == 0 else (255, 160, 255))
        draw.text((2, v + 2), str(v), fill=(0, 0, 0))
    image.save(out)
    return out
