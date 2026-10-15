"""Ask the named video model for two candidate clips of our own model, side-on, and record everything."""
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from anim.ledger import StageError, sha256_file

TEMPLATE = (Path(__file__).resolve().parent / "prompt_template.txt").read_text()
NAMES = {"#c8c8c8": "light grey", "#7f7f7f": "mid grey", "#00b140": "chroma green"}
CHECKS = ("singleAnimal", "noCuts", "sideViewHeld", "noZoom", "legCount", "sameAnimal", "plainBackground")
RESOLUTION_HEIGHT = {"480p": 480, "720p": 720, "1080p": 1080}


def size_for(info, resolution, aspect):
    if resolution not in info.get("supported_resolutions", []) or aspect not in info.get("supported_aspect_ratios", []):
        raise StageError(f"{info['id']} does not support {resolution} at {aspect}")
    a, b = (int(x) for x in aspect.split(":"))
    for size in info.get("supported_sizes", []):
        w, h = (int(x) for x in size.split("x"))
        if min(w, h) == RESOLUTION_HEIGHT[resolution] and abs(w / h - a / b) < 0.02:
            return w, h
    raise StageError(f"{info['id']} lists no {resolution} {aspect} size")


def estimate_cost(info, w, h, seconds, fps=24):
    price = (info.get("pricing_skus") or {}).get("video_tokens")
    return None if price is None else w * h * fps * seconds / 1024 * float(price)


def build_prompt(brief, background):
    name = NAMES.get(background.lower(), "flat colour")
    return TEMPLATE.format(subject=brief["subject"], motion=brief["motion"], gait=brief["gait"],
                           background=f"{name} ({background.lower()})").strip()


def data_url(png):
    return "data:image/png;base64," + base64.b64encode(Path(png).read_bytes()).decode()


def strip(video, out_png, count=8):
    """Eight evenly spaced frames side by side, for the agent to open before accepting a candidate."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=320:-2", f"{tmp}/%04d.png"],
                       check=True)
        frames = sorted(Path(tmp).glob("*.png"))
        picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(count, len(frames)))]
        images = [Image.open(p).convert("RGB") for p in picks]
        sheet = Image.new("RGB", (sum(i.width for i in images), images[0].height))
        x = 0
        for image in images:
            sheet.paste(image, (x, 0))
            x += image.width
        sheet.save(out_png)


def run_generate(client, out_dir, model_id, brief, reference_png, background, candidates=2, seed=1000,
                 max_cost=None, interval=10, timeout=1800, sleep=time.sleep):
    info = client.video_model(model_id)
    if brief["videoSeconds"] not in info.get("supported_durations", []):
        raise StageError(f"{model_id} does not support {brief['videoSeconds']} s clips")
    if "first_frame" not in info.get("supported_frame_images", []):
        raise StageError(f"{model_id} cannot start from our reference image (no first_frame)")
    w, h = size_for(info, brief["resolution"], brief["aspectRatio"])
    each = estimate_cost(info, w, h, brief["videoSeconds"])
    if each is None and max_cost is None:
        raise StageError(f"{model_id} lists no video_tokens price; pass --max-cost to accept an unknown cost")
    estimate = (each or 0) * candidates
    if max_cost is not None and estimate > max_cost:
        raise StageError(f"Estimated ${estimate:.2f} is above --max-cost ${max_cost:.2f}")
    remaining = client.limit_remaining()
    if remaining is not None and estimate * 1.5 > remaining:
        raise StageError(f"Estimated ${estimate:.2f} (x1.5 margin) exceeds the key's remaining ${remaining:.2f}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(brief, background)
    record = {"modelRequested": model_id, "estimateUsd": estimate, "prompt": prompt, "size": [w, h], "candidates": []}
    spent = 0.0
    for index in range(1, candidates + 1):
        payload = {"model": model_id, "prompt": prompt, "duration": brief["videoSeconds"],
                   "resolution": brief["resolution"], "aspect_ratio": brief["aspectRatio"], "generate_audio": True,
                   "seed": seed + index,
                   "frame_images": [{"type": "image_url", "image_url": {"url": data_url(reference_png)},
                                     "frame_type": "first_frame"}]}
        try:
            job = client.submit(payload)
            done = client.wait(job, interval=interval, timeout=timeout, sleep=sleep)
        except StageError as error:
            raise StageError(f"{error} (spent so far: ${spent:.2f}; no candidates recorded)") from error
        spent += float((done.get("usage") or {}).get("cost") or 0)
        video = client.download(done, out_dir / f"candidate-{index}.mp4")
        stats = client.generation(done.get("generation_id")) if done.get("generation_id") else None
        strip(video, out_dir / f"candidate-{index}-strip.png")
        record["candidates"].append({"index": index, "jobId": done["id"], "generationId": done.get("generation_id"),
                                     "modelReturned": (stats or {}).get("model"), "seed": seed + index,
                                     "costUsd": (done.get("usage") or {}).get("cost"), "video": str(video),
                                     "sha256": sha256_file(video), "strip": str(out_dir / f"candidate-{index}-strip.png")})
    record["spentUsd"] = spent
    (out_dir / "record.json").write_text(json.dumps(record, indent=2))
    return record


def accept(record, candidate, checks):
    chosen = next((c for c in record["candidates"] if c["index"] == candidate), None)
    if chosen is None:
        raise StageError(f"No candidate {candidate}")
    failed = [name for name in CHECKS if checks.get(name) is not True]
    if failed:
        raise StageError(f"Candidate {candidate} fails {failed}; generate again or choose the other candidate")
    if not checks.get("reviewer") or not checks.get("notes"):
        raise StageError("Acceptance needs a reviewer and notes on what was inspected")
    return {"candidate": chosen, "checks": checks}
