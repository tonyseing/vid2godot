"""Ask the named video model for two candidate clips of our own model, side-on, and record everything."""
import base64
import json
import re
import shutil
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
FPS = 24
# A resolution word in a SKU name (480p, 540p, 2k, 4k ...); a SKU naming another resolution does not price ours.
RESOLUTION_WORD = re.compile(r"\d+[pk]")
# Per-second SKUs that do not price a first-frame clip: other modes, inputs, and fees handled separately.
SKIPPED_SKU_WORDS = ("reference", "continuation", "text_to_video", "megapixel", "video_input", "minimum",
                     "image_input")


def size_for(info, resolution, aspect):
    """The frame size to request. A listed `supported_sizes` must contain one; when the listing gives none
    (null or absent), the size follows from the resolution's short side and the aspect."""
    if resolution not in (info.get("supported_resolutions") or []) or \
            aspect not in (info.get("supported_aspect_ratios") or []):
        raise StageError(f"{info['id']} does not support {resolution} at {aspect}")
    if resolution not in RESOLUTION_HEIGHT:
        raise StageError(f"{info['id']}: no known frame height for {resolution}")
    short = RESOLUTION_HEIGHT[resolution]
    a, b = (int(x) for x in aspect.split(":"))
    sizes = info.get("supported_sizes")
    if sizes is None:
        long = 2 * round(short * max(a, b) / min(a, b) / 2)
        return (long, short) if a >= b else (short, long)
    for size in sizes:
        w, h = (int(x) for x in size.split("x"))
        if min(w, h) == short and abs(w / h - a / b) < 0.02:
            return w, h
    raise StageError(f"{info['id']} lists no {resolution} {aspect} size")


def sends_audio(info):
    """Ask for sound unless the listing says the model cannot make it (`generate_audio: false`)."""
    return info.get("generate_audio") is not False


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _dollars(key, value):
    number = _number(value)
    if number is None:
        return None
    return number / 100 if "cents" in key.lower().split("_") else number


def _per_second(info, resolution, audio):
    """(dollars per second, SKU key) for a first-frame clip at this resolution, or None.

    A SKU naming any other resolution (`_540p`, `_2k`, ...) is skipped, and one naming ours beats a generic one.
    With audio, a with-audio rate is used where one is listed, otherwise the plain rate; a without-audio rate
    never prices a clip with sound, so a listing with only those is unknown. Without audio, a without-audio rate,
    otherwise the plain rate, and then the with-audio rate (as a bound).
    """
    groups = {"with": [], "plain": [], "without": []}
    for key, value in (info.get("pricing_skus") or {}).items():
        k = key.lower()
        if "second" not in k or any(word in k for word in SKIPPED_SKU_WORDS):
            continue
        named = {word for word in k.split("_") if RESOLUTION_WORD.fullmatch(word)}
        if named and resolution.lower() not in named:
            continue
        rate = _dollars(key, value)
        if rate is None:
            continue
        kind = "without" if "without_audio" in k else "with" if "with_audio" in k else "plain"
        groups[kind].append((bool(named), rate, key))
    order = ("with", "plain") if audio else ("without", "plain", "with")
    entries = next((groups[kind] for kind in order if groups[kind]), None)
    if entries is None:
        return None
    specific = [e for e in entries if e[0]] or entries
    best = max(specific, key=lambda e: e[1])
    return best[1], best[2]


def _fees(skus, cost, basis):
    """Add one image-input fee for the first frame, then raise the total to any per-generation minimum."""
    for fee_key, value in skus.items():
        fee = _dollars(fee_key, value) if "image_input" in fee_key.lower() else None
        if fee is not None:
            cost += fee
            basis += f" + {fee_key} ${fee:g} (one first frame)"
    for floor_key, value in skus.items():
        minimum = _dollars(floor_key, value) if "minimum" in floor_key.lower() else None
        if minimum is not None and minimum > cost:
            cost = minimum
            basis += f", raised to {floor_key} ${cost:g}"
    return cost, basis


def price(info, seconds, resolution, aspect, fps=FPS, audio=None, size=None):
    """(estimated dollars for one clip, pricing basis), or (None, why the price is unknown).

    `video_tokens_<resolution>` (else `video_tokens`): w x h x fps x seconds / 1024 x price. Otherwise a
    per-second SKU (keys with `cents` are in cents; see `_per_second`) x seconds. Either way, plus one
    image-input fee for the first frame, raised to any per-generation minimum. `audio` defaults to what
    `generate` sends (`sends_audio`).
    """
    audio = sends_audio(info) if audio is None else audio
    skus = info.get("pricing_skus") or {}
    token_key = next((key for key in (f"video_tokens_{resolution.lower()}", "video_tokens") if key in skus), None)
    if token_key is not None:
        rate = _number(skus[token_key])
        if rate is None:
            return None, f"unknown: {token_key} price {skus[token_key]!r} is not a number"
        if size is None:
            try:
                size = size_for(info, resolution, aspect)
            except StageError:
                return None, f"unknown: {token_key} but no {resolution} {aspect} size to count tokens"
        w, h = size
        return _fees(skus, w * h * fps * seconds / 1024 * rate,
                     f"{token_key}: {w}x{h} x {fps} fps x {seconds} s / 1024 x ${skus[token_key]}")
    per_second = _per_second(info, resolution, audio)
    if per_second is None:
        return None, f"unknown: no video_tokens or per-second SKU for {resolution} ({', '.join(skus) or 'none'})"
    rate, key = per_second
    return _fees(skus, rate * seconds, f"per second: {key} = ${rate:g}/s x {seconds} s")


def _resolution_of(w, h):
    return next((name for name, height in RESOLUTION_HEIGHT.items() if height == min(w, h)), f"{min(w, h)}p")


def estimate_cost(info, w, h, seconds, fps=FPS, resolution=None, audio=None):
    """Estimated dollars for one w x h clip, or None when the listing's pricing is unknown (see `price`)."""
    return price(info, seconds, resolution or _resolution_of(w, h), f"{w}:{h}", fps, audio, size=(w, h))[0]


def build_prompt(brief, background):
    """The strict prompt. Task 20: a brief's slowMotion adds, on the first line, slow motion at that factor with
    every step clearly visible; without it the prompt is as before."""
    name = NAMES.get(background.lower(), "flat colour")
    factor = brief.get("slowMotion")
    slow = (f" Filmed in slow motion: the whole clip plays {factor:g} times slower than real time, with every "
            f"step clearly visible, each paw lifting off and setting down distinctly.") if factor else ""
    return TEMPLATE.format(subject=brief["subject"], motion=brief["motion"], gait=brief["gait"], slow=slow,
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


def cost_of(done):
    """The job's reported cost in dollars, or None when it reported none (or not a number)."""
    cost = (done.get("usage") or {}).get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    return float(cost)


def returned_model(client, done, sleep, warnings, index):
    generation_id = done.get("generation_id")
    for attempt in range(3 if generation_id else 0):
        if attempt:
            sleep(1)
        stats = client.generation(generation_id)
        if isinstance(stats, dict) and stats.get("model"):
            return stats["model"]
    if done.get("model"):
        return done["model"]
    warnings.append(f"Candidate {index}: the model that served it is unknown (recorded as null)")
    return None


def run_generate(client, out_dir, model_id, brief, reference_png, background, candidates=2, seed=1000,
                 max_cost=None, interval=10, timeout=1800, sleep=time.sleep, audio=None):
    """`audio` None asks for sound when the listing allows it; False turns it off (an audio model is sent
    `generate_audio: false`, since leaving it out may keep the provider's default of sound)."""
    if shutil.which("ffmpeg") is None:
        raise StageError("ffmpeg is not installed; it is needed for the candidate strips, so nothing was submitted")
    info = client.video_model(model_id)
    if brief["videoSeconds"] not in (info.get("supported_durations") or []):
        raise StageError(f"{model_id} does not support {brief['videoSeconds']} s clips")
    if "first_frame" not in (info.get("supported_frame_images") or []):
        raise StageError(f"{model_id} cannot start from our reference image (no first_frame)")
    w, h = size_for(info, brief["resolution"], brief["aspectRatio"])
    can_sound = sends_audio(info)
    audio = can_sound if audio is None else bool(audio) and can_sound
    each, basis = price(info, brief["videoSeconds"], brief["resolution"], brief["aspectRatio"], audio=audio,
                        size=(w, h))
    if each is None and max_cost is None:
        raise StageError(f"{model_id} has no price the cost guard can read ({basis}); pass --max-cost to accept "
                         f"an unknown cost")
    estimate = (each or 0) * candidates
    if max_cost is not None and estimate > max_cost:
        raise StageError(f"Estimated ${estimate:.2f} is above --max-cost ${max_cost:.2f}")
    remaining = client.limit_remaining()
    if remaining is not None and estimate * 1.5 > remaining:
        raise StageError(f"Estimated ${estimate:.2f} (x1.5 margin) exceeds the key's remaining ${remaining:.2f}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(brief, background)
    warnings = []
    if each is None:
        warnings.append("first candidate's cost is unbounded (price unknown)")
    record = {"modelRequested": model_id, "estimateUsd": None if each is None else estimate, "pricingBasis": basis, "prompt": prompt,
              "size": [w, h], "generateAudio": audio,
              "candidates": [], "jobs": [], "warnings": warnings}
    spent, observed, unpriced = 0.0, 0.0, []
    for index in range(1, candidates + 1):
        try:
            if index > 1:
                if each is None and unpriced:
                    raise StageError(f"Job {unpriced[0]} reported no cost and the model lists no price, so further "
                                     f"spending cannot be bounded")
                bound = observed if each is None else max(each, observed)
                if max_cost is not None and spent + bound > max_cost:
                    raise StageError(f"Another candidate (about ${bound:.2f}) would take spending past --max-cost "
                                     f"${max_cost:.2f}")
                remaining = client.limit_remaining()
                if remaining is not None and bound * 1.5 > remaining:
                    raise StageError(f"Another candidate (x1.5 margin) exceeds the key's remaining ${remaining:.2f}")
            payload = {"model": model_id, "prompt": prompt, "duration": brief["videoSeconds"],
                       "resolution": brief["resolution"], "aspect_ratio": brief["aspectRatio"],
                       "seed": seed + index,
                       "frame_images": [{"type": "image_url", "image_url": {"url": data_url(reference_png)},
                                         "frame_type": "first_frame"}]}
            if can_sound:
                payload["generate_audio"] = audio
            job = client.submit(payload)
            record["jobs"].append({"index": index, "jobId": job["id"], "pollingUrl": job["polling_url"]})
            done = client.wait(job, interval=interval, timeout=timeout, sleep=sleep)
            cost = cost_of(done)
            if cost is None:
                unpriced.append(job["id"])
                warnings.append(f"job {job['id']} reported no cost")
            else:
                spent += cost
                observed = max(observed, cost)
            video = client.download(done, out_dir / f"candidate-{index}.mp4")
            model_returned = returned_model(client, done, sleep, warnings, index)
            strip_png = out_dir / f"candidate-{index}-strip.png"
            strip(video, strip_png)
            record["candidates"].append({"index": index, "jobId": job["id"], "generationId": done.get("generation_id"),
                                         "modelReturned": model_returned, "seed": seed + index,
                                         "costUsd": (done.get("usage") or {}).get("cost"), "video": str(video),
                                         "sha256": sha256_file(video), "strip": str(strip_png)})
        except Exception as error:
            record["spentUsd"] = spent
            partial = out_dir / "record.partial.json"
            partial.write_text(json.dumps(record, indent=2))
            reason = str(error) if isinstance(error, StageError) else f"{type(error).__name__}: {error}"
            raise StageError(f"{reason} (spent so far: ${spent:.2f}; partial record: {partial})") from error
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
