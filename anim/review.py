"""The review lab (a static page to step through frames) and the hash-bound review gate."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from anim import raster
from anim.facts import in_window, motion_contacts
from anim.ledger import StageError, sha256_file

ARTISTIC_ROUNDS = 2
TEMPLATE = (Path(__file__).resolve().parent / "lab_template.html").read_text()


def draw_overlays(work, poses, projection_npz, masks_npz, out_dir):
    projection, masks = np.load(projection_npz), np.load(masks_npz)["masks"]
    height, width = masks.shape[1:]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for entry in poses["frames"]:
        f = entry["index"]
        frame = Image.open(Path(work) / "ingest" / "registered" / f"{f:05d}.png").convert("RGB").resize((width, height))
        fitted = raster.rasterize(projection["uv"][f], projection["faces"], width, height)
        skeleton = [(points, raster.LEFT if chain.endswith("_L") else raster.RIGHT if chain.endswith("_R")
                     else (255, 255, 255)) for chain, points in entry["joints"].items()]
        path = out_dir / f"{f:05d}.png"
        raster.overlay(np.array(frame), fitted, masks[f], skeleton).save(path)
        paths.append(path)
    return paths


def contact_timeline(poses, project, rest, facts=None):
    """Each paw's frames on the ground, as facts reads them (fix round 1, ruling 8: the lab and facts agree). With
    facts.json: its declared contacts (near the ground AND world-still) at each frame's phase of the cycle (a loop:
    frame f at ((f - start) / period) mod 1) or a one-shot's contact windows. Before facts: facts' rule on the frames,
    each paw's height measured from its rest-pose tail height `rest[chain]` (facts.rest_heights) against the one
    shared ground, and its world stillness against the ground speed."""
    frames = poses["frames"]
    indices = [int(f.get("index", i)) for i, f in enumerate(frames)]
    entry = (facts or {}).get("entry") or {}
    if entry.get("loop") and isinstance(entry.get("contacts"), dict):
        phases = ((np.array(indices, float) - float(facts["start"])) / float(facts["period"])) % 1.0
        return {bone: (in_window(phases, entry["contacts"][bone]).tolist() if bone in entry["contacts"]
                       else [False] * len(frames)) for bone in project["paws"].values()}
    if entry and entry.get("loop") is False and isinstance(entry.get("contact_windows"), dict):
        last = max(1, len(frames) - 1)
        at = np.arange(len(frames)) / last
        out = {}
        for bone in project["paws"].values():
            on = np.zeros(len(frames), bool)
            for start, end in entry["contact_windows"].get(bone) or []:
                on |= (at >= start - 1e-9) & ((at < end - 1e-9) | ((end >= 1.0) & (at >= 1.0 - 1e-9)))
            out[bone] = on.tolist()
        return out
    fps = float(poses.get("fps") or 24.0)
    heights = {chain: np.array([f["paws"][chain][1] for f in frames]) - rest[chain] for chain in project["paws"]}
    vx = {chain: np.gradient(np.array([f["paws"][chain][0] for f in frames], float)) * fps
          for chain in project["paws"]}
    masks = motion_contacts(heights, vx)[3]
    return {bone: masks[chain].tolist() for chain, bone in project["paws"].items()}


def motion_summary(facts):
    """What the lab states about the baked motion (Task 19), from facts.json: the cadence and any retime factor,
    the gait check (and an accepted mismatch's reason), the contacts and their check against the baked keys (with the
    edge velocity jumps), and every warning (fix round 1: an unchecked gait, an open-mesh mass, a world pitch range)."""
    if not facts:
        return None
    return {"cadence": facts.get("cadence"), "gait": facts.get("gaitCheck"), "contactCheck": facts.get("contactCheck"),
            "contacts": (facts.get("entry") or {}).get("contacts"), "warnings": facts.get("warnings") or []}


def lab_page(clip, report, threshold, timeline, hashes, motion=None):
    data = {"clip": clip, "report": report, "threshold": threshold, "timeline": timeline, "hashes": hashes}
    if motion is not None:
        data["motion"] = motion
    # No "<" reaches the page, so a "</script>" or "<!--<script" in a clip name or flag cannot end the inline
    # script or start an HTML comment in it; JSON reads "\u003c" back as "<".
    return TEMPLATE.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))


def make_template(clip, bound, report, round_number):
    by_index = {f["index"]: f for f in report["frames"]}
    return {"schemaVersion": 1, "clip": clip,
            "bound": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in bound.items()},
            "reviewer": "", "verdict": "", "inspectedReference": False, "inspectedOverlays": False,
            "inspectedSheet": False, "observations": "", "artisticRound": round_number,
            "lowFrames": [{"frame": i, "iou": by_index[i]["iou"], "flags": by_index[i]["flags"], "action": "",
                           "note": ""} for i in report["lowFrames"]]}


def _is_frame_item(item):
    return isinstance(item, dict) and isinstance(item.get("frame"), int) and not isinstance(item["frame"], bool)


def check_template(template):
    """A template is written by `review --template`; a broken one is regenerated, not reviewed."""
    regenerate = "; regenerate it with `review --template`"
    if not isinstance(template, dict):
        raise StageError("The review template is not a JSON object" + regenerate)
    missing = [key for key in ("clip", "bound", "artisticRound", "lowFrames") if key not in template]
    if missing:
        raise StageError(f"The review template lacks {missing}" + regenerate)
    bound = template["bound"]
    if not isinstance(bound, dict) or not all(isinstance(item, dict) and isinstance(item.get("path"), str)
                                              and isinstance(item.get("sha256"), str) for item in bound.values()):
        raise StageError("The review template's bound files need a path and a sha256 each" + regenerate)
    if not isinstance(template["artisticRound"], int) or isinstance(template["artisticRound"], bool):
        raise StageError("The review template's artisticRound is not a number" + regenerate)
    if not isinstance(template["lowFrames"], list) or not all(map(_is_frame_item, template["lowFrames"])):
        raise StageError("Every low frame in the review template needs a frame number" + regenerate)


def validate(review, template):
    check_template(template)
    if not isinstance(review, dict):
        return ["The review is not a JSON object; fill in a copy of the template"]
    problems = []
    if review.get("clip") != template["clip"]:
        problems.append(f"The review is for clip {review.get('clip')!r}, not {template['clip']!r}")
    if review.get("artisticRound") != template["artisticRound"]:
        problems.append(f"The review is for round {review.get('artisticRound')!r}, but the template is round "
                        f"{template['artisticRound']}; fill in the current template")
    if review.get("bound") != template["bound"]:
        problems.append("The review's bound hashes differ from the template; regenerate it with `review --template`")
    for name, item in template["bound"].items():
        if not Path(item["path"]).is_file() or sha256_file(item["path"]) != item["sha256"]:
            problems.append(f"{name} changed after the template was made; review the new output")
    if not review.get("reviewer"):
        problems.append("reviewer is empty")
    if review.get("verdict") not in ("pass", "fail"):
        problems.append('verdict must be "pass" or "fail"')
    items = review.get("lowFrames", [])
    if not isinstance(items, list):
        problems.append("lowFrames must be the template's list of low frames")
        items = []
    if not all(map(_is_frame_item, items)):
        problems.append("Every lowFrames item needs its frame number, as in the template")
    items = [item for item in items if _is_frame_item(item)]
    listed = {item["frame"] for item in items}
    missing = [item["frame"] for item in template["lowFrames"] if item["frame"] not in listed]
    if missing:
        problems.append(f"Low or flagged frames missing from the review: {missing}")
    if review.get("verdict") == "pass":
        for key in ("inspectedReference", "inspectedOverlays", "inspectedSheet"):
            if review.get(key) is not True:
                problems.append(f"{key} must be true for a pass")
        if not review.get("observations"):
            problems.append("observations are empty")
        for item in items:
            if item.get("action") != "explained" or not item.get("note"):
                problems.append(f"Frame {item['frame']} needs an explanation, or a re-mark and refit")
    return problems


def record_round(history, template, filled):
    """Add a validated review to the history: one entry per artistic round. Returns the new history and the number
    of distinct rounds that failed."""
    if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
        raise StageError("review/history.json is not a list of rounds")
    round_number = template["artisticRound"]
    if any(item.get("round") == round_number for item in history):
        raise StageError(f"Round {round_number} was already reviewed; make a new template with `review --template` "
                         "after revising")
    history = history + [{"round": round_number, "verdict": filled["verdict"], "reviewer": filled["reviewer"],
                          "observations": filled.get("observations", "")}]
    failed = len({item.get("round") for item in history if item.get("verdict") == "fail"})
    return history, failed
