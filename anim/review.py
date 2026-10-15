"""The review lab (a static page to step through frames) and the hash-bound review gate."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from anim import raster
from anim.facts import contact_mask
from anim.ledger import sha256_file

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


def contact_timeline(poses, project):
    return {bone: contact_mask(np.array([f["paws"][chain][1] for f in poses["frames"]])).tolist()
            for chain, bone in project["paws"].items()}


def lab_page(clip, report, threshold, timeline, hashes):
    data = {"clip": clip, "report": report, "threshold": threshold, "timeline": timeline, "hashes": hashes}
    # "<\/" keeps a "</script>" inside a clip name or flag from closing the inline script.
    return TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))


def make_template(clip, bound, report, round_number):
    by_index = {f["index"]: f for f in report["frames"]}
    return {"schemaVersion": 1, "clip": clip,
            "bound": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in bound.items()},
            "reviewer": "", "verdict": "", "inspectedReference": False, "inspectedOverlays": False,
            "inspectedSheet": False, "observations": "", "artisticRound": round_number,
            "lowFrames": [{"frame": i, "iou": by_index[i]["iou"], "flags": by_index[i]["flags"], "action": "",
                           "note": ""} for i in report["lowFrames"]]}


def validate(review, template):
    problems = []
    if review.get("bound") != template["bound"]:
        problems.append("The review's bound hashes differ from the template; regenerate it with `review --template`")
    for name, item in template["bound"].items():
        if not Path(item["path"]).is_file() or sha256_file(item["path"]) != item["sha256"]:
            problems.append(f"{name} changed after the template was made; review the new output")
    if not review.get("reviewer"):
        problems.append("reviewer is empty")
    if review.get("verdict") not in ("pass", "fail"):
        problems.append('verdict must be "pass" or "fail"')
    listed = {item["frame"] for item in review.get("lowFrames", [])}
    missing = [item["frame"] for item in template["lowFrames"] if item["frame"] not in listed]
    if missing:
        problems.append(f"Low or flagged frames missing from the review: {missing}")
    if review.get("verdict") == "pass":
        for key in ("inspectedReference", "inspectedOverlays", "inspectedSheet"):
            if review.get(key) is not True:
                problems.append(f"{key} must be true for a pass")
        if not review.get("observations"):
            problems.append("observations are empty")
        for item in review.get("lowFrames", []):
            if item.get("action") != "explained" or not item.get("note"):
                problems.append(f"Frame {item['frame']} needs an explanation, or a re-mark and refit")
    return problems
