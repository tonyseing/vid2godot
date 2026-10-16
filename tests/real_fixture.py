"""The real-clip regression fixture: a paid AI video's re-ingested masks, fitted and measured.

The video and its masks are not in this repository (the clip is the project's, not the skill's). A fixture folder
holds `fixture.json` (the fit arguments, the rig's path and hash, the frames and spans to measure, the targets and
the baseline) and `masks.npz` (the ingested working masks). Build one from a pipeline work folder with

    python3 -m tests.real_fixture build --work <work folder> --out <fixture folder>

and fit and measure it with

    python3 -m tests.real_fixture run --fixture <fixture folder> --out <run folder> [--script <fit_clip.py>]

(`measure --fixture <folder> --run <run folder>` measures an existing run again.)

`tests.test_real_fixture` runs the same fit when VID2GODOT_REAL_FIXTURE names the folder (and VID2GODOT_BLENDER is
set), and holds it to the targets.

Metrics, over the measured frames: mean, per-span and minimum IoU (the gate's own scorer); the share of fitted
forepaws more than PAW_MISS_PX working pixels from every paw found in the video mask; and leg swaps, the frames where
a leg pair's fitted angles are clearly closer to the previous frame's with the pair exchanged.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anim.blender_run import run_blender  # noqa: E402
from anim import facts, ingest, paws  # noqa: E402
from anim.camera import SideCamera  # noqa: E402
from anim.ingest import background_colour, run_ingest  # noqa: E402
from anim.raster import score_fit  # noqa: E402

EVAL_FRAMES = sorted(set(range(0, 121, 8)) | {36, 48, 60, 80, 84, 92, 96, 108})
SPANS = ((0, 25), (26, 50), (51, 75), (76, 100), (101, 120))      # inclusive frame ranges
TARGETS = {"meanIoU": 0.85, "span": {"76-100": 0.80}, "frame": {"80": 0.75}, "forepawMissShare": 0.20,
           "cycle": True}
PAW_MISS_PX = 15.0
FALSE_FILL_TOL = 20.0          # a mask pixel this close to the background colour is background
FALSE_FILL_PX = 300            # working pixels of background in the mask that make a false-fill frame
SWAP_MIN_RAD = np.radians(30.0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------- paws in a mask
def find_paws(mask, max_paws=6):
    """Paw points (each leg branch's far end and the blob near it, anim.paws), in working pixels [u, v]."""
    return paws.points(paws.find_paws(mask, max_paws))


# ------------------------------------------------------------------------------------------------- the metrics
def span_name(span):
    return f"{span[0]}-{span[1]}"


def iou_metrics(ious, frames=EVAL_FRAMES, spans=SPANS):
    """Mean, per-span (inclusive ranges) and minimum IoU over `frames`; `ious` maps frame -> IoU."""
    picked = {f: ious[f] for f in frames if f in ious}
    low = min(picked, key=picked.get)
    return {"meanIoU": round(float(np.mean(list(picked.values()))), 4),
            "spans": {span_name(s): round(float(np.mean([v for f, v in picked.items() if s[0] <= f <= s[1]])), 4)
                      for s in spans if any(s[0] <= f <= s[1] for f in picked)},
            "minIoU": round(float(picked[low]), 4), "minFrame": int(low),
            "frames": {str(f): round(float(v), 4) for f, v in picked.items()}}


def forepaw_miss_share(poses, masks, chains=("front_L", "front_R"), frames=EVAL_FRAMES, limit=PAW_MISS_PX):
    """Share of fitted forepaws further than `limit` from every paw point found in the frame's mask. A fitted paw is
    the middle of the chain's last bone (the paw bone: its tail is the toe tip), in working px."""
    by_index = {e["index"]: e for e in poses["frames"]}
    misses, total, far = 0, 0, {}
    for f in frames:
        found = np.array(find_paws(masks[f]) or [[np.inf, np.inf]])
        for chain in chains:
            paw = 0.5 * (np.array(by_index[f]["joints"][chain][-1]) + np.array(by_index[f]["joints"][chain][-2]))
            distance = float(np.min(np.hypot(*(found - paw).T)))
            total += 1
            if distance > limit:
                misses += 1
                far[f"{f}:{chain}"] = round(distance, 1)
    return round(misses / max(total, 1), 4), far


def leg_swaps(poses, pairs, chains, limit=SWAP_MIN_RAD):
    """Per leg pair, the frames whose fitted angles are clearly closer to the previous frame's with the pair
    exchanged: the exchanged distance is under half the labelled one, which is over `limit` (summed |rad|)."""
    names = poses["names"]
    values = {e["index"]: np.array(e["values"]) for e in poses["frames"]}
    out = {}
    for a, b in pairs:
        ia = [names.index(n) for n in chains[a] if n in names]
        ib = [names.index(n) for n in chains[b] if n in names]
        swaps = []
        for f in sorted(values):
            if f - 1 not in values or len(ia) != len(ib) or not ia:
                continue
            x, y = values[f], values[f - 1]
            same = float(np.abs(x[ia] - y[ia]).sum() + np.abs(x[ib] - y[ib]).sum())
            cross = float(np.abs(x[ia] - y[ib]).sum() + np.abs(x[ib] - y[ia]).sum())
            if same > limit and cross < 0.5 * same:
                swaps.append(f)
        out[f"{a}/{b}"] = swaps
    return out


def period_metrics(poses):
    """facts' own cycle search on the fitted joint angles: whether a full cycle is found, its period and score."""
    values = np.array([e["values"] for e in sorted(poses["frames"], key=lambda e: e["index"])], float)[:, 2:]
    found = facts.period_search(values, float(poses["fps"]))
    return {"found": found["period"] is not None,
            "periodFrames": None if found["period"] is None else round(found["period"], 3),
            "score": round(found["score"], 4), "gate": facts.PERIOD_MAX_SCORE}


def leg_period_metrics(poses, fixture, masks):
    """The secondary report facts makes on the fitted angles, whether or not facts accepts the fit: the video's
    period (anim/cycle.py) and the well-observed leg joints' lag score at it."""
    from anim import cycle
    found = cycle.mask_period(masks, float(poses["fps"]))
    if found["period"] is None:
        return {"maskPeriod": None, "reason": found["reason"]}
    values = np.array([e["values"] for e in sorted(poses["frames"], key=lambda e: e["index"])], float)
    legs = facts.leg_period(values, poses["names"], fixture["project"], float(poses["fps"]), found["period"])
    return {"maskPeriod": round(found["period"], 4), "maskScore": round(found["score"], 4),
            "maskMargin": round(found["margin"], 3), **legs}


def rig_inspection(model, project, work, blender=None):
    """The fixture rig's contract rig.json, from the contract stage's own rig inspection."""
    return run_blender("inspect_rig", {"model": str(model), "chains": project["chains"],
                                       "headBone": project["headBone"], "pelvisBone": project["pelvisBone"]},
                       Path(work), blender)


def rig_rest(model, project, work, blender=None):
    """The fixture rig's paw rest heights (facts.rest_heights), from the contract stage's own rig inspection."""
    return facts.rest_heights(rig_inspection(model, project, work, blender), project)


# Task 19: the fixture measures the fit, not the video's gait (the brief's), so facts' gait gate is reported here,
# with this recorded reason, instead of refusing the fit.
FIXTURE_GAIT = "real-fixture fit regression: the gait is reported, not gated"


def facts_outcome(poses, project, brief, masks=None, render=None, rest=None, rig=None):
    """The facts stage itself on the fit (its period from the video masks, its averaged cycle rendered with
    `render` and checked against them, each paw measured from its rest height `rest`; with the contract `rig`, the
    contacts from the motion and the 60 Hz keys' contact check, as the pipeline runs it): "ok" with its start, period
    and the cycle evidence, or its refusal."""
    try:
        result = facts.clip_facts(poses, project, brief, rest, masks=masks, render=render, rig=rig,
                                  accept_gait=FIXTURE_GAIT)
    except facts.FactsError as error:
        return {"ok": False, "error": str(error)}
    out = {"ok": True, "start": round(result["start"], 3), "period": round(result["period"], 3)}
    for key in ("cyclesAveraged", "cycleStarts", "averagedCycleIoU", "maskPeriod", "legPeriod", "cycleCheck",
                "gaitCheck", "contactCheck"):
        if key in result:
            out[key] = result[key]
    return out


def legacy_silhouette(img, bg, threshold=40.0):
    """The ingest silhouette before Task 17b: every enclosed hole filled, whatever its colour."""
    subject = np.linalg.norm(img.astype(float) - bg, axis=2) > threshold
    mask = ndimage.binary_opening(subject, structure=np.ones((3, 3)))
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return ndimage.binary_fill_holes(labels == int(np.argmax(sizes)) + 1)


BASELINE_KEYS = ("meanIoU", "spans", "minIoU", "minFrame", "forepawMissShare", "legSwaps", "period", "facts",
                 "meanIoUAll", "minutes")


def baseline_block(metrics, fitter):
    """The fixture's baseline block from a measured run of the baseline fitter."""
    return {"fitter": fitter, **{k: metrics[k] for k in BASELINE_KEYS if k in metrics}}


def false_fill(masks, registered, tol=FALSE_FILL_TOL, limit=FALSE_FILL_PX):
    """Per frame, the working-mask pixels whose registered colour is background (within `tol`): the pixels a hole
    fill added between touching paws. Working pixel o is registered pixel 2o (the work scale is 1/2)."""
    counts = []
    for f, mask in enumerate(masks):
        img = np.asarray(Image.open(registered / f"{f:05d}.png").convert("RGB"))
        step = img.shape[0] // mask.shape[0]
        img = img[::step, ::step][:mask.shape[0], :mask.shape[1]].astype(float)
        near = np.linalg.norm(img - background_colour(img), axis=2) < tol
        counts.append(int(np.count_nonzero(mask & near)))
    return {"counts": counts, "frames": [f for f, c in enumerate(counts) if c >= limit]}


# ------------------------------------------------------------------------------------------- build, fit, measure
def build(work, out, baseline_script=None, baseline_run=None, fitter=None, blender=None):
    """Re-ingest a work folder's accepted video with this repository's ingest; write the fixture. The false fill is
    counted for these masks and for the old ingest's (every hole filled). With `baseline_script` (the baseline
    fitter's fit_clip.py) the clip is fitted with it into `baseline_run`; with `baseline_run` alone an existing
    measured run of the same masks is reused. Either way the baseline block is measured and written."""
    work, out = Path(work), Path(out)
    fit_args = json.loads((work / "fit" / "fit_clip.args.json").read_text())
    accepted = json.loads((work / "generate" / "accepted.json").read_text())["candidate"]
    reference = json.loads((work / "reference" / "reference.json").read_text())
    video = Path(accepted["video"])
    if sha256(video) != accepted["sha256"]:
        raise SystemExit(f"{video} does not match the accepted hash")
    out.mkdir(parents=True, exist_ok=True)
    camera = SideCamera.from_json(reference["camera"])
    with tempfile.TemporaryDirectory() as tmp:
        meta = run_ingest(video, work / "reference" / "mask.png", camera, tmp, fit_args["workScale"])
        masks = np.load(Path(tmp) / "masks.npz")["masks"]
        filled = false_fill(masks, Path(tmp) / "registered")
    with tempfile.TemporaryDirectory() as tmp:
        current = ingest.silhouette
        ingest.silhouette = legacy_silhouette
        try:
            run_ingest(video, work / "reference" / "mask.png", camera, tmp, fit_args["workScale"])
        finally:
            ingest.silhouette = current
        legacy = false_fill(np.load(Path(tmp) / "masks.npz")["masks"], Path(tmp) / "registered")
    np.savez_compressed(out / "masks.npz", masks=masks)
    model = Path(fit_args["model"])
    fixture = {
        "description": "Real-clip fitter regression fixture (vid2godot tests/real_fixture.py): the accepted video "
                       "re-ingested; the video itself is not stored here.",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"video": video.name, "videoSha256": accepted["sha256"], "work": work.name},
        "model": os.path.relpath(model, out), "modelSha256": sha256(model),
        "masksSha256": sha256(out / "masks.npz"),
        "project": fit_args["project"], "brief": fit_args["brief"], "camera": meta["camera"],
        "workScale": meta["workScale"], "fps": meta["fps"], "frames": meta["frames"],
        "registration": meta["registration"], "flags": meta["flags"], "marks": {},
        "falseFill": {**filled, "frameCount": len(filled["frames"]),
                      "legacy": {**legacy, "frameCount": len(legacy["frames"])}},
        "evalFrames": EVAL_FRAMES, "spans": [list(s) for s in SPANS], "targets": TARGETS,
    }
    (out / "fixture.json").write_text(json.dumps(fixture, indent=1) + "\n")
    if baseline_script or baseline_run:
        if baseline_script:
            metrics = run(out, baseline_run, blender, baseline_script)
        else:
            run_dir = Path(baseline_run)
            if not np.array_equal(np.load(run_dir / "masks.npz")["masks"], masks):
                raise SystemExit(f"{run_dir} was fitted to other masks; pass --baseline-script to refit")
            poses = json.loads((run_dir / "poses.json").read_text())
            metrics = measure(fixture, poses, run_dir / "projection.npz", run_dir / "masks.npz", model, blender,
                              run_dir)
            old = run_dir / "metrics.json"
            if old.is_file():
                metrics["minutes"] = json.loads(old.read_text()).get("minutes")
        fixture["baseline"] = baseline_block(metrics, fitter or str(baseline_script or baseline_run))
        (out / "fixture.json").write_text(json.dumps(fixture, indent=1) + "\n")
    return fixture


def load(folder):
    folder = Path(folder)
    fixture = json.loads((folder / "fixture.json").read_text())
    model = (folder / fixture["model"]).resolve()
    if sha256(model) != fixture["modelSha256"]:
        raise RuntimeError(f"{model} is not the fixture's rig")
    if sha256(folder / "masks.npz") != fixture["masksSha256"]:
        raise RuntimeError(f"{folder / 'masks.npz'} is not the fixture's masks")
    return fixture, model


def full_masks(fixture_masks, out):
    """The fitter's masks file: the masks with each frame's outside distance field."""
    masks = np.load(fixture_masks)["masks"]
    outside = np.stack([ndimage.distance_transform_edt(~m).astype(np.float32) for m in masks])
    np.savez_compressed(out, masks=masks, outside=outside)
    return out


def measure(fixture, poses, projection, masks_npz, model=None, blender=None, work=None):
    """The run's metrics. With `model`, facts runs on the fit with the rig's renderer (Blender) into `work`."""
    report = score_fit(poses, projection, masks_npz, fixture["brief"]["iouThreshold"], {})
    ious = {item["index"]: item["iou"] for item in report["frames"]}
    masks = np.load(masks_npz)["masks"]
    render = rest = rig = None
    if model is not None:
        from anim.pipeline import silhouette_renderer
        folder = Path(work or tempfile.mkdtemp()) / "facts"
        render = silhouette_renderer(model, poses, masks.shape[1:], folder, blender)
        folder.mkdir(parents=True, exist_ok=True)
        rig = rig_inspection(model, fixture["project"], folder, blender)
        rest = facts.rest_heights(rig, fixture["project"])
    frames, spans = fixture["evalFrames"], [tuple(s) for s in fixture["spans"]]
    miss, far = forepaw_miss_share(poses, masks, frames=frames)
    swaps = leg_swaps(poses, fixture["project"]["legPairs"], fixture["project"]["chains"])
    return {**iou_metrics(ious, frames, spans), "meanIoUAll": report["meanIoU"], "forepawMissShare": miss,
            "forepawMisses": far, "legSwaps": {pair: len(f) for pair, f in swaps.items()}, "legSwapFrames": swaps,
            "period": period_metrics(poses), "facts": facts_outcome(poses, fixture["project"], fixture["brief"],
                                                                    masks, render, rest, rig),
            "legPeriod": leg_period_metrics(poses, fixture, masks), "periodicPass": poses.get("periodicPass"),
            "cameraScale": poses.get("cameraScale")}


def run(folder, out, blender=None, script=None):
    """Fit the whole fixture clip and measure it; writes metrics.json into `out` and returns the metrics."""
    fixture, model = load(folder)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    masks = full_masks(Path(folder) / "masks.npz", out / "masks.npz")
    args = {"model": str(model), "project": fixture["project"], "brief": fixture["brief"],
            "camera": fixture["camera"], "masks": str(masks), "marks": fixture["marks"],
            "workScale": fixture["workScale"], "fps": fixture["fps"], "out": str(out / "poses.json"),
            "projection": str(out / "projection.npz")}
    start = time.time()
    result = run_blender(Path(script) if script else "fit_clip", args, out, blender)
    poses = json.loads((out / "poses.json").read_text())
    metrics = {**measure(fixture, poses, out / "projection.npz", masks, model, blender, out),
               "minutes": round((time.time() - start) / 60, 1), "fitSeconds": round(result.get("seconds", 0), 1)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics


def remeasure(folder, run_dir, blender=None):
    """Measure an existing fit of the fixture (a `run` folder) with this repository's metrics and facts; writes
    metrics.json there (keeping its minutes) and returns the metrics."""
    fixture, model = load(folder)
    run_dir = Path(run_dir)
    if not np.array_equal(np.load(run_dir / "masks.npz")["masks"], np.load(Path(folder) / "masks.npz")["masks"]):
        raise SystemExit(f"{run_dir} was fitted to other masks")
    poses = json.loads((run_dir / "poses.json").read_text())
    metrics = measure(fixture, poses, run_dir / "projection.npz", run_dir / "masks.npz", model, blender, run_dir)
    old = run_dir / "metrics.json"
    if old.is_file():
        previous = json.loads(old.read_text())
        metrics.update({k: previous[k] for k in ("minutes", "fitSeconds") if k in previous})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics


def failures(fixture, metrics):
    """Each target the metrics miss, and each span worse than the fixture's baseline."""
    targets, out = fixture["targets"], []
    if metrics["meanIoU"] < targets["meanIoU"]:
        out.append(f"mean IoU {metrics['meanIoU']} < {targets['meanIoU']}")
    for span, low in targets["span"].items():
        if metrics["spans"][span] < low:
            out.append(f"span {span} IoU {metrics['spans'][span]} < {low}")
    for frame, low in targets["frame"].items():
        if metrics["frames"][frame] < low:
            out.append(f"frame {frame} IoU {metrics['frames'][frame]} < {low}")
    for span, value in (fixture.get("baseline") or {}).get("spans", {}).items():
        if metrics["spans"][span] < value:
            out.append(f"span {span} IoU {metrics['spans'][span]} is worse than the baseline {value}")
    if targets.get("cycle") and not metrics["facts"]["ok"]:
        out.append(f"facts refuses the fit: {metrics['facts']['error']}")
    if "forepawMissShare" in targets and metrics["forepawMissShare"] > targets["forepawMissShare"]:
        out.append(f"forepaw miss share {metrics['forepawMissShare']} > {targets['forepawMissShare']}")
    for pair, count in metrics["legSwaps"].items():
        allowed = (fixture.get("baseline") or {}).get("legSwaps", {}).get(pair)
        if allowed is not None and count > allowed:
            out.append(f"{pair} swaps {count} times, more than the baseline's {allowed}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--baseline-script", help="the baseline fitter's fit_clip.py: fit the new masks with it")
    b.add_argument("--baseline-run", help="the folder of that fit (or of an existing fit of the same masks)")
    b.add_argument("--fitter", help="the baseline fitter's name for the block, such as 'vid2godot 71fc632'")
    b.add_argument("--blender")
    r = sub.add_parser("run")
    r.add_argument("--fixture", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--script")
    r.add_argument("--blender")
    m = sub.add_parser("measure", help="measure an existing run folder again")
    m.add_argument("--fixture", required=True)
    m.add_argument("--run", required=True)
    m.add_argument("--blender")
    a = parser.parse_args(argv)
    if a.command == "build":
        fixture = build(a.work, a.out, a.baseline_script, a.baseline_run, a.fitter, a.blender)
        print(json.dumps({"frames": fixture["frames"], "falseFillFrames": fixture["falseFill"]["frames"],
                          "legacyFalseFillFrames": fixture["falseFill"]["legacy"]["frames"],
                          "baseline": fixture.get("baseline")}))
    else:
        if a.command == "measure":
            metrics = remeasure(a.fixture, a.run, a.blender)
        else:
            metrics = run(a.fixture, a.out, a.blender, a.script)
        print(json.dumps({k: v for k, v in metrics.items() if k not in ("frames", "forepawMisses")}))


if __name__ == "__main__":
    main()
