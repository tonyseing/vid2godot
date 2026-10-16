"""vid2godot stage CLI: python3 anim/pipeline.py <command> --work <clip work folder> [options]."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anim import brief as brief_mod  # noqa: E402
from anim import contract as contract_mod  # noqa: E402
from anim import facts as facts_mod  # noqa: E402
from anim import generate as generate_mod  # noqa: E402
from anim import ingest as ingest_mod  # noqa: E402
from anim import raster as raster_mod  # noqa: E402
from anim import reference as reference_mod  # noqa: E402
from anim import review as review_mod  # noqa: E402
from anim import sheet as sheet_mod  # noqa: E402
from anim import verify as verify_mod  # noqa: E402
from anim.blender_run import run_blender  # noqa: E402
from anim.camera import SideCamera  # noqa: E402
from anim.ledger import Ledger, StageError, sha256_file  # noqa: E402
from anim.openrouter import Client  # noqa: E402

COMMANDS = {}
ARGS = {"init": [("--model", {"required": True, "type": Path}), ("--clip", {"required": True})],
        "status": [], "next": [],
        "contract": [("--project-contract", {"required": True, "type": Path})],
        "brief": [("--file", {"required": True, "type": Path})], "reference": [],
        "generate": [("--model-id", {"required": True}), ("--candidates", {"type": int, "default": 2}),
                     ("--seed", {"type": int, "default": 1000}), ("--max-cost", {"type": float})],
        "accept": [("--candidate", {"required": True, "type": int}), ("--checks", {"required": True, "type": Path})],
        "ingest": [("--work-scale", {"type": float, "default": 0.5})],
        "gridframe": [("--frame", {"required": True, "type": int})],
        "fit": [("--marks", {"type": Path}), ("--span", {})], "facts": [], "bake": [], "lab": [],
        "review": [("--template", {"action": "store_true"}), ("--file", {"type": Path})],
        "verify": [("--godot47", {"required": True, "type": Path}), ("--godot45", {"required": True, "type": Path}),
                   ("--launcher", {"required": True, "type": Path}), ("--threed-verify", {"type": Path})],
        "export": [("--destination", {"required": True, "type": Path})]}


def command(name):
    def register(function):
        COMMANDS[name] = function
        return function
    return register


def emit(data):
    print(json.dumps(data, indent=2))


@command("init")
def cmd_init(a):
    ledger = Ledger.init(a.work, a.model, a.clip)
    emit({"work": str(ledger.work), "next": ledger.next()})


@command("status")
def cmd_status(a):
    ledger = Ledger(a.work)
    emit({"clip": ledger.clip, "stages": ledger.status(), "next": ledger.next()})


@command("next")
def cmd_next(a):
    emit({"next": Ledger(a.work).next()})


@command("contract")
def cmd_contract(a):
    ledger = Ledger(a.work)
    project = contract_mod.load_project_contract(a.project_contract)
    folder = ledger.work / "contract"
    folder.mkdir(exist_ok=True)
    rig = run_blender("inspect_rig", {"model": str(ledger.model), "chains": project["chains"],
                                      "headBone": project["headBone"], "pelvisBone": project["pelvisBone"]},
                      folder, a.blender)
    (folder / "rig.json").write_text(json.dumps(rig, indent=2))
    shutil.copyfile(a.project_contract, folder / "project-contract.json")
    problems, warnings = contract_mod.check_rig(rig, project)
    (folder / "report.json").write_text(json.dumps({"problems": problems, "warnings": warnings}, indent=2))
    if problems:
        raise StageError("The rig does not meet the contract:\n- " + "\n- ".join(problems))
    ledger.record("contract", {"projectContract": folder / "project-contract.json"},
                  {"rig": folder / "rig.json", "report": folder / "report.json"}, {"warnings": warnings})
    emit({"ok": True, "warnings": warnings})


def project_contract(ledger):
    return contract_mod.load_project_contract(ledger.output("contract", "rig").parent / "project-contract.json")


def load_json(path):
    return json.loads(Path(path).read_text())


@command("brief")
def cmd_brief(a):
    ledger = Ledger(a.work)
    brief = brief_mod.load_brief(a.file, project_contract(ledger), ledger.clip)
    folder = ledger.work / "brief"
    folder.mkdir(exist_ok=True)
    shutil.copyfile(a.file, folder / "source.json")
    (folder / "brief.json").write_text(json.dumps(brief, indent=2))
    ledger.record("brief", {"source": folder / "source.json"}, {"brief": folder / "brief.json"})
    emit({"ok": True, "brief": brief})


@command("reference")
def cmd_reference(a):
    ledger = Ledger(a.work)
    brief = load_json(ledger.output("brief", "brief"))
    folder = ledger.work / "reference"
    folder.mkdir(exist_ok=True)
    result = run_blender("render_reference", {"model": str(ledger.model), "outDir": str(folder),
                                              "backgrounds": brief["backgrounds"]}, folder, a.blender)
    mask = reference_mod.model_mask(result["mask"])
    colour, report = reference_mod.choose_background({c: Path(p) for c, p in result["renders"].items()}, mask)
    shutil.copyfile(result["renders"][colour], folder / "reference.png")
    (folder / "reference.json").write_text(json.dumps({"camera": result["camera"], "background": colour,
                                                       "contrast": report}, indent=2))
    ledger.record("reference", {"brief": ledger.output("brief", "brief")},
                  {"image": folder / "reference.png", "mask": folder / "mask.png", "meta": folder / "reference.json"},
                  {"background": colour})
    emit({"ok": True, "background": colour, "contrast": report, "image": str(folder / "reference.png")})


@command("generate")
def cmd_generate(a):
    ledger = Ledger(a.work)
    brief = load_json(ledger.output("brief", "brief"))
    meta = load_json(ledger.output("reference", "meta"))
    folder = ledger.work / "generate"
    record = generate_mod.run_generate(Client(), folder, a.model_id, brief, ledger.output("reference", "image"),
                                       meta["background"], candidates=a.candidates, seed=a.seed, max_cost=a.max_cost)
    outputs = {"record": folder / "record.json"}
    outputs.update({f"candidate{c['index']}": Path(c["video"]) for c in record["candidates"]})
    ledger.record("generate", {"image": ledger.output("reference", "image")}, outputs,
                  {"model": a.model_id, "spentUsd": record["spentUsd"]})
    emit({"ok": True, "spentUsd": record["spentUsd"], "warnings": record["warnings"],
          "strips": [c["strip"] for c in record["candidates"]],
          "next": "Open every strip and several full frames of each candidate, then run `accept`."})


@command("accept")
def cmd_accept(a):
    ledger = Ledger(a.work)
    record = load_json(ledger.output("generate", "record"))
    accepted = generate_mod.accept(record, a.candidate, load_json(a.checks))
    folder = ledger.work / "generate"
    shutil.copyfile(a.checks, folder / "checks.json")
    (folder / "accepted.json").write_text(json.dumps(accepted, indent=2))
    ledger.record("accept", {"checks": folder / "checks.json", "video": Path(accepted["candidate"]["video"])},
                  {"accepted": folder / "accepted.json"}, {"candidate": a.candidate})
    emit({"ok": True, "video": accepted["candidate"]["video"]})


@command("ingest")
def cmd_ingest(a):
    ledger = Ledger(a.work)
    video = Path(load_json(ledger.output("accept", "accepted"))["candidate"]["video"])
    camera = SideCamera.from_json(load_json(ledger.output("reference", "meta"))["camera"])
    folder = ledger.work / "ingest"
    meta = ingest_mod.run_ingest(video, ledger.output("reference", "mask"), camera, folder, a.work_scale)
    outputs = {"masks": folder / "masks.npz", "meta": folder / "ingest.json"}
    if meta["audio"]:
        outputs["audio"] = Path(meta["audio"])
    ledger.record("ingest", {"video": video, "mask": ledger.output("reference", "mask")}, outputs,
                  {"frames": meta["frames"], "fps": meta["fps"], "warnings": meta["warnings"]})
    emit({"ok": True, "frames": meta["frames"], "fps": meta["fps"], "flags": meta["flags"],
          "warnings": meta["warnings"]})


@command("gridframe")
def cmd_gridframe(a):
    ledger = Ledger(a.work)
    folder = ledger.work / "ingest"
    (folder / "grid").mkdir(exist_ok=True)
    out = ingest_mod.grid_image(folder / "registered" / f"{a.frame:05d}.png", folder / "grid" / f"{a.frame:05d}.png")
    emit({"grid": str(out)})


@command("fit")
def cmd_fit(a):
    ledger = Ledger(a.work)
    meta = load_json(ledger.output("ingest", "meta"))
    brief = load_json(ledger.output("brief", "brief"))
    masks = ledger.output("ingest", "masks")
    folder = ledger.work / "fit"
    folder.mkdir(exist_ok=True)
    marks_path = folder / "marks.json"
    if a.marks:
        shutil.copyfile(a.marks, marks_path)
    elif not marks_path.exists():
        marks_path.write_text(json.dumps({"frames": {}}))
    args = {"model": str(ledger.model), "project": project_contract(ledger), "brief": brief,
            "camera": meta["camera"], "masks": str(masks), "marks": load_json(marks_path)["frames"],
            "workScale": meta["workScale"], "fps": meta["fps"], "out": str(folder / "poses.json"),
            "projection": str(folder / "projection.npz")}
    if a.span:
        start, end = (int(x) for x in a.span.split(":"))
        if not (folder / "poses.json").is_file():
            raise StageError("--span refits part of an existing fit; run a full `fit` first")
        args.update({"span": [start, end], "init": str(folder / "poses.json")})
    run_blender("fit_clip", args, folder, a.blender)
    report = raster_mod.score_fit(load_json(folder / "poses.json"), folder / "projection.npz", masks,
                                  brief["iouThreshold"], meta["flags"])
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    ledger.record("fit", {"masks": masks, "marks": marks_path},
                  {"poses": folder / "poses.json", "projection": folder / "projection.npz",
                   "report": folder / "report.json"}, {"meanIoU": report["meanIoU"], "lowFrames": report["lowFrames"]})
    emit({"ok": True, "meanIoU": report["meanIoU"], "lowFrames": report["lowFrames"],
          "next": "Run `lab`, open the overlay of every low frame, mark or refit, then run `facts`."})


def silhouette_renderer(model, poses, shape, folder, blender=None):
    """facts' renderer for the averaged cycle: value vectors (poses.json `names` order) -> the rig's silhouettes
    with the fit's camera, rasterised as the fit's scorer does, at `shape` (the working masks' height, width)."""
    def render(values):
        out = Path(folder) / "poses-uv.npz"
        run_blender("pose_silhouettes", {"model": str(model), "camera": poses["camera"], "bones": poses["bones"],
                                         "names": poses["names"], "values": [list(map(float, v)) for v in values],
                                         "out": str(out)}, folder, blender)
        data = np.load(out)
        return [raster_mod.rasterize(uv, data["faces"], shape[1], shape[0]) for uv in data["uv"]]
    return render


@command("facts")
def cmd_facts(a):
    ledger = Ledger(a.work)
    brief = load_json(ledger.output("brief", "brief"))
    poses = load_json(ledger.output("fit", "poses"))
    masks_path = ledger.output("ingest", "masks")
    masks = np.load(masks_path)["masks"]
    folder = ledger.work / "facts"
    folder.mkdir(exist_ok=True)
    render = silhouette_renderer(ledger.model, poses, masks.shape[1:], folder / "render", a.blender)
    result = facts_mod.clip_facts(poses, project_contract(ledger), brief, masks=masks, render=render)
    (folder / "facts.json").write_text(json.dumps(result, indent=2))
    (folder / "anim-entry.json").write_text(json.dumps({ledger.clip: result["entry"]}, indent=2))
    ledger.record("facts", {"poses": ledger.output("fit", "poses"), "masks": masks_path},
                  {"facts": folder / "facts.json", "entry": folder / "anim-entry.json"},
                  {"speedMps": result["speedMps"], "period": result["period"], "warnings": result["warnings"],
                   "averagedCycleIoU": result.get("averagedCycleIoU")})
    emit({"ok": True, **{k: result[k] for k in ("start", "period", "speedMps", "strideM", "warnings")},
          **{k: result[k] for k in ("cyclesAveraged", "averagedCycleIoU") if k in result}})


@command("bake")
def cmd_bake(a):
    ledger = Ledger(a.work)
    project = project_contract(ledger)
    facts_path = ledger.output("facts", "facts")
    cycle = load_json(facts_path)["cycle"]
    poses = load_json(ledger.output("fit", "poses"))
    brief = load_json(ledger.output("brief", "brief"))
    folder = ledger.work / "bake"
    folder.mkdir(exist_ok=True)
    out = folder / f"{ledger.model.stem}-{ledger.clip}-eval.glb"
    result = run_blender("bake_clip", {"model": str(ledger.model), "out": str(out), "cycle": str(facts_path),
                                       "clip": ledger.clip, "loop": brief["loop"],
                                       "pelvisBone": project["pelvisBone"], "bones": poses["bones"],
                                       "names": poses["names"]}, folder, a.blender)
    if ledger.clip not in result["actions"]:
        raise StageError(f"The re-imported GLB has no clip {ledger.clip!r}: {sorted(result['actions'])}")
    bounds = load_json(ledger.output("contract", "rig"))["bounds"]
    background = project["sheet"]["background"]
    camera = sheet_mod.game_camera(bounds, project["sheet"]["pxPerMetre"])
    n = len(cycle["frames"]) - 1
    offsets = sorted({round(i * n / 8) for i in range(8)})
    frames = run_blender("render_frames", {"model": str(out), "clip": ledger.clip, "camera": camera.to_json(),
                                           "background": background, "frames": offsets, "fps": cycle["fps"],
                                           "outDir": str(folder / "sheet-frames")}, folder / "sheet-frames", a.blender)
    sheet_mod.compose(frames["frames"], background, folder / "sheet.png")
    close = SideCamera(camera.width, camera.height, camera.px_per_m * 4, camera.center_x,
                       (0.6 * camera.height - camera.height / 2) / (camera.px_per_m * 4))
    closeup = run_blender("render_frames", {"model": str(out), "clip": ledger.clip, "camera": close.to_json(),
                                            "background": background, "frames": [0], "fps": cycle["fps"],
                                            "outDir": str(folder / "closeup-frames")}, folder / "closeup-frames",
                          a.blender)
    sheet_mod.compose(closeup["frames"], background, folder / "closeup.png")
    ledger.record("bake", {"poses": ledger.output("fit", "poses"), "facts": facts_path},
                  {"glb": out, "sheet": folder / "sheet.png", "closeup": folder / "closeup.png"},
                  {"actions": result["actions"]})
    emit({"ok": True, "glb": str(out), "sheet": str(folder / "sheet.png"), "closeup": str(folder / "closeup.png")})


def build_lab(ledger):
    folder = ledger.work / "review"
    poses = load_json(ledger.output("fit", "poses"))
    report = load_json(ledger.output("fit", "report"))
    brief = load_json(ledger.output("brief", "brief"))
    review_mod.draw_overlays(ledger.work, poses, ledger.output("fit", "projection"), ledger.output("ingest", "masks"),
                             folder / "overlays")
    hashes = {name: ledger.require("fit")["outputs"][name]["sha256"] for name in ("poses", "projection")}
    page = review_mod.lab_page(ledger.clip, report, brief["iouThreshold"],
                               review_mod.contact_timeline(poses, project_contract(ledger)), hashes)
    (folder / "lab.html").write_text(page)
    return folder / "lab.html", report


@command("lab")
def cmd_lab(a):
    lab, report = build_lab(Ledger(a.work))
    emit({"lab": str(lab), "lowFrames": report["lowFrames"],
          "next": "Open lab.html; open every low frame's overlay; mark (`gridframe`, marks file) and refit spans."})


@command("review")
def cmd_review(a):
    ledger = Ledger(a.work)
    folder = ledger.work / "review"
    history_path = folder / "history.json"
    history = load_json(history_path) if history_path.is_file() else []
    if a.template:
        lab, report = build_lab(ledger)
        bound = {"poses": ledger.output("fit", "poses"), "facts": ledger.output("facts", "facts"),
                 "glb": ledger.output("bake", "glb"), "sheet": ledger.output("bake", "sheet"),
                 "closeup": ledger.output("bake", "closeup"), "lab": lab}
        template = review_mod.make_template(ledger.clip, bound, report, len(history) + 1)
        (folder / "template.json").write_text(json.dumps(template, indent=2))
        emit({"template": str(folder / "template.json"), "lab": str(lab), "lowFrames": report["lowFrames"]})
        return
    if not a.file:
        raise StageError("review needs --template or --file")
    if not (folder / "template.json").is_file():
        raise StageError("There is no review template yet; run `review --template` first")
    template = load_json(folder / "template.json")
    filled = load_json(a.file)
    problems = review_mod.validate(filled, template)
    if problems:
        raise StageError("The review is not complete:\n- " + "\n- ".join(problems))
    history, failed = review_mod.record_round(history, template, filled)
    if filled["verdict"] != "pass":
        history_path.write_text(json.dumps(history, indent=2))
        ask = failed >= review_mod.ARTISTIC_ROUNDS
        emit({"ok": False, "failedRounds": failed,
              "next": "Ask the owner whether to continue or park this clip." if ask else "Revise and review again."})
        return
    if Path(a.file).resolve() != (folder / "review.json").resolve():
        shutil.copyfile(a.file, folder / "review.json")
    inputs = {name: Path(item["path"]) for name, item in template["bound"].items()}
    ledger.record("review", {**inputs, "template": folder / "template.json"}, {"review": folder / "review.json"},
                  {"round": template["artisticRound"]})
    history_path.write_text(json.dumps(history, indent=2))
    emit({"ok": True, "next": "verify"})


@command("verify")
def cmd_verify(a):
    ledger = Ledger(a.work)
    ledger.require("review")
    script = verify_mod.verifier(a.threed_verify)
    glb = ledger.output("bake", "glb")
    facts = load_json(ledger.output("facts", "facts"))
    outputs, problems = {}, []
    for prefix, engine in (("4.7", a.godot47), ("4.5", a.godot45)):
        report = verify_mod.run_verify(script, glb, ledger.work / "verify" / prefix, engine, a.launcher)
        # The clip's length tolerance is 1.5 frames at the baked clip's own (resampled) rate, not the ingest rate.
        problems += [f"[{prefix}] {p}" for p in verify_mod.check_report(
            report, sha256_file(glb), ledger.clip, facts["entry"]["duration_s"], facts["cycle"]["fps"], prefix,
            launcher_used=True)]
        outputs[f"report{prefix.replace('.', '')}"] = ledger.work / "verify" / prefix / "godot-report.json"
    if problems:
        raise StageError("Godot verification failed:\n- " + "\n- ".join(problems))
    ledger.record("verify", {"glb": glb}, outputs)
    emit({"ok": True, "reports": {k: str(v) for k, v in outputs.items()},
          "next": "Open every capture PNG listed in both reports, then `export`."})


@command("export")
def cmd_export(a):
    ledger = Ledger(a.work)
    ledger.require("verify")
    stem = f"{ledger.model.stem}-{ledger.clip}"
    files = {f"{stem}-eval.glb": ledger.output("bake", "glb"), "anim-entry.json": ledger.output("facts", "entry"),
             "facts.json": ledger.output("facts", "facts"), "fit-report.json": ledger.output("fit", "report"),
             "marks.json": ledger.require("fit")["inputs"]["marks"]["path"],
             "sheet.png": ledger.output("bake", "sheet"),
             "closeup.png": ledger.output("bake", "closeup"), "review.json": ledger.output("review", "review"),
             "godot-4.7.json": ledger.output("verify", "report47"),
             "godot-4.5.json": ledger.output("verify", "report45"),
             "generate.json": ledger.output("generate", "record"), "accepted.json": ledger.output("accept", "accepted"),
             "ingest.json": ledger.output("ingest", "meta"), "reference.png": ledger.output("reference", "image"),
             "brief.json": ledger.output("brief", "brief"), "rig.json": ledger.output("contract", "rig"),
             "ledger.json": ledger.path}
    if "audio" in ledger.require("ingest")["outputs"]:
        files["candidate-audio.wav"] = ledger.output("ingest", "audio")
    for prefix in ("4.7", "4.5"):
        report = load_json(ledger.output("verify", f"report{prefix.replace('.', '')}"))
        for item in report.get("captures") or []:
            path = Path(item["path"])
            name = f"captures/{prefix}/{path.name}"
            if name in files:
                raise StageError(f"Two {prefix} captures share the name {path.name}")
            if "sha256" in item and (not path.is_file() or sha256_file(path) != item["sha256"]):
                raise StageError(f"Capture changed or missing since verify: {path}")
            files[name] = path
    lab = ledger.require("review")["inputs"].get("lab")
    if lab is None:
        raise StageError("The review recorded no lab; run `review --template` and review again")
    files["lab.html"] = lab["path"]
    for overlay in sorted((Path(lab["path"]).parent / "overlays").glob("*.png")):
        files[f"overlays/{overlay.name}"] = overlay
    out = verify_mod.export_bundle({k: Path(v) for k, v in files.items()}, a.destination)
    emit({"ok": True, "bundle": str(out),
          "next": "Show the owner the sheet, close-up, captures and lab; ship only after explicit approval."})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, options in ARGS.items():
        child = sub.add_parser(name)
        child.add_argument("--work", required=True, type=Path)
        child.add_argument("--blender")
        for flag, kwargs in options:
            child.add_argument(flag, **kwargs)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except (StageError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
