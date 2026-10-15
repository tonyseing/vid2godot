"""vid2godot stage CLI: python3 anim/pipeline.py <command> --work <clip work folder> [options]."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anim import brief as brief_mod  # noqa: E402
from anim import contract as contract_mod  # noqa: E402
from anim import facts as facts_mod  # noqa: E402
from anim import generate as generate_mod  # noqa: E402
from anim import ingest as ingest_mod  # noqa: E402
from anim import raster as raster_mod  # noqa: E402
from anim import reference as reference_mod  # noqa: E402
from anim.blender_run import run_blender  # noqa: E402
from anim.camera import SideCamera  # noqa: E402
from anim.ledger import Ledger, StageError  # noqa: E402
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
        "fit": [("--marks", {"type": Path}), ("--span", {})], "facts": []}


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


@command("facts")
def cmd_facts(a):
    ledger = Ledger(a.work)
    brief = load_json(ledger.output("brief", "brief"))
    result = facts_mod.clip_facts(load_json(ledger.output("fit", "poses")), project_contract(ledger), brief)
    folder = ledger.work / "facts"
    folder.mkdir(exist_ok=True)
    (folder / "facts.json").write_text(json.dumps(result, indent=2))
    (folder / "anim-entry.json").write_text(json.dumps({ledger.clip: result["entry"]}, indent=2))
    ledger.record("facts", {"poses": ledger.output("fit", "poses")},
                  {"facts": folder / "facts.json", "entry": folder / "anim-entry.json"},
                  {"speedMps": result["speedMps"], "period": result["period"], "warnings": result["warnings"]})
    emit({"ok": True, **{k: result[k] for k in ("start", "period", "speedMps", "strideM", "warnings")}})


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
