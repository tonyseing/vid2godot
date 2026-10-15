"""vid2godot stage CLI: python3 anim/pipeline.py <command> --work <clip work folder> [options]."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anim import contract as contract_mod  # noqa: E402
from anim.blender_run import run_blender  # noqa: E402
from anim.ledger import Ledger, StageError  # noqa: E402

COMMANDS = {}
ARGS = {"init": [("--model", {"required": True, "type": Path}), ("--clip", {"required": True})],
        "status": [], "next": [],
        "contract": [("--project-contract", {"required": True, "type": Path})]}


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
