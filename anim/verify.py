"""Godot checks through game-asset-3d-pipeline's verifier, and the evaluation bundle export."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from anim.ledger import StageError, sha256_file

DEFAULT = Path.home() / ".agents/skills/game-asset-3d-pipeline/forge/godot/verify.py"


class VerifyError(StageError):
    pass


def verifier(explicit=None):
    path = Path(explicit) if explicit else DEFAULT
    if not path.is_file():
        raise VerifyError(f"Godot checks use game-asset-3d-pipeline's forge/godot/verify.py, not found at {path}; "
                          "install that skill (img2godot) or pass --threed-verify")
    return path


def run_verify(script, model, out, godot, launcher=None):
    out = Path(out)
    command = [sys.executable, str(script), "--model", str(model), "--out", str(out), "--godot", str(godot)]
    if launcher:
        command += ["--launcher", str(launcher)]
    proc = subprocess.run(command, capture_output=True, text=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify.log").write_text(proc.stdout + proc.stderr)
    report_path = out / "godot-report.json"
    if not report_path.is_file():
        raise VerifyError(f"The verifier wrote no report; see {out / 'verify.log'}")
    return json.loads(report_path.read_text())


def check_report(report, model_sha, clip, duration_s, fps, version_prefix):
    problems = []
    if report.get("status") != "passed":
        problems.append(f"Godot verification did not pass: {report.get('errors')}")
    if report.get("modelSha256") != model_sha:
        problems.append("The report is for another GLB")
    version = (report.get("engine") or {}).get("version", "")
    if not version.startswith(version_prefix):
        problems.append(f"Expected Godot {version_prefix}, the report says {version!r}")
    clips = {a["name"]: a for a in (report.get("facts") or {}).get("animations", [])}
    if clip not in clips:
        problems.append(f"Godot imported no clip {clip!r} (has {sorted(clips)})")
    else:
        if clips[clip]["tracks"] < 1:
            problems.append(f"Clip {clip!r} imported with no tracks")
        if abs(clips[clip]["lengthSeconds"] - duration_s) > 1.5 / fps:
            problems.append(f"Clip {clip!r} is {clips[clip]['lengthSeconds']} s in Godot, expected {duration_s} s")
    return problems


def export_bundle(files, destination):
    destination = Path(destination).resolve()
    if "game/assets" in destination.as_posix():
        raise StageError("export never writes game assets; shipping needs the owner's approval and the project's build")
    if destination.exists():
        raise StageError(f"Destination exists: {destination}")
    destination.mkdir(parents=True)
    manifest = {}
    for name, path in files.items():
        shutil.copyfile(path, destination / name)
        manifest[name] = sha256_file(destination / name)
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return destination
