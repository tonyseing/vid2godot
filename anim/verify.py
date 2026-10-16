"""Godot checks through game-asset-3d-pipeline's verifier, and the evaluation bundle export.

Task 19: verify shows the clip playing. For each of PHASES the verifier's folder is copied and its capture script
patched (`phase_verifier`) to seek the clip to that phase of its length before every capture, to name each PNG for the
phase and to record the phase, the clip and the clip time in each capture entry; the 3D skill's own files are never
changed. A patch marker the capture script no longer holds is a refusal, not a capture of the rest pose."""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from anim.ledger import StageError, sha256_file

DEFAULT = Path.home() / ".agents/skills/game-asset-3d-pipeline/forge/godot/verify.py"
PHASES = (0.0, 1 / 3, 2 / 3)
CAPTURE_SCRIPT = "capture_asset.gd"
SEEK = """
\tvar clip_player: AnimationPlayer = null
\tfor node in model.find_children("*", "AnimationPlayer", true, false):
\t\tif (node as AnimationPlayer).has_animation(CLIP_NAME):
\t\t\tclip_player = node as AnimationPlayer
\t\t\tbreak
\tif clip_player == null:
\t\t_fail("No AnimationPlayer in the GLB holds the clip " + CLIP_NAME)
\t\treturn
\tvar clip_seconds: float = clip_player.get_animation(CLIP_NAME).length * CLIP_PHASE
\tclip_player.play(CLIP_NAME)
\tclip_player.seek(clip_seconds, true)
\tclip_player.pause()
"""
# (what the 3D skill's capture script holds, what vid2godot puts there); each must occur exactly once.
PATCHES = (("extends SceneTree\n", "extends SceneTree\n\nconst CLIP_NAME := {clip}\nconst CLIP_PHASE := {phase}\n"
                                    "const PHASE_TAG := {tag}\n"),
           ("\tscene.add_child(model)\n", "\tscene.add_child(model)\n" + SEEK.lstrip("\n")),
           ('"captures/%s-%s.png" % [profile["name"], view["name"]]',
            '"captures/phase%03d-%s-%s.png" % [PHASE_TAG, profile["name"], view["name"]]'),
           ('captures.append({"path": filename,',
            'captures.append({"phase": CLIP_PHASE, "clip": CLIP_NAME, "clipSeconds": clip_seconds, "path": filename,'))


class VerifyError(StageError):
    pass


def verifier(explicit=None):
    path = Path(explicit) if explicit else DEFAULT
    if not path.is_file():
        raise VerifyError(f"Godot checks use game-asset-3d-pipeline's forge/godot/verify.py, not found at {path}; "
                          "install that skill (img2godot) or pass --threed-verify")
    return path


def phase_tag(phase):
    return int(round(phase * 1000))


def phase_verifier(script, clip, phase, out_dir):
    """A copy of the verifier's folder in `out_dir` whose capture script seeks `clip` to `phase` (a share of its
    length) before capturing; returns the copy's verify.py. Refuses when the capture script lacks a patch point."""
    script, out_dir = Path(script), Path(out_dir)
    source = script.parent / CAPTURE_SCRIPT
    if not source.is_file():
        raise VerifyError(f"The 3D skill's capture script is missing beside {script}: {source}")
    text = source.read_text()
    for marker, replacement in PATCHES:
        if text.count(marker) != 1:
            raise VerifyError(f"The 3D skill's capture script changed: {source} holds {text.count(marker)} of "
                              f"{marker.strip()!r}, where vid2godot seeks the clip to each phase; update "
                              f"anim/verify.py's PATCHES")
        text = text.replace(marker, replacement.replace("{clip}", json.dumps(clip))
                            .replace("{phase}", f"{phase:.6f}").replace("{tag}", str(phase_tag(phase))))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(script.parent, out_dir, ignore=shutil.ignore_patterns("__pycache__"))
    (out_dir / CAPTURE_SCRIPT).write_text(text)
    return out_dir / script.name


def combine(reports, script):
    """One engine's report from its phase runs [(phase, report path, report)]: the first run's fields, every run's
    captures (each with its phase), the phases and runs, and the 3D skill's verifier the patches were made from."""
    first = reports[0][2]
    source = Path(script).parent / CAPTURE_SCRIPT
    return {**first, "captures": [c for _, _, r in reports for c in (r.get("captures") or [])],
            "phases": [round(p, 6) for p, _, _ in reports],
            "runs": [{"phase": round(p, 6), "report": str(path), "status": r.get("status"),
                      "captureScriptSha256": sha256_file(Path(path).parent / "verifier" / CAPTURE_SCRIPT)
                      if (Path(path).parent / "verifier" / CAPTURE_SCRIPT).is_file() else None}
                     for p, path, r in reports],
            "verifier": {"path": str(script), "sha256": sha256_file(script),
                         "captureScriptSha256": sha256_file(source) if source.is_file() else None}}


def run_verify(script, model, out, godot, launcher=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "godot-report.json"
    # A previous run's report must never stand in for this run's.
    report_path.unlink(missing_ok=True)
    command = [sys.executable, str(script), "--model", str(model), "--out", str(out), "--godot", str(godot)]
    if launcher:
        command += ["--launcher", str(launcher)]
    proc = subprocess.run(command, capture_output=True, text=True)
    (out / "verify.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        errors = ""
        if report_path.is_file():
            try:
                errors = f": {json.loads(report_path.read_text()).get('errors')}"
            except (ValueError, AttributeError):
                pass
        raise VerifyError(f"The verifier exited {proc.returncode}{errors}; see {out / 'verify.log'}")
    if not report_path.is_file():
        raise VerifyError(f"The verifier wrote no report; see {out / 'verify.log'}")
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict):
        raise VerifyError(f"The verifier's report is not a JSON object: {report_path}")
    return report


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_report(report, model_sha, clip, duration_s, fps, version_prefix, launcher_used=False, phase=None):
    if not isinstance(report, dict):
        return ["The Godot report is not a JSON object"]
    problems = []
    if report.get("status") != "passed":
        problems.append(f"Godot verification did not pass: {report.get('errors')}")
    if report.get("modelSha256") != model_sha:
        problems.append("The report is for another GLB")
    engine = report.get("engine")
    version = engine.get("version") if isinstance(engine, dict) else None
    if not isinstance(version, str) or not version.startswith(version_prefix):
        problems.append(f"Expected Godot {version_prefix}, the report says {version!r}")
    if launcher_used:
        if report.get("mode") != "activation-suppressed capture":
            problems.append(f"The run was not an activation-suppressed capture (mode {report.get('mode')!r})")
        captures = report.get("captures")
        if not isinstance(captures, list) or not captures:
            problems.append("The report lists no phone/tablet captures")
        else:
            for item in captures:
                path = item.get("path") if isinstance(item, dict) else None
                if not isinstance(path, str) or not Path(path).is_file():
                    problems.append(f"Capture missing on disk: {path!r}")
                if phase is not None and isinstance(item, dict):
                    got = item.get("phase")
                    if not _number(got) or abs(got - phase) > 1e-6 or item.get("clip") != clip:
                        problems.append(f"Capture {path!r} records phase {got!r} of clip {item.get('clip')!r}, not "
                                        f"phase {phase:.6f} of {clip!r}")
    facts = report.get("facts")
    animations = facts.get("animations") if isinstance(facts, dict) else None
    if not isinstance(animations, list):
        problems.append(f"The report's facts.animations is not a list: {animations!r}")
        return problems
    names, entries = set(), []
    for entry in animations:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            problems.append(f"Malformed animation entry in the report: {entry!r}")
            continue
        names.add(entry["name"])
        if entry["name"] == clip:
            entries.append(entry)
    if not entries:
        problems.append(f"Godot imported no clip {clip!r} (has {sorted(names)})")
    # Two AnimationPlayers may report the same clip; every one of them must pass.
    for entry in entries:
        tracks, length = entry.get("tracks"), entry.get("lengthSeconds")
        if not _number(tracks):
            problems.append(f"Clip {clip!r} has no track count in the report: {tracks!r}")
        elif tracks < 1:
            problems.append(f"Clip {clip!r} imported with no tracks")
        if not _number(length):
            problems.append(f"Clip {clip!r} has no length in the report: {length!r}")
        elif abs(length - duration_s) > 1.5 / fps:
            problems.append(f"Clip {clip!r} is {length} s in Godot, expected {duration_s} s")
    return problems


def _refuse_game_assets(destination):
    """Refuse any destination inside a game/assets folder, however it is spelled or linked."""
    refusal = StageError("export never writes game assets; shipping needs the owner's approval and the project's build")
    resolved = Path(destination).resolve()
    parts = [part.casefold() for part in resolved.parts]
    if any(parts[i:i + 2] == ["game", "assets"] for i in range(len(parts) - 1)):
        raise refusal
    existing = resolved
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    chain = [existing, *existing.parents]
    game_assets = [root / "game" / "assets" for root in chain if (root / "game" / "assets").is_dir()]
    for ancestor in chain:
        if any(os.path.samefile(ancestor, folder) for folder in game_assets):
            raise refusal


def export_bundle(files, destination):
    destination = Path(destination).resolve()
    _refuse_game_assets(destination)
    if destination.exists():
        raise StageError(f"Destination exists: {destination}")
    for name in files:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name == "MANIFEST.json":
            raise StageError(f"Bad bundle file name: {name!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Build next to the destination and rename into place only once every copy and the manifest succeeded.
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        manifest = {}
        for name, path in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            manifest[name] = sha256_file(target)
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        if destination.exists():
            raise StageError(f"Destination exists: {destination}")
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
