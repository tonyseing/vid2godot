"""Godot checks through game-asset-3d-pipeline's verifier, and the evaluation bundle export.

Task 19: verify shows the clip playing. For each of PHASES the verifier's folder is copied and its capture script
patched (`phase_verifier`) to seek the clip to that phase of its length before every capture, to name each PNG for the
phase and to record the phase, the clip and the clip time in each capture entry; the 3D skill's own files are never
changed. A patch marker the capture script no longer holds is a refusal, not a capture of the rest pose.

Fix round 1, ruling 5: that the clip plays is measured, not self-reported. The patch also records the paw bones'
world positions at each phase, and `phase_evidence` requires, for each profile and view, that the captures of every
two phases differ (image hash, and a share of pixels over PIXEL_FLOOR differing by more than PIXEL_TOLERANCE in some
channel), and that some paw bone moves more than PAW_MOVE_MIN between every two phases."""
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
PIXEL_FLOOR = 0.001        # share of a capture's pixels that must differ between two phases
PIXEL_TOLERANCE = 8        # a pixel differs when a channel differs by more than this (of 255)
PAW_MOVE_MIN = 0.005       # m: some paw bone moves at least this far between two phases
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
\tvar paw_positions := {}
\tvar paw_skeleton: Skeleton3D = null
\tfor node in model.find_children("*", "Skeleton3D", true, false):
\t\tpaw_skeleton = node as Skeleton3D
\t\tbreak
\tif paw_skeleton == null:
\t\t_fail("No Skeleton3D in the GLB to read the paw bones from")
\t\treturn
\tfor paw_name in PAW_BONES:
\t\tvar bone_index: int = paw_skeleton.find_bone(str(paw_name))
\t\tif bone_index < 0:
\t\t\t_fail("The GLB's skeleton has no paw bone " + str(paw_name))
\t\t\treturn
\t\tvar paw_origin: Vector3 = paw_skeleton.global_transform * paw_skeleton.get_bone_global_pose(bone_index).origin
\t\tpaw_positions[str(paw_name)] = [paw_origin.x, paw_origin.y, paw_origin.z]
"""
# (what the 3D skill's capture script holds, what vid2godot puts there); each must occur exactly once.
PATCHES = (("extends SceneTree\n", "extends SceneTree\n\nconst CLIP_NAME := {clip}\nconst CLIP_PHASE := {phase}\n"
                                    "const PHASE_TAG := {tag}\nconst PAW_BONES := {paws}\n"),
           ("\tscene.add_child(model)\n", "\tscene.add_child(model)\n" + SEEK.lstrip("\n")),
           ('"captures/%s-%s.png" % [profile["name"], view["name"]]',
            '"captures/phase%03d-%s-%s.png" % [PHASE_TAG, profile["name"], view["name"]]'),
           ('captures.append({"path": filename,',
            'captures.append({"phase": CLIP_PHASE, "clip": CLIP_NAME, "clipSeconds": clip_seconds, "paws": paw_positions, '
            '"path": filename,'))


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


def phase_verifier(script, clip, phase, out_dir, paw_bones=()):
    """A copy of the verifier's folder in `out_dir` whose capture script seeks `clip` to `phase` (a share of its
    length) before capturing and records `paw_bones`' world positions there; returns the copy's verify.py. Refuses
    when the capture script lacks a patch point."""
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
                            .replace("{phase}", f"{phase:.6f}").replace("{tag}", str(phase_tag(phase)))
                            .replace("{paws}", json.dumps([str(b) for b in paw_bones])))
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
    # The worst run's status: one failed phase fails the engine's report.
    status = next((r.get("status") for _, _, r in reports if r.get("status") != "passed"), "passed")
    return {**first, "status": status, "captures": [c for _, _, r in reports for c in (r.get("captures") or [])],
            "phases": [round(p, 6) for p, _, _ in reports],
            "runs": [{"phase": round(p, 6), "report": str(path), "status": r.get("status"),
                      "captureScriptSha256": sha256_file(Path(path).parent / "verifier" / CAPTURE_SCRIPT)
                      if (Path(path).parent / "verifier" / CAPTURE_SCRIPT).is_file() else None}
                     for p, path, r in reports],
            "verifier": {"path": str(script), "sha256": sha256_file(script),
                         "captureScriptSha256": sha256_file(source) if source.is_file() else None}}


def pixels_differ(a, b, tolerance=PIXEL_TOLERANCE):
    """The share of pixels that differ by more than `tolerance` in some channel (1.0 for different sizes)."""
    import numpy as np
    from PIL import Image
    with Image.open(a) as first, Image.open(b) as second:
        x, y = np.asarray(first.convert("RGB"), int), np.asarray(second.convert("RGB"), int)
    if x.shape != y.shape:
        return 1.0
    return float(np.mean(np.abs(x - y).max(axis=2) > tolerance))


def phase_evidence(runs):
    """That the clip plays, from one engine's phase runs [(phase, report)]: (evidence, problems). For each
    profile/view, every two phases' captures must differ (hash, and pixels over PIXEL_FLOOR); for every two phases,
    some paw bone must move over PAW_MOVE_MIN."""
    views, paws, problems = {}, {}, []
    for phase, report in runs:
        for item in report.get("captures") or []:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            views.setdefault(f"{item.get('profile')}/{item.get('view')}", {})[phase] = item["path"]
            if isinstance(item.get("paws"), dict) and phase not in paws:
                paws[phase] = item["paws"]
    evidence = {"pixelFloor": PIXEL_FLOOR, "pixelTolerance": PIXEL_TOLERANCE, "pawMoveMin": PAW_MOVE_MIN,
                "views": {}, "paws": []}
    phases = [phase for phase, _ in runs]
    for key, by_phase in sorted(views.items()):
        pairs = []
        for i, a in enumerate(phases):
            for b in phases[i + 1:]:
                if a not in by_phase or b not in by_phase:
                    problems.append(f"{key}: no capture at phase {a if a not in by_phase else b:.2f}")
                    continue
                files = [Path(by_phase[a]), Path(by_phase[b])]
                if not all(f.is_file() for f in files):
                    continue          # check_report names the missing file
                same = sha256_file(files[0]) == sha256_file(files[1])
                share = 0.0 if same else pixels_differ(*files)
                pairs.append({"phases": [round(a, 6), round(b, 6)], "sameHash": same,
                              "pixelsDiffer": round(share, 6)})
                if same or share <= PIXEL_FLOOR:
                    problems.append(f"{key}: phases {a:.2f} and {b:.2f} render the same image "
                                    f"({'same hash' if same else 'different hash'}; {share:.4f} of the pixels differ, "
                                    f"floor {PIXEL_FLOOR:g}): the clip is not playing in the captures")
        evidence["views"][key] = pairs
    for i, a in enumerate(phases):
        if a not in paws:
            problems.append(f"phase {a:.2f} records no paw positions: the capture script was not patched to read them")
            continue
        for b in phases[i + 1:]:
            if b not in paws:
                continue
            bones = sorted(set(paws[a]) & set(paws[b]))
            moves = {bone: math.dist(paws[a][bone], paws[b][bone]) for bone in bones}
            largest = max(moves.values(), default=0.0)
            evidence["paws"].append({"phases": [round(a, 6), round(b, 6)], "maxMoveM": round(largest, 6),
                                     "moves": {k: round(v, 6) for k, v in moves.items()}})
            if largest <= PAW_MOVE_MIN:
                problems.append(f"phases {a:.2f} and {b:.2f}: no paw bone moved (largest {largest:.4f} m, floor "
                                f"{PAW_MOVE_MIN:g} m; {', '.join(bones) or 'no bones'}): the clip is not playing")
    return evidence, problems


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
