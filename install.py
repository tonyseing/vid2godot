#!/usr/bin/env python3
"""Install the vid2godot skill atomically; preserve displaced skills for rollback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

NAME = "game-asset-animation-pipeline"
RESOURCES = ("SKILL.md", "README.md", "LICENSE", "PROVENANCE.json", "reference", "anim")
MARKER = ".vid2godot-install.json"


def link_state(link, target):
    if not link.exists() and not link.is_symlink():
        return "absent"
    if link.is_symlink() and Path(os.readlink(link)) == target:
        return "ours"
    return "foreign"


def ensure_link(link, target):
    if link_state(link, target) == "absent":
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    return str(link)


def try_link(link, target):
    """Return (link path, None) or (None, error message); never raises, so the caller can still report."""
    try:
        return ensure_link(link, target), None
    except OSError as error:
        return None, str(error)


def report(payload, link_result):
    """Print the JSON result with the Claude link outcome; return the process exit code."""
    linked, error = link_result
    payload["claudeLink"] = linked
    if error:
        payload["claudeLinkError"] = error
    print(json.dumps(payload))
    return 1 if error else 0


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink: {path}")
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_file() and path != root / MARKER:
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def owned(path):
    if path.is_symlink() or not path.is_dir():
        return False
    marker = path / MARKER
    if not marker.is_file():
        return False
    data = json.loads(marker.read_text())
    return data.get("name") == NAME and data.get("pipeline") == "vid2godot" and data.get("files") == inventory(path)


def intact_backup(path):
    if not owned(path):
        return False
    expected = f"{NAME}-{digest(inventory(path))}"
    if path.name == expected:
        return True
    return path.name.startswith(expected + "-") and path.name[len(expected) + 1:].isdigit()


def preserve(path, backups):
    files = inventory(path)
    backup = backups / f"{NAME}-{digest(files)}"
    if backup.exists():
        if inventory(backup) != files:
            raise ValueError(f"Backup collision: {backup}")
        # Preserve even identical installations without deleting their directory.
        index = 1
        while backup.with_name(f"{backup.name}-{index}").exists():
            index += 1
        backup = backup.with_name(f"{backup.name}-{index}")
    path.rename(backup)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents-home", type=Path, default=Path.home() / ".agents")
    parser.add_argument("--claude-home", type=Path, default=Path.home() / ".claude")
    parser.add_argument("--rollback", type=Path, help="Exact backup directory printed by an earlier installation")
    args = parser.parse_args()
    home = args.agents_home.expanduser().absolute()
    skills = home / "skills"
    backups = home / "skill-backups"
    target = skills / NAME
    link = args.claude_home.expanduser().absolute() / "skills" / NAME
    if link_state(link, target) == "foreign":
        raise ValueError(f"Refusing unrelated Claude skill entry: {link}")
    for path in (home, skills, backups):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink directory: {path}")
    if target.exists() or target.is_symlink():
        if not owned(target):
            raise ValueError(f"Refusing unrelated or modified skill directory: {target}")
    if args.rollback:
        source = args.rollback.expanduser().absolute()
        if source.parent != backups or not source.is_dir() or not intact_backup(source):
            raise ValueError("Rollback must name an unmodified recognized backup under .agents/skill-backups")
        inventory(source)
        skills.mkdir(parents=True, exist_ok=True)
        backups.mkdir(parents=True, exist_ok=True)
        displaced = preserve(target, backups) if target.exists() else None
        try:
            source.rename(target)
        except Exception:
            if displaced:
                displaced.rename(target)
            raise
        return report({"restored": str(target), "displacedBackup": str(displaced) if displaced else None},
                      try_link(link, target))
    source = Path(__file__).resolve().parent
    for resource in RESOURCES:
        if not (source / resource).exists():
            raise ValueError(f"Missing fork resource: {resource}")
    skills.mkdir(parents=True, exist_ok=True)
    backups.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{NAME}-", dir=home))
    displaced = None
    try:
        for resource in RESOURCES:
            src = source / resource
            if src.is_symlink():
                raise ValueError(f"Refusing symlink resource: {src}")
            if src.is_dir():
                inventory(src)
                shutil.copytree(src, stage / resource, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, stage / resource)
        files = inventory(stage)
        marker = {"name": NAME, "pipeline": "vid2godot", "contentSha256": digest(files), "files": files}
        (stage / MARKER).write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
        if target.exists():
            if inventory(target) == files:
                return report({"installed": str(target), "unchanged": True}, try_link(link, target))
            displaced = preserve(target, backups)
        try:
            stage.rename(target)
        except Exception:
            if displaced:
                displaced.rename(target)
            raise
        return report({"installed": str(target), "contentSha256": digest(files),
                       "rollbackBackup": str(displaced) if displaced else None}, try_link(link, target))
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    try:
        code = main()
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(str(error))
    raise SystemExit(code)
