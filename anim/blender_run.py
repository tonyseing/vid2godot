"""Run an anim/blender script headless with JSON arguments and return the JSON result it writes."""
import json
import os
import subprocess
from pathlib import Path

from anim.ledger import StageError

SCRIPTS = Path(__file__).resolve().parent / "blender"
DEFAULT = "/Applications/Blender.app/Contents/MacOS/Blender"


class BlenderError(StageError):
    pass


def blender_path(explicit=None):
    path = Path(explicit or os.environ.get("VID2GODOT_BLENDER") or DEFAULT)
    if not path.is_file():
        raise BlenderError(f"Blender not found at {path}; pass --blender or set VID2GODOT_BLENDER")
    return path


def run_blender(script, args, log_dir, blender=None, timeout=4 * 3600):
    script = Path(script) if isinstance(script, Path) else SCRIPTS / f"{script}.py"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    result = log_dir / f"{script.stem}.result.json"
    result.unlink(missing_ok=True)
    args_path = log_dir / f"{script.stem}.args.json"
    args_path.write_text(json.dumps({**args, "result": str(result)}, indent=2))
    command = [str(blender_path(blender)), "--background", "--factory-startup", "--python-exit-code", "1",
               "--python", str(script), "--", str(args_path)]
    log = log_dir / f"{script.stem}.log"

    def text(data):
        return data.decode(errors="replace") if isinstance(data, bytes) else (data or "")

    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        log.write_text(text(error.stdout) + text(error.stderr))
        raise BlenderError(f"Blender {script.stem} timed out after {timeout}s; see {log}") from error
    except OSError as error:
        log.write_text(str(error))
        raise BlenderError(f"Blender {script.stem} could not start ({error}); see {log}") from error
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not result.is_file():
        raise BlenderError(f"Blender {script.stem} failed (exit {proc.returncode}); see {log}")
    return json.loads(result.read_text())
