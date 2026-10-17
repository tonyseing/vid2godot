# vid2godot

vid2godot is the agent skill `game-asset-animation-pipeline`. It turns a video of your own rigged model into a
game animation clip: it renders the rig exactly side-on, has a named AI video model film it moving from that
frame, fits the bones frame by frame until the rig's silhouette matches the video, measures one clean cycle and
its contacts from the motion, checks its gait and body-size cadence, bakes a 60 Hz evaluation GLB with the planted
paws pinned, gates it on a hash-bound review and on Godot 4.7 and 4.5 captures of the clip playing, and exports a
bundle for the owner to approve ([reference/motion.md](reference/motion.md)). It never writes `game/assets`. Agents follow [SKILL.md](SKILL.md).

## Install

```bash
git clone --branch anim-pipeline-2026-10-16.5 https://github.com/tonyseing/vid2godot.git ~/.vid2godot/repo
python3 ~/.vid2godot/repo/install.py
```

The installer puts the skill at `~/.agents/skills/game-asset-animation-pipeline`, links it from
`~/.claude/skills/`, and moves a recognized earlier install to `~/.agents/skill-backups/`. It refuses an
unrelated or modified directory.

Rollback: `python3 ~/.vid2godot/repo/install.py --rollback <backup directory printed by the install>`.

## Prerequisites

- Python 3 with numpy, scipy and Pillow
- ffmpeg and ffprobe
- Blender 5.2 (`--blender <path>` or `VID2GODOT_BLENDER`)
- The Godot 4.7 and 4.5 engines, and the project's activation-suppressed launcher
- `OPENROUTER_API_KEY` for `generate`
- game-asset-3d-pipeline installed (img2godot); `verify` uses its `forge/godot/verify.py` from
  `~/.agents/skills/game-asset-3d-pipeline`, or the path given to `--threed-verify`

## Tests

```bash
python3 -m unittest discover -s tests -t . -v
VID2GODOT_BLENDER=/Applications/Blender.app/Contents/MacOS/Blender python3 -m unittest discover -s tests -t . -v
```

Blender tests are skipped unless `VID2GODOT_BLENDER` is set. Isolated installer checks pass both
`--agents-home` and `--claude-home` with temporary directories. The skill text's agent tests are in
`tests/skill/` (scenarios, and the baseline without the skill).

## Provenance

[PROVENANCE.json](PROVENANCE.json) records the origin, the license (Apache-2.0) and the game-asset-3d-pipeline
version this depends on.
