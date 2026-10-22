---
name: game-asset-animation-pipeline
description: Use when a rigged 3D creature or character needs a game animation clip (walk, trot, lope, gallop, attack) copied from an AI-generated video, or when someone asks to animate a GLB with "the video method", AI mocap, silhouette fitting, or a video model such as Seedance, Kling, Veo or Wan.
---

# vid2godot: rigged GLB → AI side-view video → fitted clip → Godot

A video model films your rig side-on, moving; the bones are fitted to its silhouette. The tool exists: your plan
is its commands, in order. Write no fitter or GLB patcher; keyframe nothing by code.

## Check the request first

| If | Then |
|---|---|
| No skeleton or weights (a Tripo mesh), or `contract` reports a problem | Stop. **REQUIRED SKILL: game-asset-3d-pipeline** rigs it first |
| The motion leaves the side plane (a person lifting, a held prop, a turn) | Refuse: a side camera sees no depth, so the outline fits a wrong pose; that needs the two-view method, not yet supported |
| The owner named a video model | Pass that exact id as `--model-id` |
| The owner delegated by a stated criterion (e.g. best performance per price) | Apply it and record the result |
| Neither | Ask the owner which model |

## The plan

From the project root: `python3 <skill>/anim/pipeline.py <command> --work asset-work/<area>/<slug>/anim/<clip>`
(optional `--blender <path>`). `status`, `next` show where you are.

| Command | What you do |
|---|---|
| `init --model <glb> --clip <name>` | Start the ledger |
| `contract --project-contract <json>` | Check the rig |
| `brief --file <json>` | `clip subject motion gait loop planar videoSeconds`; a loop needs two cycles. Small fast animal: `slowMotion` |
| `reference` | Open `reference/reference.png` |
| `generate --model-id <id> [--max-cost USD] [--no-audio]` | Two candidates (`--no-audio` sends `generate_audio: false`) |
| `accept --candidate N --checks <json>` | After opening both strips and full frames (`reference/prompting.md`) |
| `ingest` | Drops pulldown repeats; read `flags`, `warnings` |
| `preflight --source-observations <json> [--witness <diagnostic-poses.json>]` | Independently solve representative source frames with the existing fitter before a full fit. Open `articulation/preflight/lab.html`; inspect source, projected mesh/skeleton, uncertainty and residuals. Never writes canonical fit or ledger stages |
| `fit [--source-observations <json>] [--seed N]` | Clip-wide camera scale and leg identity; restarts, swaps, a periodic second pass. `--seed` picks the restart draw (recorded; omitted = seed 0). With observations, requires current passing preflight and writes independent articulation assessment in addition to `lowFrames` |
| `lab` | Open `review/lab.html`, every low frame's overlay |
| `assess` | Reassess current canonical fit against stored source observations; does not refit or waive failures |
| `gridframe --frame N`; `fit --marks <json> --span A:B` | Fix flips (below) |
| `facts [--mass-kg KG] [--accept-gait REASON] [--accept-retime REASON]` | Period and averaged cycle from the video (refused under 0.85 IoU); contacts from the motion (grounded, world-still, leg-scaled). Refuses footfalls unlike the brief's `gait` unless accepted; refuses a stride under 10 frames (`slowMotion`); retimes a slow video to the gait's body-size cadence band (brief `massKg` or `--mass-kg`); keys (24+ per stride) pin planted paws above ground, trunk within 60 Hz jerk; rechecked. `reference/motion.md` |
| `bake` | Keys facts' cycle; refuses if a paw misses facts' pin. Open `bake/sheet.png`, `bake/closeup.png` |
| `review --template`; `review --file <json>` | Fill it from what you saw; explain or refit low frames |
| `verify --godot47 <bin> --godot45 <bin> --launcher <launcher>` | Phone/tablet captures of the clip playing on both engines via game-asset-3d-pipeline's `verify.py`. Open every capture |
| `export --destination asset-work/<area>/<slug>/evaluation/<clip>` | The owner's bundle |

## Leg identity

Near and far legs cast one silhouette. A pair exchanged for the whole clip is a valid mirrored lead; a flip
partway through is not: a leg jumps sides at a `leg_crossing:` flag, or `facts` says a paw touches down twice.
Recipe: `gridframe --frame N` just after the crossing, read both paws off `ingest/grid/<NNNNN>.png`,
add them to one marks file (`--marks` replaces the last), `fit --marks marks.json --span A:B` (B exclusive), `lab`.

```json
{"frames": {"47": {"paw_fl": [612, 540], "paw_fr": [655, 538]}}}
```

`frames` targets contract paw **tails**. If paws match but the fold is wrong, optional `jointFrames`
(`frame → fitted non-root bone → [x,y]`) targets bone **heads**: forearm = elbow, wrist = wrist.
Use visible, inspected source joints in registered-image pixels; never infer hidden anatomy from toes.
Keep earlier marks when replacing the file. Schema, validation, provenance and refit instructions:
`reference/failures.md`, “Articulated landmarks”.

For new animation work, inspect representative source support/stance, push-off, fold/recovery,
extension and crossing poses as applicable; record uncertainty-bearing observations and run `preflight`
before the full fit. The optional CLI argument preserves legacy compatibility, not a recommendation to
skip the checkpoint. See `reference/failures.md`, “Source articulation checkpoint”.
`jointFrames` alone is a fitting input, not evidence that articulation passed.
A matched toe or high IoU cannot excuse a missed elbow.
Recorded hidden/ambiguous/tentative correspondence remains unresolved even when projected points match.
`facts`, `bake`, `review`, `verify` and `export` block when recorded articulation evidence fails or is stale.
Legacy workspaces remain usable and explicitly show `not_assessed`, never an implied articulation pass.
One fold observation only supports that fold; it cannot verify recovery over unobserved motion phases.

### Restart noise: one fit is one draw

Restarts are random, so any change to marks, spans or inputs reshuffles every later restart: a refit that scores
worse (fit IoU, facts' averaged cycle, identity spread) may be a bad draw, not a bad change. Before calling a
change worse or better, or a clip blocked, run each side with at least three `--seed` values in separate work copies
and compare the spread, not single numbers. A gate still judges the one fit you ship; seeds never average a gate.
See `reference/failures.md`, “Restart noise”.

## Shipping

`export` refuses `game/assets`. Show the owner the bundle. Passing `review` and `verify`
is mechanical, not approval: the clip enters the game only once the owner approves. Owner notes ("frame 47, legs
jump back") become marks or a span refit and a new review; after two failed artistic rounds, ask whether to go on.

Do not bypass these stages with a substitute fitter, angle keys or GLB patcher.
See `reference/failures.md` for detailed failure remedies and pipeline guardrails.
