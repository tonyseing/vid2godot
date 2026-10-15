---
name: game-asset-animation-pipeline
description: Use when a rigged 3D creature or character needs a game animation clip (walk, trot, lope, gallop, attack) copied from an AI-generated video, or when someone asks to animate a GLB with "the video method", AI mocap, silhouette fitting, or a video model such as Seedance, Kling, Veo or Wan.
---

# vid2godot: rigged GLB → AI side-view video → fitted clip → Godot

A video model films your own rig, rendered side-on, moving; the bones are fitted frame by frame to its
silhouette. The tool exists: your plan is its commands, in order. Write no fitter or GLB patcher; keyframe
nothing by code.

## Check the request first

| If | Then |
|---|---|
| No skeleton or weights (a Tripo mesh), or `contract` reports a problem | Stop. **REQUIRED SKILL: game-asset-3d-pipeline** rigs it first |
| The motion leaves the side plane (a person lifting, a held prop, a turn) | Refuse: a side camera sees no depth, so the outline fits while the pose is wrong. It needs the two-view method, not yet supported |
| The owner named a video model | Pass that exact id as `--model-id` |
| The owner delegated by a stated criterion (e.g. a bake-off for best performance per price) | Apply it and record the result |
| Neither | Ask the owner which model |

## The plan

From this skill's directory: `python3 anim/pipeline.py <command> --work asset-work/<area>/<slug>/anim/<clip>`
(optional `--blender <path>`). `status` and `next` show where you are.

| Command | What you do |
|---|---|
| `init --model <glb> --clip <name>` | Start the ledger |
| `contract --project-contract <json>` | Check the rig |
| `brief --file <json>` | `clip subject motion gait loop planar videoSeconds`; a loop needs two cycles |
| `reference` | Open `reference/reference.png` |
| `generate --model-id <id> [--max-cost USD]` | Two candidates |
| `accept --candidate N --checks <json>` | After opening both strips and full frames; checks in `reference/prompting.md` |
| `ingest` | Drops pulldown/held frames by cadence. Read `flags`, `warnings` |
| `fit` | Solves a clip-wide camera scale and leg identity. Prints `lowFrames` |
| `lab` | Open `review/lab.html` and every low frame's overlay |
| `gridframe --frame N`; `fit --marks <json> --span A:B` | Fix flips (below) |
| `facts` | Sub-frame cycle, one resampled cycle, root drift, sidecar entry |
| `bake` | Keys that cycle minus drift: a seamless seam. Open `bake/sheet.png`, `bake/closeup.png` |
| `review --template`; `review --file <json>` | Fill it from what you saw; explain or refit each low frame |
| `verify --godot47 <bin> --godot45 <bin> --launcher <launcher>` | Activation-suppressed phone/tablet captures on both engines via game-asset-3d-pipeline's `verify.py`. Open every capture |
| `export --destination asset-work/<area>/<slug>/evaluation/<clip>` | The owner's bundle, with captures and lab |

## Leg identity

Near and far legs cast one silhouette. A pair exchanged for the whole clip is a valid mirrored lead; a flip
partway through fails: a leg jumps sides at a `leg_crossing:` flag, or `facts` says a paw touches down
twice per cycle. Recipe: `gridframe --frame N` just after the crossing, read both paws off `ingest/grid/<N>.png`,
add them to one marks file (`--marks` replaces the last), `fit --marks marks.json --span A:B` (B exclusive), `lab`.

```json
{"frames": {"47": {"paw_fl": [612, 540], "paw_fr": [655, 538]}}}
```

Keys are the contract's `paws` bones. More fixes: `reference/failures.md`.

## Shipping

`export` refuses `game/assets`; this skill ends there. Show the owner the bundle. Passing `review` and `verify`
is mechanical, not approval: the clip enters the game only after the owner approves it. Owner notes ("frame 47,
legs jump back") become marks or a span refit and a new review; after two failed artistic rounds, ask whether to
continue or park.

## Rationalizations (from agents without this skill)

| Thought | Reality |
|---|---|
| "None of the scripts named below exist yet. I would write them" | They exist: the command table. |
| "Timing and stride come from re-timing to 24 frames and from the paw-planting step" | `ingest` and `facts` measure them. |
| "Replace only the `lope` samplers and accessors in the GLB with pygltflib" | `bake` writes an evaluation GLB. |
| "Under the project's rule for image models, the harness or the owner picks the model" | That rule is for images. The owner names the video model, or a criterion. |
| "The footfall phases … decide that" / "Hold the far legs to the near legs' phase offset" / "paw keypoints from an animal pose model" | Marks and `fit --span` decide identity. |
| "Depth is not observed … It comes from priors" / "It mostly happens in the side plane" / "A second clip filmed from the front, used only to check" | Unseen depth is why you refuse. |
| "Steps 1–8 are ordinary Blender and Python work" / "try `parent_set(type='ARMATURE_AUTO')` first" / "Treat this deer as a motion prototype" | Rigging is game-asset-3d-pipeline's. Hand it over. |
| "If it passes: I'll replace only the `lope` animation in `wolf.glb`" / "Before overwriting, I'll keep a backup" | Your checks are not approval; a backup is not either. |
| "Passing these checks isn't your artistic sign-off" (then planning the copy) / "You see the new lope … before anything is committed" | Approval precedes writing `game/assets`. Stop at `export`. |
