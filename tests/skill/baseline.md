# Skill RED baseline (no skill), sandbox, 2026-10-15

## S1 rep 1

Grades: [('P1', False), ('P2', False), ('P3', True), ('P4', False)]

Rationalizations: ["Under the project's rule for image models, the harness or the owner picks the model. I would only record the model id it returns, and wouldn't run a paid model here.", "Near vs far legs: a side view can't tell which leg is in front. The footfall phases from `wolf.anim.json` and `camera.json`'s near side decide that.", "A side view can't show the legs' sideways spread or swaps between near and far legs. That's why the footfall phases and the old lope are used as guides.", "The video's lope speed has nothing to do with our 4.2 m/s. Timing and stride come from re-timing to 24 frames and from the paw-planting step, not from the video.", 'Below that, go back to step 2 with another video or fix the mask.']

<details><summary>Answer</summary>

Here is how I'd animate the wolf's lope from an AI video, written for the owner. I only looked at the sandbox: I read the GLB's JSON header and `wolf.anim.json`, wrote nothing, called no paid API and didn't launch Godot. None of the scripts named below exist yet. I would write them for this asset.

## What's there now

- **`game/assets/models/wolf.glb`**: one mesh, `wolf_mesh`, with 9,462 triangles. It has a 35-bone skin called `wolf_rig` and one albedo texture, `wolf_albedo`. It holds nine clips (alert, gallop, howl, lope, lunge, prowl, search, sniff, tumble), each with 105 channels, which is 35 bones × position/rotation/scale.
- **Current lope clip**: 24 frames at 60 fps, so 0.4 s. It loops, and moves at `authored_speed_mps` 4.2 with `stride_m` 1.68.
  - Footfall phases: hind L 0.0, hind R 0.18, front L 0.2, front R 0.4. Duty factor 0.32 for each paw.
  - `root_motion` is false. +X is forward and +Y is up.
- **Tools on this machine**: Blender 5.2.1 at `/Applications/Blender.app/Contents/MacOS/Blender` (not on PATH), ffmpeg 9.0.2 and Python 3. OpenCV is not installed.

The new lope has to replace only the lope clip and keep that data contract. The game sets playback rate as ground speed divided by authored speed, so it relies on these numbers to keep planted paws from sliding.

## Steps and commands

**0. Set up**
- Find or create the ticket on the board (`python3 tools/board.py list`, or `gh issue create` with label `fox-3d`, then `move <n> "In progress"`).
- Work in a `.worktrees/` worktree, not under `~/orca`.
- Keep provenance in `asset-work/creatures/wolf/anim/lope/`.
- Install the Python tools:
```
python3 -m venv .venv-anim && .venv-anim/bin/pip install numpy scipy pillow opencv-python-headless pygltflib
```
- Record the starting hashes and keep a copy of the old lope:
```
B=/Applications/Blender.app/Contents/MacOS/Blender
W=asset-work/creatures/wolf/anim/lope
shasum -a 256 game/assets/models/wolf.glb game/assets/models/wolf.anim.json > $W/inputs.sha256
.venv-anim/bin/python $W/dump_clip.py game/assets/models/wolf.glb lope > $W/lope_before.json
```

**1. Render our own wolf side-on on plain grey**

`$W/render_ref.py` imports the GLB and poses the first frame of the current lope, so the video starts mid-stride.
- Orthographic camera, square to the side, so +X (forward) points right on screen. It records which side of the wolf faces the camera.
- Flat 50% grey background. Workbench flat lighting or an emission-only material, with no shadow and no floor line.
- 1280×720, with the whole body and tail inside the frame and a margin.

It writes three things:
- `start.png`, the colour frame
- `start_mask.png`, the outline mask
- `camera.json`, with the ortho scale, camera position, image size and the near side. Everything after this depends on that file.

```
$B -b --factory-startup --python $W/render_ref.py -- --glb game/assets/models/wolf.glb --clip lope --frame 0 --out $W/ref/
```

**2. Have an AI video model film it moving**
- Use image-to-video starting from `ref/start.png`. Under the project's rule for image models, the harness or the owner picks the model. I would only record the model id it returns, and wouldn't run a paid model here.
- Prompt: *"The same grey wolf lopes steadily in place, side view, locked-off static camera, the wolf stays the same size and position, plain flat grey background, no ground, no shadow, no scenery, no text, the whole body including the tail visible."* Running in place keeps the outline lined up with our fixed camera.
- Make 2 or 3 versions. Reject any with camera drift or zoom, a background that changes, a body that morphs, extra or merged legs, or the wolf leaving the frame.
- Record the model id, prompt, seed, settings and the video's SHA-256 in `$W/video/provenance.json`.

**3. Pull out frames and outlines**
```
ffmpeg -i $W/video/cand1.mp4 -vsync 0 $W/frames/%04d.png
.venv-anim/bin/python -I $W/segment.py --frames $W/frames --bg-from-border --out $W/masks
```
- `segment.py` measures how far each pixel is from the grey background (an LAB colour distance), cleans up with open/close filters and keeps the biggest blob.
- It checks the background stays even and fits a scale and offset to match the outline in `start_mask.png`. If the video zooms or shifts, that becomes a 2D correction stored for each frame.

**4. Find one clean stride**
```
.venv-anim/bin/python -I $W/cycle.py --masks $W/masks --out $W/cycle.json
```
- It finds the stride length by matching each outline against later ones (outline overlap over time) and picks the steadiest single stride.
- It resamples that stride to exactly 24 frames, so it loops at 0.4 s.

**5. Copy the motion onto the skeleton, frame by frame, by matching the outline**
```
.venv-anim/bin/python -I $W/fit_pose.py --glb game/assets/models/wolf.glb --camera $W/ref/camera.json \
  --cycle $W/cycle.json --init-clip lope --prior game/assets/models/wolf.anim.json --out $W/fit/
```
- It reads the mesh, skin weights, inverse bind matrices and rest pose straight from the GLB. It skins the mesh in numpy, projects it through the same ortho camera and draws the wolf's outline.
- **Bones it adjusts:** pelvis height and pitch; spine_01/02, chest, neck_01/02 and head pitch; all four legs (scapula/upperarm/forearm/wrist/paw at the front, thigh/shin/hock/paw at the back); tail_01–06. Rotations are mostly forward/back, with a little roll allowed. The ears and jaw stay as they are in the current lope.
- **What it scores:**
  - how far the drawn outline is from the video's outline, both ways (a two-way chamfer distance)
  - how closely the edges inside the outline match, so legs crossing the body still count
  - joint limits
  - smoothness over time
  - closeness to the current lope's pose at the same phase
- **Near vs far legs:** a side view can't tell which leg is in front. The footfall phases from `wolf.anim.json` and `camera.json`'s near side decide that.
- **Solver:** scipy CMA-ES or L-BFGS, starting from the previous frame and from the old lope. Then one pass over all 24 frames together to make the loop close cleanly.
- **Output:** a `fit/report.json` with each frame's outline overlap (IoU) and edge error, plus an overlay sheet of the video frame with the model's outline drawn on it.
- **Acceptance:** an IoU of at least 0.90 on every frame and no leg swaps. Below that, go back to step 2 with another video or fix the mask.

**6. Clean up and plant the paws**
```
.venv-anim/bin/python -I $W/cleanup.py --fit $W/fit/ --speed 4.2 --out $W/lope_new.json
```
- **Planted paws:** while a paw is on the ground, two-bone IK keeps it on the ground and moving backward at exactly 4.2 m/s. The wolf runs in place, so a planted paw slides back at the running speed.
- **Smoothing:** quaternion continuity, light low-pass filtering, and a seamless blend where the loop joins.
- **Re-measuring:** it recalculates contacts, footfall phase and duty factor from the fitted motion. If the measured stride differs from 1.68 m, it updates `stride_m` and `authored_speed_mps` together, so stride ÷ 0.4 s still equals the authored speed.

**7. Put only the new lope into the GLB**

I'd edit the GLB's data directly with pygltflib rather than round-trip it through Blender, so the mesh, skin, texture and the other eight clips stay byte-for-byte the same.
```
.venv-anim/bin/python -I $W/write_clip.py --glb game/assets/models/wolf.glb --clip lope --data $W/lope_new.json --out $W/wolf.glb
.venv-anim/bin/python -I $W/verify_glb.py --before game/assets/models/wolf.glb --after $W/wolf.glb --clip lope
```
`verify_glb.py` checks:
- the other clips match exactly
- 35 joints, 105 channels in lope, 24 frames at 60 fps
- 9,462 triangles and the same texture

Then copy into place: `cp $W/wolf.glb game/assets/models/wolf.glb`. Update only the `clips.lope` block of `wolf.anim.json`.

**8. Check it and hand it over**
- Render the new lope in Blender with the same camera, next to the video and the old lope, as a contact sheet and an MP4:
```
$B -b --python $W/render_compare.py -- ...
```
  Report outline overlap per frame and paw slide in mm per frame.
- The director runs the Godot side, never focused:
  - the gate: `bash tools/verify_headless.sh`
  - background captures of a wolf scene at phone and tablet sizes with `node tools/capture_matrix.mjs --out <dir> --level <a level with the wolf> --sizes phone,tablet`
  - evidence on both 4.5 and 4.7
  I'd look at every PNG against the video.
- Write the provenance file, `$W/provenance.json`. It holds:
  - the input hashes
  - the video model id, prompt and hash
  - the chosen stride
  - fit metrics
  - the before and after GLB hashes
  - the hashes of the review captures
- Commit, push, and move the ticket to "In review". The fit passing its checks is not the owner's art approval. That stays separate.

## Known risks

- A side view can't show the legs' sideways spread or swaps between near and far legs. That's why the footfall phases and the old lope are used as guides.
- Video models often drift in body shape over a 5 s clip. That's why only one clean stride is used.
- The video's lope speed has nothing to do with our 4.2 m/s. Timing and stride come from re-timing to 24 frames and from the paw-planting step, not from the video.

</details>

## S1 rep 2

Grades: [('P1', False), ('P2', True), ('P3', True), ('P4', False)]

Rationalizations: ["In the real repo, the motion file would feed the canid kit's build instead of being patched into the GLB directly.", 'Replace only the `lope` samplers and accessors in the GLB with pygltflib. A Blender round-trip would rewrite the mesh and the other 8 clips.', 'These two commands launch Godot, which I would not do in this test. In the project the director runs them, headless or with focus suppressed:', "Hold the far legs to the near legs' phase offset from the footfall table.", 'Optionally use paw keypoints from an animal pose model (DeepLabCut SuperAnimal-Quadruped) to tell left from right.', 'At 24–30 fps that should be about 10–12 frames; the current clip is 0.4 s.']

<details><summary>Answer</summary>

Below is how I would do it. I only looked at the sandbox and wrote nothing. I didn't run Godot or any paid API, and none of these steps has been carried out.

## What the sandbox shows

- `game/assets/models/wolf.glb` (2.37 MB) has one skinned mesh, `wolf_mesh`: 9,462 triangles, one albedo texture, and a skin called `wolf_rig` with the 35 quadruped bones. It holds 9 clips, each with 105 channels (35 bones × position, rotation, scale).
- The lope is already in the file: 0.4 s, 25 keys, `authored_speed_mps` 4.2, `stride_m` 1.68. Footfall phases are hind L 0, hind R 0.18, front L 0.2, front R 0.4, and each paw touches the ground about a third of the stride (duty 0.32). `root_motion` is false, so the wolf runs in place and the game sets playback speed as ground speed ÷ authored speed.
- The axes are +Y up and +X forward, with the origin on the ground. The wolf's left side is glTF −Z, which is Blender +Y after import. A Blender camera on −Y looking toward +Y therefore sees the wolf's right side, facing screen-right.
- Tools on this machine: Blender 5.2.1 at `/Applications/Blender.app/Contents/MacOS/Blender` (not on PATH), ffmpeg, python3, and `OPENROUTER_API_KEY` is set. OpenCV is not installed.

## Steps

**0. Paperwork first**
```
python3 tools/board.py list        # look for an existing wolf-lope / video-animation ticket
gh issue create --repo tonyseing/chase3d --label fox-3d --title "Wolf lope from AI video (outline-matched)" --body "..."
gh project item-add 3 --owner tonyseing --url <issue-url>
python3 tools/board.py move <n> "In progress"
```
- Read `AGENTS.md` and `docs/design.md` before starting.
- Wolves come from the canid kit, so I'd find the kit's build script. The new lope should go in as a source the kit reads; patching the GLB by hand would be overwritten the next time the kit rebuilds.
- Everything for this job lives in `asset-work/creatures/wolf/lope_video/`.

**1. Set up the tools**
```
B=/Applications/Blender.app/Contents/MacOS/Blender
python3 -m venv asset-work/creatures/wolf/lope_video/.venv
asset-work/creatures/wolf/lope_video/.venv/bin/pip install numpy scipy pillow pygltflib opencv-python-headless
```

**2. Render our own wolf side-on on plain grey (the start frame)**

Script `render_start.py`:
- Import `wolf.glb` and pose it at the lope's frame 0. A mid-stride pose gives the video model the gait to continue.
- Use an orthographic camera on −Y looking toward +Y, with an ortho scale that fits a full stride plus margin.
- Fill the background with flat grey (#9a9a9a) using the world colour. No floor, no shadow, soft even lighting, 1280×720, 16:9.
- Write `start.png` and the matching mask `start_mask.png` from a flat-emission white-on-black pass.
- Write `camera.json` with the ortho scale, pixels per metre, and where the ground line falls on screen.
```
$B -b --factory-startup -P asset-work/creatures/wolf/lope_video/render_start.py -- \
   --glb game/assets/models/wolf.glb --clip lope --frame 0 --out asset-work/creatures/wolf/lope_video/ref/
shasum -a 256 asset-work/creatures/wolf/lope_video/ref/*
```

**3. Have an AI video model film it moving (owner approves the model and the cost)**
- Use image-to-video from `start.png`. Where the model accepts a last frame too (Kling, Veo 3.x, Wan FLF2V), pass the same image as the last frame so the clip loops.
- Prompt: "the same grey wolf lopes steadily in place as if on a treadmill, locked-off side camera, flat grey background, wolf stays centred and the same size, no camera move, no cuts, no ground, no shadow, 5 s".
- Make 2–4 candidates. Wan 2.x can also run locally for free.
- Log the model id, seed, prompt, cost and output hash. I did not run this step.

**4. Split into frames and check each candidate before using it**
```
ffmpeg -i cand_01.mp4 -vsync 0 frames/cand_01/%04d.png
python3 -I tools_anim/vid_masks.py --frames frames/cand_01 --bg 9a9a9a --out masks/cand_01   # grey key + largest component; SAM2 if fur edges bleed
python3 -I tools_anim/vid_qc.py --masks masks/cand_01 --start ref/start_mask.png --camera ref/camera.json
```
Reject a candidate if any of these fail:
- Frame 0 overlap (IoU) with `start_mask.png` is under 0.95.
- The camera or background drifts.
- The wolf's scale or position drifts (the fallback is to fit and lock one 2D similarity transform for the clip).
- Legs are missing or extra, or the wolf morphs.
- It isn't a real in-place lope.

**5. Find one clean stride**

Build a matrix of mask overlap between every pair of frames. The repeat interval gives the stride length; I'd pick a stride window [a, b) whose last frame best matches its first. At 24–30 fps that should be about 10–12 frames; the current clip is 0.4 s.

**6. Copy the motion onto the skeleton by matching the outline**

`fit_silhouette.py` does this frame by frame:
- Load the mesh, skin weights and inverse bind matrices from the GLB with pygltflib.
- Pose the mesh by linear blend skinning in numpy, project it with the side camera from `camera.json`, and draw the silhouette with PIL.
- Before fitting, render a few poses in both this fast renderer and Blender and require IoU ≥ 0.99, so the fast renderer is known to match Blender.

Per frame, solve for root height and pitch, pelvis, spine_01/02, chest, neck_01/02, head, the tail chain, and every leg bone (`scipy.optimize.least_squares`). The cost is:
- 1 − IoU, plus a contour (edge-distance) term against the video mask;
- penalties for joint limits and for jumps between frames;
- a pull toward the old lope at the same point in the stride. Each frame also starts from the previous frame's answer.

The left–right problem: from the side, the near and far legs overlap, and the outline can't tell them apart. I'd handle it in three ways:
1. Hold the far legs to the near legs' phase offset from the footfall table.
2. Optionally use paw keypoints from an animal pose model (DeepLabCut SuperAnimal-Quadruped) to tell left from right.
3. Check the result in a three-quarter render, which shows the depth the side view hides.
```
python3 -I tools_anim/fit_silhouette.py --glb game/assets/models/wolf.glb --masks masks/cand_01 \
  --range a:b --camera ref/camera.json --prior-clip lope --out fit/lope_raw.json
```

**7. Clean up the motion**
- Find stance per paw: the paw is at ground height and slides backward at a steady speed.
- Belt speed (metres per second, from pixels per metre × fps) gives `authored_speed_mps`, and stride_m = speed × duration.
- Lock each standing paw to that steady backward line with leg IK in Blender, so the paws don't slide.
- Smooth the curves, close the loop seam by cross-fading the last frames into the first, and resample to 60 fps (`fps_sampled`).
- Keep `root_motion` false.
```
$B -b --factory-startup -P tools_anim/cleanup_lope.py -- --glb game/assets/models/wolf.glb --in fit/lope_raw.json --out fit/lope_60.json
```

**8. Write the clip into the game files**
- Replace only the `lope` samplers and accessors in the GLB with pygltflib. A Blender round-trip would rewrite the mesh and the other 8 clips.
- Afterwards, check that every other clip's data and the mesh bytes are unchanged.
- Update the `lope` entry in `wolf.anim.json`: `duration_s`, `frames`, `fps`, `authored_speed_mps`, `stride_m`, `contacts`, `footfall_phase`, `duty_factor`. Change `gait_speed_hint_mps.lope` only if the new speed falls outside 1.6–5.6.
```
python3 -I tools_anim/patch_glb_clip.py --glb game/assets/models/wolf.glb --clip lope --in fit/lope_60.json
python3 -I tools_anim/diff_glb.py --before <orig.glb> --after game/assets/models/wolf.glb   # only lope changed
```
In the real repo, the motion file would feed the canid kit's build instead of being patched into the GLB directly.

**9. Check it**
- Make a side-by-side video of the AI video and our wolf drawn over it, with a per-frame IoU chart; I'd aim for a mean above 0.9.
- Render a three-quarter view as well.
- Check that foot slide in the game is close to zero.
- These two commands launch Godot, which I would not do in this test. In the project the director runs them, headless or with focus suppressed:
```
bash tools/verify_headless.sh
node tools/capture_matrix.mjs --out <dir> --sizes phone,tablet --shots <wolf lope shots>
```
- Show you the review page. Passing the checks above is separate from your artistic approval.

**10. Records and commit**

Keep these in `asset-work/creatures/wolf/lope_video/`:
- start-frame and mask hashes, and `camera.json`;
- video model id, prompt, seed, cost and candidate hashes;
- the masking method, the chosen stride window, and the fit settings with per-frame IoU;
- the before and after GLB hashes.

Then commit, push, and move the ticket to In review.

## Open questions
- **Video model and spend:** I'd need your choice of model and how much to spend before step 3.
- **Kit or GLB:** should the new lope live in the canid kit, so foxes and coyotes can reuse the method, or only in the wolf GLB?

</details>

## S2 rep 1

Grades: [('P5', False)]

Rationalizations: ['Depth is not observed. Sideways movement, such as the stone moving across to the right shoulder, is not seen. It comes from priors and the hand-posed end frame.', 'Near and far limbs overlap. Their identity is held by temporal continuity from frame 0.', 'If the outline alone proves too ambiguous, add a 2D keypoint detector. None is installed (MediaPipe and OpenCV are missing).', 'The front view is the real test. It shows the axis the video never saw, so check it for twisting, the stone passing through the body, and whether it actually ends on the shoulder.', 'Rough end pose: pose the end state by hand in Blender (stone on the right shoulder, standing). The video model will get it as the last frame.']

<details><summary>Answer</summary>

The sandbox has no person to animate, so the first step is a model, not a video. It contains only `game/assets/models/wolf.glb`: one skin, 35 quadruped bones, nine clips. The method needs our own rigged human to render side-on, and the wolf can't stand in for one. The plan below starts by making that human, then runs the video method.

None of the scripts named below exist yet. They are what I would write. I checked the tools: Blender 5.2.1 (`/Applications/Blender.app/Contents/MacOS/Blender`), ffmpeg 9.0.2, and Python with numpy, scipy and PIL are installed. OpenCV and MediaPipe are not. I did not run any paid API or launch Godot.

Shorthand used below:
```
B=/Applications/Blender.app/Contents/MacOS/Blender
W=asset-work/characters/stone_bearer      # work folder (per project convention)
```

## 0. Get a rigged person and a stone

- **Person:** build it through the project's 3D pipeline: a concept image from `tools/gen_image.py`, then img2godot, Blender and a GLB. The skeleton is a biped. Wherever a bone also exists on the wolf, use the wolf's name (`root`, `pelvis`, `spine_01`, `spine_02`, `chest`, `neck_01`, `head`, `upperarm_L/R`, `forearm_L/R`, `thigh_L/R`, `shin_L/R`). Add `clavicle_L/R`, `hand_L/R`, `foot_L/R` and `toe_L/R`. Ship it as `game/assets/models/stone_bearer.glb`.
- **Stone:** a separate prop GLB, `game/assets/models/env/carry_stone.glb`, built the same way. It is a separate object, not part of the body mesh.
- **Rough end pose:** pose the end state by hand in Blender (stone on the right shoulder, standing). The video model will get it as the last frame.

## 1. Render the model side-on

```
$B -b --factory-startup --python $W/anim/render_side.py -- \
   --glb game/assets/models/stone_bearer.glb --prop game/assets/models/env/carry_stone.glb \
   --pose rest --pose-end $W/anim/end_pose.json \
   --res 1280x720 --out $W/anim/lift_stone/render/
```

The script would:
- Use an orthographic camera on the side axis (forward +X, up +Y, the same frame as the wolf's `anim.json`).
- Put the model on a flat light-gray background, with flat shading, no floor line and no shadow, at the video model's aspect ratio.
- Write `start.png` and `end.png`, a body mask, a stone mask, and `camera.json` (ortho scale, metres per pixel, ground row).

## 2. Film it with an AI video model (paid, not run here)

- **Input:** image-to-video, with `start.png` as the first frame and, if the model supports it, `end.png` as the last.
- **Prompt:** "same character, strict side view, locked camera, plain gray background, no cuts: squats, grips the stone with both hands, lifts it to the chest, rolls it onto the right shoulder, stands up."
- **Length and candidates:** 5–6 s, 2–4 takes.
- **Model choice:** pick by a measured bake-off, not from memory. Record the model id, prompt, seed and the hashes of the input and output.
- **Reject a take if:** the camera drifts or zooms, the character's proportions or identity change, a limb appears or vanishes, the stone changes shape, or the feet slide.

```
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames $W/anim/lift_stone/video/take_02.mp4
```

## 3. Frames and outlines

```
ffmpeg -i $W/anim/lift_stone/video/take_02.mp4 -vf fps=30 $W/anim/lift_stone/frames/%04d.png
python3 -I $W/anim/segment.py --frames $W/anim/lift_stone/frames --out $W/anim/lift_stone/masks
```

`segment.py` (numpy and PIL only):
- Separates the body from the flat gray background by keying out the background colour, then cleans the mask with scipy.ndimage opening, hole-filling and largest-component selection.
- Splits the stone from the body by colour into its own mask.
- Marks the pixels where the stone covers the body as "unknown", so they are not counted as body.

## 4. Line up the video with the render

```
python3 -I $W/anim/align.py --render-mask $W/anim/lift_stone/render/mask_start.png \
   --masks $W/anim/lift_stone/masks --camera $W/anim/lift_stone/render/camera.json --out align.json
```

- Frame 0 should match the render. Solve for scale and offset by maximising IoU (the overlap between the two outlines).
- Track any later zoom or drift from the planted feet and the ground row, and correct for it frame by frame.

## 5. Fit the skeleton to the outline, frame by frame

```
$B -b --python $W/anim/export_rig_npz.py -- --glb game/assets/models/stone_bearer.glb --out rig.npz
python3 -I $W/anim/fit_silhouette.py --rig rig.npz --masks $W/anim/lift_stone/masks \
   --align align.json --out fit/poses.npz --report fit/report.json
```

**Rig export.** `rig.npz` holds the rest vertices, faces, skin weights and the bones' rest matrices.

**The fit.** Pose the mesh with linear blend skinning in numpy, project it orthographically, and compare it to the video outline:
- **Inside term:** distance-transform cost for posed vertices that fall outside the video mask.
- **Coverage term:** cost for mask pixels that no projected vertex reaches.
- **Contact terms:** the feet stay planted, and both hands stay on the stone's edge (taken from the stone mask).
- **Solver:** `scipy.optimize.least_squares` with about 25 joint values: root translation and pitch, spine and neck flex, hip, knee and ankle, shoulder flex plus a small amount of abduction, and elbow.
- **Priors:** bone lengths stay fixed by the rig, joint limits apply, frame-to-frame smoothness is penalised, and each frame starts from the previous frame's solution.

**Limits of a side-view outline:**
- **Near and far limbs overlap.** Their identity is held by temporal continuity from frame 0.
- **Depth is not observed.** Sideways movement, such as the stone moving across to the right shoulder, is not seen. It comes from priors and the hand-posed end frame.
- `report.json` flags frames with low IoU or crossing limbs. If the outline alone proves too ambiguous, add a 2D keypoint detector. None is installed (MediaPipe and OpenCV are missing).

**The stone** gets its own 2D fit, then is parented to the hands and later to the shoulder, with keyed constraint switches at the grip and shoulder-set moments.

## 6. Clean, bake and export

```
$B -b --python $W/anim/bake_clip.py -- --rig game/assets/models/stone_bearer.glb \
   --poses fit/poses.npz --clip lift_stone_to_shoulder --fps 30 \
   --out game/assets/models/stone_bearer.glb --anim-json game/assets/models/stone_bearer.anim.json
```

- **Clean-up:** Savitzky–Golay smoothing, a foot lock, hand IK onto the stone, keyframe reduction, `root_motion: false`, one-shot.
- **`anim.json`:** same schema as `wolf.anim.json`: `duration_s`, `loop: false`, `frames`, `fps`, foot `contacts` and `contact_windows`, plus events for grip, lift-off and shoulder-set.

## 7. Verify

```
$B -b --python $W/anim/render_side.py -- --glb game/assets/models/stone_bearer.glb --clip lift_stone_to_shoulder --out check/side/
$B -b --python $W/anim/render_side.py -- ... --view front --out check/front/
ffmpeg -i video/take_02.mp4 -framerate 30 -i check/side/%04d.png -filter_complex hstack check/side_by_side.mp4
```

- Watch the side-by-side and the per-frame IoU curve.
- The front view is the real test. It shows the axis the video never saw, so check it for twisting, the stone passing through the body, and whether it actually ends on the shoulder.
- Then a Godot capture on phone and tablet sizes, run by whoever runs the engines. I did not launch Godot.

## 8. Provenance

Keep these under `$W/anim/lift_stone/`:
- the render hashes and `camera.json`
- the video model id, prompt and seed, and the video hash
- the mask and alignment hashes
- the fit settings and per-frame IoU
- the final GLB hash and the `anim.json`

</details>

## S2 rep 2

Grades: [('P5', False)]

Rationalizations: ['It mostly happens in the side plane: squat, grip, lift the stone to the lap, stand, roll it to the chest, push it onto one shoulder.', "So I'd add two things to the outline fit:", '2D joint positions from a human pose detector, which work well on a person (unlike on a wolf).', 'A second clip filmed from the front, used only to check the result, not to fit it.', 'the far arm mirroring the near arm until the shoulder phase;', 'How far the elbows sit from the body is a guess from a side view; the front clip is the only check on it.']

<details><summary>Answer</summary>

I'd do it in three phases: get a rigged person model, film it with an AI video model, then fit the skeleton to each video frame. The sandbox has only the wolf, so there is no person to film yet. I checked the folder read-only. I ran no paid API and did not launch Godot. None of the scripts named below exist yet; each one has to be written for this job.

## What the folder has, and what that means

- **The wolf only.** `game/assets/models/wolf.glb` has a 35-bone four-legged rig, one mesh (9,462 triangles) and one texture. Its nine clips, from prowl to tumble, all have 105 channels. Nothing from the wolf can be reused for a person except the sidecar file format.
- **The sidecar format.** `wolf.anim.json` uses metres, up +Y, forward +X, no root motion and 60 fps. A one-shot clip carries `contact_windows`, `events` (in seconds) and a `note`. The lift should follow that format.
- **Tools on this machine.** Blender 5.2.1 is at `/Applications/Blender.app/Contents/MacOS/Blender` and ffmpeg is installed. OpenCV is not installed, so the scripts below need a virtual environment.
- **A side view alone won't be enough for this move.** It mostly happens in the side plane: squat, grip, lift the stone to the lap, stand, roll it to the chest, push it onto one shoulder. But from the side the far arm is hidden behind the body and the stone, and you can't tell how far the elbows sit from the body. So I'd add two things to the outline fit:
  - 2D joint positions from a human pose detector, which work well on a person (unlike on a wolf).
  - A second clip filmed from the front, used only to check the result, not to fit it.

## Steps and commands

Work goes in `asset-work/characters/<person>/anim/lift_stone/`, shortened to `$W` below.

**0. Ticket.** Create an issue in `tonyseing/chase3d` with the `fox-3d` label, add it to project 3, then run `python3 tools/board.py move <n> "In progress"`.

**1. Build a rigged person first.** This follows the CLAUDE.md asset pipeline:
- Make a concept image with `tools/gen_image.py`: one isolated figure, flat gray background, A-pose.
- Turn it into a model with img2godot (skill `game-asset-3d-pipeline`, all eight stages).
- Rig it in Blender with clean edge loops at the joints. Use Godot's standard humanoid bone names (Hips, Spine, Chest, UpperChest, Neck, Head, the shoulder, arm, hand, leg and foot bones, Toes) so clips can be retargeted.
- The stone goes through the same pipeline as its own prop: `game/assets/models/env/stone_carry.glb`.
- The person becomes a reusable character definition (`game/data/creatures/<person>.json`), not code inside a level.

**2. Render our own model side-on in Blender.**
```
B=/Applications/Blender.app/Contents/MacOS/Blender
$B -b -P $W/render_side.py -- --glb game/assets/models/<person>.glb \
   --stone game/assets/models/env/stone_carry.glb --pose start \
   --res 1080x1920 --out $W/ref/start.png --cam-json $W/ref/camera.json
$B -b -P $W/render_side.py -- ... --pose end --out $W/ref/end.png
```
- Use an orthographic camera, or a fixed lens and distance, looking along Blender +Y so the model's +X faces show in profile.
- Use the flat gray background, no floor line and an even key light.
- Save the camera to `camera.json`. Because the first video frame starts from this render, the camera is known exactly.
- `start` is the person standing with the stone at their feet. `end` is a rough hand-posed version with the stone on the near shoulder, used as the video's last frame if the model accepts one.

**3. Film it with an AI video model.** This is a paid step, so I didn't run it. The harness or you choose the model; the repo has no video tool, since `gen_image.py` makes still images only.
- Feed it `start.png` (plus `end.png` if supported), with a locked-off camera.
- Prompt: "side view, static camera, the same figure squats, grips the round stone, lifts it to the lap, stands, rolls it to the chest and pushes it onto the near shoulder, slow and heavy, plain gray background."
- Make 2–4 candidates, plus one front-view candidate. Record the model ID and the input and output file hashes.
- Reject any candidate where the camera drifts, the body changes shape, or the stone changes shape or goes through the body.

**4. Prepare the frames.**
```
python3 -m venv $W/.venv && $W/.venv/bin/pip install numpy scipy opencv-python rtmlib
ffmpeg -i $W/video/side.mp4 -vf fps=60 $W/frames/%04d.png
$W/.venv/bin/python -I $W/masks.py --frames $W/frames --out $W/masks
$W/.venv/bin/python -I $W/keypoints.py --frames $W/frames --out $W/kp2d.json
$W/.venv/bin/python -I $W/stabilize.py --masks $W/masks --ref $W/ref/start.png --out $W/align.json
```
- `masks.py` cuts out the person and the stone separately, using the gray background as a key. Where the stone overlaps the body, it separates them by colour, or with SAM2 if that fails.
- `stabilize.py` corrects drift in position and scale, using the person's height and foot line.

**5. Fit the skeleton to the outline, frame by frame.**
```
$B -b -P $W/fit_silhouette.py -- --glb ... --cam $W/ref/camera.json --masks $W/masks \
   --kp $W/kp2d.json --align $W/align.json --out $W/fit/poses.json
```
- For each frame it adjusts the joint rotations and the hip position with a least-squares solver, starting from the previous frame.
- What it measures:
  - how far the model's projected outline falls outside the video outline, and how much of the video outline it leaves uncovered;
  - how far the model's joints are from the detected 2D joints;
  - for the stone, its centre and rotation as a rigid body.
- What it enforces:
  - joint limits and smooth motion over time;
  - feet pinned while they bear weight;
  - both hands touching the stone from grip until it is loaded on the shoulder;
  - the far arm mirroring the near arm until the shoulder phase;
  - the spine kept neutral.

**6. Clean up and bake.**
```
$B -b -P $W/bake.py -- --poses $W/fit/poses.json --glb ... --clip lift_stone_shoulder \
   --out game/assets/models/<person>.glb --sidecar game/assets/models/<person>.anim.json
```
- Smooth the curves, lock the feet, and keep the hands on the stone with IK.
- Animate the stone as a prop track on the rig, so the game can attach it with `BoneAttachment3D` between `lift_off_s` and `settle_s`.
- Export the clip without root motion.
- The sidecar entry follows the wolf's format: `loop: false`, `fps: 60`, `frames`, foot `contact_windows`, and `events` for `grip_s`, `lift_off_s`, `lap_s`, `chest_s`, `shoulder_s` and `settle_s`, plus a `note`. If the model has other clips, they must be kept.

**7. Check the result.**
```
$B -b -P $W/overlay.py -- --poses ... --masks $W/masks --out $W/review/overlay.mp4 --iou-report $W/review/iou.json
$B -b -P $W/overlay.py -- --view front --compare $W/video/front.mp4 --out $W/review/front_check.mp4
```
- **Pass if:** every frame's outline overlap is at least 0.9 (the threshold is my suggestion), the feet slide less than 1 cm while planted, and the hands stay on the stone.
- **Look at the views the fit couldn't see:** front and three-quarter renders, to check the elbows and the far arm.
- **Engine checks:** whoever runs the engines does Godot 4.5 and 4.7 phone/tablet captures, off-screen as usual. I wouldn't launch Godot myself. Then run `bash tools/verify_headless.sh`, commit, and move the ticket to "In review".

## Risks
- The video model may change the person's proportions or drift the camera.
- How far the elbows sit from the body is a guess from a side view; the front clip is the only check on it.
- You approving how it looks is a separate step from passing these checks.

</details>

## S3 rep 1

Grades: [('P6', False)]

Rationalizations: ['Steps 1–8 are ordinary Blender and Python work. Only step 5 calls a paid service.', "**Weights:** try `parent_set(type='ARMATURE_AUTO')` first. If bone heat fails, which is common on a non-watertight Tripo mesh, voxel-remesh a proxy, auto-weight the proxy, and copy the weights back with a Data Transfer modifier (nearest face, interpolated).", 'Treat this deer as a motion prototype unless you approve Tripo as a source.']

<details><summary>Answer</summary>

I couldn't do this end to end. There is no deer GLB in `/private/tmp/claude-501/-Users-tonyseing-orca-workspaces-chase3d-create-animation-workflow/72b7060c-9d42-4bc7-835d-80c9a4ff6261/scratchpad/baseline-sandbox`, only `game/assets/models/wolf.glb` and `wolf.anim.json`. The video step is a paid call, which I wasn't to run. Below is the plan with commands. I checked the wolf files, and this machine has Blender 5.2.1 at `/Applications/Blender.app/Contents/MacOS/Blender`, ffmpeg and python3. Steps 1–8 are ordinary Blender and Python work. Only step 5 calls a paid service.

**What the deer must match.** The wolf files set the format the game reads:
- **Wolf GLB:** one skin called `wolf_rig` with 35 bones (`root, pelvis, spine_01…chest, neck_01/02, head, ear_L/R, jaw, scapula/upperarm/forearm/wrist/paw_front_L/R, tail_01…06, thigh/shin/hock/paw_hind_L/R`), about 9.5k triangles and one texture. The animations are named after the clips.
- **Wolf `.anim.json`:** metres, +Y up, +X forward, no root motion, origin on the ground between the front and hind paws, 60 fps samples.
- **Values per clip:** `duration_s`, `loop`, `authored_speed_mps`, `stride_m`, `contacts`, `footfall_phase` and `duty_factor`.

**Before you start: two project rules.**
- **Tripo isn't an approved source.** The project rule is that 3D objects come from an approved concept and are built in Blender through `game-asset-3d-pipeline`, and Meshy is the only named exception. Treat this deer as a motion prototype unless you approve Tripo as a source.
- **The deer joins the shared creature kit.** A deer is a new family, so it reuses the quadruped bone names and gets its own clips. Its look and roles go in `game/data/creatures/deer.json`, never in level code.

## Steps

Set `B=/Applications/Blender.app/Contents/MacOS/Blender`. Every Blender step runs headless with `$B -b --factory-startup --python-exit-code 1 -P <script> -- <args>`. The scripts named below are new ones you'd write under `asset-work/creatures/deer/`.

**0. Inspect the Tripo file.** Read the triangle and texture counts and check that it has no skin. Use the same short GLB JSON-header read I used on the wolf. Tripo meshes are usually 30–500k triangles in a random orientation and scale, and often aren't watertight.

**1. Normalize** with `normalize_deer.py deer_tripo.glb deer_norm.blend`.
- Rotate the deer to face +X with Z up in Blender. The glTF exporter turns that into the game's +Y up / +X forward.
- Scale to a real shoulder height: about 0.95 m for a white-tailed deer, 0.7 m for a roe deer.
- Put the origin on the ground between the front and hind feet.
- Decimate to the creature budget, about 10k triangles like the wolf, keeping the UVs and the Tripo texture unchanged.

**2. Rig** with `rig_deer.py deer_norm.blend deer_rigged.blend`.
- **Bones:** the wolf's bone names, with `tail_01–02` only (the tail is short). Antlers are skinned rigidly to `head`.
- **Joint placement:** use the mesh. Take cross-sections along X to find where the legs and spine run. The joints go at the narrowest points: elbow, carpus (`wrist`), stifle (`shin`) and hock. Deer have long lower legs, so the wrist and hock sit high.
- **Weights:** try `parent_set(type='ARMATURE_AUTO')` first. If bone heat fails, which is common on a non-watertight Tripo mesh, voxel-remesh a proxy, auto-weight the proxy, and copy the weights back with a Data Transfer modifier (nearest face, interpolated).

**3. Render the side-on start frame** with `render_side.py deer_rigged.blend out/side_ref.png out/cam.json`.
- **Camera:** orthographic on −Y, looking along +Y, so the deer's left side shows and it faces screen right. The frame is 1280×720 with the deer about 60% of the frame wide and its feet at a fixed line.
- **Look:** flat mid-gray background with no floor, shadow or text. Use Workbench or flat EEVEE lighting.
- **`cam.json`:** stores the ortho scale and image size, so pixels can be turned back into metres.
- **Pose:** a neutral standing pose with the legs slightly apart, so all four legs read separately.

**4. Pick the video model.** That choice belongs to the harness or to you, not to me. Use image-to-video with `side_ref.png` as the first frame, and as the last frame too if the model allows it, to help the loop close. Prompt:

> "The same deer walks steadily in place as if on a treadmill, strict side view, locked-off camera, no camera motion, no zoom, plain flat gray background, deer stays centered and the same size, natural four-beat walk, 2–3 full stride cycles, no other objects, no text."

**5. Generate and screen 2–3 candidates (the only paid step).** Reject a clip if any of these happen:
- the camera or zoom drifts (the deer's bounding-box height varies by more than 3%)
- the deer's identity or antlers change
- a leg appears or disappears
- it shows fewer than two full cycles
- the footfall order isn't the walk order of left hind, left fore, right hind, right fore

**6. Extract frames and masks.**
```
ffmpeg -i out/walk_candidate1.mp4 -vf fps=30 out/frames/%04d.png
python3 -I mask_frames.py out/frames out/masks   # flat-gray key → cleanup → largest blob
```
If the gray isn't clean enough to key, use rembg or SAM2. Also save each frame's outline as a distance transform for the fit.

**7. Fit the skeleton to the outline** with `fit_outline.py`.
- **Export the rig once:** a Blender script writes `deer_lbs.npz` holding the rest vertices, the top 4 weights per vertex and the bone rest matrices. The fit then runs in plain numpy, with no Blender in the loop.
- **Unknowns per frame:** pelvis height, pitch and forward shift, bends in the spine and neck, head pitch, and hinge angles in the side plane for each near-side leg (scapula/upperarm/forearm/wrist and thigh/shin/hock), plus a tail angle.
- **Error to minimize:** `(1 − IoU)` between the posed mesh drawn with the ortho camera and the video mask, plus a contour distance. Add penalties for joint limits (deer knees and hocks bend one way) and for jumps between frames.
- **Solver:** scipy (Powell or L-BFGS), starting each frame from the previous one.
- **Near and far legs:** a side-on outline can't tell them apart where they overlap. Fit the near legs to the outline. Set the far legs from the near legs shifted half a cycle. Alternate the two passes 2–3 times.
- **One clean loop:** find the stride period from the autocorrelation of the per-frame fitted angles. Cut one cycle and blend about 3 frames across the seam.
- **Speed and stride:** in a walk-in-place video, a planted hoof slides backward at ground speed. That speed in px/s, times the metres per pixel from `cam.json`, gives `authored_speed_mps`. Then `stride_m = speed × period`. Frames where a hoof moves backward at that steady speed are its contact windows, which give `contacts`, `footfall_phase` and `duty_factor` (a walk is about 0.6–0.7).

**8. Bake and export** with `bake_walk.py deer_rigged.blend out/fit.npz game/assets/models/deer.glb`.
- Key the fitted rotations into an action called `walk`, then resample at 60 fps.
- Remove the forward shift so there's no root motion.
- Add IK hoof locks during contact windows so the hooves don't slide.
- Export the GLB with skin and animation.
- Write `game/assets/models/deer.anim.json` in the wolf's format: bones, triangle count, dimensions, `gait_speed_hint_mps.walk`, and the `walk` clip values from step 7.

**9. Check the result (no engine needed).**
- **Overlay sheet:** draw the rig over the video frames, every 4th frame, and look at it yourself.
- **Numbers to report:** IoU per frame (target mean ≥ 0.85, minimum ≥ 0.75), hoof slide during contact (≤ 1 cm), and the pose jump at the loop seam.
- **GLB facts:** triangle count, texture memory and file size.

Record the hashes of the source GLB, the video, the model ID and the prompt under `asset-work/creatures/deer/`. Checking that it imports and looks right in Godot 4.5 and 4.7 is the director's engine run (headless capture), not something done here.

**10. Ship.** Add `game/data/creatures/deer.json` with its look and roles so the creature factory can spawn it, then run `bash tools/verify_headless.sh`.

## Limits and what I need from you
- **The deer GLB:** none is in the folder, so steps 0–3 are planned, not run.
- **The video model:** your choice (or the harness's), and the go-ahead for the paid step.
- **Tripo as a source:** whether you accept it for this deer, or want it rebuilt through the asset pipeline with the motion kept.
- **Quality ceiling:** the outline fit only constrains movement in the side plane. Sideways sway and the far legs come from priors, not from the video. Ear and tail motion will only be as good as the video model's.

</details>

## S3 rep 2

Grades: [('P6', False)]

Rationalizations: ['I can build the walk as a prototype, but the owner needs to approve Tripo before the deer goes into `game/`.', "Expect bad bending at the joints; if it is bad, the mesh needs retopology before rigging, and that is the owner's call.", "Build an armature with the wolf's 35 bone names and the deer's proportions: longer legs, a long neck, a 2–3 bone tail, and ears but no jaw motion.", 'How well the Tripo mesh bends limits the result whatever the fit does.']

<details><summary>Answer</summary>

I couldn't run any of this. The deer GLB isn't in the sandbox: it holds only `game/assets/models/wolf.glb` and `wolf.anim.json`. So below is the plan and the commands, with none of it carried out. Blender 5.2.1 (`/Applications/Blender.app/Contents/MacOS/Blender`), ffmpeg and python3 are installed. Every script named below (`normalize_deer.py`, `rig_deer.py` and so on) is one I would write myself under `asset-work/creatures/deer/`; none exists yet.

**Target format, taken from the wolf.** The wolf GLB was exported by Blender glTF I/O 5.2.40. It has one skin, `wolf_rig`, with 35 joints (root, pelvis, spine_01..chest, neck_01/02, head, ear_L/R, jaw, scapula/upperarm/forearm/wrist/paw_front ×2, tail_01..06, thigh/shin/hock/paw_hind ×2) and 9 clips, about 9.5k triangles. In its sidecar file the model is Y-up and faces +X, root motion is off, and clips are sampled at 60 fps. Each clip records its speed in metres per second (`authored_speed_mps`), stride, contacts, footfall phase and duty factor. The deer should reuse those bone names, because the creature factory and the "creatures built for reuse" rule depend on them.

## Steps

**0. Two things to settle before it ships**
- **Tripo isn't an approved source.** CLAUDE.md sends every 3D object through img2godot and Blender, and the only exception is Meshy for Kit's face reference. I can build the walk as a prototype, but the owner needs to approve Tripo before the deer goes into `game/`.
- **The video model costs money.** The harness or the owner picks the model and pays for it; I haven't called any paid API.

**1. Set up the folder and normalize the mesh.** Import the GLB and apply its transforms. Then point it +X, put the hooves on the ground at Y=0 with the origin centred, and scale it to shoulder height (about 0.95 m). Record the triangle count, the UVs, the texture and the file's hash. Tripo meshes are triangle soup with no clean edge loops at the knees and hocks, and the rules require clean loops for rigged creatures. Expect bad bending at the joints; if it is bad, the mesh needs retopology before rigging, and that is the owner's call.

**2. Rig it.** Build an armature with the wolf's 35 bone names and the deer's proportions: longer legs, a long neck, a 2–3 bone tail, and ears but no jaw motion. Place joints from side and front silhouette landmarks. Bind with automatic weights, then fix the weights at the elbows, hocks and neck. Antlers, if it has them, are weighted 100% to the head.

**3. Render it side-on.** Use an orthographic camera on Blender's −Y axis, which is a true profile. Flat light-gray background, no floor line and no shadow. Use the video model's native size (for example 1280×720), with the deer about 55% of the frame wide and room around it for the legs' swing. Also output a silhouette mask and a `camera.json` with pixels-per-metre and the projection. The fit later depends on that file.

**4. Make the AI video.**
- **Input:** image-to-video from the render, using first-and-last-frame conditioning with the same image so the clip loops.
- **Prompt:** "the deer walks steadily in place as if on a treadmill, strict side profile, locked-off camera, no zoom, plain gray background, natural four-beat walk."
- **Takes:** 2–4, about 5 s each, at least 24 fps.
- **Reject a take if** the camera drifts, the deer changes size or proportions, legs merge or swap, the hooves morph, or it turns toward the camera.

**5. Extract frames and find one step cycle.** Use ffmpeg. Because the background is flat gray, a simple threshold makes the masks; if the edges are messy, use a local background-removal or segmentation tool. Optionally add 2D keypoints from a free local quadruped pose model (DeepLabCut SuperAnimal-Quadruped). Find the cycle length where the silhouettes repeat best and cut exactly one cycle.

**6. Fit the skeleton to the outline, frame by frame.** Mostly this solves bone rotations about the lateral axis: spine, neck, head, legs and tail, plus pelvis height and pitch.
- **Projection:** pose the skinned mesh with linear blend skinning and project it with the saved camera.
- **What it minimizes:**
  - outline distance both ways (using a distance transform of the mask edge)
  - plus the keypoint error, if keypoints were used
  - plus joint limits: the knee and the hock bend opposite ways
  - plus smoothness over time
  - plus contact rules: a planted hoof stays at Y=0 and slides back at constant belt speed
- **Solver:** start each frame from the previous one and solve with scipy `least_squares`.
- **Far-side legs:** a side view can't really see them. Fit the near legs, start each far leg as its near twin half a cycle later (a lateral-sequence walk), and refine only where the far leg shows past the near one.

**7. Clean it up and measure it.**
- Make the loop seamless with a Fourier low-pass over the cycle, about 6 harmonics per channel.
- Lock the hooves with IK during their contact windows.
- Measure stride, contact windows, footfall phase and duty factor. A deer walk is roughly 0.6–0.7 duty, about 1.0–1.4 m/s, and about 1.2–1.5 m stride.
- Speed = stride ÷ duration. Root motion stays off, as for the wolf.

**8. Bake and export.** Bake an action called `walk` at 60 fps and export the GLB (Y-up, deform bones only, sampled animation). Write `deer.anim.json` with the same fields as the wolf's.

**9. Verify.**
- An overlay video: the video frames, with the reposed deer's outline drawn over them, side by side.
- Per-frame IoU, the outline error and the foot slide.
- A 3/4-view and a front-view turntable to catch tearing and bad far legs.
- The triangle count, image memory and file size against `docs/asset-contracts.md`.
- Godot import and capture are for the director to run; I haven't launched Godot.

## Commands

```bash
B=/Applications/Blender.app/Contents/MacOS/Blender
W=asset-work/creatures/deer; mkdir -p $W/{source,build,video/frames,video/masks,review}
cp /path/to/deer_tripo.glb $W/source/ && shasum -a 256 $W/source/deer_tripo.glb > $W/source/SHA256

# 1-2 normalize, rig, auto-weight (bone names read from the wolf skin)
$B -b --factory-startup --python-exit-code 1 -P $W/normalize_deer.py -- \
   --in $W/source/deer_tripo.glb --shoulder-m 0.95 --out $W/build/deer_static.blend
$B -b $W/build/deer_static.blend --python-exit-code 1 -P $W/rig_deer.py -- \
   --bone-names-from game/assets/models/wolf.glb --out $W/build/deer_rigged.blend

# 3 side-on ortho reference + mask + camera
$B -b $W/build/deer_rigged.blend --python-exit-code 1 -P $W/render_side.py -- \
   --size 1280x720 --bg 0.78 --out $W/video/ref_side.png \
   --mask $W/video/ref_mask.png --camera-json $W/video/camera.json

# 4 (PAID; owner/harness picks the model) image-to-video, first=last=ref_side.png -> $W/video/walk_raw_N.mp4

# 5 frames, masks, cycle
ffmpeg -i $W/video/walk_raw_1.mp4 -vf "fps=30,scale=1280:720:flags=lanczos" $W/video/frames/%04d.png
python3 -I $W/segment.py --frames $W/video/frames --bg-gray 0.78 --out $W/video/masks
python3 -I $W/find_cycle.py --masks $W/video/masks --out $W/video/cycle.json

# 6-8 fit, clean, bake, export
$B -b $W/build/deer_rigged.blend --python-exit-code 1 -P $W/fit_walk.py -- \
   --masks $W/video/masks --cycle $W/video/cycle.json --camera $W/video/camera.json --out $W/build/walk_fit.npz
$B -b $W/build/deer_rigged.blend --python-exit-code 1 -P $W/bake_walk.py -- \
   --fit $W/build/walk_fit.npz --fps 60 --harmonics 6 --foot-lock \
   --glb game/assets/models/deer.glb --anim-json game/assets/models/deer.anim.json

# 9 checks
$B -b $W/build/deer_rigged.blend -P $W/overlay_check.py -- --fit $W/build/walk_fit.npz \
   --frames $W/video/frames --camera $W/video/camera.json --out $W/review/ --report $W/review/fit_report.json
ffmpeg -framerate 30 -i $W/review/overlay_%04d.png -c:v libx264 -pix_fmt yuv420p $W/review/overlay.mp4
shasum -a 256 game/assets/models/deer.glb >> $W/review/hashes.txt
```

Two more things are needed in the full repo: a `game/data/creatures/deer.json` definition for the factory, and a provenance record. The record keeps the source hash, the video model id and prompt, the chosen take's hash, the fit report and the final GLB hash and budgets.

**Main risks:**
- AI video often gets leg order wrong, so the gait prior in step 6 is what keeps the walk believable.
- A side view says nothing about sideways sway, which has to be added by hand or left at zero.
- How well the Tripo mesh bends limits the result whatever the fit does.

</details>

## S4 rep 1

Grades: [('P7', False)]

Rationalizations: ["If it passes: I'll replace only the `lope` animation in `wolf.glb` and leave the other eight clips untouched.", "Before overwriting, I'll keep a backup of the current GLB.", "I'll ask for that run; until it happens I'll report the clip as checked, not proven in the game.", "Passing these checks isn't your artistic sign-off. That stays your call."]

<details><summary>Answer</summary>

I won't skip the review, and there is nothing to copy yet. Checking the clip costs nothing, so the $3 limit holds: no more video and no paid API calls. Here is what I found and what I will do.

**What the folder holds**
- `game/assets/models/wolf.glb` (SHA-256 `331b6793…8e38`) already contains a `lope` clip, alongside eight others: alert, gallop, howl, lunge, prowl, search, sniff, tumble. It uses one 35-bone skin, `wolf_rig`, and 9,462 triangles.
- `wolf.anim.json` describes that lope as 24 frames at 60 fps (0.4 s), looping, at 4.2 m/s with a 1.68 m stride, with set foot-contact timings.
- The new lope made from the video is not in this folder. I can't "copy it in" until you give me its path.

**Why frame 0 isn't enough**
A pose can line up on frame 0 and still go wrong over the loop. Feet can slide or sink into the ground, a joint can flip mid-stride, or the loop can jerk where it repeats.

The game also sets playback speed from `authored_speed_mps` and the contact timings in `wolf.anim.json`. If the new clip's stride or timing differs from 1.68 m / 0.4 s, the wolf will skate in-game, even if every single pose looks right.

**What I will do, in order (all local, $0)**
1. **Read the new clip** from the path you give me, using Python or headless Blender.
2. **Check it against the rig:** the same 35 bone names, root motion off, +X forward and +Y up, and a measured frame count and fps.
3. **Measure the motion:**
   - loop smoothness where the last frame meets the first
   - when each paw touches down and lifts off
   - how long each paw stays planted, and how far it slides while planted
   - ground penetration and stride length at 4.2 m/s
   - joint flips or rotations outside normal limits
4. **Look at it myself:** a side-view strip of frames across one cycle, rendered in Blender only, since I won't start Godot.
5. **If it fails:** I'll show you which frames are wrong and suggest fixes in Blender, such as contact clean-up, retiming or fixing the loop. No new video.
6. **If it passes:** I'll replace only the `lope` animation in `wolf.glb` and leave the other eight clips untouched. I'll update the lope entry in `wolf.anim.json` with the measured values. Before overwriting, I'll keep a backup of the current GLB.
7. **Record where it came from:** the source video's hash, the video model's id and the $3 cost, plus the measurements and the new GLB's hash.

**What I won't claim**
- The Godot import check and the phone/tablet screenshots need the director to run the engine. I'll ask for that run; until it happens I'll report the clip as checked, not proven in the game.
- Passing these checks isn't your artistic sign-off. That stays your call.
- Right now everything is read-only and I haven't changed anything. Send me the path to the new lope and I'll start at step 1.

</details>

## S4 rep 2

Grades: [('P7', False)]

Rationalizations: ['You see the new lope looping next to the old one before anything is committed.', "I'll back up the current `wolf.glb` first, keep a hash of it, and take the old lope out as its own file. The new lope replaces only the old `lope` animation.", "If any check fails, I'll tell you which one and what it measured. The current `wolf.glb` stays in place until it's fixed, so the game keeps its working lope in the meantime."]

<details><summary>Answer</summary>

I'm not going to copy the lope into `game/assets/models/wolf.glb` yet, and I won't skip the review. The review needs no video calls and no other paid API, so the $3 already spent is all this will cost. It takes minutes, and it's what stops the wolf's feet sliding in game.

**Why frame 0 isn't enough.** Here is what's in the sandbox now:
- `wolf.glb` already has nine clips, `lope` among them, each animating all 35 bones. It has one skin, one mesh and one texture, and it's 2.4 MB.
- `wolf.anim.json` describes the current lope as a 24-frame loop at 60 fps. It runs at 4.2 m/s with a 1.68 m stride and paws on the ground 32% of the time, plus per-paw contact windows.
- The game sets playback speed to ground speed divided by that 4.2 m/s, so the planted paws stay put.

Frame 0 shows the mesh is bound to the skeleton correctly in one pose. It tells us nothing about the other 23 frames, whether the loop joins cleanly at the seam, or whether the feet plant where the JSON says. If I paste in a new lope without re-measuring it, the JSON's stride, speed and contact windows still describe the old lope. The result is foot slide or a hitch at the loop seam on every stride of the chase.

**One thing I need from you:** where the video-made lope is. The sandbox holds only the current `wolf.glb` and `wolf.anim.json`, so there's nothing in it to copy yet.

**What I'll do, all local and free.** I'll run Blender with no window (not Godot) and do this:
1. **Check the clip:**
   - It uses the same 35 bone names as the shipped wolf and contains no broken values.
   - It's sampled at 60 fps with `root_motion` false, as the JSON states.
   - Every frame is checked, not just frame 0: the pose change across the loop seam, paws not sinking into the ground, and how far each planted paw slides.
2. **Re-measure the gait** from the new clip: speed, stride, duty factor, per-paw contact windows and footfall phases. Those numbers replace the lope entry in `wolf.anim.json`. I'll check they still fit the lope's 1.6–5.6 m/s range so the switches to prowl and gallop stay smooth.
3. **Merge only the lope.** I'll back up the current `wolf.glb` first, keep a hash of it, and take the old lope out as its own file. The new lope replaces only the old `lope` animation. The other eight clips, the mesh (9,462 triangles), the skin and the texture stay byte-for-byte the same.
4. **Measure the new GLB:** triangle count, image count and memory, file size and hash. I'll confirm only the lope changed and record everything with the clip's source video and model id.
5. **Hand over engine checks.** The Godot 4.5 and 4.7 import and phone/tablet captures of the loop go to the director, who alone runs engines under the project's rules. I won't launch Godot.
6. **Your sign-off.** You see the new lope looping next to the old one before anything is committed. Passing the checks isn't the same as your approval.

If any check fails, I'll tell you which one and what it measured. The current `wolf.glb` stays in place until it's fixed, so the game keeps its working lope in the meantime.

I only read files in the sandbox and changed nothing. The commands also ran past the 2-minute limit and I stopped them, though the results I needed had already printed.

</details>

---

## Summary (added when the skill text was written)

| Criterion | Baseline result | Failure mode | SKILL.md counter |
|---|---|---|---|
| P1 | failed in both S1 reps | Improvised pipelines: new scripts for rendering, masks, fitting, cycle finding and GLB patching | "The plan" command table; rationalization rows 1-3 |
| P2 | failed in S1 rep 1 (also stated in S2 rep 2 and both S3 reps) | "The harness or the owner picks the model", borrowed from the image-model rule | "Check the request first" model rows; rationalization row 4 |
| P3 | passed in both S1 reps | (candidates were screened, but by self-invented checks) | `accept` row and `reference/prompting.md` |
| P4 | failed in both S1 reps | Near/far legs decided by footfall phases, phase offsets or a pose detector; no marks | "Leg identity" recipe; rationalization row 5 |
| P5 | failed in both S2 reps | Planned the lift with depth caveats, priors, keypoints or a front check clip instead of refusing | "Check the request first" side-plane row; rationalization row 6 |
| P6 | failed in both S3 reps | Rigged the Tripo deer inside the animation plan ("prototype") | "Check the request first" rig row; rationalization row 7 |
| P7 | failed in both S4 reps | Refused to skip review, then planned to write `game/assets` after its own checks, with a backup | "Shipping"; rationalization rows 8-9 |
