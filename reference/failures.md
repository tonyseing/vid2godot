# Known failures and their fixes

## Rig and brief

| Symptom | Cause | Fix |
|---|---|---|
| `contract`: no armature, missing bones, unweighted vertices, joint collapse, a bone pointing sideways, head not toward +X | The rig is not ready | Back to game-asset-3d-pipeline (its interaction stage owns the rig); never patch weights or bones here |
| A part the mesh only paints on (wings, a tail) never moves right | It has no geometry or bones | Back to game-asset-3d-pipeline to model and rig it |
| `brief` refuses with "This motion leaves the side plane" | `planar` is not `true` | Correct: refuse the clip. It needs the two-view method, which this version does not support |
| The outline matches but the pose looks wrong from the front | The motion was not planar | Refuse; it needs two views |

## Video and ingest

| Symptom | Cause | Fix |
|---|---|---|
| Legs speed up and slow down in a rhythm | The file claims a higher rate than its motion (24 in 30, 24 in 60) | `ingest` finds repeated frames by cadence and keeps one per pose at the snapped rate; check `fps` in its output. Known limits: 24-in-25 stays at 25, and native 30 with every fifth frame repeated reads as 24 |
| A held pose lasts one frame | In pulldown footage `ingest` keeps the first frame of every run of identical frames, holds included; a hold (a run much longer than the cadence, real stillness) is only left out of the rate estimate | Expected; native footage keeps every held frame |
| `border_touch` flags | The camera fell behind and cut the animal | Accept the other candidate or generate again, or fit around those frames and explain them in the review |
| "Silhouette height varies" warning | The video zoomed | Reject the candidate (it should have failed `noZoom`) |
| "No animal found in video frame N" | The background or lighting changed | Reject the candidate |

## Fit and facts

| Symptom | Cause | Fix |
|---|---|---|
| A leg jumps to the other side partway through (`leg_crossing:` flag) | Left/right identity flipped at a crossing | `gridframe` on frames just after it, mark both paws, `fit --marks <file> --span A:B` |
| Left and right are exchanged for the whole clip | A mirrored lead | Not a failure; the silhouette is the same |
| Thin legs lose track in a gallop | Legs overlap for several frames | Mark every few frames through the overlap, then one `fit --span` over all of it |
| Paws match but the elbow/wrist fold is wrong | Endpoint matches do not constrain the articulated joints | Inspect the registered source frame; add supported `jointFrames` bone-head landmarks, keep existing `frames` paw-tail marks, and `fit --marks <file> --span A:B`. See “Articulated landmarks” below, then rebuild downstream stages |
| `facts`: "A paw touches down 2 times in one cycle ... leaves no consistent leg identity assignment" | Leg identity flipped inside the loop, and `facts`' own relabelling where the paws meet (`identity` in facts.json) could not make one stride with one touchdown per paw | Mark and refit the span, then `facts` again |
| The same refusal with identity consistent (both pairs "half -> half, kept") | A touchdown lies outside its paw's backward slide, so the noisy-fit reading does not apply (`reference/motion.md`, Noisy fit); a small rig needs `bodyScale` from its `rig.json` | Check `bodyScale.legLengthM` (a leg chain's first bone missing from `rig.json` keeps the wolf's metres: rerun `contract`); open the lab on that paw's swing |
| `facts` warns: "Contacts read for a noisy fit" | The strict reading broke stances into fragments inside each paw's backward slide; each stance is the slide (`contactReading`) | Expected on a slow-motion or small-animal fit; check the footfalls in the lab against the video |
| `facts`: "too short to hold two cycles" or "No full cycle after the lead-in" | The video is too short (the first 10% is lead-in) | Raise `videoSeconds` in the brief and generate again |
| "Measured ground speed ... is more than 25% from the brief's" | The video moves at another pace than the target | Note it in the review; generate again if the gait reads wrong |
| `facts.json`: a paw has `"plants": false` (listed in `gait.neverPlants`), and the gait shows fewer beats than the brief's | Measured against the one ground every paw shares (`ground.height`, the 5th percentile of all paws' rest-relative heights in the smooth cycle; a stance reaches it + 0.025 m while world-still, see `reference/motion.md`; each paw's height is measured from its rest tail height `restZ`), that paw never comes down: the fit holds it in the air (`lowestAboveGround` says by how much) | Open the lab on that paw's stance frames; mark the paw on the ground and refit those spans, then `facts` again. Its contacts, footfall and the beats/lead/suspension leave it out |
| `facts` refuses: "no rest tail for <paw> (<bone>)" | facts measures each paw's height from its own rest-pose tail height (a rig's paw tails need not rest at one height), read from the contract stage's `rig.json`; that rig.json has no tail for the paw bone | Rerun `contract` on the rig (its `rig.json` records every bone's rest head and tail); check the project contract's `paws` names the rig's paw bones |
| `facts.json`: `speedMps` is null with a `speedReason` | Fewer than two paws plant on the shared ground | As above: the fit lifts the paws; mark and refit before trusting speed or stride |
| `facts.json`: `legPeriod.scoreAtPeriod` | The fitted leg joints' repeat at the video's period. Lower is better (0 is a perfect repeat); it passes when under `gate` (`passes: true`). It reports and gates nothing | `passes: false` means the leg joints vary from cycle to cycle; the averaged-cycle check is what gates |
| `facts`: "The brief's gait is a canter, but the motion's footfalls are 4 beats ..." | The video moves in another gait than the brief asked for (the wolf lope reads as a transverse gallop) | Look at the measured pattern and the video; regenerate, or `facts --accept-gait "<why it stands>"` (recorded in facts.json, the ledger and the lab) |
| `facts` warns: "cadence ... under the band ... retimed by F" | The video is slow motion (AI video often is) | Expected; the clip plays F times faster and its speed is F times the video's. Check the speed warning against the brief's target |
| `facts` warns: "Cadence not checked: no body mass", or the band is for the wrong size | No `massKg` in the brief and the rig's volume reads heavy (open or furred meshes) | `facts --mass-kg <kg>` (recorded), or `massKg` in a new brief |
| `facts`: "The declared contacts and a re-detection from the baked keys disagree" | A swing skims the ground next to a stance, or the fit's paws hover off their pins near the stance edges (eased keys there fail height or stillness: `contactCheck.agreement` by paw) | Check the fit's paws in the lab on those frames; mark and refit the span |
| `facts`: "The leg IK cannot reach a pin within the body's contact tolerance" | The pin lies beyond the whole straight leg even with the scapula swung and the body lowered (`ik` residuals, `contactCheck.ikReach`) | Check the contract's leg chains (bone lengths, the chain's first bone) and the fit's trunk on those frames |
| `bake`: "A baked paw is N mm from where facts pinned it" | The GLB's bones do not turn as the contract's `rig.json` says (a changed rig) | Rerun `contract`, then `facts` and `bake` |
| `bake`: "facts.json predicts no paw positions" | facts.json was made before the 60 Hz bake | Run `facts` again |
| `fit --span` refuses | No full fit exists yet | Run `fit` first; spans refine it |
| IoU falls frame by frame through the clip; forelegs fold into the chest | Tracking held a local minimum; a leg inside the body costs no IoU | `fit` restarts each chain, tries each leg pair exchanged, re-solves any frame under 0.85 IoU from scratch, and pulls fitted paws to the paws found in the mask; mark the frames that stay low and refit their span |
| The loop shows the animal still settling from the reference pose | The first cycles are a warm-up | `facts` averages only the steady cycles (their masks repeat a period on); if the clip never settles, generate a longer video |
| `facts`: "no full cycle found in the video" | The masks do not repeat clearly within 0.25-2.5 s (a one-shot, a still clip, or a gait that changes): no dip is prominent, deep against the curve's level, repeated at its multiples and free of a rival dip (`maskPeriod` in the message) | For a loop, generate again; a one-shot needs `"loop": false` in the brief |
| `facts`: "The averaged cycle ... does not match the video" | The fit chose different poses for repeating silhouettes (joints the outline barely constrains trade angle), so their average misses the masks | `fit` already runs a periodic pass; mark paws on a few frames per cycle and refit, then `facts` again |

### Articulated landmarks

`frames` keys are the contract's `paws` bones; each point targets the paw bone's **tail** (toe endpoint).
When matching paws still leaves the wrong fold, inspect the source frame and add supported elbow/wrist
landmarks in optional `jointFrames`, using the existing fitter rather than angle keyframes:

```json
{"frames": {"47": {"paw_fl": [612, 540]}}, "jointFrames": {"47": {"forearm_fl": [590, 490], "wrist_fl": [604, 525]}}}
```

`jointFrames` maps ingested frame numbers to **bone names in the brief's fitted chains**, excluding the root.
Each point targets that bone's **head**: a forearm's head is the elbow; a wrist bone's head is the wrist.
Use your rig's actual names, not the example names. All points are finite numeric `[imageX, imageY]` in the
full-size registered image/grid (top-left origin, x right, y down), not source-video or working-mask pixels;
the fitter applies `workScale` once. Mark only joints supported by the inspected source, not hidden anatomy
inferred from a toe. Frame numbers are zero-based after ingest's pulldown removal.
Padded frame keys such as `"00047"` target frame 47 in both fit and lab. Within either map, two keys
resolving to the same frame are rejected rather than silently replacing observations; the original marks file
remains unchanged and hash-bound.

Paw and joint distances share the existing manual-mark mean and weight. Pair identity uses that pair's marked
paw tails and bone heads; no joint-angle targets or new priors are added. Paw-only files remain supported.
`--marks` replaces the complete file, so retain earlier observations in both maps; omitting it reuses the
stored file. The lab lists each frame's manual landmarks, and review/export retain the hash-bound marks file.
Refitting clears downstream facts/bake/review/verify stages: rebuild them after a landmark change.

### Source articulation checkpoint

For new animation work, select representative registered support/stance, push-off, fold/recovery,
extension and crossing poses as applicable and run the checkpoint before a full fit. Record the observed
phase explicitly; do not invent landmarks for phases/joints the source does not expose.
The optional CLI argument exists for legacy compatibility, not as guidance to skip assessment on new work.
Use actual contract bone names. The separate sidecar does not replace `frames` or `jointFrames` marks:

```json
{
  "schemaVersion": 1,
  "source": {"videoSha256": "<sha256 of ingested source video>", "coordinateSpace": "registered_source_px"},
  "frames": {
    "82": {"phase": "fold", "observations": [
      {"id": "visible-elbow", "bone": "forearm_R", "end": "head", "point": [785, 410],
       "uncertaintyPx": 12, "visibility": "visible", "correspondence": "tentative"},
      {"id": "visible-toe", "bone": "paw_front_R", "end": "tail", "point": [728, 469],
       "uncertaintyPx": 4, "visibility": "visible", "correspondence": "tentative"}
    ]}
  }
}
```

The example preserves the retained wolf probe's tentative screen-side correspondence: it **cannot pass**.
Do not relabel that correspondence as confirmed without source-grounded evidence. Coordinates are finite
registered-image positions, not raw video pixels or working-mask pixels. Every visible position needs
positive finite `uncertaintyPx`, a source-pixel radius. `head` supports fitted non-root chain bones;
`tail` supports contract paws. IDs and targets are distinct within each frame. Unsupported/duplicate targets,
invalid frames and source hash mismatches are rejected. `visibility` is `visible`, `occluded` or
`out_of_frame`; `correspondence` is `confirmed`, `tentative` or `ambiguous`. Hidden points may omit
coordinates and uncertainty; no hidden target is invented.

Run `preflight --source-observations <file>` after ingest. It solves only the specified frames with the
existing camera/search/objective and stores isolated poses, mesh projection, report, assessment and lab under
`articulation/preflight/`; no full fit is required, and canonical fit files/ledger are untouched. Open its lab:
source outline, projected mesh, actual rig skeleton, source uncertainty circles and projected-point residual
lines are visible together. Per-point residuals are also reported numerically in source pixels and uncertainty
units, separately from IoU and objective terms. Phase/frame coverage is explicit; a fold-only checkpoint
does not claim full-motion recovery.

Only visible, confirmed observations contribute a squared normalized residual to the existing objective.
Independent assessment passes an observation only when its projected landmark is within the recorded
uncertainty radius (boundary included). Missing projections and hidden/tentative/ambiguous observations
remain unresolved, not zero-error passes. Neither silhouette score nor matching toes overrides a joint miss.
No uncertainty-free acceptance, weight/limit relaxation or explanation override exists.

For these usable source observations, the existing fitter also takes coupled numerical steps in
its own root/camera/joint parameters. A numerical Cartesian landmark Jacobian and the curvature of
distance to the original uncertainty disks guide bounded restoration; first-hit parameter bounds
can be released when another coupled direction is available. No artificial interior point targets,
new pose priors or expanded uncertainty disks define acceptance. Once feasible, refinement improves
the unchanged silhouette/marks/prior objective while retaining feasibility. Current rig limits and
fixed-camera passes remain binding.
Before writing output, the fitter checks the actual Blender bone projections used for the saved
landmarks. A small bounded correction can restore a NumPy/Blender boundary mismatch using that
exact oracle, with the shared camera fixed and the original uncertainty disks unchanged. Failure
remains unresolved; neither floating-point disagreement nor a nearly passing point expands acceptance.

The output still uses one shared camera, not a separately chosen camera for each observed frame.
With usable observations, the median is only a seed: camera scale and observed poses are restored
jointly, and candidate selection judges all their source disks before the unchanged objective.
Without usable observations, camera selection and search retain their legacy behavior. Frame and
relabel candidate selection likewise prefer disk feasibility over lower soft-objective or identity
cost; relabelling cannot trade away an established source disk.

Per-frame `constraintSearch` diagnostics recompute final normalized error after the shared camera,
relabelling and refinement. Its local passes include `initialRestoration`: bounded per-round errors,
linearized singular values, bound-active outcomes, proposed steps, actual line-search merit and a
stopping reason. The pass's `iterations` counts objective-refinement iterations, not restoration
rounds: `iterations: 1` can mean restoration remained unresolved before objective refinement began.
`unresolved` means this bounded search did not find feasibility, never that the rig/source is
globally impossible. Numerical search diagnostics do not override independent landmark assessment.
Passing every disk is still insufficient without the existing whole-body silhouette gate and visual
review; a displaced torso or head remains unacceptable. Phase labels identify source evidence only:
they do not manufacture pose priors or targets for unobserved joints.

For a constructive diagnostic only, `preflight --witness <poses.json>` can project retained plausible or
fixture poses using the current rig and registered camera, validate parameter names, finite values,
current limits and camera scale, independently assess the witness observations, and compare both poses under
the same full-resolution objective with no temporal/periodic prior. It does not seed the search, alter a fit
or author shipping keys. A passing witness proves that recorded configuration feasible; a strictly better
witness objective identifies a search miss, while a strictly preferred failed fit identifies objective
rejection. Ties, invalid/nonpassing witnesses and unsuccessful searches without a witness remain unresolved.
No finite failed search is reported as rig impossibility.
Keep witnesses outside `articulation/preflight/`: paths resolving into the output folder (including symlinks)
are rejected before any preflight write, so the hashed witness remains the exact input used in diagnosis.

`fit --source-observations <file>` (or reuse stored observations) requires that same sidecar's current passing
preflight, then automatically assesses the canonical fit. `assess` recomputes its assessment without solving.
Evidence binds the observation file, source video and selected registered frames, model, rig/project
contract, brief/limits, ingest metadata/camera/masks, pose/mesh projection and implementation hashes.
Changing any binding invalidates acceptance. `facts`, `bake`, `review`, `verify` and `export` require the
current canonical assessment to pass; changing marks/fit still invalidates the ordinary ledger stages.
The review template binds both observations and assessment, and export includes them. Preflight records
independent `articulation/enrollment.json` before solving; retained assessments also preserve the observation
binding. A deleted recorded sidecar is missing evidence even before the first full fit, never a way to regain
legacy status. Workspaces that never recorded observations retain their old workflow, with `not_assessed`
explicitly shown in the lab.

### Visible contour checkpoint (without invented joints)

When a folded limb is visibly wrong but the elbow or wrist is hidden, record its visible silhouette,
not a guessed anatomical landmark. `preflight --silhouette-regions <file>` works without
`--source-observations`; both may also be supplied together. Use schema version 1:

```json
{
  "schemaVersion": 1,
  "source": {
    "videoSha256": "<accepted source video SHA-256>",
    "coordinateSpace": "registered_source_px"
  },
  "frames": {
    "85": [{
      "id": "folded-recovery",
      "bounds": [730, 435, 820, 525],
      "visibility": "visible",
      "minIoU": 0.85,
      "weight": 1.0
    }]
  }
}
```

Bounds are registered-source pixels with exclusive right/bottom. The same crop applies to source
and fitted masks: floor left/top and ceil right/bottom after multiplying by working scale.
Each named region adds its weighted local `100 * (1 - IoU)` loss to the existing fitter objective;
the whole-body silhouette, marks, source uncertainty disks, joint limits and priors are unchanged.
`weight` balances recorded regions, never changes their individual acceptance threshold.
Unknown, occluded, out-of-frame or empty-source regions cannot certify acceptance.

On region-observed frames, candidate selection preserves the original source-disk priority, then
whole-body feasibility, then local-region feasibility before comparing the augmented objective.
An initially invalid body can improve its full-working-resolution overlap toward the unchanged
threshold; there is no infinite rejection barrier trapping the initial pose. Once body-feasible,
soft local gains cannot buy a body-gate failure. Grid/coordinate search, restart candidates,
joint restoration proposals, relabels and shared-camera selection use this regional ordering.
The shared camera also checks the full-fit mean, not only the region-observed frames.
These decisions use the existing full-resolution sampled occupancy as a search surrogate, with
bounded per-pose caching; only the independently rasterized final Blender geometry certifies a pass.
Coarse pyramid crops zero-extend a twice-rounded missing edge identically in both masks, never
inventing intersection or changing the source region. Final acceptance crops are not padded.

`fit --silhouette-regions <file>` requires that exact sidecar's current passing preflight; subsequent
fits reuse enrolled regions. The pipeline independently rasterizes the saved actual Blender mesh
projection, rather than trusting sampled optimizer occupancy or a claimed pose metric. Every
visible region must meet its `minIoU`; the unchanged brief `iouThreshold` must also pass for the
whole-fit mean and every region-observed frame. A locally improved paw with a displaced body fails.
The reports, box overlays, lab and review template expose both local and global results.
An unsuccessful finite search is not proof of infeasibility; do not lower either bar or fabricate a pass.

Region evidence reuses articulation's hash/current checks and binds source, selected registered frames,
rig/project, limits/brief, masks/camera metadata, actual poses/projection and implementation. Enrollment
is retained before Blender runs in `silhouette-regions/enrollment.json`; deleting a recorded sidecar
does not restore legacy status. `assess` recomputes the region assessment, and `facts`, `bake`, `review`,
`verify` and `export` require it to be current and passing. Review explanations cannot override it.
Export includes both the recorded region sidecar and its independent assessment. Workspaces with no
region enrollment retain their existing fitting and acceptance behavior.



### Restart noise

**Symptom:** a refit with more (correct, honoured) marks scores lower, swaps "move" to other frames, or a span refit
looks stuck, and each next attempt seems to make things worse. **Cause:** `fit` uses random restarts (`RESTARTS` per
chain per tracked frame) from one stream; the default seed is fixed, so identical inputs repeat exactly, but any input
change reshuffles every later restart. A single before/after pair confounds the change with restart luck.
**Fix:** treat each fit as one draw. Copy the work folder per run and fit each condition with `fit --seed N` for
N = 1, 2, 3 (plus the seed-0 run you have), run `facts` on each, and compare distributions (facts averaged-cycle IoU,
fit meanIoU, per-paw cross-cycle spread). Decide on the spread; never declare a block from one draw. The seed is in
the ledger's fit facts and `poses.json` (`restartSeed`). Choosing the best seed to ship is allowed only by the same
unchanged gates and review; it is not averaging and never lowers a threshold.

## Review, verify, export

| Symptom | Cause | Fix |
|---|---|---|
| "changed after the template was made" | An output changed after `review --template` | Run `review --template` again and review the new output |
| "Round N was already reviewed" | A revised output needs a new round | `review --template`, then review |
| `facts`: "The brief ... names no gait facts can check" | The loop brief's `gait` names none of walk, trot, pace, canter/lope, gallop (or their -ing forms) | Name the gait in the brief, or `facts --accept-gait "<why>"` (a warning in facts.json, the ledger and the lab) |
| `facts`: "stride of N frames ... under 10 frames", or no clear period with a best candidate under 10 frames | A small, fast animal's steps are undersampled at 24 fps (the field vole's trot: 4.5 frames) | Set the brief's `slowMotion` to the factor `facts` names (or more) and regenerate (`reference/motion.md`) |
| `facts`: "The cadence retime F ... is outside 0.5-2.0" | The video is far slower than the body size (or the brief's `slowMotion`) says, or the mass is wrong; the message names the stride the leg evidence gives | Check the mass (brief `massKg`, `--mass-kg`) and the stride (`cadence.strideEvidence`); if it stands, `facts --accept-retime "<why>"` |
| `facts` warns: "Body mass ... of an open mesh (N open edges)" | The mass is the rig's mesh volume and the mesh is not closed | Give the brief's `massKg` or `facts --mass-kg`; nothing is retimed on that mass |
| `facts`: "A paw's velocity jumps at a contact edge" | The pin does not ease in or out (`contactCheck.edgeJumps` against the swing's median) | Check the leg chains and the `ik` residuals; the pins should ease over 4 keys |
| `facts` warns: "keeps under 0.95 of the video's world pitch range" | A trunk or tail bone's world pitch is a small sum of joints the smoothing moved apart | Look at that bone in the sheet; its own joint range is kept |
| `verify`: "phases ... render the same image" / "no paw bone moved" | The captures do not show the clip playing (the seek did nothing, or the clip does not move those bones) | Open the captures; check the patched `capture_asset.gd` and the GLB's clip |
| `verify`: "The 3D skill's capture script changed" | game-asset-3d-pipeline's `capture_asset.gd` no longer holds a line vid2godot patches to seek the clip to each phase | Update `PATCHES` in `anim/verify.py` to the new script; never verify the rest pose alone |
| `verify`: "Capture ... records phase ..., not phase ..." | A capture was not taken at its run's clip phase | Rerun `verify`; if it persists, check the patched `verify/<engine>/phaseNNN/verifier/capture_asset.gd` |
| "Frame N needs an explanation, or a re-mark and refit" | A low or flagged frame has no `action: "explained"` and note | Explain what the overlay shows, or mark and refit |
| `verify`: "not an activation-suppressed capture" or "lists no phone/tablet captures" | The run did not go through the launcher | Pass the project's activation-suppressed launcher; never run an engine in the foreground |
| `verify`: "Expected Godot 4.7" or "4.5" | The wrong binary for that flag | Pass each engine to its own flag |
| `export` refuses the destination | It is inside `game/assets`, or it exists | Export to a new folder under `asset-work`; shipping is the owner's decision |

## Rationalizations (from agents without this skill)

| Thought | Reality |
|---|---|
| "None of the scripts named below exist yet. I would write them" | They exist: the command table. |
| "Timing and stride come from re-timing to 24 frames and from the paw-planting step" | `ingest` and `facts` measure them. |
| "Replace only the `lope` samplers and accessors in the GLB with pygltflib" | `bake` writes an evaluation GLB. |
| "Under the project's rule for image models, the harness or the owner picks the model" | That rule is for images. The owner names the video model, or a criterion. |
| "The footfall phases … decide that" / "Hold the far legs to the near legs' phase offset" | Marks and `fit --span` decide identity. |
| "Depth is not observed … It comes from priors" / "It mostly happens in the side plane" | Unseen depth is why you refuse. |
| "Steps 1–8 are ordinary Blender and Python work" / "try `parent_set(type='ARMATURE_AUTO')` first" | Rigging is game-asset-3d-pipeline's. Hand it over. |
| "If it passes: I'll replace only the `lope` animation in `wolf.glb`" / "Before overwriting, I'll keep a backup" | Your checks are not approval; a backup is not either. |
| "Passing these checks isn't your artistic sign-off" (then planning the copy) | Approval precedes writing `game/assets`. Stop at `export`. |
