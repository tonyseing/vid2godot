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
| `facts`: "A paw touches down 2 times in one cycle" | Leg identity flipped inside the loop | Mark and refit the span, then `facts` again |
| `facts`: "too short to hold two cycles" or "No full cycle after the lead-in" | The video is too short (the first 10% is lead-in) | Raise `videoSeconds` in the brief and generate again |
| "Measured ground speed ... is more than 25% from the brief's" | The video moves at another pace than the target | Note it in the review; generate again if the gait reads wrong |
| `facts.json`: a paw has `"plants": false` (listed in `gait.neverPlants`), and the gait shows fewer beats than the brief's | Measured against the one ground every paw shares (`ground.height`, the 5th percentile of all paws' rest-relative heights in the averaged cycle, + 0.025 m; each paw's height is measured from its rest tail height `restZ`; a one-sample lift inside a stance is bridged), that paw never comes down: the fit holds it in the air (`lowestAboveGround` says by how much) | Open the lab on that paw's stance frames; mark the paw on the ground and refit those spans, then `facts` again. Its contacts, footfall and the beats/lead/suspension leave it out |
| `facts` refuses: "no rest tail for <paw> (<bone>)" | facts measures each paw's height from its own rest-pose tail height (a rig's paw tails need not rest at one height), read from the contract stage's `rig.json`; that rig.json has no tail for the paw bone | Rerun `contract` on the rig (its `rig.json` records every bone's rest head and tail); check the project contract's `paws` names the rig's paw bones |
| `facts.json`: `speedMps` is null with a `speedReason` | Fewer than two paws plant on the shared ground | As above: the fit lifts the paws; mark and refit before trusting speed or stride |
| `facts.json`: `legPeriod.scoreAtPeriod` | The fitted leg joints' repeat at the video's period. Lower is better (0 is a perfect repeat); it passes when under `gate` (`passes: true`). It reports and gates nothing | `passes: false` means the leg joints vary from cycle to cycle; the averaged-cycle check is what gates |
| `fit --span` refuses | No full fit exists yet | Run `fit` first; spans refine it |
| IoU falls frame by frame through the clip; forelegs fold into the chest | Tracking held a local minimum; a leg inside the body costs no IoU | `fit` restarts each chain, tries each leg pair exchanged, re-solves any frame under 0.85 IoU from scratch, and pulls fitted paws to the paws found in the mask; mark the frames that stay low and refit their span |
| The loop shows the animal still settling from the reference pose | The first cycles are a warm-up | `facts` averages only the steady cycles (their masks repeat a period on); if the clip never settles, generate a longer video |
| `facts`: "no full cycle found in the video" | The masks do not repeat clearly within 0.25-2.5 s (a one-shot, a still clip, or a gait that changes) | For a loop, generate again; a one-shot needs `"loop": false` in the brief |
| `facts`: "The averaged cycle ... does not match the video" | The fit chose different poses for repeating silhouettes (joints the outline barely constrains trade angle), so their average misses the masks | `fit` already runs a periodic pass; mark paws on a few frames per cycle and refit, then `facts` again |

## Review, verify, export

| Symptom | Cause | Fix |
|---|---|---|
| "changed after the template was made" | An output changed after `review --template` | Run `review --template` again and review the new output |
| "Round N was already reviewed" | A revised output needs a new round | `review --template`, then review |
| "Frame N needs an explanation, or a re-mark and refit" | A low or flagged frame has no `action: "explained"` and note | Explain what the overlay shows, or mark and refit |
| `verify`: "not an activation-suppressed capture" or "lists no phone/tablet captures" | The run did not go through the launcher | Pass the project's activation-suppressed launcher; never run an engine in the foreground |
| `verify`: "Expected Godot 4.7" or "4.5" | The wrong binary for that flag | Pass each engine to its own flag |
| `export` refuses the destination | It is inside `game/assets`, or it exists | Export to a new folder under `asset-work`; shipping is the owner's decision |
