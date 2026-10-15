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
| `fit --span` refuses | No full fit exists yet | Run `fit` first; spans refine it |

## Review, verify, export

| Symptom | Cause | Fix |
|---|---|---|
| "changed after the template was made" | An output changed after `review --template` | Run `review --template` again and review the new output |
| "Round N was already reviewed" | A revised output needs a new round | `review --template`, then review |
| "Frame N needs an explanation, or a re-mark and refit" | A low or flagged frame has no `action: "explained"` and note | Explain what the overlay shows, or mark and refit |
| `verify`: "not an activation-suppressed capture" or "lists no phone/tablet captures" | The run did not go through the launcher | Pass the project's activation-suppressed launcher; never run an engine in the foreground |
| `verify`: "Expected Godot 4.7" or "4.5" | The wrong binary for that flag | Pass each engine to its own flag |
| `export` refuses the destination | It is inside `game/assets`, or it exists | Export to a new folder under `asset-work`; shipping is the owner's decision |
