# How `facts`, `bake` and `verify` make the clip (Task 19)

`facts` turns the fitted video into the clip the game plays: the contacts, a gait check, the cadence, and 60 Hz
keys with the planted paws pinned. `bake` writes those keys; `verify` films them playing in Godot.

## Contacts from the motion

The motion is the forward kinematics of the smooth cycle (below) on the contract's `rig.json`, read at 240 samples
per cycle. Every fitted joint turns about its bone's side axis, which is world +Y at rest, so the side view is exact
in two dimensions (`anim/motion.py`, `PlanarRig`); the bake checks it against Blender (below).

A paw is on the ground while it is **near the ground and world-still**:

- near: each paw's height is measured from its own rest tail height (`restZ`), against one ground for all paws
  (the 5th percentile of every paw's height): within 0.025 m (`CONTACT_TOLERANCE`) somewhere in the stance, and
  within 0.06 m (`STANCE_TOLERANCE`) throughout it. The fit can hold a planted paw a few cm up (a side silhouette
  does not pin it); the real lope's front_L is world-still at 3 to 6 cm for 0.15 of a cycle;
- world-still: the clip plays in place, so a planted paw slides back at the ground speed. Its in-place x speed is
  within 0.3 x (`SLIP_SHARE`) of minus the ground speed. The ground speed is the median backward speed of the
  paws near the ground, measured again from the stances.

Gaps up to 0.05 of a cycle inside a stance are bridged. `gait` reads beats, lead and suspensions from these
stances. Each stance is rounded to the key grid; `entry.contacts` reach half a key past its first and last key (the
paw lands and lifts between keys; the midpoint is the estimate), so the phase origin's footfall phase is half a key
before its first pinned key (0.979 at 24 keys). The pin holds the keys strictly inside, from half a key after the
rounded touchdown to half a key before the rounded liftoff (see the keys).

The loop's first key is not the phase origin's touchdown (Task 19 round 4): `motion.start_key` picks the key with the
most room outside every paw's pin approach and release, so the seam is where no paw eases into or out of its pin
(on the wolf lope the origin's touchdown key lay inside hind_L's approach and hind_R's; the start moved 5 keys).
`contactCheck.cycleStart` records the key and its room; `cycle.startKey`, `cycle.startPhase` and `start` (the video
frame of key 0) move with it, `entry.contacts` are the clip's phases, and `entry.footfall_phase` stays the phase
origin's (each contact's start plus `startPhase`). The clip is cyclic, so only the seam's place changes.

## The gait check

The brief's `gait` text names the expected gait by its earliest whole word: walk, trot, pace, canter or lope (one
gait), gallop, transverse gallop, rotary gallop, and their inflections (walking, trotting, pacing, cantering,
loping, galloping). A loop brief whose gait names none of them is refused too, unless accepted with
`facts --accept-gait`; the acceptance is a warning in `facts.json`, the ledger and the lab. The measured footfalls
read as:

| Gait | Pattern |
|---|---|
| walk | 4 single beats, each hind followed by its own side's fore, no suspension |
| trot / pace | 2 beats, each a diagonal / a lateral pair |
| canter (lope) | 3 beats, one a diagonal pair, a fore lead, one suspension |
| gallop | 4 single beats, hinds together and fores together, a suspension |
| transverse / rotary gallop | a gallop whose fore lead is on the hind lead's side / the other side |

A mismatch refuses with the measured pattern. Regenerate, or run `facts --accept-gait "<why this motion stands>"`:
the reason goes to `facts.json` (`gaitCheck.accepted`), the ledger and the lab. Wolves and coyotes mostly lope in
a transverse gallop (Biancardi & Minetti 2012), and the Seedance wolf lope reads as one.

## The cadence

For a canter or gallop, the stride frequency is checked against the band of the preferred-gallop frequency
4.44 M^-0.156 Hz (Heglund & Taylor 1988) over the body mass M: the brief's `massKg` (a number m means
[0.75 m, 1.25 m]; or `[lightest, heaviest]`), else `facts --mass-kg`, else the rig's mesh volume at 1000 kg/m3.
The volume is a mass only for a closed mesh: with open (non-manifold) edges the cadence is checked against it and
warned about, naming the source and the open-edge count, and never retimed (the chase3d wolf has 1590 open edges and
reads 85 kg, so give its mass). 30-50 kg is 2.41-2.61 Hz.

A video under the band whose steady cycles each repeat within 10 % of the clip's period (one time scale: AI video
is often slow motion) is **retimed** to the band centre: the clip's duration divides by the factor and its speed
multiplies by it; the stride stays. The factor is `cadence.retime` in `facts.json`, a warning and the lab's Motion
section. Over the band, or with cycles that disagree, it warns and does not retime. A factor outside 0.5-2.0 is
refused (the mass is likely wrong) unless `facts --accept-retime "<why>"` records why (`cadence.retimeAccepted`, the
ledger). A retimed speed can then be far from the brief's `targetSpeedMps` (the stride is the video's): read that
warning.

## The keys

- **Smooth cycle**: each column of the averaged cycle as a truncated Fourier series. Trunk, neck, tail and root
  take the lowest order keeping 95 % of their sampled range; if its jerk is over 3 times a sine's of the same range
  at its dominant harmonic (a neck nodding at twice the stride rate; a root that bobs twice per stride is all second
  harmonic, `smooth.dominantHarmonic`), the harmonics above the dominant one are tapered until it is not. The series
  is then scaled about its mean up to the sampled range when it falls short (`smooth.overtoneTaper`, `smooth.gain`,
  `smooth.jerkRatio`, `smooth.keptRange`). Legs keep every harmonic. Each trunk and tail bone's world pitch range
  against the sampled cycle's is `smooth.worldPitchKept`; under 0.95 it is a warning (a joint's own range is kept,
  but a small world pitch that is a sum of differently-phased joints can shrink: the wolf's tail_03 at 0.46).
- **60 Hz**: round(60 x duration) keys over one cycle; the last repeats the first (a closed seam), drift zero.
- **Ground**: the root is lifted so the shared ground is the rig's rest ground (`cycle.groundLift`).
- **Pins**: over each stance's keys less half a key at each end, two-bone IK (the two bones above the last two; the
  foot keeps its world orientation, the knee its bend; where the leg cannot reach, the foot turns about the pinned
  toe, the heel lifting) holds the paw on a world-still line: on the ground, sliding back one stride per cycle.
  Outside, the paw follows the free motion plus the correction where the pin lets go, faded out by a quintic over
  max(0.05 of a cycle, 3 key intervals). The correction is set by `motion.PIN_RELEASE` (span, the derivatives
  matched per axis at touchdown and at liftoff, and the inset; recorded as `contactCheck.pinRelease`). The default
  matches position, velocity and acceleration in x and position and velocity in z at both edges, so liftoff and
  touchdown have no velocity step. It came from a sweep on the wolf lope, measured with chase3d's
  `tools/anim_compare.py` (Task 19 rounds 3 and 4; 4 keys in fix round 1, 2.6 in round 3): of the settings that pass
  bar criteria 1-6 and facts' gates, 3 keys pops least at the contact edges. None passes the per-paw edge criterion
  (the front paws' lift-offs). `ik` reports each paw's pinned and eased keys and its largest residual (a leg that
  cannot reach its pin leaves one).
- **Check**: the keys are played as the clip plays them (linear between keys) and detected again with the same
  rule, without the 0.06 m allowance (a pinned paw is on the ground). Under 0.95 agreement for any paw, `facts`
  refuses (`contactCheck`); `plantedStillShare` reports how much of each stance is under 0.10 m/s.
- **Edges**: `contactCheck.edgeJumps` is each paw's velocity change (x and z at the keys, m/s per key; in place
  equals world for a constant body speed, as chase3d's harness measures it) against its swing's median per-key
  change. `touchdownMps` and `liftoffMps` are the largest across the whole approach and release span, where the kink
  is (round 4); `touchdownPinnedMps` and `liftoffPinnedMps` are at the first and last pinned key (round 3). An edge
  over the median at the pinned keys refuses (the pin pops). Across the span the rule is recorded in
  `contactCheck.edgeSpanGate` and warned; it refuses only with `facts.EDGE_GATE_SPAN` on, which is off pending a
  ruling: it refuses every clip measured, the old hand-made wolf lope's own motion included.

A one-shot (`"loop": false`) keys its fitted frames; with the rig, `facts` predicts each paw at every key as the
bake plays it (dx less its mean), so `bake` checks a one-shot the same way.

`bake` keys facts' cycle and reads every paw back from the re-imported GLB: more than 1 mm from facts' prediction
refuses (the GLB's bones do not turn as `rig.json` says; rerun `contract`).

## Verify shows the clip playing

For phases 0, 1/3 and 2/3 of the clip, `verify` copies game-asset-3d-pipeline's `forge/godot` folder into the work
folder and patches its `capture_asset.gd` to play the clip, seek to that phase and pause before every capture.
Each capture is named `phaseNNN-<profile>-<view>.png` and records `phase`, `clip` and `clipSeconds`; each
engine's `godot-report.json` lists `phases`, `runs` and the verifier's hash, and takes the worst run's status. If
the 3D skill's capture script changes so a patch point is gone, `verify` refuses instead of capturing the rest pose.

That the clip plays is measured: the patch also records every paw bone's world position at each phase (`paws` on
each capture). For each profile and view, every two phases' captures must differ (another image hash, and over
0.1 % of the pixels differing by more than 8 of 255 in a channel), and between every two phases some paw bone must
move over 5 mm; otherwise `verify` fails with the numbers (`phaseEvidence`). Open every capture.
