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
stances. The keys pin each paw over its stance rounded to the key grid; `entry.contacts` reach half a key past the
first and last pinned key (the paw lands and lifts between keys; the midpoint is the estimate), so the phase origin's
footfall is half a key before 0 (0.979 at 24 keys).

## The gait check

The brief's `gait` text names the expected gait by its earliest word: walk, trot, pace, canter or lope (one gait),
gallop, transverse gallop, rotary gallop. The measured footfalls read as:

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
[0.75 m, 1.25 m]; or `[lightest, heaviest]`), else `facts --mass-kg`, else the rig's mesh volume at 1000 kg/m3
(rough: an open or furred mesh reads heavy; the chase3d wolf reads 85 kg, so give its mass). 30-50 kg is
2.41-2.61 Hz.

A video under the band whose steady cycles each repeat within 10 % of the clip's period (one time scale: AI video
is often slow motion) is **retimed** to the band centre: the clip's duration divides by the factor and its speed
multiplies by it; the stride stays. The factor is `cadence.retime` in `facts.json`, a warning and the lab's Motion
section. Over the band, or with cycles that disagree, it warns and does not retime. A retimed speed can then be far
from the brief's `targetSpeedMps` (the stride is the video's): read that warning.

## The keys

- **Smooth cycle**: each column of the averaged cycle as a truncated Fourier series. Trunk, neck, tail and root
  take the lowest order keeping 95 % of their sampled range; if its jerk is over 3 times a sine's of the same range
  (a neck nodding at twice the stride rate), its overtones are tapered until it is not and the series is scaled
  about its mean back to the range (`smooth.overtoneTaper`, `smooth.gain`, `smooth.jerkRatio`). Legs keep every
  harmonic.
- **60 Hz**: round(60 x duration) keys over one cycle; the last repeats the first (a closed seam), drift zero.
- **Ground**: the root is lifted so the shared ground is the rig's rest ground (`cycle.groundLift`).
- **Pins**: over each stance's keys, two-bone IK (the two bones above the last two; the foot keeps its world
  orientation, the knee its bend; where the leg cannot reach, the foot turns about the pinned toe, the heel lifting)
  holds the paw on a world-still line: on the ground, sliding back one stride per cycle. It eases in and out over 0.05 of a cycle outside the window. `ik` reports each paw's largest residual (a
  leg that cannot reach its pin leaves one).
- **Check**: the keys are played as the clip plays them (linear between keys) and detected again with the same
  rule, without the 0.06 m allowance (a pinned paw is on the ground). Under 0.95 agreement for any paw, `facts`
  refuses (`contactCheck`); `plantedStillShare` reports how much of each stance is under 0.10 m/s.

`bake` keys facts' cycle and reads every paw back from the re-imported GLB: more than 1 mm from facts' prediction
refuses (the GLB's bones do not turn as `rig.json` says; rerun `contract`).

## Verify shows the clip playing

For phases 0, 1/3 and 2/3 of the clip, `verify` copies game-asset-3d-pipeline's `forge/godot` folder into the work
folder and patches its `capture_asset.gd` to play the clip, seek to that phase and pause before every capture.
Each capture is named `phaseNNN-<profile>-<view>.png` and records `phase`, `clip` and `clipSeconds`; each
engine's `godot-report.json` lists `phases`, `runs` and the verifier's hash. If the 3D skill's capture script
changes so a patch point is gone, `verify` refuses instead of capturing the rest pose. Open every capture.
