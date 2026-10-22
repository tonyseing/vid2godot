# How `facts`, `bake` and `verify` make the clip (Task 19)

`facts` turns the fitted video into the clip the game plays: the contacts, a gait check, the cadence, and 60 Hz
keys with the planted paws pinned. `bake` writes those keys; `verify` films them playing in Godot.

## The period and the stride (vole round 3)

The period comes from the masks' self-similarity (`maskPeriod`): per lag, the median aligned 1 - IoU of frame
pairs that far apart. A lag is the period when its dip scores at most 0.25, dips by at least 0.2 of the curve's
median level (`prominence`, `level`), its family (it and its multiples) beats the median of the other lags by 2x
(`margin`), every multiple dips under that median, and no other dip comes within 1.5x (`rival`); the first such lag
is chosen, so the fundamental wins when its multiples dip too. A low, rigid body (the field vole changes its outline
by at most 17 %) scores every lag under 0.25; the old first-dip rule took a 0.003-deep noise dip at 16 frames where
the vole repeats at 30 (0.025) and 60 (0.011).

The stride is the period, or twice it when the masks show half a stride (near and far legs look alike). Each leg
pair's half-period test is read on its paw tracks (`identity.before`): "half" (a paw matches its partner a period
on), "whole", or "mixed". When a pair is mixed or the pairs disagree, or a paw touches down twice in the cycle,
`facts` resolves identity: where a pair's paws meet, the fit may bounce them back instead of letting them cross; the
paws keep their velocities through a meeting (one swings forward, one sweeps back), so the labels are exchanged
there when that carries the velocities on (`identity.pairs`: meetings and boundaries). For a brief trot or pace the
other pairs are then labelled so the origin's diagonal (trot) or lateral (pace) partner moves with it
(`identity.oriented`; a side view cannot tell). The relabelling is used only when every pair's test then agrees
(`identity.after`); otherwise the clip stays as fitted, and a paw that still touches down twice is refused. A clean
clip (the wolf lope: both pairs "whole") is untouched. The stride's evidence is `cadence.strideEvidence`.

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
  paws near the ground while they move back (a swing never enters it), measured again from the stances only; with
  no stance in two paws there is none.

**Body size (round 4, the field vole).** Those metres are the wolf's. They scale with the rig's leg length, the
mean rest height of the leg chains' top joints (hips, withers) in the contract's `rig.json`, over the wolf's
0.7225 m (`REFERENCE_LEG_M`): at the wolf they are exactly 0.025 m, 0.06 m and 0.10 m (`STILL_SPEED`), at the
4.6 cm vole 1.6 mm, 3.8 mm and 6.4 mm/s. `facts.json` records them, the scale and the source as `bodyScale`. The
stillness test reads a velocity without the averaged cycle's per-key jitter: the paws' kinematics of the smooth
cycle capped at 7 harmonics (`VELOCITY_ORDER`, the most a 15-frame cycle like the wolf's holds, so it is unchanged;
a slow-motion 60-frame cycle carries jitter up to its 29th, twice the vole's ground speed).

**Noisy fit.** When that strict reading still breaks a paw's stance into several touchdowns and every one lies
inside the paw's backward slide, they are one stance: each paw's stance is then its slide from its most forward to
its most backward point (a treadmill gait's anterior and posterior extreme positions) on its x capped at 4
harmonics, planted when it comes within the stance tolerance of the ground, and the speed is measured from those.
`contactReading` records which reading was used (`strict` or `noisy fit`, the strict touchdowns, the extremes),
and a warning says so. A touchdown outside the slide (a paw that plants again mid-swing) keeps the strict reading,
which is refused. The vole round-3 fit needs it: its planted paws float 4-17 mm against a 17-19 mm lift.

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

The stride frequency is checked against a band for the gait the brief names (else the measured footfalls' gait)
over the body mass M: the brief's `massKg` (a number m means [0.75 m, 1.25 m]; or `[lightest, heaviest]`), else
`facts --mass-kg`, else the rig's mesh volume at 1000 kg/m3. The band runs from the law at the heaviest mass to
the law at the lightest (Task 20; before it, only a canter or gallop had a band):

| Gait | Law (Hz, M in kg) | Source | 40 kg (30-50) | 0.03 kg (0.0225-0.0375) |
|---|---|---|---|---|
| walk | 2.11 M^-0.091 | derived, below | 1.48-1.55 (centre 1.51) | 2.84-2.98 (2.91) |
| trot, pace | 3.35 M^-0.130 | H&T 1988 Table 3, preferred trotting speed | 2.01-2.15 (2.08) | 5.13-5.49 (5.31) |
| canter/lope, gallop (transverse, rotary) | 4.44 M^-0.156 | H&T 1988 Table 3, preferred galloping speed | 2.41-2.61 (2.51) | 7.41-8.02 (7.72) |

- **Heglund & Taylor 1988** (H&T): N. C. Heglund and C. R. Taylor, "Speed, stride frequency and energy cost per
  stride: how do they change with body size and gait?", J. Exp. Biol. 138:301-318. Treadmill stride frequencies of
  16 quadruped species, 30 g mice to a 170 kg horse; Table 3 gives f = a M^b at five equivalent speeds (95 %
  limits on the preferred trot: a 2.90-3.50, b -0.099 to -0.149; preferred gallop: a 4.08-4.84, b -0.131 to -0.180).
  The canter/gallop law is unchanged, so the wolf lope's 2.41-2.61 Hz band and its 1.581 retime stand.
- **Pace**: two lateral beats at trotting speeds; H&T measured no pace, so it takes the trot's law.
- **Walk**: H&T have too few constant-speed walks for an equation. The walk law is their minimum trotting speed's
  frequency, 2.42 M^-0.091 (the slowest trot: near the walk-trot change, which quadrupeds make at a Froude number
  u^2/(g h) of about 0.5; R. McN. Alexander and A. S. Jayes 1983, "A dynamic similarity hypothesis for the gaits
  of quadrupedal mammals", J. Zool. 201:135-152), moved to a comfortable walk's Froude number, about 0.25
  (R. McN. Alexander 1989, "Optimization and gaits in the locomotion of vertebrates", Physiol. Rev. 69:1199-1227).
  At one leg length h the relative stride is 2.3 Fr^0.3 (R. McN. Alexander 1976, "Estimates of speeds of
  dinosaurs", Nature 261:129-130), so the stride frequency u / stride goes as Fr^0.2: a = 2.42 x 0.5^0.2 = 2.11.
  It is an estimate from dynamic similarity, not a fitted equation; read a walk's retime with that in mind.

The volume is a mass only for a closed mesh: with open (non-manifold) edges the cadence is checked against it and
warned about, naming the source and the open-edge count, and never retimed (the chase3d wolf has 1590 open edges and
reads 85 kg, so give its mass). With no mass it warns and does not check.

A video under the band whose steady cycles each repeat within 10 % of the clip's period (one time scale: AI video
is often slow motion) is **retimed** to the band centre, whatever the gait: the clip's duration divides by the
factor and its speed multiplies by it; the stride stays. The factor is `cadence.retime` in `facts.json` (with
`cadence.gait` and `cadence.law`), a warning and the lab's Motion section. Over the band, or with cycles that
disagree, it warns and does not retime. A factor outside 0.5-2.0 of the expected retime (below) is refused (the
mass is likely wrong) unless `facts --accept-retime "<why>"` records why (`cadence.retimeAccepted`, the ledger); the
refusal names the stride the leg evidence gives, and the bound is never widened for slow motion. A
retimed speed can then be far from the brief's `targetSpeedMps` (the stride is the video's): read that warning.

### Small, fast animals: `slowMotion`

A stride must span at least 10 video frames (`facts.MIN_STRIDE_FRAMES`). Under it the legs change between frames
faster than the video samples them, and no period can be read: the field vole's real-time trot (5.3 Hz) is 4.5
frames at 24 fps, and Seedance's vole shuffled with a 5-6 frame stride and no clear period (Task 20). `facts`
refuses a measured stride under 10 frames, and a clip with no clear period whose best candidate is under 10 frames,
with the band's centre and the smallest factor that would do (10 x centre Hz / fps; the vole: 2.2).

The brief's `"slowMotion": F` (over 1, at most 8; default none) asks for it: `generate` adds to the prompt's first
line "Filmed in slow motion: the whole clip plays F times slower than real time, with every step clearly visible,
each paw lifting off and setting down distinctly". The rest of the prompt is unchanged (strict side view, the
camera following, one shot, no cuts or zoom, four legs, the same animal). `facts` records F as
`cadence.expectedRetime` beside the measured `cadence.retime`, and `cadence.retimeVsExpected` (measured / F): the
measured cadence decides, and the 0.5-2.0 bound is on that ratio. A video that is not slow enough (not under the
band) is not retimed and warns that the slow motion asked for did not come. Without `slowMotion` the expected
retime is 1 and nothing changes. Pick F so the band's centre gives at least 10 frames, with room (the vole trot:
3, about 13.6 frames). A wolf galloping in real time (2.51 Hz) is 9.6 frames at 24 fps, so it needs F too.

## The keys

- **Smooth cycle**: each column of the averaged cycle as a truncated Fourier series. Trunk, neck, tail and root
  take the lowest order keeping 95 % of their sampled range; if its jerk is over 3 times a sine's of the same range
  at its dominant harmonic (a neck nodding at twice the stride rate; a root that bobs twice per stride is all second
  harmonic, `smooth.dominantHarmonic`), the harmonics above the dominant one are tapered until it is not. The series
  is then scaled about its mean up to the sampled range when it falls short (`smooth.overtoneTaper`, `smooth.gain`,
  `smooth.jerkRatio`, `smooth.keptRange`). Legs keep every harmonic. Each trunk and tail bone's world pitch range
  against the sampled cycle's is `smooth.worldPitchKept`; under 0.95 it is a warning (a joint's own range is kept,
  but a small world pitch that is a sum of differently-phased joints can shrink: the wolf's tail_03 at 0.46).
- **60 Hz, at least 24 keys per stride**: max(24, round(60 x duration)) keys over one cycle; the last repeats the
  first (a closed seam), drift zero. A fast stride's key rate rises above 60 Hz (the 5.31 Hz vole: 24 keys at
  127 Hz, not 11); the floor is the wolf lope's own 24 (60 x 0.398 s = 23.9), so the wolf keys as before. Recorded
  as `cycle.keysPerStride`, `cycle.minKeysPerStride` and `cycle.playedKeyRateHz`; Godot plays any key rate.
- **As played (vole round 6)**: the bar measures the played clip (keys linear between, differenced at 60 Hz)
  against a sine at the stride rate, so round 5's bound of 3 d^3 at the series' dominant harmonic d let the
  vole's tail through at 21.55 (`motion.for_keys`, `smooth.played`). Every trunk, neck and tail column holds no
  harmonic above the keys' own (11 at 24 keys: above it they alias into steps); a column whose played jerk is over
  3 has its overtones low-passed (a Gaussian whose width shrinks) until it is under 2.85, then is scaled back to
  the sampled range of the closed cycle at the keys (a peak between keys: the vole's tail_02 at 0.917); a column so
  changed whose world pitch range is under 0.95 is scaled up until kept (the vole's spine_01 at 0.689). A column
  whose range is a second harmonic cannot be under 3 and keep its range: it nods once per stride. A short stride
  needs a lower order: at 60 Hz the vole's stride spans 11 samples, the wolf's 24. Columns these rules leave alone
  are unchanged (the wolf lope's every column; its tail_03 world pitch warning stands). `smooth.worldPitchKept` is
  the baked cycle's, `smooth.seriesWorldPitchKept` the series'.
- **Ground**: the root is lifted so the shared ground is the rig's rest ground (`cycle.groundLift`).
- **Pins**: over each stance's keys less half a key at each end, IK holds the paw on a world-still line: on the
  ground, sliding back one stride per cycle. A chain with a fitted girdle first searches its one redundant girdle
  angle using the existing `fitconstraints.refine`. The existing analytic pair derives the upper, lower and wrist
  angles; counterrotation at the wrist preserves the original world foot frame. Actual FK must reach within
  1e-6 m and **every derived chain angle of this preservation candidate** must stay in the brief's original
  `limitsDeg` fit-search box (missing bones use [-60, 60] degrees, exactly as in the fitter). These are not new
  downstream acceptance limits. The objective is uniform squared full-chain angle change from the incoming pose,
  not a bend target; a reachable pin no longer mandates a fixed girdle and straightened elbow.
  If refinement from the incoming girdle angle is unresolved, deterministic scalar seeds cover the original girdle
  interval at at most one-degree spacing (`IK_GIRDLE_SAMPLE_DEG`); the lowest uniform-angle-change feasible seed is
  refined with the same helper, retaining the certified seed if refinement cannot improve it. This is numerical
  branch discovery, not an acceptance threshold or proof that an unsampled branch cannot exist.
  Body/root, unrelated joints and terminal paw angles stay frozen. If a frozen angle is already outside its
  fit-search box, the preservation search is skipped for that pose rather than changing the frozen angle.
  This is bounded local refinement, not a global optimum or infeasibility proof.
  When preservation search is unresolved or ineligible, the original analytic reach/toe-pivot correction is
  retained unchanged, including historically permitted angles outside the fit-search boxes. No fallback angle is
  clipped, and the wrapper returns the actual FK endpoint for the existing downstream contact/motion gates.
  `pivot=False` disables pivoting in the original analytic fallback, not fixed-foot redundancy. Short chains
  without a fitted girdle retain their original analytic behavior. The original fit-search boxes flow from the
  brief through facts, recursive seam planning, contact pins, ground clearance, edge settling and swing nudges;
  fitter boxes and downstream acceptance thresholds are unchanged.
  Primary `plan_keys` pins and standalone `solve_leg` calls use `preserve_source=True`: they preserve the fitted
  source articulation even when a fixed-pair endpoint alone would be reachable. Secondary ground clearance, edge
  settling (initial and iterative), and swing corrections act on an already-corrected pose and pass
  `preserve_source=False`. When the existing fixed-foot analytic pair reaches their target within the same 1e-6 m,
  it keeps the corrected support rather than redistributing the girdle again; redundant preservation recovery is
  still available when that pair misses. This stage distinction avoids unnecessary girdle work during secondary
  nudges without bypassing source-articulation recovery or changing played-ground/contact gates.
  With played-ground clearance enabled, primary preservation is also checked on the complete cyclic key matrix
  before clearance nudges. If its played chain dips below the existing penetration tolerance, the original analytic
  correction is compared on the same source poses, targets and body. It replaces that chain's active angles only
  when its actual endpoints reach within 1e-6 m and its played dip improves; unrelated angles and body stay frozen.
  This is candidate acceptance against the existing grounding constraint, not a new temporal solver or a change
  to pin timing, contact/ground thresholds, or key counts. Remaining dips still run through existing clearance.
  Past the whole straight leg's reach, the body still comes down by the least smooth amount with 2 % to spare
  (`ik.<paw>.bodyDropMaxM`); girdle motion is reported as `ik.<paw>.girdleSwingMaxDeg`. No eased or swinging key
  goes below its paw's ground (a noisy fit's mid-swing dip is lifted onto it). Vole round 6: the knee
  bends to the rig's rest side at every key (round 5's nearly straight vole elbows flipped between keys and the
  played paw swung 6.9 mm into the ground), and no paw goes more than 5 mm times the body's leg-length scale
  (0.32 mm at the vole) under its ground anywhere in the played cycle, between keys too: an interval that dips
  deeper has both keys' paws lifted by the dip and half that tolerance and the legs solved again
  (`ik.<paw>.clearanceLiftMaxM`). A paw whose agreement (below) is under 0.97 has its declared contact's edges
  settled: the unpinned keys inside it take the pin, the key just outside takes the pin's line two contact
  tolerances up, so the played paw comes down on its spot at the declared edge (`ik.<paw>.settledKeys`; the
  round-5 vole slid back into its touchdowns at 0.953-0.969, the wolf passes at 0.974 and up). Vole round 7: the
  key just outside is solved so the played paw crosses the bar's contact height (below) exactly at the declared
  edge, from the inside key's own height and the leg's kinematics between the keys (round 6's two contact
  tolerances, 2 x 1.59 mm at the vole, crossed the bar's 1.27 mm a third of a key late); a settled paw's swing that
  comes back down under that height wholly outside its declared contact is lifted to 1.5 contact heights there
  (`motion.clear_swing`, `ik.<paw>.swingLiftMaxM`: the round-6 vole's swings dragged back to 0.0-0.8 mm a few
  keys after liftoff), and the edges are settled again. A clip that fails the bar's agreement settles every paw
  under 0.97 on it; a clip that passes both readings (the wolf lope, 0.957 and up on the bar's) is untouched.
  Outside, the paw follows the free motion plus the correction where the pin lets go, faded out by a quintic over
  max(0.05, 0.125) of the stride (`PIN_RELEASE.span_stride`: round 4's 3 key intervals at the wolf's 24 keys, now a
  share of the stride so a short stride's stance is not eaten). The correction is set by `motion.PIN_RELEASE` (span, the derivatives
  matched per axis at touchdown and at liftoff, and the inset; recorded as `contactCheck.pinRelease`). The default
  matches position, velocity and acceleration in x and position and velocity in z at both edges, so liftoff and
  touchdown have no velocity step. It came from a sweep on the wolf lope, measured with chase3d's
  `tools/anim_compare.py` (Task 19 rounds 3 and 4; 4 keys in fix round 1, 2.6 in round 3): of the settings that pass
  bar criteria 1-6 and facts' gates, 3 keys pops least at the contact edges. None passes the per-paw edge criterion
  (the front paws' lift-offs). `ik` reports each paw's pinned and eased keys and its largest residual (a leg that
  cannot reach its pin leaves one). A residual over the body's contact tolerance refuses (`contactCheck.ikReach`).
- **Check**: the keys are played as the clip plays them (linear between keys) and detected again with the same
  rule, without the stance allowance (a pinned paw is on the ground) and with the body's scaled contact tolerance
  (`contactCheck.redetectionTolerance`). Under 0.95 agreement for any paw, `facts` refuses (`contactCheck`);
  `plantedStillShare` reports how much of each stance is under the body's still speed (0.10 m/s at the wolf).
  Vole round 7: the bar's own reading too (`contactCheck.barContact`; chase3d `tools/anim_compare.py` criterion
  5): the keys played at 600 samples a 24-key cycle, a paw in contact while under 2 cm times the body's leg-length
  scale above its rest height (1.27 mm at the vole), height alone, against its declared contact. Under 0.95 for
  any paw, `facts` refuses.
- **Loop seam (vole round 6)**: per paw and trunk, neck and tail column, the velocity jump across the seam is at
  most the cycle's median per-key change (`contactCheck.loopSeam`). Where it is not, the loop starts at the
  nearest key with the most room outside the eases where it is (`loopSeam.movedFrom`: the round-5 vole started
  mid-swing of front_L, jump 6.1 times the median); where no key does, a warning. The wolf passes where it starts.
- **Edges**: `contactCheck.edgeJumps` is each paw's velocity change (x and z at the keys, m/s per key; in place
  equals world for a constant body speed, as chase3d's harness measures it) against its swing's median per-key
  change. `touchdownMps` and `liftoffMps` are the largest across the whole approach and release span, where the kink
  is (round 4); `touchdownPinnedMps` and `liftoffPinnedMps` are at the first and last pinned key (round 3). An edge
  over the median at the pinned keys refuses (the pin pops). Across the span the rule is recorded in
  `contactCheck.edgeSpanGate` and warned; it refuses only with `facts.EDGE_GATE_SPAN` on, which is off pending a
  ruling: it refuses every clip measured, the old hand-made wolf lope's own motion included.

A one-shot (`"loop": false`) keys its fitted frames; with the rig, `facts` predicts each paw at every key as the
bake plays it (dx less its mean), so `bake` checks a one-shot the same way.

`bake` keys facts' cycle and reads every paw back from the re-imported GLB: more than 1 mm (times the body's
leg-length scale, 0.064 mm at the vole) from facts' prediction refuses (the GLB's bones do not turn as `rig.json` says; rerun `contract`).

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
