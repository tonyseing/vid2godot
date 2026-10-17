"""Clip facts from the fitted poses: one clean loop, paw contacts, ground speed and stride, the sidecar entry.

A loop's period comes from the video (the registered masks' self-similarity, anim/cycle.py), not from the fitted
angles: joints the silhouette barely constrains trade angle for the same outline from frame to frame. The cycle is
the phase-aligned average over the steady cycles (warm-up left out, leg identity aligned per cycle, angles by their
circular mean), rolled to the phase origin's touchdown, and it is rendered at every frame's phase of those cycles and
checked against the masks (mean IoU at least CYCLE_MIN_IOU) before it is accepted.

Paw contacts are measured against one ground shared by every paw (`ground_height`), never each paw's own lowest
point: a paw that never comes within CONTACT_TOLERANCE of that ground has no contact (`plants: false`), and the beats,
lead, suspension and ground speed come from the planted paws only. A one-sample lift inside a stance is bridged.
Each paw's height is measured from its own rest-pose tail height (`rest_heights`, from the contract's rig.json) before
the ground and the tolerance apply: a rig's paw tails need not rest at one height (the chase3d wolf's front tails
rest 2.7 cm over its hind tails), and absolute heights would read the higher-resting paws as never planting.

Task 19: a paw is on the ground while it is near that ground AND world-still: in the in-place clip a planted paw slides
back at the body's ground speed, so its in-place x speed is within SLIP_SHARE of minus the ground speed (measured from
the planted paws). A stance is seeded within CONTACT_TOLERANCE of the ground and carries on while the paw stays still
within STANCE_TOLERANCE (a side silhouette lets the fit hold a planted paw a few cm up). With the contract's rig
(`rig`), the motion is the forward kinematics of the smooth periodic cycle (anim/motion.py) at DENSE samples; the
keys pin every planted paw over that detection on the 60 Hz key grid (two-bone IK), the declared contacts reach half a
key past the pinned keys at each end (where between keys the paw lands and lifts; so the phase origin's footfall phase
is 1 - 0.5 / n_keys, half a key before its first pinned key). The loop's first key is the one with the most room
outside every paw's pin approach and release (round 4: motion.start_key; the cycle is rolled there, the start frame
moves with it, and the footfall phases stay the phase origin's while the contacts are the clip's); and the
keys, played as the clip plays them, are detected again (the same rule, without the fit-error allowance): under
CONTACT_AGREEMENT per paw facts refuses. The footfall
pattern is checked against the brief's `gait` (`expected_gait`, `gait_reads`), and the cadence against the
Heglund-Taylor band for the gait and the body mass (`cadence_check`, CADENCE_LAWS; Task 20: walk, trot or pace,
canter or gallop): a slow video with one consistent period is retimed to the band centre, recorded in `cadence` and the
warnings. The brief's `slowMotion` is the expected retime (the measured one decides; both are recorded), and a stride
under MIN_STRIDE_FRAMES video frames is refused: the video undersamples the steps.

Round 4 (the field vole): the metre tolerances are the wolf's and scale with the rig's leg length (`tolerances`,
recorded as `bodyScale`); stillness is read from the smooth cycle capped at VELOCITY_ORDER harmonics; the ground
speed comes from stances only; and a stance the strict reading breaks into touchdowns that all lie inside the paw's
backward slide is read as that slide (`read_cycle`, recorded as `contactReading`).

Vole round 6: the keys are held to the bar as played (motion.for_keys, `smooth.played`), cleared of the ground
(`penetration`), settled at a paw's contact edges under SETTLE_BELOW agreement, and the loop starts where its seam
has no velocity jump (`contactCheck.loopSeam`).

Vole round 7: the keys are also read as the bar reads contacts (`bar_agreement`, `contact_height`: under 2 cm x leg
scale, height alone; `contactCheck.barContact`), settled at that height, their swings lifted out of it, and gated on
it."""
import re
from typing import NamedTuple, Optional

import numpy as np

from anim import cycle, motion
from anim.ledger import StageError

LEAD_IN = 0.10
# The fitted angles' period score (`period_search`) gates nothing since Task 17b fix round 2; it is reported, on the
# well-observed leg joints, against this threshold.
PERIOD_MAX_SCORE = 0.2
# The averaged cycle, rendered at each frame's phase, must match the video masks at least this well (mean IoU).
CYCLE_MIN_IOU = 0.85
# A paw is on the ground while it is within CONTACT_TOLERANCE (metres) of the shared ground. The ground is this low
# percentile of every paw's height over the measured stretch (a loop: the averaged cycle; a one-shot: the clip). A
# percentile, not the minimum, so one sample that dips through the floor does not set it; a low one, so it lies inside
# the planted paws' stance heights (each planted paw spends a quarter or more of a cycle on the ground).
CONTACT_TOLERANCE = 0.025
GROUND_PERCENTILE = 5
# Touchdowns closer than this (a fraction of the cycle, and never less than one cycle sample) are one beat.
BEAT_TOLERANCE = 0.07
# Task 19. A planted paw's world speed (its in-place x speed plus the ground speed) is under SLIP_SHARE of the ground
# speed: a swinging paw passes the ground at about twice the body's speed, a planted one ~0. A stance seeded within
# CONTACT_TOLERANCE carries on while the paw stays that still within STANCE_TOLERANCE of the ground. Measured on the
# real lope (FK of the smooth cycle): the paws are world-still at 3-6 cm for up to 0.15 of a cycle around their
# lowest point (front_L), sliding at 0.0-0.9 m/s against swings at 2-8 m/s (ground speed 3.6 m/s).
SLIP_SHARE = 0.3
STANCE_TOLERANCE = 0.06
# Free gaps up to this share of the cycle inside a stance are bridged on the dense samples (one sample of the
# ~15-sample averaged cycle before).
BRIDGE_PHASE = 0.05
DENSE = 240                  # samples per cycle the rig's motion is read at
CONTACT_AGREEMENT = 0.95     # declared contacts vs a re-detection from the baked keys, per paw
SETTLE_BELOW = CONTACT_AGREEMENT + 0.02   # vole round 6: a paw under this agreement has its contact edges settled
# Vole round 7: the bar's contact (chase3d tools/anim_compare.py criterion 5, PRIMARY_THR): a paw under 2 cm above its
# rest height, height alone, at the wolf; scaled with leg length as the round-4 tolerances are (the vole's 1.27 mm).
# Under CONTACT_TOLERANCE, so a key planted within this is within facts' own contact too.
BAR_CONTACT = 0.02
# Round 4 ruling: the edge gate (an edge's velocity change at most the paw's swing median per-key change) measured
# across each pin's whole approach and release span, where the bar's harness finds the kink (motion.edge_jumps).
# Measured, recorded and warned on every clip; it refuses only while EDGE_GATE_SPAN is True. It is False pending the
# controller's decision because it refuses every clip measured with every pin-release setting: the wolf lope, the
# toy and the real fixture, and the old hand-made wolf lope itself (edges 2.1-2.4 m/s against swing medians
# 1.6-1.8). The round-3 gate (the same rule at the first and last pinned key) is enforced either way.
EDGE_GATE_SPAN = False
STILL_SPEED = 0.10           # m/s: the report's "still" for a planted paw in the played clip
# Round 4 (vole): the metre tolerances above (CONTACT_TOLERANCE, STANCE_TOLERANCE, STILL_SPEED) are the reference
# body's, the chase3d wolf's: they scale with the rig's leg length (`leg_length`: the mean rest height of the leg
# chains' top joints, hips and withers, from the contract's rig.json) over REFERENCE_LEG_M, the wolf's own (its
# rig.json: hips 0.645 m, withers 0.800 m), so at the wolf they are exactly those values.
REFERENCE_LEG_M = 0.7225002944469452
# Stillness is read from a velocity without the averaged cycle's per-key jitter: the paws' forward kinematics of
# the smooth cycle with every column capped at VELOCITY_ORDER harmonics (the most a 15-frame cycle such as the
# wolf's holds, so its velocity is unchanged; a slow-motion 60-frame cycle carries fit jitter up to its 29th).
VELOCITY_ORDER = 7
# When that reading still breaks a planted paw's stance into several touchdowns, all inside the paw's backward slide,
# the contacts are read for a noisy fit (`read_cycle`): each stance is the slide between the paw's anterior and
# posterior extreme positions on its x track capped at NOISY['order'] harmonics (the vole round-3 fit holds planted
# paws 4-17 mm up against a 17-19 mm lift, and its stance speeds jitter by twice the ground speed). Recorded, warned.
NOISY = {"order": 4}
MASS_SPREAD = 0.25           # a single massKg m is the range [m (1 - spread), m (1 + spread)]
CADENCE_CONSISTENCY = 0.10   # every steady cycle's own period within this share of the clip's: one time scale
RETIME_BOUND = (0.5, 2.0)    # a retime factor outside this needs `facts --accept-retime <reason>` (fix round 1)
GALLOPS = ("canter", "gallop", "transverse gallop", "rotary gallop")
# Task 20: stride frequency f = a M^b Hz (M in kg) per gait, each the centre of its band over the mass range. Heglund &
# Taylor 1988 (J Exp Biol 138:301-318, Table 3; 16 species, 30 g mice to 170 kg horses): the preferred trotting speed
# for a trot (a pace, the lateral two-beat, takes the trot's), the preferred galloping speed for a canter/lope and the
# gallops (the band the wolf lope was checked against, unchanged). Heglund & Taylor have no walk equation; the walk is
# their minimum trotting speed's (2.42 M^-0.091, about the walk-trot change at Froude number 0.5, Alexander & Jayes
# 1983) at the comfortable walk's Froude number 0.25 (Alexander 1989): at one leg length the stride frequency goes as
# Froude^0.2 (relative stride 2.3 Froude^0.3, Alexander 1976), so a = 2.42 x 0.5^0.2. reference/motion.md.
CADENCE_LAWS = {
    "walk": (2.42 * 0.5 ** 0.2, -0.091, "a walk: Heglund & Taylor 1988's minimum trotting speed 2.42 M^-0.091 Hz at "
                                        "half its Froude number (x 0.5^0.2; Alexander & Jayes 1983, Alexander 1989)"),
    "trot": (3.35, -0.130, "Heglund & Taylor 1988: stride frequency at the preferred trotting speed 3.35 M^-0.130 Hz"),
    "gallop": (4.44, -0.156, "Heglund & Taylor 1988: stride frequency at the preferred galloping speed "
                             "4.44 M^-0.156 Hz"),
}
GAIT_LAW = {"walk": "walk", "trot": "trot", "pace": "trot", **{g: "gallop" for g in GALLOPS}}
# Task 20: a stride must span at least this many video frames: under it (the field vole's ~5-6 frames at 24 fps) the
# legs change between frames faster than the video samples them, and no repeating period can be read.
MIN_STRIDE_FRAMES = 10
# The brief's `gait` vocabulary: the earliest of these whole words (or their inflections) names the expected gait.
GAIT_WORDS = ((r"transverse gallop(?:s|ing|ed)?", "transverse gallop"), (r"rotary gallop(?:s|ing|ed)?", "rotary gallop"),
              (r"gallop(?:s|ing|ed)?", "gallop"), (r"canter(?:s|ing|ed)?", "canter"), (r"lop(?:e|es|ing|ed)", "canter"),
              (r"trot(?:s|ting|ted)?", "trot"), (r"pac(?:e|es|ing|ed)", "pace"), (r"walk(?:s|ing|ed)?", "walk"))
VOCABULARY = "walk, trot, pace, canter/lope, gallop, transverse gallop or rotary gallop (or their -ing forms)"


class FactsError(StageError):
    pass


class TouchdownError(FactsError):
    """A paw touches down more than once in one cycle (vole round 3: facts then resolves leg identity and retries)."""


class Tolerances(NamedTuple):
    """Round 4: the contact tolerances for one body: the reference (wolf) metres times `scale`, the rig's leg length
    over REFERENCE_LEG_M."""
    contact: float
    stance: float
    still_speed: float
    scale: float
    leg_m: Optional[float]
    source: str

    def block(self):
        return {"legLengthM": None if self.leg_m is None else round(self.leg_m, 6), "referenceLegM": REFERENCE_LEG_M,
                "scale": round(self.scale, 6), "contactToleranceM": round(self.contact, 6),
                "stanceToleranceM": round(self.stance, 6), "stillSpeedMps": round(self.still_speed, 6),
                "slipShare": SLIP_SHARE, "velocityOrder": VELOCITY_ORDER, "source": self.source}


REFERENCE = Tolerances(CONTACT_TOLERANCE, STANCE_TOLERANCE, STILL_SPEED, 1.0, None,
                       "the reference (wolf) tolerances: no rig with leg chains to scale them")


def leg_length(rig, project):
    """The rig's leg length: the mean rest height (world z) of each leg chain's top joint, the head of its first bone
    (the hips; the withers for a chain that starts at the scapula), from the contract's rig.json. (None, why) when
    the rig names no leg chain's first bone."""
    bones = {b.get("name"): b for b in (rig or {}).get("bones") or [] if isinstance(b, dict)}
    chains = [c for pair in project.get("legPairs", []) for c in pair]
    heights, missing = [], []
    for chain in chains:
        first = (project.get("chains", {}).get(chain) or [None])[0]
        head = bones.get(first, {}).get("head")
        if isinstance(head, (list, tuple)) and len(head) == 3:
            heights.append(float(head[2]))
        else:
            missing.append(f"{chain} ({first})")
    if not chains or missing:
        return None, f"no rest head for {', '.join(missing) or 'any leg chain'}"
    return float(np.mean(heights)), (f"contract rig.json: mean rest head height of {', '.join(chains)}'s first "
                                     f"bones ({', '.join(project['chains'][c][0] for c in chains)})")


def tolerances(rig, project):
    """The contact tolerances scaled to the rig (round 4): CONTACT_TOLERANCE, STANCE_TOLERANCE and STILL_SPEED times
    leg_length / REFERENCE_LEG_M (exactly 1 at the wolf); the reference values when the rig has no leg chains."""
    length, source = leg_length(rig, project)
    if length is None or length <= 0:
        return REFERENCE._replace(source=f"the reference (wolf) tolerances: the rig.json has {source}")
    scale = length / REFERENCE_LEG_M
    return Tolerances(CONTACT_TOLERANCE * scale, STANCE_TOLERANCE * scale, STILL_SPEED * scale, scale, length,
                      f"{source}, over the wolf's {REFERENCE_LEG_M} m")


def period_search(values, fps, min_s=0.25, max_s=2.5):
    """The fundamental period in frames, at sub-frame precision, and its score: {"period": P or None, "score": s}.

    Used for the report on the fitted leg joints (`leg_period`); the period itself comes from the video.

    The lag score (mean squared difference between the clip and itself shifted by the lag) is normalised by twice the
    mean total variance, so 0 is a perfect repeat and about 1 is no relation. A parabola through each whole-frame local
    minimum and its neighbours gives the sub-frame period and the score there; the first local minimum in range whose
    refined score is below PERIOD_MAX_SCORE is the fundamental (its multiples come later). Gating on the refined score,
    not the whole-frame one, keeps a short period that falls between frames from losing to its double. With no such
    minimum the period is None and the score is the lowest refined score in range (how far from periodic it is)."""
    x = values - values.mean(axis=0)
    low, high = max(2, int(np.ceil(min_s * fps))), min(int(max_s * fps), len(x) // 2)
    if high < low + 1:
        raise FactsError(f"The clip ({len(x)} frames) is too short to hold two cycles")
    variance = float(np.mean(np.sum(x ** 2, axis=1)))
    scores = {lag: float(np.mean(np.sum((x[:-lag] - x[lag:]) ** 2, axis=1))) / (2 * variance) if variance else np.inf
              for lag in range(low - 1, high + 2) if 0 < lag < len(x)}
    lowest = np.inf
    for lag in range(low, high + 1):
        before, here, after = scores.get(lag - 1, np.inf), scores[lag], scores.get(lag + 1, np.inf)
        if here <= before and here <= after:
            curve = before - 2 * here + after
            fitted = np.isfinite(curve) and curve > 0
            vertex = here - (before - after) ** 2 / (8 * curve) if fitted else here
            if vertex < PERIOD_MAX_SCORE:
                offset = 0.5 * (before - after) / curve if fitted else 0.0
                return {"period": float(lag + min(0.5, max(-0.5, offset))), "score": float(vertex)}
            lowest = min(lowest, float(vertex))
    if not np.isfinite(lowest):
        lowest = min(scores[lag] for lag in range(low, high + 1))
    return {"period": None, "score": float(lowest)}


def resample(series, times):
    """Linear interpolation in frame time of every column of `series` (T, K) at `times`."""
    return cycle.resample(series, times)


def rest_heights(rig, project):
    """Each paw chain's rest-pose tail height (world z) from the contract's rig.json (`bones[].tail`). Refuses when
    a paw bone or its tail is missing: facts never falls back to absolute heights."""
    bones = {b.get("name"): b for b in (rig or {}).get("bones") or [] if isinstance(b, dict)}
    rest, missing = {}, []
    for chain, bone in project["paws"].items():
        tail = bones.get(bone, {}).get("tail")
        if not isinstance(tail, (list, tuple)) or len(tail) != 3:
            missing.append(f"{chain} ({bone})")
            continue
        rest[chain] = float(tail[2])
    if missing:
        raise FactsError(f"facts measures each paw from its rest-pose tail height, and the contract's rig.json has "
                         f"no rest tail for {', '.join(missing)}; rerun `contract`")
    return rest


def rest_relative(paws, rest, project):
    """The paws' (x, height) with each paw's height measured from its rest-pose tail height."""
    if rest is None:
        raise FactsError("facts measures each paw from its rest-pose tail height (the contract's rig.json), and no "
                         "rest heights were given")
    missing = [f"{c} ({project['paws'][c]})" for c in paws if c not in rest]
    if missing:
        raise FactsError(f"facts measures each paw from its rest-pose tail height, and there is none for "
                         f"{', '.join(missing)}")
    return {c: p - np.array([0.0, float(rest[c])] + [0.0] * (p.shape[1] - 2)) for c, p in paws.items()}


def ground_height(heights):
    """The one ground every paw is measured against: GROUND_PERCENTILE of all paws' heights together."""
    return float(np.percentile(np.concatenate([np.asarray(z, float) for z in heights.values()]), GROUND_PERCENTILE))


def contact_mask(z, ground, tol=CONTACT_TOLERANCE):
    return np.asarray(z, float) <= ground + tol


def runs(mask, circular=False):
    """The True runs of a mask as (start, length); with `circular` a run across the end joins the first."""
    mask = np.asarray(mask, bool)
    n = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, n)]
    first = int(np.argmin(mask)) if circular else 0
    out, i = [], 0
    while i < n:
        if mask[(first + i) % n]:
            j = i
            while j < n and mask[(first + j) % n]:
                j += 1
            out.append(((first + i) % n, j - i))
            i = j
        else:
            i += 1
    return out


def bridge(mask, circular=False, gap=1):
    """Free runs of at most `gap` samples between two contact samples are part of the stance: fit noise lifts a
    planted paw a few millimetres over the tolerance for a sample (the real lope's hind_L, Task 17b fix round 3), and a
    paw cannot lift and plant again that fast. `circular` joins the ends (one loop cycle); a clip's ends are open."""
    mask = np.asarray(mask, bool)
    out = mask.copy()
    n = len(mask)
    if n < 3 or not mask.any():
        return out
    for start, length in runs(~mask, circular):
        bounded = circular or (start > 0 and start + length < n)
        if length <= gap and bounded:
            out[(start + np.arange(length)) % n] = True
    return out


def contact_masks(heights, ground=None, circular=False):
    """Each paw's contact mask against the shared ground (computed from `heights` when not given), one-sample lifts
    bridged."""
    ground = ground_height(heights) if ground is None else ground
    return {c: bridge(contact_mask(z, ground), circular) for c, z in heights.items()}


def stance_masks(heights, vx, ground, speed, circular=False, gap=1, extent=None, tol=None, slip=SLIP_SHARE):
    """Each paw's stance (Task 19): world-still (in-place x speed within `slip` * speed of -speed) and near the
    shared ground: a run of still samples within the stance tolerance (`extent`, a number or one per paw; default
    tol.stance) that reaches the contact tolerance (tol.contact) somewhere. `tol` is the body's Tolerances (round 4;
    default the reference). Without a ground speed (under two planted paws) stillness is not known and every sample
    counts as still."""
    tol = tol or REFERENCE
    extent = tol.stance if extent is None else extent
    out = {}
    for chain, z in heights.items():
        z = np.asarray(z, float)
        v = np.asarray(vx[chain], float)
        still = np.abs(v + speed) <= slip * speed if speed and speed > 0 else np.ones(len(z), bool)
        core = still & (z <= ground + tol.contact)
        reach = still & (z <= ground + (extent[chain] if isinstance(extent, dict) else extent))
        keep = np.zeros(len(z), bool)
        for start, length in runs(reach, circular):
            index = (start + np.arange(length)) % len(z)
            if core[index].any():
                keep[index] = True
        out[chain] = bridge(keep, circular, gap)
    return out


def speed_from(vx, masks):
    """(speed, reason): minus the median in-place x speed of the planted paws while planted, or (None, why) when
    fewer than two paws plant (one paw's stance is too little to trust for the clip's speed)."""
    planted = [c for c in vx if masks[c].any()]
    if len(planted) < 2:
        names = ", ".join(planted) or "none"
        return None, (f"{len(planted)} paw{'' if len(planted) == 1 else 's'} plant{'s' if len(planted) == 1 else ''} "
                      f"({names}); ground speed needs at least two planted paws")
    return float(-np.median(np.concatenate([np.asarray(vx[c], float)[masks[c]] for c in planted]))), None


def motion_contacts(heights, vx, circular=False, gap=1, tol=None, still=None, slip=SLIP_SHARE, extent=None):
    """Contacts from the motion: the shared ground, then a first ground speed from the paws near it while they move
    back (in place a planted paw slides back, a swing passes forward: round 4, the swing never enters the speed), then
    each paw's stance (`stance_masks`), the speed measured again from those stances only and the stances read with
    it. `vx` is the in-place x speed the ground speed is measured from; `still` (default `vx`) the smoothed one the
    stillness test reads (round 4). With no stance in two paws there is no speed. Returns (ground, speed, speed
    reason, masks)."""
    tol = tol or REFERENCE
    still = vx if still is None else still
    ground = ground_height(heights)
    near = {c: bridge(contact_mask(z, ground, tol.contact), circular, gap) & (np.asarray(still[c], float) < 0)
            for c, z in heights.items()}
    speed, reason = speed_from(vx, near)
    masks = stance_masks(heights, still, ground, speed, circular, gap, extent, tol, slip)
    if speed is not None:
        again, why = speed_from(vx, masks)
        if again is not None and again > 0:
            speed, reason = again, why
            masks = stance_masks(heights, still, ground, speed, circular, gap, extent, tol, slip)
        else:
            speed, reason = None, why or "the paws' stances do not move back: no ground speed"
    return ground, speed, reason, masks


def lowpass(x, order):
    """One periodic cycle of samples with only its first `order` harmonics (round 4: a fit's per-key jitter lives in
    the higher ones)."""
    coeffs = np.fft.rfft(np.asarray(x, float))
    coeffs[np.arange(len(coeffs)) > order] = 0.0
    return np.fft.irfft(coeffs, len(x))


def cycle_touchdowns(masks):
    """Per paw, how many times it touches down in one cycle of samples (a run across the seam is one)."""
    return {c: len(runs(m, circular=True)) if np.asarray(m).any() and not np.asarray(m).all() else int(np.any(m))
            for c, m in masks.items()}


def slide_stances(heights, x, ground, tol):
    """The noisy-fit stance of each paw (round 4): its backward slide, from the most forward point of its in-place x
    (smoothed: `x`) to the most backward (a treadmill gait's anterior and posterior extreme positions), when the
    paw comes within the stance tolerance of the shared ground in it; else no stance. Returns (masks, extremes)."""
    masks, extremes = {}, {}
    for chain, z in heights.items():
        xs, z = np.asarray(x[chain], float), np.asarray(z, float)
        n = len(xs)
        front, back = int(np.argmax(xs)), int(np.argmin(xs))
        mask = np.zeros(n, bool)
        mask[(front + np.arange((back - front) % n)) % n] = True
        if not mask.any() or float(np.min(z[mask])) > ground + tol.stance:
            mask[:] = False
        masks[chain] = mask
        extremes[chain] = [round(front / n, 4), round(back / n, 4)]
    return masks, extremes


def read_cycle(heights, vx, still, smooth_x, gap, tol, reading=None):
    """One cycle's contacts (round 4). The strict reading is `motion_contacts` with the body's tolerances and the
    smoothed stillness velocity `still`. When it gives a planted paw more than one touchdown and every one of them
    lies inside that paw's backward slide, they are fragments of one stance that the fit's height and speed noise
    broke up, and the noisy-fit reading is taken: each stance is the paw's backward slide (`slide_stances` on
    `smooth_x()`, the paw x capped at NOISY['order'] harmonics), the ground speed measured from those stances.
    Otherwise the strict reading stands (and the caller refuses a paw that touches down twice). `reading` forces one
    ("strict" or "noisy fit"). Returns (ground, speed, reason, masks, report)."""
    ground, speed, reason, masks = motion_contacts(heights, vx, True, gap, tol, still)
    counts = cycle_touchdowns(masks)
    report = {"reading": "strict", "touchdowns": counts}
    if reading == "strict" or (reading is None and all(v <= 1 for v in counts.values())):
        return ground, speed, reason, masks, report
    slides, extremes = slide_stances(heights, smooth_x(), ground, tol)
    outside = {c: int(np.sum(masks[c] & ~slides[c])) for c in masks if counts[c] > 1}
    if reading is None and any(outside.values()):
        return ground, speed, reason, masks, {**report, "slideCheck": {"samplesOutsideSlide": outside,
                                                                        "extremes": extremes}}
    speed, why = speed_from(vx, slides)
    report = {"reading": "noisy fit", "touchdowns": cycle_touchdowns(slides), "strictTouchdowns": counts,
              "order": NOISY["order"], "extremes": extremes,
              "rule": "the strict reading broke a stance into several touchdowns, all inside the paw's backward "
                      f"slide: each stance is the slide from the paw's most forward to its most backward point "
                      f"(its in-place x capped at {NOISY['order']} harmonics), planted when it comes within the "
                      f"stance tolerance of the ground; the ground speed from those stances"}
    return ground, speed, why, slides, report


def reading_warning(report):
    if report["reading"] != "noisy fit":
        return []
    strict = ", ".join(f"{c} {v}" for c, v in report["strictTouchdowns"].items() if v != 1)
    return [f"Contacts read for a noisy fit (round 4): the strict reading touched down {strict} times in a cycle, "
            f"every touchdown inside the paw's backward slide; each stance is that slide (anterior to posterior "
            f"extreme of the paw's x at {report['order']} harmonics)"]


def circular_speed(x, dt):
    """The x speed of one periodic cycle of samples (central differences across the seam)."""
    x = np.asarray(x, float)
    return (np.roll(x, -1) - np.roll(x, 1)) / (2 * dt)


# ------------------------------------------------------------------------------------------- the gait gate
def expected_gait(text):
    """The gait the brief's `gait` text names: the earliest whole word of GAIT_WORDS in it (its inflections too:
    loping, pacing, trotting, cantering, walking, galloping), or None."""
    text = (text or "").lower()
    found = []
    for pattern, gait in GAIT_WORDS:
        match = re.search(r"\b(?:" + pattern + r")\b", text)
        if match:
            found.append((match.start(), match.start() - match.end(), gait))
    return min(found)[2] if found else None


def _kind(chain):
    return "front" if chain.startswith("front") else "hind" if chain.startswith("hind") else chain


def _side(chain):
    return chain.rsplit("_", 1)[-1]


def gait_reads(gait):
    """The vocabulary's gaits a measured pattern (`gait_pattern` plus plantedPaws) reads as: walk (four single beats,
    each hind followed by its own side's fore, no suspension), trot (two diagonal-pair beats), pace (two lateral-pair
    beats), canter (three beats, one a diagonal pair, a fore lead, one suspension), gallop (four single beats, the
    hinds together and the fores together, a suspension) and transverse or rotary gallop (its fore lead on the hind
    lead's side or the other)."""
    beats, suspensions, lead = gait["beatPaws"], len(gait["suspensions"]), gait["lead"]
    if len(gait.get("plantedPaws", [])) < 4:
        return []
    pair = [b for b in beats if len(b) == 2 and _kind(b[0]) != _kind(b[1])]
    diagonal = [b for b in pair if _side(b[0]) != _side(b[1])]
    lateral = [b for b in pair if _side(b[0]) == _side(b[1])]
    reads = []
    if len(beats) == 4 and all(len(b) == 1 for b in beats):
        order = [b[0] for b in beats]
        hinds = [i for i, c in enumerate(order) if _kind(c) == "hind"]
        if len(hinds) == 2 and (hinds[1] - hinds[0]) % 4 in (1, 3):
            if suspensions >= 1:
                reads.append("gallop")
                front, hind = lead.get("front"), lead.get("hind")
                if front and hind:
                    reads.append("transverse gallop" if _side(front) == _side(hind) else "rotary gallop")
        elif len(hinds) == 2 and suspensions == 0 and all(
                _kind(order[(i + 1) % 4]) == "front" and _side(order[(i + 1) % 4]) == _side(order[i]) for i in hinds):
            reads.append("walk")
    if len(beats) == 2 and len(diagonal) == 2:
        reads.append("trot")
    if len(beats) == 2 and len(lateral) == 2:
        reads.append("pace")
    if (len(beats) == 3 and sorted(len(b) for b in beats) == [1, 1, 2] and diagonal and lead.get("front")
            and suspensions == 1):
        reads.append("canter")
    return reads


def describe_gait(gait):
    beats = " | ".join("+".join(b) for b in gait["beatPaws"]) or "no planted paw"
    lead = gait["lead"]
    return (f"{gait['beats']} beats ({beats}), lead front {lead.get('front')}, hind {lead.get('hind')}, "
            f"{len(gait['suspensions'])} suspension{'' if len(gait['suspensions']) == 1 else 's'}")


def gait_check(gait, brief, accept_gait=None):
    """The measured pattern against the brief's gait. A mismatch, or a brief gait that names nothing in the
    vocabulary (fix round 1), refuses unless `accept_gait` gives the reason. Returns (check, warnings)."""
    text = brief.get("gait")
    expected = expected_gait(text)
    reads = gait_reads(gait)
    out = {"expected": expected, "checked": expected is not None,
           "measured": {"text": describe_gait(gait), "reads": reads},
           "matches": None if expected is None else expected in reads, "accepted": None}
    warnings = []
    named = f"the brief's gait {text!r}" if text else "the brief"
    if expected is None:
        if not accept_gait:
            raise FactsError(f"{named[0].upper()}{named[1:]} names no gait facts can check ({VOCABULARY}); the motion's "
                             f"footfalls are {describe_gait(gait)}, reading as {', '.join(reads) or 'none of them'}. "
                             f"Name the gait in the brief, or record why the clip stands unchecked with "
                             f"`facts --accept-gait <reason>`")
        out["accepted"] = accept_gait
        warnings.append(f"Gait not checked: {named} names none of {VOCABULARY}; accepted: {accept_gait}")
    elif expected not in reads:
        if not accept_gait:
            raise FactsError(f"The brief's gait is a {expected}, but the motion's footfalls are {describe_gait(gait)}: "
                             f"they read as {', '.join(reads) or 'none of walk, trot, pace, canter/lope, gallop'}. "
                             f"Regenerate the video, or record why this motion stands with "
                             f"`facts --accept-gait <reason>`")
        out["accepted"] = accept_gait
        warnings.append(f"Gait mismatch accepted: the brief's {expected}, measured {describe_gait(gait)}; "
                        f"accepted: {accept_gait}")
    return out, warnings


# ------------------------------------------------------------------------------------------------ the cadence
def mass_range(kg):
    """[lightest, heaviest] kg: a single mass m is [m (1 - MASS_SPREAD), m (1 + MASS_SPREAD)]."""
    if isinstance(kg, (list, tuple)):
        return [float(kg[0]), float(kg[1])]
    return [round(float(kg) * (1 - MASS_SPREAD), 6), round(float(kg) * (1 + MASS_SPREAD), 6)]


def stride_hz(mass_kg, gait):
    """The stride frequency (Hz) of the gait's law (CADENCE_LAWS) at a body mass."""
    if gait not in GAIT_LAW:
        raise FactsError(f"No cadence band for a {gait!r}: the bands are for {VOCABULARY}")
    a, b, _ = CADENCE_LAWS[GAIT_LAW[gait]]
    return a * float(mass_kg) ** b


def cadence_band(kg, gait):
    """The gait's stride-frequency band (Hz) over a mass range: [f(heaviest), f(lightest)]."""
    low, high = mass_range(kg)
    return [stride_hz(high, gait), stride_hz(low, gait)]


def cadence_check(period, fps, gait_name, mass, periods, accept_retime=None, slow_motion=None, stride=None):
    """The stride frequency against the gait's body-size band (Task 20: every gait in GAIT_LAW): the block for
    facts.json and the warnings. A video under the band whose steady cycles share one period (each within
    CADENCE_CONSISTENCY of the clip's) is retimed to the band centre; the factor scales the clip's speed.
    Fix round 1: a mass from an open mesh's volume (`openEdges` > 0) is checked against but never retimes; a factor
    outside RETIME_BOUND refuses unless `accept_retime` records why. Task 20: `slow_motion` (the brief's slowMotion)
    is the expected retime: recorded beside the measured one, and the bound is on measured / expected. Vole round 3:
    `stride` is the leg evidence for the measured stride (the masks' period, whether it was doubled, leg identity
    resolved, touchdowns per paw), recorded as `strideEvidence` and named in a refusal; the bound is never widened."""
    hz = fps / period
    expected = float(slow_motion) if slow_motion else 1.0
    law = GAIT_LAW.get(gait_name)
    out = {"videoHz": round(hz, 4), "gait": gait_name, "massKg": None, "massSource": None, "bandHz": None,
           "centreHz": None, "checked": False, "consistent": None, "cyclePeriods": [round(p, 3) for p in periods],
           "retime": 1.0, "expectedRetime": slow_motion if slow_motion else 1.0, "hz": round(hz, 4),
           "law": CADENCE_LAWS[law][2] if law else None}
    if stride is not None:
        out["strideEvidence"] = stride
    warnings = []
    asked = f"slowMotion {slow_motion:g}" if slow_motion else None
    open_edges = (mass or {}).get("openEdges") or 0
    if mass is not None:
        out["massKg"], out["massSource"] = mass_range(mass["kg"]), mass["source"]
        if open_edges:
            out["massOpenEdges"] = int(open_edges)
    if law is None:
        out["reason"] = f"no cadence band for a {gait_name or 'gait the brief does not name'}"
        return out, warnings
    if mass is None:
        out["reason"] = "no body mass"
        warnings.append("Cadence not checked: no body mass (the brief's massKg or `facts --mass-kg`)")
        return out, warnings
    band = cadence_band(mass["kg"], gait_name)
    centre = sum(band) / 2
    consistent = bool(periods) and all(abs(p - period) / period <= CADENCE_CONSISTENCY for p in periods)
    out.update({"bandHz": [round(b, 4) for b in band], "centreHz": round(centre, 4), "checked": True,
                "consistent": consistent})
    kg = "-".join(f"{m:g}" for m in out["massKg"])
    where = f"{band[0]:.2f}-{band[1]:.2f} Hz {gait_name} band for {kg} kg"
    untrusted = (f"the mass is the {mass['source']} of an open mesh ({int(open_edges)} open edges); give the brief's "
                 f"massKg or `facts --mass-kg`") if open_edges else None
    if untrusted:
        warnings.append(f"Body mass {kg} kg is the {mass['source']} of an open mesh ({int(open_edges)} open edges): "
                        f"the cadence is checked against it and not retimed")
    if hz < band[0] and consistent and untrusted:
        out["retimeBlocked"] = untrusted
        warnings.append(f"The video's cadence {hz:.2f} Hz is under the {where} and not retimed: {untrusted}")
    elif hz < band[0] and consistent:
        factor = centre / hz
        low, high = RETIME_BOUND
        ratio = factor / expected
        accepted = ""
        if slow_motion:
            out["retimeVsExpected"] = round(ratio, 6)
        if not low <= ratio <= high:
            if not accept_retime:
                against = (f" against the brief's {asked} (the measured over the expected retime is {ratio:.2f})"
                           if slow_motion else "")
                far = "from the slow motion asked for" if slow_motion else "slower than the body size says"
                evidence = ""
                if stride:
                    half = (f": the masks repeat every {stride['maskPeriodFrames']:.2f} frames, half of it (each leg "
                            f"matches its partner there, not itself)") if stride.get("doubled") else ""
                    resolved = "; leg identity resolved across the clip" if stride.get("identityResolved") else ""
                    evidence = (f" The leg evidence gives a stride of {stride['strideFrames']:.1f} frames{half}, and "
                                f"each paw touches down once in it{resolved}.")
                raise FactsError(f"The cadence retime {factor:.2f} (the video's {hz:.2f} Hz to {centre:.2f} Hz, the "
                                 f"{where}){against} is outside {low}-{high}: the video is far {far} or the mass is "
                                 f"wrong.{evidence} Check the mass (the brief's massKg or `facts --mass-kg`), or "
                                 f"record why this retime stands with `facts --accept-retime <reason>`")
            out["retimeAccepted"] = accept_retime
            accepted = f"; outside {low}-{high}, accepted: {accept_retime}"
        out["retime"], out["hz"] = round(factor, 6), round(centre, 4)
        expected_note = f" (the brief's {asked} expected {expected:g})" if slow_motion else ""
        warnings.append(f"The video's cadence {hz:.2f} Hz is under the {where}: retimed by {factor:.3f}"
                        f"{expected_note} to {centre:.2f} Hz (the speed by the same factor){accepted}")
    elif hz < band[0]:
        warnings.append(f"The video's cadence {hz:.2f} Hz is under the {where} and not retimed: the steady cycles' "
                        f"own periods ({out['cyclePeriods']}) are not one time scale")
    elif hz > band[1]:
        warnings.append(f"The video's cadence {hz:.2f} Hz is above the {where}; not retimed (only a slow video is)")
    if slow_motion and hz >= band[0]:
        warnings.append(f"The brief's {asked} expected a retime of {expected:g}, but the video's cadence {hz:.2f} Hz "
                        f"is not under the {where}: not retimed (the measured cadence decides)")
    return out, warnings


def paw_report(heights, masks, ground, project, rest):
    """Per paw: whether it plants, its lowest rest-relative height over the shared ground, its rest tail height."""
    return {c: {"bone": project["paws"][c], "plants": bool(masks[c].any()),
                "lowestAboveGround": round(float(np.min(heights[c]) - ground), 4),
                "restZ": round(float(rest[c]), 4)} for c in heights}


def contact_window(mask):
    length = len(mask)
    if mask.all():
        return [0.0, 1.0], 1.0
    starts = [i for i in range(length) if mask[i] and not mask[i - 1]]
    if len(starts) != 1:
        raise TouchdownError(f"A paw touches down {len(starts)} times in one cycle; check leg identity and marks")
    end = starts[0]
    while mask[end % length]:
        end += 1
    return [round(starts[0] / length, 4), round((end % length) / length, 4)], (end - starts[0]) / length


def windows(mask):
    """Contact runs of a one-shot as [start, end] fractions of its duration: frame i sits at i / (len - 1), the same
    denominator as the duration, and a run ends at its first free frame (or at the last frame)."""
    last = max(1, len(mask) - 1)
    runs, start = [], None
    for i, on in enumerate(list(mask) + [False]):
        if on and start is None:
            start = i
        if not on and start is not None:
            runs.append([round(start / last, 4), round(min(i, last) / last, 4)])
            start = None
    return runs


def gait_pattern(masks, touchdowns, project, n):
    """Beats, lead and suspension of one cycle from the planted paws only. `masks` are the n-sample contact masks,
    `touchdowns` each planted paw's touchdown phase. Touchdowns within the beat tolerance are one beat; a leg pair's
    lead is the paw that lands second, less than half a cycle after its partner (none when either paw never plants,
    when they land together or half a cycle apart); a suspension is a run of samples with no planted paw down."""
    tol = max(BEAT_TOLERANCE, 1.0 / n)
    order = sorted(touchdowns, key=lambda c: touchdowns[c])
    groups = []
    for c in order:
        if groups and touchdowns[c] - touchdowns[groups[-1][-1]] <= tol:
            groups[-1].append(c)
        else:
            groups.append([c])
    if len(groups) > 1 and touchdowns[groups[0][0]] + 1.0 - touchdowns[groups[-1][-1]] <= tol:
        groups[0] = groups.pop() + groups[0]
    lead = {}
    for a, b in project.get("legPairs", []):
        side = "front" if a.startswith("front") else "hind" if a.startswith("hind") else f"{a}/{b}"
        if a not in touchdowns or b not in touchdowns:
            lead[side] = None
            continue
        lag = (touchdowns[b] - touchdowns[a]) % 1.0
        together = lag <= tol or lag >= 1.0 - tol or abs(lag - 0.5) <= tol
        lead[side] = None if together else (b if lag < 0.5 else a)
    down = np.zeros(n, bool)
    for c in touchdowns:
        down |= masks[c]
    suspensions = []
    if touchdowns and not down.all():
        first = int(np.argmax(down))
        i = first
        while i < first + n:
            if not down[i % n]:
                begin = i
                while not down[i % n]:
                    i += 1
                end = i % n if i % n else n
                suspensions.append([round((begin % n) / n, 4), round(end / n, 4)])
            i += 1
    return {"beats": len(groups), "beatPhases": [touchdowns[g[0]] for g in groups], "beatPaws": groups,
            "lead": lead, "suspensions": suspensions, "beatTolerance": round(tol, 4)}


def clip_facts(poses, project, brief, rest, masks=None, render=None, rig=None, mass=None, accept_gait=None,
               accept_retime=None):
    """The clip's facts. `rest` is each paw chain's rest-pose tail height (`rest_heights`); the paws' heights are
    measured from it. A loop needs the video: `masks` (the ingest's working masks, (T, H, W) bool) and `render`
    (value vectors -> the rig's silhouettes at the masks' size) for the averaged cycle's check.

    Task 19: `rig` (the contract's rig.json) reads a loop's contacts from the forward kinematics of its smooth cycle
    and makes the bake's 60 Hz keys (the pipeline always passes it; without it the averaged cycle's paw columns are
    read and its samples are the keys, as before). `mass` ({"kg": m or [lo, hi], "source"}; else the brief's massKg)
    sets the cadence band; `accept_gait` is the recorded reason a gait mismatch (or an unnamed gait) stands, and
    `accept_retime` the reason a retime outside RETIME_BOUND does. With `rig`, a one-shot's facts predict each paw at
    every key as the bake plays it (`cycle.paws`), which `bake` checks."""
    fps = float(poses["fps"])
    frames = poses["frames"]
    every = np.array([f["values"] for f in frames], float)
    chains = list(project["paws"])
    paws = rest_relative({c: np.array([f["paws"][c] for f in frames], float) for c in chains}, rest, project)
    warnings = []
    if mass is None and brief.get("massKg") is not None:
        mass = {"kg": brief["massKg"], "source": "brief massKg"}
    if brief["loop"]:
        found = loop_cycle(poses, project, every, paws, chains, masks, render, rest, rig, brief, mass, accept_gait,
                           accept_retime)
        start, period, entry, cycle_out, extra = found
        speed, speed_reason = entry["authored_speed_mps"], extra.pop("speedReason", None)
        warnings += extra.pop("warnings", [])
    else:
        start, period = 0, len(frames) - 1
        heights = {c: paws[c][:, 1] for c in chains}
        vx = {c: np.gradient(paws[c][:, 0]) * fps for c in chains}
        tol = REFERENCE if rig is None else tolerances(rig, project)
        ground, speed, speed_reason, masks = motion_contacts(heights, vx, tol=tol)
        all_windows = {project["paws"][c]: windows(masks[c]) for c in chains}
        entry = {"duration_s": period / fps, "loop": False,
                 "authored_speed_mps": None if speed is None else round(speed, 3),
                 "stride_m": 0.0, "contacts": {b: w[0] for b, w in all_windows.items() if w},
                 "frames": period, "fps": fps, "contact_windows": all_windows}
        cycle_out = {"frames": every.tolist(), "fps": fps, "periodFrames": period, "start": start,
                     "drift": {"dx": 0.0, "dz": 0.0}}
        if rig is not None:
            # As bake_clip keys a one-shot: dx less its mean over keys 0..n-1, every other value as fitted.
            played = every.copy()
            played[:, poses["names"].index("dx")] -= float(played[:period, poses["names"].index("dx")].mean())
            predicted = motion.PlanarRig(rig).paws(played, poses["names"], project)
            cycle_out["paws"] = {project["paws"][c]: np.round(p, 7).tolist() for c, p in predicted.items()}
        extra = {"ground": ground_block(ground, "every frame, all paws", tol),
                 "paws": paw_report(heights, masks, ground, project, rest), "bodyScale": tol.block()}
    target = brief.get("targetSpeedMps")
    if speed is not None and target and abs(speed - target) / target > 0.25:
        warnings.append(f"Measured ground speed {speed:.2f} m/s is more than 25% from the brief's {target} m/s")
    out = {"start": start, "period": period, "fps": fps, "speedMps": speed, "strideM": entry["stride_m"],
           "entry": entry, "warnings": warnings, "cycle": cycle_out}
    if speed is None:
        out["speedReason"] = speed_reason
    out.update(extra)
    return out


def ground_block(ground, source, tol=None):
    tol = tol or REFERENCE
    return {"height": round(ground, 4), "tolerance": round(tol.contact, 6), "percentile": GROUND_PERCENTILE,
            "source": source}


def leg_period(values, names, project, fps, period):
    """The secondary report on the fitted angles: over the well-observed leg joints (`cycle.leg_joints`), the
    normalised lag score at the video's period (interpolated between whole frames; lower is better, `passes` when it is
    under the gate) and the angle-only period search."""
    joints = cycle.leg_joints(names, project)
    if not joints:
        return {"joints": [], "scoreAtPeriod": None, "search": None, "gate": PERIOD_MAX_SCORE, "passes": False}
    x = values[:, [names.index(j) for j in joints]]
    x = x - x.mean(axis=0)
    variance = float(np.mean(np.sum(x ** 2, axis=1))) or 1.0
    times = np.arange(0, len(x) - 1 - period)
    later = resample(x, times + period)
    score = float(np.mean(np.sum((x[times.astype(int)] - later) ** 2, axis=1))) / (2 * variance) if len(times) else None
    try:
        search = period_search(x, fps)
    except FactsError as error:
        search = {"period": None, "score": None, "error": str(error)}
    # Lower is better: 0 is a perfect repeat at the video's period; `passes` is the score under the gate.
    return {"joints": joints, "scoreAtPeriod": None if score is None else round(score, 4),
            "passes": score is not None and score < PERIOD_MAX_SCORE,
            "search": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.items()},
            "gate": PERIOD_MAX_SCORE}


def slow_motion_advice(brief, mass=None, fps=24.0):
    """What to do about a stride the video undersamples (Task 20): regenerate in slow motion. With the brief's gait
    and a body mass, name the band's centre and the smallest factor that gives a stride MIN_STRIDE_FRAMES frames."""
    brief = brief or {}
    asked = brief.get("slowMotion")
    gait = expected_gait(brief.get("gait"))
    hint = ""
    if gait in GAIT_LAW and mass is not None:
        centre = sum(cadence_band(mass["kg"], gait)) / 2
        hint = (f" (the {gait} band for {'-'.join(f'{m:g}' for m in mass_range(mass['kg']))} kg centres on "
                f"{centre:.2f} Hz, {fps / centre:.1f} frames at {fps:g} fps: a slowMotion of at least "
                f"{MIN_STRIDE_FRAMES * centre / fps:.1f})")
    if asked:
        return (f"The brief's slowMotion {asked:g} was not slow enough: raise it so a stride spans at least "
                f"{MIN_STRIDE_FRAMES} frames{hint}, and regenerate")
    return (f"Regenerate with the brief's slowMotion set so a stride spans at least {MIN_STRIDE_FRAMES} frames{hint}")


# ------------------------------------------------------------------------------------------------- leg identity
def named_pairs(names, project):
    """The project's leg pairs with their value columns: [(chain a, chain b, columns a, columns b)]."""
    out = []
    for a, b in project.get("legPairs", []):
        ia = [names.index(x) for x in project["chains"].get(a, []) if x in names]
        ib = [names.index(x) for x in project["chains"].get(b, []) if x in names]
        if ia and len(ia) == len(ib):
            out.append((a, b, ia, ib))
    return out


def orient_pairs(x, pairs, origin, gait):
    """A side silhouette cannot tell a pair's near leg from its far one, so after resolving identity each other pair
    is labelled the way the brief's two-beat gait names: for a trot the phase origin moves with its diagonal partner,
    for a pace with its lateral one. x: chain -> in-place paw x. Returns the "a/b" keys of the pairs to exchange
    over the whole clip (none for any other gait, or when the origin's partner already moves with it)."""
    if gait not in ("trot", "pace") or origin not in x:
        return []
    home = np.gradient(np.asarray(x[origin], float))
    out = []
    for a, b in pairs:
        if origin in (a, b) or a not in x or b not in x:
            continue
        same = a if _side(a) == _side(origin) else b if _side(b) == _side(origin) else None
        if same is None:
            continue
        other = b if same == a else a
        partner, rival = (same, other) if gait == "pace" else (other, same)

        def corr(chain):
            v = np.gradient(np.asarray(x[chain], float))
            return float(np.corrcoef(home, v)[0, 1]) if np.std(v) > 0 and np.std(home) > 0 else 0.0
        if corr(rival) > corr(partner):
            out.append(f"{a}/{b}")
    return out


def exchange(paws, a, b, flip):
    """The paw tracks with chains a and b exchanged on the flipped frames."""
    out = dict(paws)
    pa, pb = np.array(paws[a], float), np.array(paws[b], float)
    out[a], out[b] = np.where(flip[:, None], pb, pa), np.where(flip[:, None], pa, pb)
    return out


def leg_identity(every, paws, names, project, period, brief, force=False):
    """Vole round 3: leg identity across the clip before the stride is decided. Each pair's half-period test
    (`cycle.pair_verdicts` on its paw tracks, the leg evidence) says whether the video's period is half the stride
    (the pair matches its partner a period on) or the whole of it. When every pair says the same, nothing changes (the
    clean wolf lope). When a pair's test is mixed or the pairs disagree (or `force`: a paw touched down twice in one
    cycle), each pair is relabelled where its paws meet (`cycle.resolve_identity`, the paw tracks' continuity), the
    other pairs oriented to the brief's trot or pace (`orient_pairs`), and the tests read again: a relabelling that
    leaves every pair saying the same is used; otherwise the clip stays as fitted, decided by the summed test
    (`cycle.half_period`).

    Returns (values, paws, report, doubled)."""
    named = named_pairs(names, project)
    cols = [(ia, ib) for _, _, ia, ib in named]
    keys = [f"{a}/{b}" for a, b, _, _ in named]
    on_paws = all(a in paws and b in paws for a, b, _, _ in named)

    def verdicts_of(values, tracks):
        """Each pair's half-period test on its paw tracks (x, height: the leg evidence), or on its value columns
        when a pair has no paws."""
        if not on_paws:
            return cycle.pair_verdicts(values, cols, period)
        table = np.concatenate([np.asarray(tracks[c], float)[:, :2] for a, b, _, _ in named for c in (a, b)], axis=1)
        return cycle.pair_verdicts(table, [([4 * i, 4 * i + 1], [4 * i + 2, 4 * i + 3]) for i in range(len(named))],
                                   period)
    before = verdicts_of(every, paws)
    report = {"resolved": False, "before": {k: v["verdict"] for k, v in zip(keys, before)}, "after": None,
              "pairs": {}, "oriented": [], "trigger": None, "testedOn": "paw tracks" if on_paws else "leg values"}
    legacy = bool(cycle.half_period(every, cycle.leg_pairs(names, project), period))
    verdicts = {v["verdict"] for v in before}
    if not named or (not force and "mixed" not in verdicts and len(verdicts) == 1):
        return every, paws, report, legacy
    report["trigger"] = ("a paw touched down more than once in one cycle" if force else
                         "the leg pairs' half-period tests are mixed or disagree")
    values, tracks, flips = np.array(every, float), dict(paws), []
    for (a, b, _, _), key in zip(named, keys):
        if a in paws and b in paws:
            found = cycle.resolve_identity(paws[a][:, 0], paws[b][:, 0])
        else:
            found = {"flip": np.zeros(len(every), bool), "boundaries": [], "meetings": []}
        flips.append(found["flip"])
        report["pairs"][key] = {"boundaries": found["boundaries"],
                                "meetings": [{"frame": m["frame"], "swap": m["swap"],
                                              "exchangedOverKept": round(m["exchanged"] / max(m["kept"], 1e-12), 4)}
                                             for m in found["meetings"]]}
        if a in paws and b in paws:
            tracks = exchange(tracks, a, b, found["flip"])
    values = cycle.relabel(values, cols, flips)
    gait = expected_gait(brief.get("gait"))
    oriented = orient_pairs({c: tracks[c][:, 0] for c in tracks}, [[a, b] for a, b, _, _ in named],
                            project.get("phaseOrigin"), gait)
    for (a, b, ia, ib), key in zip(named, keys):
        if key in oriented:
            whole = np.ones(len(every), bool)
            values = cycle.relabel(values, [(ia, ib)], [whole])
            tracks = exchange(tracks, a, b, whole)
    report["oriented"] = [f"{k} (labelled for the brief's {gait}: a side view cannot tell near from far)"
                          for k in oriented]
    after = verdicts_of(values, tracks)
    report["after"] = {k: v["verdict"] for k, v in zip(keys, after)}
    changed = any(f.any() for f in flips) or bool(oriented)
    agreed = {v["verdict"] for v in after}
    if changed and "mixed" not in agreed and len(agreed) == 1:
        report["resolved"] = True
        return values, tracks, report, agreed == {"half"}
    report["reason"] = ("no pair's paws meet where exchanging them keeps their velocities" if not changed else
                        f"relabelled, the pairs' half-period tests read {report['after']}: not one stride")
    return every, paws, report, legacy


def describe_identity(report):
    if not report:
        return "not tried"
    parts = []
    for key, pair in (report.get("pairs") or {}).items():
        swaps = pair["boundaries"]
        parts.append(f"{key} {report['before'].get(key)} -> {(report.get('after') or {}).get(key)}"
                     f"{', exchanged from frames ' + ', '.join(map(str, swaps)) if swaps else ', kept'}")
    text = "; ".join(parts) or "no leg pairs"
    if report.get("reason"):
        text += f"; {report['reason']}"
    return text


def loop_cycle(poses, project, every, paws, chains, masks, render, rest, rig=None, brief=None, mass=None,
               accept_gait=None, accept_retime=None):
    """The loop's cycle from the video: period, steady cycles, phase average, the averaged cycle's check against the
    masks, the contacts, the gait and cadence checks, and (with `rig`) the 60 Hz keys. Returns (start, period, entry,
    cycle block, extra facts).

    Vole round 3: when a paw touches down more than once in the cycle, leg identity is resolved across the clip
    (`leg_identity` with force) and the cycle measured again; facts refuses only when that leaves none consistent."""
    if masks is None or render is None:
        raise FactsError("facts finds a loop's period in the video: it needs the ingest masks and the rig's renderer")
    brief = brief or {}
    fps = float(poses["fps"])
    found = cycle.mask_period(masks, fps)
    if found["period"] is None:
        best = found.get("fundamentalLag")
        short = (f"; the best candidate, {best:.2f} frames, is under {MIN_STRIDE_FRAMES} frames: the steps may be "
                 f"too fast for {fps:g} fps. {slow_motion_advice(brief, mass, fps)}") if best is not None and \
            best < MIN_STRIDE_FRAMES else ""
        raise FactsError(f"no full cycle found in the video between 0.25 s and 2.5 s: {found['reason']}; the clip "
                         f"must hold at least two cycles{short}")
    try:
        return loop_from_period(found, poses, project, every, paws, chains, masks, render, rest, rig, brief, mass,
                                accept_gait, accept_retime, force=False)
    except TouchdownError as error:
        first = getattr(error, "identity", None) or {}
        if not first.get("resolved"):
            try:
                return loop_from_period(found, poses, project, every, paws, chains, masks, render, rest, rig, brief,
                                        mass, accept_gait, accept_retime, force=True)
            except TouchdownError as again:
                error, first = again, getattr(again, "identity", None) or {}
        raise FactsError(f"{error}: resolving leg identity across the clip from the paw tracks "
                         f"({describe_identity(first)}) leaves no consistent leg identity assignment with one "
                         f"touchdown per paw per stride. Mark the legs and refit the span (gridframe marks, "
                         f"`fit --marks --span`), or regenerate") from error


def loop_from_period(found, poses, project, every, paws, chains, masks, render, rest, rig, brief, mass, accept_gait,
                     accept_retime, force=False):
    """`loop_cycle` from the video's period on: leg identity (`leg_identity`; with `force` resolved even when the
    half-period test is decisive), the stride, the steady cycles, the average and the contacts."""
    fps = float(poses["fps"])
    names = poses["names"]
    count = len(every)
    period = found["period"]
    every, paws, identity, doubled = leg_identity(every, paws, names, project, period, brief, force)
    pairs = cycle.leg_pairs(names, project)
    try:
        return loop_rest(found, period, doubled, identity, poses, project, every, paws, chains, masks, render, rest,
                         rig, brief, mass, accept_gait, accept_retime, fps, names, count, pairs)
    except TouchdownError as error:
        error.identity = identity
        raise


def loop_rest(found, period, doubled, identity, poses, project, every, paws, chains, masks, render, rest, rig, brief,
              mass, accept_gait, accept_retime, fps, names, count, pairs):
    """The stride (the video's period, doubled when the masks show half of it), its steady cycles, their average, the
    contacts, gait, cadence and keys, from leg-identity-resolved values and paws."""
    mask_frames = period
    if doubled:
        period *= 2
    if period < MIN_STRIDE_FRAMES:
        raise FactsError(f"The video's stride of {period:.2f} frames ({fps / period:.2f} Hz at {fps:g} fps) is under "
                         f"{MIN_STRIDE_FRAMES} frames: the video undersamples the steps. "
                         f"{slow_motion_advice(brief, mass, fps)}")
    found = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in found.items()}, "doubled": doubled}
    n = max(2, int(round(period)))
    first = int(np.ceil(LEAD_IN * count))
    steady = cycle.steady_cycles(masks, period, first)
    if not steady["kept"]:
        raise FactsError(f"No full cycle of {period:.1f} frames fits after the lead-in (the first {first} frames); "
                         f"generate a longer video")
    kept = steady["kept"]
    ref = kept.index(steady["ref"])
    # One table: the values (dx, dz, joints) and each paw's x and height; paws move with the root.
    width = every.shape[1]
    paw_cols = {c: (width + 2 * i, width + 2 * i + 1) for i, c in enumerate(chains)}
    series = np.concatenate([every] + [paws[c][:, :2] for c in chains], axis=1)
    kinds = ["x", "z"] + ["angle"] * (width - 2) + ["x", "z"] * len(chains)
    swaps = []
    for (ia, ib), (a, b) in zip(pairs, [p for p in project.get("legPairs", [])
                                       if p[0] in project["chains"] and p[1] in project["chains"]]):
        extra_a = list(paw_cols[a]) if a in paw_cols else []
        extra_b = list(paw_cols[b]) if b in paw_cols else []
        swaps.append((ia + extra_a, ib + extra_b))
    samples = cycle.cycle_samples(series, kept, period, n)
    samples, exchanged = cycle.align_identity(samples, swaps, ref)
    # Root and paws are moved to the first averaged cycle, the one the start is read from.
    avg = cycle.phase_average(samples, kinds, 0)
    check = check_cycle(avg, samples, exchanged, swaps, kept, period, n, kinds, width, masks, render)
    if check["meanIoU"] < CYCLE_MIN_IOU:
        phases = ", ".join(f"{v:.2f}" for v in check["phaseIoU"])
        raise FactsError(f"The averaged cycle ({len(kept)} cycles of {period:.2f} frames) does not match the video: "
                         f"rendered at each frame's phase it scores mean IoU {check['meanIoU']:.3f} against the "
                         f"masks, under the gate {CYCLE_MIN_IOU}; per phase: {phases}. The fitted poses differ "
                         f"from cycle to cycle where the video repeats; refit (marks, spans) or regenerate")
    # The root's travel over one period (averaged over the cycles): the bake keys the cycle in place.
    drift = {"dx": float(avg[n][0] - avg[0][0]), "dz": float(avg[n][1] - avg[0][1])}
    origin = project.get("phaseOrigin", chains[0])
    tol = REFERENCE if rig is None else tolerances(rig, project)
    if rig is None:
        contacts = averaged_contacts(avg, n, kinds, paw_cols, chains, origin, fps, period, kept, project, rest, tol)
    else:
        contacts = rig_contacts(avg, n, drift, names, project, rig, rest, chains, origin, fps, period, kept, tol)
    masks_of, touchdowns = contacts["masks"], {}
    for c in chains:
        if masks_of[c].any():
            touchdowns[c] = contact_window(masks_of[c])[0][0]
    gait = {"plantedPaws": list(touchdowns), "neverPlants": [c for c in chains if c not in touchdowns],
            "phaseOrigin": contacts["origin"], **gait_pattern(masks_of, touchdowns, project, len(masks_of[chains[0]]))}
    gate, warnings = gait_check(gait, brief, accept_gait)
    stride = {"maskPeriodFrames": round(mask_frames, 3), "strideFrames": round(period, 3), "doubled": bool(doubled),
              "identityResolved": bool(identity["resolved"]), "touchdownsPerPaw": {c: 1 for c in touchdowns}}
    cadence, more = cadence_check(period, fps, gate["expected"] or next(iter(gate["measured"]["reads"]), None),
                                  mass, cycle.cycle_periods(masks, kept, period), accept_retime,
                                  brief.get("slowMotion"), stride=stride)
    warnings += more + contacts.get("warnings", [])
    factor = cadence["retime"]
    speed = contacts["speed"]
    duration = period / fps / factor
    extra = {"periodSource": "video masks", "maskPeriod": found,
             "legPeriod": leg_period(every, names, project, fps, period),
             "cycleStarts": [round(s, 4) for s in kept], "cyclesAveraged": len(kept),
             "cycleScores": {f"{s:.4f}": round(v, 4) for s, v in steady["scores"].items()},
             "identityExchanged": exchanged, "identity": identity, "averagedCycleIoU": check["meanIoU"],
             "cycleCheck": check,
             "ground": contacts["ground"], "paws": contacts["paws"], "gait": gait, "gaitCheck": gate,
             "cadence": cadence, "warnings": warnings, "bodyScale": tol.block(), "contactReading": contacts["reading"]}
    if speed is None:
        extra["speedReason"] = contacts["speedReason"]
    start, shift = contacts["start"], 0.0
    if rig is None:
        keys, n_keys, declared = contacts["keys"], n, contacts["windows"]
        cycle_out = {"frames": keys, "fps": n / duration, "periodFrames": period, "start": start,
                     "drift": contacts["drift"]}
    else:
        n_keys = motion.key_count(duration)
        keys, declared, bake = rig_keys(contacts, n_keys, names, project, rig, rest, period / fps, factor)
        extra.update({"smooth": {**contacts["smoothReport"], "seriesWorldPitchKept": contacts["smoothReport"][
                          "worldPitchKept"], "worldPitchKept": bake["worldPitchKept"], "played": bake["played"]},
                      "contactCheck": bake["check"], "ik": bake["ik"]})
        warnings += bake["warnings"]
        # Round 4: the clip starts startKey keys after the phase origin's touchdown.
        shift = bake["startKey"] / n_keys
        start = contacts["start"] + shift * period
        cycle_out = {"frames": keys, "fps": n_keys / duration, "periodFrames": period, "start": start,
                     "startKey": bake["startKey"], "startPhase": shift, "originStart": contacts["start"],
                     "drift": {"dx": 0.0, "dz": 0.0}, "videoDrift": drift, "keyRate": motion.KEY_RATE,
                     "keysPerStride": n_keys, "minKeysPerStride": motion.MIN_KEYS_PER_STRIDE,
                     "playedKeyRateHz": round(motion.key_rate(n_keys, duration), 6),
                     "retime": factor, "groundLift": round(-contacts["groundHeight"], 6), "paws": bake["paws"]}
    entry = {"duration_s": duration, "loop": True,
             "authored_speed_mps": None if speed is None else round(speed * factor, 3),
             "stride_m": None if speed is None else round(speed * period / fps, 3), "contacts": {},
             "frames": n_keys, "fps": n_keys / duration, "footfall_phase": {}, "duty_factor": {}}
    for c in chains:
        bone = project["paws"][c]
        if c not in declared:
            entry["duty_factor"][bone] = 0.0
            continue
        on, off = declared[c]
        entry["contacts"][bone] = [round(on, 4), round(off, 4)]
        entry["footfall_phase"][bone] = round((on + shift) % 1.0, 4)     # the phase origin's phase (round 4)
        entry["duty_factor"][bone] = round((off - on) % 1.0 or 1.0, 4)
    if factor != 1.0 and speed is not None:
        extra["videoSpeedMps"] = round(speed, 3)
    return start, period, entry, cycle_out, extra


def averaged_contacts(avg, n, kinds, paw_cols, chains, origin, fps, period, kept, project, rest, tol=None):
    """Without the rig: contacts from the averaged cycle's paw columns at its n samples, rolled to the phase
    origin's touchdown; the keys are the rolled average (as before Task 19). Round 4: `read_cycle` (the noisy-fit
    reading's slides from the paw track's first NOISY['order'] harmonics)."""
    rate = n * fps / period
    tol = tol or REFERENCE
    chosen = {}

    def read(table):
        heights = {c: table[:n, paw_cols[c][1]] for c in chains}
        vx = {c: np.gradient(table[:, paw_cols[c][0]])[:n] * rate for c in chains}
        noisy = lambda: {c: lowpass(table[:n, paw_cols[c][0]], NOISY["order"]) for c in chains}
        *found, report = read_cycle(heights, vx, vx, noisy, 1, tol, chosen.get("reading"))
        chosen.setdefault("reading", report["reading"])
        chosen["report"] = report
        return tuple(found), heights
    (ground, _, _, contacts_of), _ = read(avg)
    # The phase origin's touchdown starts the cycle; if it never plants, the first planted paw's does.
    origin = next((c for c in [origin] + chains if contacts_of[c].any()), origin)
    touchdown = int(round(contact_window(contacts_of[origin])[0][0] * n)) if contacts_of[origin].any() else 0
    rolled = cycle.roll(avg, touchdown, kinds)
    (ground, speed, reason, masks), heights = read(rolled)
    width = len(kinds) - 2 * len(chains)
    keys = rolled[:, :width].copy()
    # The root's travel over one period, taken before the closing key is overwritten: the bake removes
    # drift * k / n, so a periodic sway is left untouched.
    drift = {"dx": float(keys[-1][0] - keys[0][0]), "dz": float(keys[-1][1] - keys[0][1])}
    keys[-1] = keys[0]
    windows_of = {c: tuple(contact_window(masks[c])[0]) for c in chains if masks[c].any()}
    return {"masks": masks, "speed": speed, "speedReason": reason, "origin": origin, "windows": windows_of,
            "start": float(kept[0] + touchdown * period / n), "keys": keys.tolist(), "drift": drift,
            "groundHeight": ground, "ground": ground_block(ground, "averaged cycle, all paws", tol),
            "paws": paw_report(heights, masks, ground, project, rest), "reading": chosen["report"],
            "warnings": reading_warning(chosen["report"])}


def rig_contacts(avg, n, drift, names, project, rig, rest, chains, origin, fps, period, kept, tol=None, reading=None):
    """With the rig: the smooth periodic cycle (motion.SmoothCycle of the averaged values, root drift removed),
    its paws by forward kinematics at DENSE samples, the contacts from that motion, all rolled to the phase origin's
    touchdown. Round 4: `tol` is the body's Tolerances (default: scaled to `rig`); the heights and the ground speed
    are read from the smooth cycle, the stillness from it capped at VELOCITY_ORDER harmonics (`read_cycle`; the
    noisy-fit reading's slides from it capped at NOISY['order']). `reading` forces the strict or noisy-fit one."""
    tol = tol or tolerances(rig, project)
    width = len(names)
    table = np.array(avg[:n, :width], float)
    k = np.arange(n) / n
    table[:, 0] -= drift["dx"] * k
    table[:, 1] -= drift["dz"] * k
    smooth = motion.SmoothCycle(table, names, project)
    # Vole round 6: the cycle with its closing sample (one cycle averaged need not end where it starts): the range
    # the played keys are held to (motion.for_keys), as the bar's reference reads the video fit's.
    closed = np.array(avg[:n + 1, :width], float) if len(avg) > n else table
    k_closed = np.arange(len(closed)) / n
    closed[:, 0] -= drift["dx"] * k_closed
    closed[:, 1] -= drift["dz"] * k_closed
    planar = motion.PlanarRig(rig)
    grid = np.arange(DENSE) / DENSE
    gap = max(1, int(round(BRIDGE_PHASE * DENSE)))
    dt = period / fps / DENSE

    chosen = {"reading": reading}

    def capped_x(smooth_cycle, order):
        paws = planar.paws(smooth_cycle.capped(order).at(grid), names, project)
        return {c: paws[c][:, 0] for c in chains}

    def speeds(smooth_cycle, order):
        return {c: circular_speed(x, dt) for c, x in capped_x(smooth_cycle, order).items()}

    def read(smooth_cycle):
        paws = planar.paws(smooth_cycle.at(grid), names, project)
        heights = {c: paws[c][:, 1] - float(rest[c]) for c in chains}
        vx = {c: circular_speed(paws[c][:, 0], dt) for c in chains}
        *found, report = read_cycle(heights, vx, speeds(smooth_cycle, VELOCITY_ORDER),
                                    lambda: capped_x(smooth_cycle, NOISY["order"]), gap, tol, chosen["reading"])
        chosen["reading"], chosen["report"] = report["reading"], report
        return tuple(found), heights
    (_, _, _, first), _ = read(smooth)
    origin = next((c for c in [origin] + chains if first[c].any()), origin)
    touchdown = contact_window(first[origin])[0][0] if first[origin].any() else 0.0
    smooth = smooth.roll(touchdown)
    (ground, speed, reason, masks), heights = read(smooth)
    trunk = [b for chain, bones in project["chains"].items() if chain not in cycle.leg_chains(project)
             for b in bones if b in names]
    pitch = motion.world_pitch_kept(planar, table, smooth, names, trunk)
    # Vole round 6: the world pitch warning is the baked keys' (rig_keys, after motion.for_keys), not this series'.
    warnings = reading_warning(chosen["report"])
    report = {"orders": smooth.orders, "keptRange": smooth.kept_range(), "keep": motion.RANGE_KEEP,
              "worldPitchKept": pitch, "dominantHarmonic": {k: v for k, v in smooth.dominant.items() if v != 1},
              "overtoneTaper": {k: v for k, v in smooth.taper.items() if v != 1.0},
              "gain": {k: round(v, 4) for k, v in smooth.gain.items() if v != 1.0},
              "jerkRatio": smooth.jerk_ratio(), "jerkMax": motion.JERK_MAX, "samplesPerCycle": DENSE}
    return {"masks": masks, "speed": speed, "speedReason": reason, "origin": origin, "smooth": smooth,
            "smoothReport": report, "start": float(kept[0] + touchdown * period), "groundHeight": ground,
            "ground": ground_block(ground, "smooth cycle (forward kinematics of the rig), all paws", tol),
            "paws": paw_report(heights, masks, ground, project, rest), "planar": planar, "warnings": warnings,
            "reading": chosen["report"], "tol": tol, "table": table, "closed": closed, "trunk": trunk}


def pitch_warning(pitch):
    """The warning for trunk and tail bones whose world pitch range is kept under motion.RANGE_KEEP (ruling 7)."""
    short = {b: v for b, v in pitch.items() if v < motion.RANGE_KEEP}
    return [f"The smooth cycle keeps under {motion.RANGE_KEEP} of the video's world pitch range on "
            f"{', '.join(f'{b} {v:.3f}' for b, v in short.items())} (each joint's own range is kept; a warning, "
            f"not a refusal)"] if short else []


def penetration(tol):
    """How far under its ground a played paw may go (vole round 6): motion.PENETRATION_M (the bar's 5 mm at the
    wolf) scaled with the body's leg length, as the round-4 contact tolerances are (the vole's 0.32 mm)."""
    return float(motion.PENETRATION_M * tol.scale)


def contact_height(tol):
    """The bar's contact height for this body (vole round 7): BAR_CONTACT (2 cm at the wolf) times the body's leg
    scale (the vole's 1.27 mm), the height under which chase3d's criterion 5 counts a paw in contact."""
    return float(BAR_CONTACT * tol.scale)


def bar_agreement(keys, planar, names, project, rest, declared, height, subsamples=motion.CLEARANCE_SUBSAMPLES):
    """The bar's criterion 5 on the keys as the clip plays them (vole round 7): the closed keys linear at
    `subsamples` per key interval (600 a 24-key cycle, the harness's samples), each paw in contact while its tail is
    under `height` (contact_height) above its rest height, height alone, against its declared contact. Returns
    {paw bone: agreement}."""
    values, phases = motion.played(np.asarray(keys, float), subsamples)
    out = {}
    for chain, track in planar.paws(values, names, project).items():
        h = track[:, 1] - float(rest[chain])
        want = in_window(phases, declared[chain]) if chain in declared else np.zeros(len(phases), bool)
        out[project["paws"][chain]] = round(float(np.mean(want == (h < height))), 4)
    return out


def ik_reach_tolerance(tol):
    """How near a pinned paw the leg IK must bring it (vole round 5): the body's contact tolerance (round 4's
    CONTACT_TOLERANCE scaled with leg length; exactly CONTACT_TOLERANCE at the wolf)."""
    return float(tol.contact)


def widened(window, n_keys):
    """A pinned window's declared contact: half a key interval wider at each end. Between the last key in the air
    and the first pinned key the paw comes down somewhere; the midpoint is the estimate (likewise at liftoff)."""
    on, off = window
    if (on, off) == (0.0, 1.0):
        return window
    half = 0.5 / n_keys
    return round((on - half) % 1.0, 6), round((off + half) % 1.0, 6)


def snapped(mask, n_keys):
    """A dense contact mask's window on the key grid: [on, off] as key phases, at least one key interval long."""
    (on, off), _ = contact_window(mask)
    if mask.all():
        return 0.0, 1.0
    a, b = int(round(on * n_keys)) % n_keys, int(round(off * n_keys)) % n_keys
    if a == b:
        b = (a + 1) % n_keys
    return a / n_keys, b / n_keys


def rolled(window, key, n_keys):
    """A key-grid window [on, off] as phases of the clip that starts `key` keys later (round 4)."""
    if tuple(window) == (0.0, 1.0):
        return window
    on, off = (int(round(x * n_keys)) for x in window)
    return ((on - key) % n_keys) / n_keys, ((off - key) % n_keys) / n_keys


def in_window(phases, window):
    on, off = window
    if (on, off) == (0.0, 1.0):
        return np.ones(len(phases), bool)
    return (phases >= on) & (phases < off) if off > on else (phases >= on) | (phases < off)


def redetect(keys, planar, names, project, rest, declared, speed, tol, period_s):
    """The keys played as the clip plays them (linear between keys), their paws detected again with the same rule
    (near the ground within the body's contact tolerance and world-still; no stance tolerance) and compared with the
    declared contacts. Returns (agreement per paw bone, heights, vx, phases, detected masks)."""
    played, phases = motion.played(keys)
    dt = period_s / len(played)
    heights, vx = {}, {}
    for chain, track in planar.paws(played, names, project).items():
        heights[chain] = track[:, 1] - float(rest[chain])
        vx[chain] = circular_speed(track[:, 0], dt)
    # The same rule without the fit-error allowance (the stance tolerance): a pinned paw in the keys is on the
    # ground. Round 4: the body's contact tolerance.
    again = stance_masks(heights, vx, 0.0, speed, circular=True, gap=max(1, int(round(BRIDGE_PHASE * len(played)))),
                         extent=tol.contact, tol=tol)
    agreement = {}
    for chain in heights:
        want = in_window(phases, declared[chain]) if chain in declared else np.zeros(len(phases), bool)
        agreement[project["paws"][chain]] = round(float(np.mean(want == again[chain])), 4)
    return agreement, heights, vx, phases, again


def rig_keys(contacts, n_keys, names, project, rig, rest, period_s, factor, start=None):
    """The bake's keys from the smooth cycle: every planted paw pinned (motion.plan_keys) over its detection on the
    key grid, the declared contacts half a key wider at each end (`widened`), and the check: the keys played as the clip plays them (linear between
    keys), their paws detected again with the same rule (near the ground within CONTACT_TOLERANCE and world-still;
    no stance tolerance, which allows for the fit's height error), against the declared contacts. Refuses under
    CONTACT_AGREEMENT for any paw. Returns (keys, declared windows, {"check", "ik", "paws"})."""
    planar, speed, tol = contacts["planar"], contacts["speed"], contacts["tol"]
    pinned = {c: snapped(m, n_keys) for c, m in contacts["masks"].items() if m.any()}
    # Round 4: the loop starts at the key with the most room outside every pin's approach and release.
    key0, start_info = motion.start_key(pinned, n_keys, force=start)
    smooth = contacts["smooth"]
    if key0:
        pinned = {c: rolled(w, key0, n_keys) for c, w in pinned.items()}
        smooth = smooth.roll(key0 / n_keys)
    declared = {c: widened(w, n_keys) for c, w in pinned.items()}
    # Vole round 6: the trunk, neck and tail columns as the keys play them (motion.for_keys): within the bar's 60 Hz
    # jerk and keeping their range and world pitch range at the keys. The wolf lope's columns are unchanged.
    table, trunk = contacts.get("table"), contacts.get("trunk", [])
    smooth, played_report = motion.for_keys(smooth, n_keys, period_s / factor, motion.leg_bones(project), planar,
                                            contacts.get("closed", table), trunk)
    pitch = motion.world_pitch_kept(planar, table, smooth, names, trunk) if table is not None else {}
    lift = -contacts["groundHeight"]
    stride = (speed or 0.0) * period_s
    keys, paws, ik = motion.plan_keys(planar, smooth, names, project, pinned, stride, rest, lift, n_keys,
                                      clearance=penetration(tol))
    agreement, heights, vx, phases, again = redetect(keys, planar, names, project, rest, declared, speed, tol,
                                                     period_s)
    releases = motion.release_windows(pinned, n_keys, motion.PIN_RELEASE.inset_keys)
    plant = contact_height(tol)
    bar = bar_agreement(keys, planar, names, project, rest, declared, plant)
    low = [c for c in heights if c in declared and agreement[project["paws"][c]] < SETTLE_BELOW]
    if min(bar.values(), default=1.0) < CONTACT_AGREEMENT:
        # Vole round 7: a clip that fails the bar's criterion 5 also settles every paw under SETTLE_BELOW on it.
        low = [c for c in heights if c in declared and (c in low or bar[project["paws"][c]] < SETTLE_BELOW)]
    if low:
        # Vole round 6: a paw whose keys fail the agreement gate or pass it by under SETTLE_BELOW has the keys at its
        # declared contact's edges settled onto its pin (motion.settle_edges), the ground cleared again, and is
        # detected again (never the wolf lope's: 0.974 at least, and 0.957 at least on the bar's rule). Vole round 7:
        # settled at the bar's contact height (round 6 used 2 x facts' contact tolerance), its swing lifted out of
        # contact (motion.clear_swing), and the edges settled again on the lifted swing.
        open_keys = np.asarray(keys, float)[:n_keys]
        grounds = {c: float(rest[c]) for c in releases}
        for settle_pass in range(2):
            settled = {}
            for chain in low:
                open_keys, settled[chain] = motion.settle_edges(planar, open_keys, names, project, chain,
                                                                releases[chain], declared[chain], n_keys, plant,
                                                                grounds[chain])
            open_keys, _ = motion.clear_ground(planar, open_keys, names, project, grounds, penetration(tol))
            if settle_pass == 0:
                swings = {c: (declared[c], set(np.flatnonzero(motion._pinned_mask(releases[c], n_keys)[0]).tolist())
                              | set(settled[c])) for c in low}
                open_keys, swung = motion.clear_swing(planar, open_keys, names, project, swings, grounds, plant)
        for chain in low:
            if settled[chain]:
                ik.setdefault(chain, {})["settledKeys"] = settled[chain]
            if chain in swung:
                ik.setdefault(chain, {})["swingLiftMaxM"] = swung[chain]
        keys = np.vstack([open_keys, open_keys[:1]])
        paws = planar.paws(keys, names, project)
        agreement, heights, vx, phases, again = redetect(keys, planar, names, project, rest, declared, speed, tol,
                                                         period_s)
        bar = bar_agreement(keys, planar, names, project, rest, declared, plant)
    # Vole round 6: the loop seam joins with no velocity jump over the cycle's median per-key change, per paw and
    # trunk, neck and tail column (motion.seam_jumps); else the loop starts at the nearest key, with the most room
    # outside the pins' eases, where it does (the same motion rolled; the wolf lope's seam passes where it is).
    seam = motion.seam_jumps(keys, planar, names, project, motion.seam_columns(names, project))
    seam_pass = max(seam.values(), default=0.0) <= 1.0
    if not seam_pass and start is None:
        spans = motion.ease_spans(pinned, n_keys)
        room = [motion.ease_clearance(k / n_keys, spans) for k in range(n_keys)]
        shift = motion.seam_shift(keys, planar, names, project, room)
        if shift:
            out = rig_keys(contacts, n_keys, names, project, rig, rest, period_s, factor,
                           start=(key0 + shift) % n_keys)
            out[2]["check"]["loopSeam"]["movedFrom"] = {"key": key0, "worstRatio": round(max(seam.values()), 4)}
            return out
    jumps = motion.edge_jumps(paws, motion.release_windows(pinned, n_keys, motion.PIN_RELEASE.inset_keys), n_keys,
                              period_s / factor / n_keys, span=motion.PIN_RELEASE.span(n_keys))
    edge_jumps = {project["paws"][c]: j for c, j in jumps.items()}
    still = {}
    for chain in heights:
        bone = project["paws"][chain]
        want = in_window(phases, declared[chain]) if chain in declared else np.zeros(len(phases), bool)
        if want.any() and speed:
            world = np.abs(vx[chain] + speed) * factor
            still[bone] = round(float(np.mean(world[want] < tol.still_speed)), 4)
    check = {"agreement": agreement, "gate": CONTACT_AGREEMENT, "plantedStillShare": still,
             "stillSpeedMps": round(tol.still_speed, 6), "samples": len(phases), "keys": n_keys, "edgeJumps": edge_jumps,
             "pinRelease": {**motion.PIN_RELEASE._asdict(),
                            "touchdown_orders": list(motion.PIN_RELEASE.touchdown_orders),
                            "liftoff_orders": list(motion.PIN_RELEASE.liftoff_orders),
                            "spanKeys": round(motion.PIN_RELEASE.span(n_keys) * n_keys, 3)},
             "redetectionTolerance": {"contactToleranceM": round(tol.contact, 6), "stillShareOfSpeed": SLIP_SHARE,
                                      "scale": round(tol.scale, 6)},
             "edgeJumpRule": "an edge's velocity change at most the paw's swing median per-key change (x and z at "
                             "the 60 Hz keys; in place = world for a constant body speed): touchdownMps and "
                             "liftoffMps across the whole approach and release span (round 4), touchdownPinnedMps and "
                             "liftoffPinnedMps at the first and last pinned key (round 3)"}
    span_over = {b: j for b, j in edge_jumps.items() if max(j["touchdownMps"], j["liftoffMps"]) > j["swingMedianMps"]}
    pinned_over = {b: j for b, j in edge_jumps.items()
                   if max(j["touchdownPinnedMps"], j["liftoffPinnedMps"]) > j["swingMedianMps"]}
    check["edgeSpanGate"] = {"pass": not span_over, "enforced": EDGE_GATE_SPAN, "over": sorted(span_over)}
    check["barContact"] = {"agreement": bar, "gate": CONTACT_AGREEMENT, "heightM": round(plant, 6),
                           "pass": min(bar.values(), default=1.0) >= CONTACT_AGREEMENT,
                           "rule": "chase3d tools/anim_compare.py criterion 5 on the keys as played (600 samples a "
                                   "24-key cycle): a paw in contact while under BAR_CONTACT x body scale above its "
                                   "rest height, height alone, against its declared contact"}
    check["loopSeam"] = {"pass": seam_pass, "worstRatio": round(max(seam.values(), default=0.0), 4),
                         "ratios": seam, "rule": "per paw and trunk, neck and tail column, the velocity jump across "
                                                 "the seam over the cycle's median per-key change, at most 1"}
    check["cycleStart"] = {**start_info, "phase": key0 / n_keys, "originStart": contacts["start"],
                           "origin": contacts["origin"],
                           "rule": "the key with the most room outside every paw's pin approach and release span "
                                   "(motion.start_key); footfall phases stay the phase origin's"}
    warnings = pitch_warning(pitch)
    if not seam_pass:
        warnings.append("The loop seam's velocity jumps over the cycle's median per-key change at every start key: "
                        + ", ".join(f"{k} {v:.2f}" for k, v in seam.items() if v > 1.0) + " (a warning)")
    if span_over and not EDGE_GATE_SPAN:
        warnings.append("Across the pins' approach and release spans the paw velocity changes more than the swing's "
                        "median per-key change (round 4 edge gate, measured, not enforced): " + "; ".join(
                            f"{b} touchdown {j['touchdownMps']:.2f}, liftoff {j['liftoffMps']:.2f} against "
                            f"{j['swingMedianMps']:.2f} m/s per key" for b, j in span_over.items()))
    jumpy = span_over if EDGE_GATE_SPAN else pinned_over
    if jumpy:
        where = "across its pin's approach or release span" if EDGE_GATE_SPAN else "at its first or last pinned key"
        key = ("touchdownMps", "liftoffMps") if EDGE_GATE_SPAN else ("touchdownPinnedMps", "liftoffPinnedMps")
        raise FactsError(f"A paw's velocity jumps at a contact edge ({where}) more than anywhere typical in its swing: "
                         + "; ".join(f"{b} touchdown {j[key[0]]:.2f} m/s, liftoff {j[key[1]]:.2f} m/s against a swing "
                                     f"median of {j['swingMedianMps']:.2f} m/s per key" for b, j in jumpy.items())
                         + f" (leg IK: {ik}). The pins do not ease in and out; check the leg chains and the IK "
                           f"residuals")
    # Vole round 5: the IK reaches every pin within the body's contact tolerance (scaled with leg length as the
    # round-4 contact tolerances are; the wolf's 25 mm, the vole's 1.6 mm).
    reach_tol = ik_reach_tolerance(tol)
    worst = {project["paws"][c]: r["maxResidualM"] for c, r in ik.items()}
    check["ikReach"] = {"toleranceM": round(reach_tol, 6), "maxResidualM": worst,
                        "pass": all(v <= reach_tol for v in worst.values())}
    if not check["ikReach"]["pass"]:
        raise FactsError("The leg IK cannot reach a pin within the body's contact tolerance "
                         f"({reach_tol * 1000:.2f} mm): " + ", ".join(f"{b} misses by {v * 1000:.2f} mm"
                                                                      for b, v in worst.items() if v > reach_tol)
                         + f" (leg IK: {ik}). The pinned paw lies beyond the whole leg's straight reach even with "
                           f"the girdle swung and the body lowered; check the rig's leg chains and the fit's trunk")
    low = {b: a for b, a in agreement.items() if a < CONTACT_AGREEMENT}
    if low:
        raise FactsError(f"The declared contacts and a re-detection from the baked keys disagree: they agree on "
                         f"{', '.join(f'{b} {a:.2f}' for b, a in low.items())} of the played cycle, under the gate "
                         f"{CONTACT_AGREEMENT} (leg IK: {ik}). The keys do not plant those paws where facts "
                         f"declared them; check the leg chains, the fit's paws and the IK residuals")
    under = {b: a for b, a in bar.items() if a < CONTACT_AGREEMENT}
    if under:
        raise FactsError(f"The declared contacts and the bar's contact height ({plant * 1000:.2f} mm, criterion 5: "
                         f"height alone) disagree on the played keys: they agree on "
                         f"{', '.join(f'{b} {a:.3f}' for b, a in under.items())} of the cycle, under the gate "
                         f"{CONTACT_AGREEMENT} (leg IK: {ik}). A swing comes down into contact or a planted paw is "
                         f"over that height; check the fit's paws and the settled edges")
    return keys.tolist(), declared, {"check": check, "ik": ik, "startKey": key0, "warnings": warnings,
                                     "played": played_report, "worldPitchKept": pitch,
                                     "paws": {project["paws"][c]: np.round(p, 6).tolist() for c, p in paws.items()}}


def check_cycle(avg, samples, exchanged, swaps, kept, period, n, kinds, width, masks, render):
    """Render the averaged cycle at every whole frame of the averaged cycles, at that frame's phase, each cycle
    placed at its own root (its start, its own travel) and leg labelling, and score it against the frame's mask.
    Returns {"meanIoU", "phaseIoU" (per phase sample, the mean over the frames nearest it), "frames", "gate"}."""
    from anim.raster import iou
    translate = cycle.translation(samples, kinds, 0)
    travel = samples[:, n] - samples[:, 0]
    mean_travel = travel.mean(axis=0)
    values, frames, phases = [], [], []
    count = len(masks)
    for c, s in enumerate(kept):
        for f in range(int(np.ceil(s - 1e-9)), count):
            if f >= s + period - 1e-9:
                break
            phase = (f - s) / period * n
            v = cycle.at_phase(avg, phase)
            v = v + translate[c] + (travel[c] - mean_travel) * phase / n * np.array(
                [kind in ("x", "z") for kind in kinds])
            for (a, b), flag in zip(swaps, exchanged[c]):
                if flag:
                    v[a], v[b] = v[b].copy(), v[a].copy()
            values.append(v[:width].tolist())
            frames.append(f)
            phases.append(int(round(phase)) % n)
    rendered = render(values)
    ious = [float(iou(np.asarray(r, bool), masks[f])) for r, f in zip(rendered, frames)]
    per = [round(float(np.mean([v for v, p in zip(ious, phases) if p == k])), 4) if k in phases else None
           for k in range(n)]
    return {"meanIoU": round(float(np.mean(ious)), 4), "phaseIoU": per, "frames": len(frames),
            "gate": CYCLE_MIN_IOU}
