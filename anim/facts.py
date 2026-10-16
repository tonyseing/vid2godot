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
key past the pinned keys at each end (where between keys the paw lands and lifts; so the phase origin's footfall is
1 - 0.5 / n_keys, half a key before key 0); and the
keys, played as the clip plays them, are detected again (the same rule, without the fit-error allowance): under
CONTACT_AGREEMENT per paw facts refuses. The footfall
pattern is checked against the brief's `gait` (`expected_gait`, `gait_reads`), and the cadence against the
Heglund-Taylor band for the body mass (`cadence_check`): a slow video with one consistent period is retimed to the band
centre, recorded in `cadence` and the warnings."""
import re

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
STILL_SPEED = 0.10           # m/s: the report's "still" for a planted paw in the played clip
MASS_SPREAD = 0.25           # a single massKg m is the range [m (1 - spread), m (1 + spread)]
CADENCE_CONSISTENCY = 0.10   # every steady cycle's own period within this share of the clip's: one time scale
RETIME_BOUND = (0.5, 2.0)    # a retime factor outside this needs `facts --accept-retime <reason>` (fix round 1)
GALLOPS = ("canter", "gallop", "transverse gallop", "rotary gallop")
# The brief's `gait` vocabulary: the earliest of these whole words (or their inflections) names the expected gait.
GAIT_WORDS = ((r"transverse gallop(?:s|ing|ed)?", "transverse gallop"), (r"rotary gallop(?:s|ing|ed)?", "rotary gallop"),
              (r"gallop(?:s|ing|ed)?", "gallop"), (r"canter(?:s|ing|ed)?", "canter"), (r"lop(?:e|es|ing|ed)", "canter"),
              (r"trot(?:s|ting|ted)?", "trot"), (r"pac(?:e|es|ing|ed)", "pace"), (r"walk(?:s|ing|ed)?", "walk"))
VOCABULARY = "walk, trot, pace, canter/lope, gallop, transverse gallop or rotary gallop (or their -ing forms)"


class FactsError(StageError):
    pass


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


def stance_masks(heights, vx, ground, speed, circular=False, gap=1, extent=STANCE_TOLERANCE):
    """Each paw's stance (Task 19): world-still (in-place x speed within SLIP_SHARE * speed of -speed) and near the
    shared ground: a run of still samples within STANCE_TOLERANCE that reaches CONTACT_TOLERANCE somewhere. Without
    a ground speed (under two planted paws) stillness is not known and every sample counts as still."""
    out = {}
    for chain, z in heights.items():
        z = np.asarray(z, float)
        v = np.asarray(vx[chain], float)
        still = np.abs(v + speed) <= SLIP_SHARE * speed if speed and speed > 0 else np.ones(len(z), bool)
        core = still & (z <= ground + CONTACT_TOLERANCE)
        reach = still & (z <= ground + extent)
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


def motion_contacts(heights, vx, circular=False, gap=1):
    """Contacts from the motion: the shared ground, then the ground speed from the paws near it, then each paw's
    stance (`stance_masks`), the speed measured again from those stances and the stances read with it. Returns
    (ground, speed, speed reason, masks)."""
    ground = ground_height(heights)
    near = {c: bridge(contact_mask(z, ground), circular, gap) for c, z in heights.items()}
    speed, reason = speed_from(vx, near)
    masks = stance_masks(heights, vx, ground, speed, circular, gap)
    if speed is not None:
        again, why = speed_from(vx, masks)
        if again is not None and again > 0:
            speed, reason = again, why
            masks = stance_masks(heights, vx, ground, speed, circular, gap)
    return ground, speed, reason, masks


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


def cadence_band(kg):
    """The Heglund-Taylor stride-frequency band (Hz) over a mass range: [f(heaviest), f(lightest)]."""
    low, high = mass_range(kg)
    return [motion.heglund_taylor_hz(high), motion.heglund_taylor_hz(low)]


def cadence_check(period, fps, gait_name, mass, periods, accept_retime=None):
    """The stride frequency against the body-size band (galloping gaits: canter, gallop): the block for facts.json
    and the warnings. A video under the band whose steady cycles share one period (each within
    CADENCE_CONSISTENCY of the clip's) is retimed to the band centre; the factor scales the clip's speed.
    Fix round 1: a mass from an open mesh's volume (`openEdges` > 0) is checked against but never retimes; a factor
    outside RETIME_BOUND refuses unless `accept_retime` records why."""
    hz = fps / period
    out = {"videoHz": round(hz, 4), "massKg": None, "massSource": None, "bandHz": None, "centreHz": None,
           "checked": False, "consistent": None, "cyclePeriods": [round(p, 3) for p in periods], "retime": 1.0,
           "hz": round(hz, 4), "law": "Heglund & Taylor 1988: stride frequency at the preferred galloping speed "
                                      "4.44 M^-0.156 Hz"}
    warnings = []
    open_edges = (mass or {}).get("openEdges") or 0
    if mass is not None:
        out["massKg"], out["massSource"] = mass_range(mass["kg"]), mass["source"]
        if open_edges:
            out["massOpenEdges"] = int(open_edges)
    if gait_name not in GALLOPS:
        out["reason"] = f"no Heglund-Taylor band for a {gait_name or 'gait the brief does not name'}"
        return out, warnings
    if mass is None:
        out["reason"] = "no body mass"
        warnings.append("Cadence not checked: no body mass (the brief's massKg or `facts --mass-kg`)")
        return out, warnings
    band = cadence_band(mass["kg"])
    centre = sum(band) / 2
    consistent = bool(periods) and all(abs(p - period) / period <= CADENCE_CONSISTENCY for p in periods)
    out.update({"bandHz": [round(b, 4) for b in band], "centreHz": round(centre, 4), "checked": True,
                "consistent": consistent})
    kg = "-".join(f"{m:g}" for m in out["massKg"])
    where = f"{band[0]:.2f}-{band[1]:.2f} Hz band for {kg} kg"
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
        accepted = ""
        if not low <= factor <= high:
            if not accept_retime:
                raise FactsError(f"The cadence retime {factor:.2f} (the video's {hz:.2f} Hz to {centre:.2f} Hz, the "
                                 f"{where}) is outside {low}-{high}: the video is far slower than the body size says "
                                 f"or the mass is wrong. Check the mass (the brief's massKg or `facts --mass-kg`), or "
                                 f"record why this retime stands with `facts --accept-retime <reason>`")
            out["retimeAccepted"] = accept_retime
            accepted = f"; outside {low}-{high}, accepted: {accept_retime}"
        out["retime"], out["hz"] = round(factor, 6), round(centre, 4)
        warnings.append(f"The video's cadence {hz:.2f} Hz is under the {where}: retimed by {factor:.3f} to "
                        f"{centre:.2f} Hz (the speed by the same factor){accepted}")
    elif hz < band[0]:
        warnings.append(f"The video's cadence {hz:.2f} Hz is under the {where} and not retimed: the steady cycles' "
                        f"own periods ({out['cyclePeriods']}) are not one time scale")
    elif hz > band[1]:
        warnings.append(f"The video's cadence {hz:.2f} Hz is above the {where}; not retimed (only a slow video is)")
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
        raise FactsError(f"A paw touches down {len(starts)} times in one cycle; check leg identity and marks")
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
        ground, speed, speed_reason, masks = motion_contacts(heights, vx)
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
        extra = {"ground": ground_block(ground, "every frame, all paws"),
                 "paws": paw_report(heights, masks, ground, project, rest)}
    target = brief.get("targetSpeedMps")
    if speed is not None and target and abs(speed - target) / target > 0.25:
        warnings.append(f"Measured ground speed {speed:.2f} m/s is more than 25% from the brief's {target} m/s")
    out = {"start": start, "period": period, "fps": fps, "speedMps": speed, "strideM": entry["stride_m"],
           "entry": entry, "warnings": warnings, "cycle": cycle_out}
    if speed is None:
        out["speedReason"] = speed_reason
    out.update(extra)
    return out


def ground_block(ground, source):
    return {"height": round(ground, 4), "tolerance": CONTACT_TOLERANCE, "percentile": GROUND_PERCENTILE,
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


def loop_cycle(poses, project, every, paws, chains, masks, render, rest, rig=None, brief=None, mass=None,
               accept_gait=None, accept_retime=None):
    """The loop's cycle from the video: period, steady cycles, phase average, the averaged cycle's check against the
    masks, the contacts, the gait and cadence checks, and (with `rig`) the 60 Hz keys. Returns (start, period, entry,
    cycle block, extra facts)."""
    if masks is None or render is None:
        raise FactsError("facts finds a loop's period in the video: it needs the ingest masks and the rig's renderer")
    brief = brief or {}
    fps = float(poses["fps"])
    names = poses["names"]
    count = len(every)
    found = cycle.mask_period(masks, fps)
    if found["period"] is None:
        raise FactsError(f"no full cycle found in the video between 0.25 s and 2.5 s: {found['reason']}; the clip "
                         f"must hold at least two cycles")
    period = found["period"]
    pairs = cycle.leg_pairs(names, project)
    doubled = cycle.half_period(every, pairs, period)
    if doubled:
        period *= 2
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
    if rig is None:
        contacts = averaged_contacts(avg, n, kinds, paw_cols, chains, origin, fps, period, kept, project, rest)
    else:
        contacts = rig_contacts(avg, n, drift, names, project, rig, rest, chains, origin, fps, period, kept)
    masks_of, touchdowns = contacts["masks"], {}
    for c in chains:
        if masks_of[c].any():
            touchdowns[c] = contact_window(masks_of[c])[0][0]
    gait = {"plantedPaws": list(touchdowns), "neverPlants": [c for c in chains if c not in touchdowns],
            "phaseOrigin": contacts["origin"], **gait_pattern(masks_of, touchdowns, project, len(masks_of[chains[0]]))}
    gate, warnings = gait_check(gait, brief, accept_gait)
    cadence, more = cadence_check(period, fps, gate["expected"] or next(iter(gate["measured"]["reads"]), None),
                                  mass, cycle.cycle_periods(masks, kept, period), accept_retime)
    warnings += more + contacts.get("warnings", [])
    factor = cadence["retime"]
    speed = contacts["speed"]
    duration = period / fps / factor
    extra = {"periodSource": "video masks", "maskPeriod": found,
             "legPeriod": leg_period(every, names, project, fps, period),
             "cycleStarts": [round(s, 4) for s in kept], "cyclesAveraged": len(kept),
             "cycleScores": {f"{s:.4f}": round(v, 4) for s, v in steady["scores"].items()},
             "identityExchanged": exchanged, "averagedCycleIoU": check["meanIoU"], "cycleCheck": check,
             "ground": contacts["ground"], "paws": contacts["paws"], "gait": gait, "gaitCheck": gate,
             "cadence": cadence, "warnings": warnings}
    if speed is None:
        extra["speedReason"] = contacts["speedReason"]
    if rig is None:
        keys, n_keys, declared = contacts["keys"], n, contacts["windows"]
        cycle_out = {"frames": keys, "fps": n / duration, "periodFrames": period, "start": contacts["start"],
                     "drift": contacts["drift"]}
    else:
        n_keys = motion.key_count(duration)
        keys, declared, bake = rig_keys(contacts, n_keys, names, project, rig, rest, period / fps, factor)
        extra.update({"smooth": contacts["smoothReport"], "contactCheck": bake["check"], "ik": bake["ik"]})
        cycle_out = {"frames": keys, "fps": n_keys / duration, "periodFrames": period, "start": contacts["start"],
                     "drift": {"dx": 0.0, "dz": 0.0}, "videoDrift": drift, "keyRate": motion.KEY_RATE,
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
        entry["contacts"][bone], entry["footfall_phase"][bone] = [round(on, 4), round(off, 4)], round(on, 4)
        entry["duty_factor"][bone] = round((off - on) % 1.0 or 1.0, 4)
    if factor != 1.0 and speed is not None:
        extra["videoSpeedMps"] = round(speed, 3)
    return contacts["start"], period, entry, cycle_out, extra


def averaged_contacts(avg, n, kinds, paw_cols, chains, origin, fps, period, kept, project, rest):
    """Without the rig: contacts from the averaged cycle's paw columns at its n samples, rolled to the phase
    origin's touchdown; the keys are the rolled average (as before Task 19)."""
    rate = n * fps / period

    def read(table):
        heights = {c: table[:n, paw_cols[c][1]] for c in chains}
        vx = {c: np.gradient(table[:, paw_cols[c][0]])[:n] * rate for c in chains}
        return motion_contacts(heights, vx, circular=True), heights
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
            "groundHeight": ground, "ground": ground_block(ground, "averaged cycle, all paws"),
            "paws": paw_report(heights, masks, ground, project, rest)}


def rig_contacts(avg, n, drift, names, project, rig, rest, chains, origin, fps, period, kept):
    """With the rig: the smooth periodic cycle (motion.SmoothCycle of the averaged values, root drift removed),
    its paws by forward kinematics at DENSE samples, the contacts from that motion, all rolled to the phase origin's
    touchdown."""
    width = len(names)
    table = np.array(avg[:n, :width], float)
    k = np.arange(n) / n
    table[:, 0] -= drift["dx"] * k
    table[:, 1] -= drift["dz"] * k
    smooth = motion.SmoothCycle(table, names, project)
    planar = motion.PlanarRig(rig)
    grid = np.arange(DENSE) / DENSE
    gap = max(1, int(round(BRIDGE_PHASE * DENSE)))
    dt = period / fps / DENSE

    def read(smooth_cycle):
        paws = planar.paws(smooth_cycle.at(grid), names, project)
        heights = {c: paws[c][:, 1] - float(rest[c]) for c in chains}
        vx = {c: circular_speed(paws[c][:, 0], dt) for c in chains}
        return motion_contacts(heights, vx, circular=True, gap=gap), heights
    (_, _, _, first), _ = read(smooth)
    origin = next((c for c in [origin] + chains if first[c].any()), origin)
    touchdown = contact_window(first[origin])[0][0] if first[origin].any() else 0.0
    smooth = smooth.roll(touchdown)
    (ground, speed, reason, masks), heights = read(smooth)
    trunk = [b for chain, bones in project["chains"].items() if chain not in cycle.leg_chains(project)
             for b in bones if b in names]
    pitch = motion.world_pitch_kept(planar, table, smooth, names, trunk)
    short = {b: v for b, v in pitch.items() if v < motion.RANGE_KEEP}
    warnings = [f"The smooth cycle keeps under {motion.RANGE_KEEP} of the video's world pitch range on "
                f"{', '.join(f'{b} {v:.3f}' for b, v in short.items())} (each joint's own range is kept; a warning, "
                f"not a refusal)"] if short else []
    report = {"orders": smooth.orders, "keptRange": smooth.kept_range(), "keep": motion.RANGE_KEEP,
              "worldPitchKept": pitch, "dominantHarmonic": {k: v for k, v in smooth.dominant.items() if v != 1},
              "overtoneTaper": {k: v for k, v in smooth.taper.items() if v != 1.0},
              "gain": {k: round(v, 4) for k, v in smooth.gain.items() if v != 1.0},
              "jerkRatio": smooth.jerk_ratio(), "jerkMax": motion.JERK_MAX, "samplesPerCycle": DENSE}
    return {"masks": masks, "speed": speed, "speedReason": reason, "origin": origin, "smooth": smooth,
            "smoothReport": report, "start": float(kept[0] + touchdown * period), "groundHeight": ground,
            "ground": ground_block(ground, "smooth cycle (forward kinematics of the rig), all paws"),
            "paws": paw_report(heights, masks, ground, project, rest), "planar": planar, "warnings": warnings}


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


def in_window(phases, window):
    on, off = window
    if (on, off) == (0.0, 1.0):
        return np.ones(len(phases), bool)
    return (phases >= on) & (phases < off) if off > on else (phases >= on) | (phases < off)


def rig_keys(contacts, n_keys, names, project, rig, rest, period_s, factor):
    """The bake's keys from the smooth cycle: every planted paw pinned (motion.plan_keys) over its detection on the
    key grid, the declared contacts half a key wider at each end (`widened`), and the check: the keys played as the clip plays them (linear between
    keys), their paws detected again with the same rule (near the ground within CONTACT_TOLERANCE and world-still;
    no STANCE_TOLERANCE, which allows for the fit's height error), against the declared contacts. Refuses under
    CONTACT_AGREEMENT for any paw. Returns (keys, declared windows, {"check", "ik", "paws"})."""
    planar, speed = contacts["planar"], contacts["speed"]
    pinned = {c: snapped(m, n_keys) for c, m in contacts["masks"].items() if m.any()}
    declared = {c: widened(w, n_keys) for c, w in pinned.items()}
    lift = -contacts["groundHeight"]
    stride = (speed or 0.0) * period_s
    keys, paws, ik = motion.plan_keys(planar, contacts["smooth"], names, project, pinned, stride, rest, lift,
                                      n_keys)
    played, phases = motion.played(keys)
    dt = period_s / len(played)
    heights, vx = {}, {}
    for chain, track in planar.paws(played, names, project).items():
        heights[chain] = track[:, 1] - float(rest[chain])
        vx[chain] = circular_speed(track[:, 0], dt)
    # The same rule without the fit-error allowance (STANCE_TOLERANCE): a pinned paw in the keys is on the ground.
    again = stance_masks(heights, vx, 0.0, speed, circular=True, gap=max(1, int(round(BRIDGE_PHASE * len(played)))),
                         extent=CONTACT_TOLERANCE)
    jumps = motion.edge_jumps(paws, motion.release_windows(pinned, n_keys, motion.PIN_RELEASE.inset_keys), n_keys,
                              period_s / factor / n_keys)
    edge_jumps = {project["paws"][c]: j for c, j in jumps.items()}
    agreement, still = {}, {}
    for chain in heights:
        bone = project["paws"][chain]
        want = in_window(phases, declared[chain]) if chain in declared else np.zeros(len(phases), bool)
        agreement[bone] = round(float(np.mean(want == again[chain])), 4)
        if want.any() and speed:
            world = np.abs(vx[chain] + speed) * factor
            still[bone] = round(float(np.mean(world[want] < STILL_SPEED)), 4)
    check = {"agreement": agreement, "gate": CONTACT_AGREEMENT, "plantedStillShare": still,
             "stillSpeedMps": STILL_SPEED, "samples": len(played), "keys": n_keys, "edgeJumps": edge_jumps,
             "pinRelease": {**motion.PIN_RELEASE._asdict(),
                            "touchdown_orders": list(motion.PIN_RELEASE.touchdown_orders),
                            "liftoff_orders": list(motion.PIN_RELEASE.liftoff_orders),
                            "spanKeys": round(motion.PIN_RELEASE.span(n_keys) * n_keys, 3)},
             "edgeJumpRule": "the velocity change at the first and last pinned key at most the swing's median "
                             "per-key change"}
    jumpy = {b: j for b, j in edge_jumps.items() if max(j["touchdownMps"], j["liftoffMps"]) > j["swingMedianMps"]}
    if jumpy:
        raise FactsError("A paw's velocity jumps at a contact edge more than anywhere typical in its swing: " + "; ".join(
            f"{b} touchdown {j['touchdownMps']:.2f} m/s, liftoff {j['liftoffMps']:.2f} m/s against a swing median of "
            f"{j['swingMedianMps']:.2f} m/s per key" for b, j in jumpy.items()) + f" (leg IK: {ik}). The pins do not "
            f"ease in and out; check the leg chains and the IK residuals")
    low = {b: a for b, a in agreement.items() if a < CONTACT_AGREEMENT}
    if low:
        raise FactsError(f"The declared contacts and a re-detection from the baked keys disagree: they agree on "
                         f"{', '.join(f'{b} {a:.2f}' for b, a in low.items())} of the played cycle, under the gate "
                         f"{CONTACT_AGREEMENT} (leg IK: {ik}). The keys do not plant those paws where facts "
                         f"declared them; check the leg chains, the fit's paws and the IK residuals")
    return keys.tolist(), declared, {"check": check, "ik": ik,
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
