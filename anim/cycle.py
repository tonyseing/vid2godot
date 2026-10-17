"""The video's cycle, apart from the fitted angles: the period from the registered masks' own self-similarity, the
steady cycles after the warm-up, and the phase-aligned average of a clip's values over them.

A side silhouette barely constrains some joints (a paw, the neck, the chest, a tail tip): the fit can trade their
angles for the same outline, so the fitted angles need not repeat where the video does. The period is therefore
read from the masks, and the cycle the bake keys is the average over the steady cycles, which `facts` checks against
the masks. The fitter's periodic pass uses the same average as a weak prior.

numpy only (anim/blender/fit_clip.py imports it inside Blender)."""
import math

import numpy as np

# The period's gate: the fundamental's refined aligned 1 - IoU (median over frame pairs) at most MASK_PERIOD_MAX, and
# the median of the other lags at least MASK_PERIOD_MARGIN times the best score of it and its multiples. Measured
# (Task 17b fix round 2): the real Seedance lope scores 0.17 at 15.1 frames against 0.45 elsewhere (margin 2.7x;
# half its frame pairs hold one of its time-displaced frames); the toy gaits in tests score 0.12-0.19 with margins
# 2.9-4.4x; the 24-frame synthetic walk (S1-S3) scores ~0; a still clip has no margin, a one-shot no low lag.
MASK_PERIOD_MAX = 0.25
MASK_PERIOD_MARGIN = 2.0
# Vole round 3: a candidate dips by at least MASK_PERIOD_PROMINENCE of the curve's median level (the vole's noise dip
# at 16 frames: 0.003 against 0.14; its period at 30: 0.14; the wolf's at 15: 0.33 against 0.45), and no other dip
# comes within MASK_PERIOD_RIVAL of the period's family score (the aperiodic round-1 vole: 0.044 at 57 against 0.034
# at 60, 1.3x; the round-2 vole 16x; the wolf has no rival dip).
MASK_PERIOD_PROMINENCE = 0.2
MASK_PERIOD_RIVAL = 1.5
# A cycle is steady when its mask score (the mean 1 - IoU of its frames against the same phase a period on) is
# within STEADY_RATIO of the steadiest cycle's, or STEADY_SLACK of it (a clean clip scores ~0).
STEADY_RATIO = 1.5
STEADY_SLACK = 0.03
# Leg identity exchanged between cycles only when exchanging is clearly closer; the half-period test likewise.
HALF_PERIOD_RATIO = 0.5
# Vole round 3: a pair's paws meet where their distance has a local minimum under MEET_SHARE of its 95th percentile;
# the labels are exchanged there when the exchanged velocities (slopes over IDENTITY_WINDOW frames each side) change
# under IDENTITY_RATIO as much as the kept ones (the vole's front bounces: 0.17 and 0.10; its crossings: 6.1 and more).
MEET_SHARE = 0.25
IDENTITY_WINDOW = 3
IDENTITY_RATIO = 0.5
# The periodic prior's per-joint weight: 0 at or below INCONSISTENT_LOW (the joint repeats: the silhouette pins it),
# 1 at or above INCONSISTENT_HIGH (it varies from cycle to cycle about as much as over the whole clip).
INCONSISTENT_LOW, INCONSISTENT_HIGH = 0.3, 0.8


# -------------------------------------------------------------------------------------------- the masks' period
def aligned(masks):
    """The masks with each frame's subject moved to a common column centroid (the animal may drift across the
    frame), cropped to the rows and columns any frame uses: (flat (T, K) bool, area (T,))."""
    masks = np.asarray(masks, bool)
    count = len(masks)
    centres, extents = [], []
    for m in masks:
        cols = np.nonzero(m.any(axis=0))[0]
        if not len(cols):
            centres.append(m.shape[1] // 2)
            extents.append(0)
            continue
        c = int(round(float(np.nonzero(m)[1].mean())))
        centres.append(c)
        extents.append(max(c - int(cols[0]), int(cols[-1]) - c))
    half = max(extents) + 1
    rows = np.nonzero(masks.any(axis=(0, 2)))[0]
    top, bottom = (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, 1)
    out = np.zeros((count, bottom - top, 2 * half + 1), bool)
    for f, (m, c) in enumerate(zip(masks, centres)):
        lo, hi = max(c - half, 0), min(c + half + 1, m.shape[1])
        out[f, :, lo - (c - half):hi - (c - half)] = m[top:bottom, lo:hi]
    flat = out.reshape(count, -1)
    return flat, flat.sum(axis=1)


def pair_distance(flat, area, a, b):
    """1 - IoU of the aligned frames a[i] and b[i] (index arrays)."""
    inter = np.count_nonzero(flat[a] & flat[b], axis=1)
    union = area[a] + area[b] - inter
    return np.where(union > 0, 1.0 - inter / np.maximum(union, 1), 0.0)


def lag_score(flat, area, lag):
    """The median, over frame pairs a lag apart, of their aligned 1 - IoU. The median, not the mean: an AI video
    can hold a minority of time-displaced frames (the real lope has one every fourth frame), and the mean of their
    larger distances drags the parabola's vertex off the period."""
    t = np.arange(len(flat) - lag)
    return float(np.median(pair_distance(flat, area, t, t + lag)))


def refine(scores, lag):
    """The parabola through a whole-frame lag and its neighbours: (sub-frame lag, score at the vertex)."""
    before, here, after = scores.get(lag - 1, np.inf), scores[lag], scores.get(lag + 1, np.inf)
    curve = before - 2 * here + after
    if not (np.isfinite(curve) and curve > 0):
        return float(lag), float(here)
    offset = 0.5 * (before - after) / curve
    return float(lag + min(0.5, max(-0.5, offset))), max(0.0, float(here - (before - after) ** 2 / (8 * curve)))


def multiples(flat, area, period, high):
    """The period refined over its multiples: the parabola vertex (`refine`) at each whole multiple k of it that
    fits in the lag range, fitted through the origin (sum k L_k / sum k^2). The 1 - IoU curve is V-shaped, not
    parabolic, so one parabola pulls a sub-frame period toward the nearest whole frame (a toy 15.2 reads 15.06); at
    k periods that error is shared by k (measured: 15.2 -> 15.20, 12.4 -> 12.42, 9.6 -> 9.61, 7.2 -> 7.20)."""
    num = den = 0.0
    k = 1
    while k * period + 1 <= high:
        centre = int(round(k * period))
        scores = {lag: lag_score(flat, area, lag) for lag in range(centre - 2, centre + 3) if 0 < lag < len(flat)}
        near = [lag for lag in (centre - 1, centre, centre + 1) if lag - 1 in scores and lag + 1 in scores]
        if not near:
            break
        lag = min(near, key=scores.get)
        num += k * refine(scores, lag)[0]
        den += k * k
        k += 1
    return num / den if den else period


def prominence(scores, lag):
    """How far a whole-frame minimum dips below its neighbourhood: on each side, the highest score before the curve
    first falls under it (or the scored range ends); the lower of the two sides, less the minimum's score."""
    here = scores[lag]

    def side(step):
        peak, g = here, lag + step
        while g in scores and scores[g] >= here:
            peak = max(peak, scores[g])
            g += step
        return peak
    return float(min(side(-1), side(1)) - here)


def period_from_scores(scores, low, high):
    """The fundamental lag from the lag scores (lag -> median aligned 1 - IoU, over [low - 1, high + 1]).

    Vole round 3: every whole-frame local minimum in [low, high] is refined (`refine`) and given its prominence
    (`prominence`). A candidate scores at most MASK_PERIOD_MAX and dips by at least MASK_PERIOD_PROMINENCE of the
    curve's level (the median score in range): a low, rigid body (the field vole changes its outline by at most 17 %)
    scores every lag under the gate, and a 0.003-deep noise dip must not pass as the period. A candidate is the period
    when, against the median of the lags more than a frame from any multiple of it, the best score of it and its
    multiples has a margin of MASK_PERIOD_MARGIN; every multiple in range also dips under that median (the gait
    repeats there too); and no other minimum comes within MASK_PERIOD_RIVAL of that best score (a rival dip means no
    lag is clearly the period). The first such candidate is chosen: the fundamental, when its multiples dip as well
    (a fast gait between whole frames can score worse at its period than at twice it).

    Returns {"chosen", "fundamentalLag", "score", "familyScore", "median", "margin", "prominence", "level", "rival"
    (the deepest other minimum's score, None without one), "rivalRatio", "candidates" (every minimum: lag, score,
    prominence), the gates, "reason"}; without a period the numbers are the best candidate's."""
    minima = []
    for lag in range(low, high + 1):
        if scores[lag] <= scores.get(lag - 1, np.inf) and scores[lag] <= scores.get(lag + 1, np.inf):
            refined, score = refine(scores, lag)
            minima.append({"lag": refined, "score": score, "prominence": prominence(scores, lag)})
    level = float(np.median([scores[g] for g in range(low, high + 1)]))
    out = {"chosen": False, "fundamentalLag": None, "score": None, "familyScore": None, "median": None,
           "margin": None, "prominence": None, "level": level, "rival": None, "rivalRatio": None,
           "candidates": [{k: round(v, 4) for k, v in m.items()} for m in minima],
           "gate": MASK_PERIOD_MAX, "marginGate": MASK_PERIOD_MARGIN, "prominenceGate": MASK_PERIOD_PROMINENCE,
           "rivalGate": MASK_PERIOD_RIVAL, "reason": None}
    if not minima:
        out["reason"] = "the masks' self-similarity has no minimum in range"
        return out

    def judge(m):
        lag = m["lag"]

        def near(x):
            return min(abs(x - k * lag) for k in range(1, int(high // max(lag, 1.0)) + 2)) <= 1.0
        family = min(o["score"] for o in minima if near(o["lag"]))
        others = [scores[g] for g in range(low, high + 1) if not near(g)]
        median = float(np.median(others)) if others else 0.0
        rivals = [o["score"] for o in minima if not near(o["lag"])]
        rival = min(rivals) if rivals else None
        grid = np.arange(low - 1, high + 2)
        values = np.array([scores.get(int(g), np.nan) for g in grid])
        repeats = all(float(np.interp(k * lag, grid, values)) < median
                      for k in range(2, int(high // max(lag, 1.0)) + 1) if k * lag <= high)
        verdict = {"fundamentalLag": lag, "score": m["score"], "familyScore": family, "median": median,
                   "margin": median / max(family, 1e-6), "prominence": m["prominence"], "rival": rival,
                   "rivalRatio": None if rival is None else rival / max(family, 1e-6), "repeats": repeats}
        verdict["ok"] = (verdict["margin"] >= MASK_PERIOD_MARGIN and repeats
                         and (rival is None or verdict["rivalRatio"] >= MASK_PERIOD_RIVAL))
        return verdict
    gated = [m for m in minima if m["score"] <= MASK_PERIOD_MAX and m["prominence"] >= MASK_PERIOD_PROMINENCE * level]
    judged = [judge(m) for m in gated]
    chosen = next((j for j in judged if j["ok"]), None)
    if chosen is not None:
        out.update({k: v for k, v in chosen.items() if k not in ("ok", "repeats")}, chosen=True)
        return out
    best = max(judged, key=lambda j: j["margin"]) if judged else judge(min(minima, key=lambda m: m["score"]))
    out.update({k: v for k, v in best.items() if k not in ("ok", "repeats")})
    why = []
    if not judged:
        why.append(f"no dip scores under {MASK_PERIOD_MAX} and dips by {MASK_PERIOD_PROMINENCE} of the curve's level "
                   f"{level:.2f} (the best dips {best['prominence']:.3f})")
    else:
        if best["margin"] < MASK_PERIOD_MARGIN:
            why.append(f"margin {best['margin']:.1f}x, needed {MASK_PERIOD_MARGIN}x")
        if not best["repeats"]:
            why.append("its multiples do not dip under that median")
        if best["rival"] is not None and best["rivalRatio"] < MASK_PERIOD_RIVAL:
            why.append(f"another dip scores {best['rival']:.3f} ({best['rivalRatio']:.2f}x, needed "
                       f"{MASK_PERIOD_RIVAL}x)")
    out["reason"] = (f"no lag is clearly the period: the best candidate, {best['fundamentalLag']:.2f} frames, scores "
                     f"{best['score']:.2f} (gate {MASK_PERIOD_MAX}; {best['familyScore']:.2f} with its multiples) "
                     f"against a median of {best['median']:.2f} elsewhere ({'; '.join(why)})")
    return out


def mask_period(masks, fps, min_s=0.25, max_s=2.5):
    """The video's fundamental period in frames, from the masks' self-similarity over lags in [min_s, max_s].

    Each whole-frame lag scores the median aligned 1 - IoU between frames that far apart (`lag_score`); the
    fundamental is chosen from that curve's clear minima (`period_from_scores`) and refined over its multiples
    (`multiples`).

    Returns {"period" (None when nothing passes), "fundamentalLag", "score" (its vertex score), "familyScore" (the
    best over it and its multiples), "median", "margin", "prominence", "level", "rival", "rivalRatio", the gates,
    "reason"}; without a period, the numbers are the best candidate's."""
    count = len(masks)
    low, high = max(2, int(math.ceil(min_s * fps))), min(int(max_s * fps), count // 2)
    out = {"period": None, "fundamentalLag": None, "score": None, "familyScore": None, "median": None,
           "margin": None, "prominence": None, "level": None, "rival": None, "rivalRatio": None,
           "gate": MASK_PERIOD_MAX, "marginGate": MASK_PERIOD_MARGIN, "prominenceGate": MASK_PERIOD_PROMINENCE,
           "rivalGate": MASK_PERIOD_RIVAL, "reason": None}
    if high < low + 1:
        out["reason"] = f"the clip ({count} frames) is too short to hold two cycles"
        return out
    flat, area = aligned(masks)
    scores = {lag: lag_score(flat, area, lag) for lag in range(low - 1, high + 2) if 0 < lag < count}
    found = period_from_scores(scores, low, high)
    out.update({k: v for k, v in found.items() if k not in ("chosen", "candidates")})
    if found["chosen"]:
        out["period"] = multiples(flat, area, found["fundamentalLag"], high)
    return out


def frame_steadiness(masks, period):
    """Per frame, its aligned 1 - IoU against the frame a period later (interpolated between the two whole-frame
    lags around it), or a period earlier when the clip ends first; NaN when neither exists."""
    flat, area = aligned(masks)
    count = len(flat)
    lo, hi = int(math.floor(period)), int(math.ceil(period))
    w = period - lo
    out = np.full(count, np.nan)
    for f in range(count):
        for sign in (1, -1):
            a, b = f + sign * lo, f + sign * hi
            if 0 <= a < count and 0 <= b < count:
                da = pair_distance(flat, area, np.array([f]), np.array([a]))[0]
                db = pair_distance(flat, area, np.array([f]), np.array([b]))[0]
                out[f] = (1 - w) * da + w * db
                break
    return out


def steady_cycles(masks, period, first):
    """The cycles on the grid first + k * period that fit in the clip, and the steady ones among them: a cycle's
    score is the mean steadiness of its frames (every cycle holds its share of any odd frames, so the mean compares
    cycles fairly; real lope: warm-up 0.33 and 0.20, steady 0.11-0.16), and a steady cycle scores within STEADY_RATIO (or STEADY_SLACK)
    of the steadiest. Returns {"starts", "scores" (per start), "kept", "ref" (the steadiest start)}."""
    count = len(masks)
    starts = []
    s = float(first)
    while s + period <= count - 1 + 1e-9:
        starts.append(round(s, 6))
        s += period
    if not starts:
        return {"starts": [], "scores": {}, "kept": [], "ref": None}
    steady = frame_steadiness(masks, period)
    scores = {}
    for s in starts:
        frames = [f for f in range(int(math.ceil(s - 1e-9)), count) if f < s + period - 1e-9]
        values = steady[frames]
        values = values[np.isfinite(values)]
        scores[s] = float(np.mean(values)) if len(values) else np.inf
    best = min(scores.values())
    limit = max(STEADY_RATIO * best, best + STEADY_SLACK)
    kept = [s for s in starts if scores[s] <= limit]
    return {"starts": starts, "scores": scores, "kept": kept, "ref": min(kept, key=scores.get)}


def cycle_periods(masks, starts, period, spread=0.2):
    """Each cycle's own period (Task 19, the cadence's one-time-scale check): for the frames of the cycle starting at
    s, the whole-frame lag within `spread` of `period` whose median aligned 1 - IoU against the frame that lag later
    (earlier, when the clip ends first) is lowest, refined to a sub-frame lag (`refine`)."""
    flat, area = aligned(masks)
    count = len(flat)
    low, high = max(2, int(math.floor(period * (1 - spread)))), int(math.ceil(period * (1 + spread)))
    out = []
    for s in starts:
        frames = np.array([f for f in range(int(math.ceil(s - 1e-9)), count) if f < s + period - 1e-9])
        if not len(frames):
            continue
        scores = {}
        for lag in range(low - 1, high + 2):
            later = frames[frames + lag < count]
            if len(later) >= max(1, len(frames) // 2):
                scores[lag] = float(np.median(pair_distance(flat, area, later, later + lag)))
                continue
            earlier = frames[frames - lag >= 0]
            if len(earlier):
                scores[lag] = float(np.median(pair_distance(flat, area, earlier, earlier - lag)))
        inside = [lag for lag in range(low, high + 1) if lag in scores]
        if inside:
            out.append(refine(scores, min(inside, key=scores.get))[0])
    return out


# ------------------------------------------------------------------------------------------------- leg chains
def leg_chains(project):
    return [c for pair in project.get("legPairs", []) for c in pair]


def leg_pairs(names, project):
    """Per leg pair, the value columns of its two chains' joints: [(columns a, columns b)] (equal lengths)."""
    out = []
    for a, b in project.get("legPairs", []):
        ia = [names.index(x) for x in project["chains"].get(a, []) if x in names]
        ib = [names.index(x) for x in project["chains"].get(b, []) if x in names]
        if ia and len(ia) == len(ib):
            out.append((ia, ib))
    return out


def leg_joints(names, project):
    """The joints a side silhouette observes best, by chain role: in a leg that carries a girdle bone (the longer
    leg chains, a foreleg's scapula), the two bones above the paw (forearm and wrist); in the other legs the first
    two (thigh and shin). With every leg chain the same length, the first two bones of each. The paw (the chain's
    `paws` bone, or its last bone) is never one of them."""
    chains = leg_chains(project)
    paws = project.get("paws", {})

    def body(chain):
        bones = [b for b in project["chains"].get(chain, []) if b in names]
        paw = paws.get(chain) or (bones[-1] if bones else None)
        return [b for b in bones if b != paw] if paw in bones else bones
    lengths = {c: len(project["chains"].get(c, [])) for c in chains}
    girdled = len(set(lengths.values())) > 1
    out = []
    for chain in chains:
        bones = body(chain)
        if girdled and lengths[chain] == max(lengths.values()):
            out += bones[-2:]
        else:
            out += bones[:2]
    return out


def half_period(values, pairs, period):
    """Whether `period` is half the gait's: a side silhouette cannot tell left legs from right, so a gait whose
    pairs move half a cycle apart repeats in the video every half period. Then each leg matches its partner, not
    itself, a period on: true when the exchanged comparison is under HALF_PERIOD_RATIO of the direct one."""
    if not pairs:
        return False
    values = np.asarray(values, float)
    times = np.arange(0, len(values) - 1 - period)
    if not len(times):
        return False
    now, later = values[times.astype(int)], resample(values, times + period)
    direct = sum(float(np.sum((now[:, a] - later[:, a]) ** 2) + np.sum((now[:, b] - later[:, b]) ** 2))
                 for a, b in pairs)
    crossed = sum(float(np.sum((now[:, a] - later[:, b]) ** 2) + np.sum((now[:, b] - later[:, a]) ** 2))
                  for a, b in pairs)
    return crossed < HALF_PERIOD_RATIO * direct


def pair_verdicts(values, pairs, period):
    """Per leg pair, the half-period test on its own (`half_period` sums the pairs): {"direct", "crossed",
    "verdict"}, the verdict "half" when the pair matches its partner a period on (crossed under HALF_PERIOD_RATIO of
    direct: the period is half the stride), "whole" when it matches itself (the reverse), else "mixed" (neither: the
    pair's identity is mixed along the clip)."""
    values = np.asarray(values, float)
    times = np.arange(0, len(values) - 1 - period)
    out = []
    if not len(times):
        return [{"direct": None, "crossed": None, "verdict": "mixed"} for _ in pairs]
    now, later = values[times.astype(int)], resample(values, times + period)
    for a, b in pairs:
        direct = float(np.sum((now[:, a] - later[:, a]) ** 2) + np.sum((now[:, b] - later[:, b]) ** 2))
        crossed = float(np.sum((now[:, a] - later[:, b]) ** 2) + np.sum((now[:, b] - later[:, a]) ** 2))
        verdict = ("half" if crossed < HALF_PERIOD_RATIO * direct else
                   "whole" if direct < HALF_PERIOD_RATIO * crossed else "mixed")
        out.append({"direct": direct, "crossed": crossed, "verdict": verdict})
    return out


def _slope(x, lo, hi):
    lo, hi = max(lo, 0), min(hi, len(x) - 1)
    if hi - lo < 1:
        return None
    t = np.arange(lo, hi + 1)
    return float(np.polyfit(t, x[lo:hi + 1], 1)[0])


def resolve_identity(xa, xb, window=IDENTITY_WINDOW, gap=1):
    """A leg pair's identity across the clip from its paws' in-place x tracks (vole round 3). Where the two paws meet
    (a local minimum of their distance under MEET_SHARE of its 95th percentile), a side silhouette shows one leg and
    the fit may bounce the paws back instead of letting them cross. The paws keep their velocities through a meeting
    (one swings forward, the other sweeps back on the ground), so at each meeting, in order, the labels are exchanged
    from the frame after it when the exchanged velocities (slopes over `window` frames each side, `gap` frames from
    the meeting) carry on under IDENTITY_RATIO of the kept ones' squared change.

    Returns {"flip" (per frame, the labels exchanged), "boundaries" (frames where the exchange starts or ends),
    "meetings" ([{"frame", "distance", "kept", "exchanged", "swap"}])}."""
    xa, xb = np.array(xa, float), np.array(xb, float)
    count = len(xa)
    flip = np.zeros(count, bool)
    distance = np.abs(xa - xb)
    scale = float(np.percentile(distance, 95)) if count else 0.0
    meetings, last = [], -np.inf
    if scale <= 1e-9:
        return {"flip": flip, "boundaries": [], "meetings": []}
    for m in range(1, count - 1):
        if not (distance[m] <= distance[m - 1] and distance[m] <= distance[m + 1] and distance[m] < MEET_SHARE * scale):
            continue
        if m - last <= window:
            continue
        last = m
        b = m + 1
        before = (b - 1 - gap - window, b - 1 - gap)
        after = (b + gap, b + gap + window)
        va0, vb0, va1, vb1 = _slope(xa, *before), _slope(xb, *before), _slope(xa, *after), _slope(xb, *after)
        if None in (va0, vb0, va1, vb1):
            continue
        kept = (va0 - va1) ** 2 + (vb0 - vb1) ** 2
        exchanged = (va0 - vb1) ** 2 + (vb0 - va1) ** 2
        swap = exchanged < IDENTITY_RATIO * kept
        meetings.append({"frame": m, "distance": round(float(distance[m]), 6), "kept": kept, "exchanged": exchanged,
                         "swap": bool(swap)})
        if swap:
            xa[b:], xb[b:] = xb[b:].copy(), xa[b:].copy()
            flip[b:] = ~flip[b:]
    boundaries = [int(f) for f in np.nonzero(np.diff(flip.astype(int)))[0] + 1]
    if flip[0]:
        boundaries.insert(0, 0)
    return {"flip": flip, "boundaries": boundaries, "meetings": meetings}


def relabel(values, pairs, flips):
    """The values with each pair's columns exchanged on the frames its flip mask marks."""
    out = np.array(values, float)
    for (a, b), flip in zip(pairs, flips):
        rows = np.nonzero(np.asarray(flip, bool))[0]
        if len(rows):
            x = out[np.ix_(rows, a)].copy()
            out[np.ix_(rows, a)] = out[np.ix_(rows, b)]
            out[np.ix_(rows, b)] = x
    return out


# --------------------------------------------------------------------------------------------- the phase average
def resample(series, times):
    """Linear interpolation in frame time of every column of `series` (T, K) at `times`."""
    frame = np.arange(len(series))
    return np.stack([np.interp(times, frame, series[:, k]) for k in range(series.shape[1])], axis=1)


def cycle_samples(series, starts, period, n):
    """Each cycle's n + 1 phase samples, s + i * period / n for i = 0..n: (cycles, n + 1, K)."""
    series = np.asarray(series, float)
    return np.array([resample(series, s + np.arange(n + 1) * period / n) for s in starts])


def align_identity(samples, swaps, ref):
    """A leg pair may come out exchanged from one cycle to the next (a side silhouette cannot tell its legs apart):
    per cycle and pair, exchange the pair's columns when that is closer to the reference (cycle `ref` first, then
    the mean of the aligned cycles). swaps: [(columns a, columns b)], each pair's columns to exchange together.
    Returns (aligned samples, per cycle the per-pair exchanged flags)."""
    out = np.array(samples, float)
    flags = [[False] * len(swaps) for _ in range(len(out))]
    for step in range(2):
        target = out[ref] if step == 0 else out.mean(axis=0)
        for c in range(len(out)):
            for p, (a, b) in enumerate(swaps):
                same = float(np.sum((out[c][:, a] - target[:, a]) ** 2) + np.sum((out[c][:, b] - target[:, b]) ** 2))
                cross = float(np.sum((out[c][:, b] - target[:, a]) ** 2) + np.sum((out[c][:, a] - target[:, b]) ** 2))
                if cross < same:
                    x = out[c].copy()
                    out[c][:, a], out[c][:, b] = x[:, b], x[:, a]
                    flags[c][p] = not flags[c][p]
    return out, flags


def translation(samples, kinds, ref, anchor=(0, 1)):
    """Per cycle, the offset that moves it so its root at phase 0 sits where cycle `ref`'s does: (cycles, K), zero
    in angle columns, the root dx difference in "x" columns and dz in "z" columns (anchor: the dx, dz columns)."""
    shift = np.zeros((len(samples), samples.shape[2]))
    for k, kind in enumerate(kinds):
        if kind in ("x", "z"):
            col = anchor[0 if kind == "x" else 1]
            shift[:, k] = samples[:, 0, col] - samples[ref, 0, col]
    return shift


def phase_average(samples, kinds, ref, anchor=(0, 1)):
    """The phase-aligned average of the cycles: angle columns by their circular mean, "x"/"z" columns by the
    mean after moving each cycle to the reference cycle's root (`translation`). (n + 1, K)."""
    moved = samples - translation(samples, kinds, ref, anchor)[:, None, :]
    avg = moved.mean(axis=0)
    for k, kind in enumerate(kinds):
        if kind == "angle":
            avg[:, k] = np.arctan2(np.sin(samples[:, :, k]).mean(axis=0), np.cos(samples[:, :, k]).mean(axis=0))
    return avg


def roll(avg, t, kinds):
    """The average started t samples later: past the seam, "x"/"z" columns continue by one cycle's travel
    (avg[n] - avg[0]) and angles wrap. (n + 1, K)."""
    n = len(avg) - 1
    travel = np.array([avg[n, k] - avg[0, k] if kind in ("x", "z") else 0.0 for k, kind in enumerate(kinds)])
    ext = np.concatenate([avg[:n], avg[:n + 1] + travel])
    return ext[t:t + n + 1].copy()


def at_phase(avg, phase):
    """The average at a (real) phase sample in [0, n]: linear interpolation of every column."""
    n = len(avg) - 1
    return np.array([np.interp(phase, np.arange(n + 1), avg[:, k]) for k in range(avg.shape[1])])


def inconsistency(samples, columns):
    """Per column, how much it varies from cycle to cycle at the same phase relative to how much it varies over
    the whole clip: the mean over phases of its variance across cycles over its total variance. ~0 for a joint
    that repeats, ~1 for one that does not (0 for a column that never moves)."""
    out = []
    for k in columns:
        x = samples[:, :-1, k]
        total = float(np.var(x))
        out.append(float(np.mean(np.var(x, axis=0))) / total if total > 1e-10 else 0.0)
    return np.array(out)


def periodic_prior(values, masks, fps, pairs, first):
    """The fitter's second-pass prior from a first pass's values (T, N: dx, dz, joints): the video's period
    (`mask_period`), the phase average of the steady cycles (identity aligned, angles by circular mean) and, per
    frame, that average at the frame's phase (with each leg pair labelled as the frame has it), and per column a
    weight in [0, 1] that grows with the column's cycle-to-cycle inconsistency (0 for the root). None without a
    period or a steady cycle."""
    values = np.asarray(values, float)
    found = mask_period(masks, fps)
    if found["period"] is None:
        return None
    period = found["period"]
    if half_period(values, pairs, period):
        period *= 2
    n = max(2, int(round(period)))
    steady = steady_cycles(masks, period, first)
    if not steady["kept"]:
        return None
    kept = steady["kept"]
    kinds = ["x", "z"] + ["angle"] * (values.shape[1] - 2)
    samples = cycle_samples(values, kept, period, n)
    samples, _ = align_identity(samples, pairs, kept.index(steady["ref"]))
    avg = phase_average(samples, kinds, kept.index(steady["ref"]))
    weights = np.zeros(values.shape[1])
    joints = list(range(2, values.shape[1]))
    spread = inconsistency(samples, joints)
    weights[2:] = np.clip((spread - INCONSISTENT_LOW) / (INCONSISTENT_HIGH - INCONSISTENT_LOW), 0.0, 1.0)
    targets = values.copy()
    for f in range(len(values)):
        target = at_phase(avg, ((f - kept[0]) / period % 1.0) * n)
        for a, b in pairs:
            same = float(np.sum((values[f, a] - target[a]) ** 2) + np.sum((values[f, b] - target[b]) ** 2))
            cross = float(np.sum((values[f, a] - target[b]) ** 2) + np.sum((values[f, b] - target[a]) ** 2))
            if cross < same:
                target[a], target[b] = target[b].copy(), target[a].copy()
        targets[f, 2:] = target[2:]
    return {"period": period, "n": n, "kept": kept, "targets": targets, "weights": weights,
            "inconsistency": spread, "mask": found}
