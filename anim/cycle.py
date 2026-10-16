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
# A cycle is steady when its mask score (the mean 1 - IoU of its frames against the same phase a period on) is
# within STEADY_RATIO of the steadiest cycle's, or STEADY_SLACK of it (a clean clip scores ~0).
STEADY_RATIO = 1.5
STEADY_SLACK = 0.03
# Leg identity exchanged between cycles only when exchanging is clearly closer; the half-period test likewise.
HALF_PERIOD_RATIO = 0.5
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


def mask_period(masks, fps, min_s=0.25, max_s=2.5):
    """The video's fundamental period in frames, from the masks' self-similarity over lags in [min_s, max_s].

    Each whole-frame lag scores the median aligned 1 - IoU between frames that far apart (`lag_score`); each local
    minimum is refined to a sub-frame lag and its vertex score (`refine`). The fundamental is the first refined
    minimum whose score is at most MASK_PERIOD_MAX (its multiples come later). It must be clearly the period: the
    median of the other lags (every lag in range more than a frame from a multiple of it) is at least
    MASK_PERIOD_MARGIN times the best score among it and its multiples. (A fast gait whose period falls between
    whole frames scores worse at the period than at twice it; judging the margin on the fundamental alone would hand
    the period to its double.) The period is then refined over its multiples (`multiples`).

    Returns {"period" (None when nothing passes), "fundamentalLag" (the first minimum's refined lag), "score" (its
    vertex score), "familyScore" (the best over it and its multiples), "median", "margin", "gate", "marginGate",
    "reason"}; without a period, the numbers are the best candidate's."""
    count = len(masks)
    low, high = max(2, int(math.ceil(min_s * fps))), min(int(max_s * fps), count // 2)
    out = {"period": None, "fundamentalLag": None, "score": None, "familyScore": None, "median": None,
           "margin": None, "gate": MASK_PERIOD_MAX, "marginGate": MASK_PERIOD_MARGIN, "reason": None}
    if high < low + 1:
        out["reason"] = f"the clip ({count} frames) is too short to hold two cycles"
        return out
    flat, area = aligned(masks)
    scores = {lag: lag_score(flat, area, lag) for lag in range(low - 1, high + 2) if 0 < lag < count}
    minima = [refine(scores, lag) for lag in range(low, high + 1)
              if scores[lag] <= scores.get(lag - 1, np.inf) and scores[lag] <= scores.get(lag + 1, np.inf)]
    if not minima:
        out["reason"] = "the masks' self-similarity has no minimum in range"
        return out
    passing = [m for m in minima if m[1] <= MASK_PERIOD_MAX]
    lag, score = passing[0] if passing else min(minima, key=lambda m: m[1])

    def near_multiple(x):
        return min(abs(x - k * lag) for k in range(1, int(high // max(lag, 1.0)) + 2)) <= 1.0
    family = min(m[1] for m in minima if near_multiple(m[0]))
    others = [scores[g] for g in range(low, high + 1) if not near_multiple(g)]
    median = float(np.median(others)) if others else 0.0
    margin = median / max(family, 1e-6)
    out.update({"fundamentalLag": lag, "score": score, "familyScore": family, "median": median, "margin": margin})
    if passing and margin >= MASK_PERIOD_MARGIN:
        out["period"] = multiples(flat, area, lag, high)
        return out
    out["reason"] = (f"no lag is clearly the period: the best candidate, {lag:.2f} frames, scores {score:.2f} "
                     f"(gate {MASK_PERIOD_MAX}; {family:.2f} with its multiples) against a median of {median:.2f} "
                     f"elsewhere (margin {margin:.1f}x, needed {MASK_PERIOD_MARGIN}x)")
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
