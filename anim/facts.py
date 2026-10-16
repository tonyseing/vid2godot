"""Clip facts from the fitted poses: one clean loop, paw contacts, ground speed and stride, the sidecar entry."""
import numpy as np

from anim.ledger import StageError

LEAD_IN = 0.10
PERIOD_MAX_SCORE = 0.2
# Cycles whose normalised difference from their neighbour cycles is within this of the steadiest are equally steady.
STEADY_TOL = 0.01
# A cycle moved back whole periods (to start at the phase origin's touchdown) may be at most this much less steady
# (normalised score) than the steadiest: more means it reaches back into the warm-up.
WRAP_TOL = 0.05


class FactsError(StageError):
    pass


def period_search(values, fps, min_s=0.25, max_s=2.5):
    """The fundamental period in frames, at sub-frame precision, and its score: {"period": P or None, "score": s}.

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


def period_frames(values, fps, min_s=0.25, max_s=2.5):
    """The fundamental period in frames (see `period_search`); FactsError when the clip holds no full cycle."""
    found = period_search(values, fps, min_s, max_s)
    if found["period"] is None:
        raise FactsError(f"no full cycle found between {min_s} s and {max_s} s (best period score "
                         f"{found['score']:.2f}, gate {PERIOD_MAX_SCORE}); the clip must hold at least two cycles")
    return found["period"]


def resample(series, times):
    """Linear interpolation in frame time of every column of `series` (T, K) at `times`."""
    frame = np.arange(len(series))
    return np.stack([np.interp(times, frame, series[:, k]) for k in range(series.shape[1])], axis=1)


def best_start(values, period):
    """The whole frame after the lead-in whose pose best matches the pose one (sub-frame) period later."""
    first = int(np.ceil(LEAD_IN * len(values)))
    candidates = range(first, int(np.floor(len(values) - 1 - period)) + 1)
    if not candidates:
        raise FactsError("No full cycle after the lead-in; generate a longer video")
    later = resample(values, np.array(candidates, float) + period)
    return min(candidates, key=lambda s: float(np.sum((values[s] - later[s - first]) ** 2)))


def steadiness(values, period, start):
    """How far the cycle starting at (real) frame `start` is, phase for phase (n samples per period), from the cycle
    one period later and the one a period earlier, wherever the clip holds them: the mean squared difference,
    normalised like the period score (by twice the mean total variance). None when the clip holds neither."""
    n = max(2, int(round(period)))
    phase = np.arange(n) * period / n
    here = resample(values, start + phase)
    terms = [float(np.mean(np.sum((here - resample(values, start + shift + phase)) ** 2, axis=1)))
             for shift in (period, -period) if 0 <= start + shift and start + shift + period <= len(values) - 1]
    if not terms:
        return None
    x = values - values.mean(axis=0)
    return float(np.mean(terms)) / (2 * float(np.mean(np.sum(x ** 2, axis=1))) or 1.0)


def steadiest_start(values, period):
    """The whole frame after the lead-in that starts the steadiest full cycle.

    An AI video starts from the reference pose and settles over a cycle or two, and a small warm-up swing can match
    its own next swing better than a full one does, so the seam alone (`best_start`) can pick the warm-up. Each start
    is scored instead by its `steadiness`. Only starts with a neighbouring cycle are scored; a clip too short for any
    falls back to the seam. Among the starts within STEADY_TOL of the steadiest, the cleanest seam wins (as
    `best_start`), so a clip that repeats exactly keeps the seam choice."""
    first, length = int(np.ceil(LEAD_IN * len(values))), len(values)
    candidates = list(range(first, int(np.floor(length - 1 - period)) + 1))
    if not candidates:
        raise FactsError("No full cycle after the lead-in; generate a longer video")
    scores = {s: v for s in candidates if (v := steadiness(values, period, s)) is not None}
    if not scores:
        return best_start(values, period)
    low = min(scores.values())
    steady = [s for s, v in scores.items() if v <= low + STEADY_TOL]
    later = resample(values, np.array(steady, float) + period)
    return min(steady, key=lambda s: float(np.sum((values[s] - later[steady.index(s)]) ** 2)))


def contact_mask(z, tol=0.025):
    return z <= z.min() + tol


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


def ground_speed(paw_x, masks, fps):
    velocities = []
    for chain, x in paw_x.items():
        v = np.gradient(x) * fps
        velocities.extend(v[masks[chain]])
    if not velocities:
        raise FactsError("No paw is ever planted; the fit or the video is wrong")
    return float(-np.median(velocities))


def clip_facts(poses, project, brief):
    fps = float(poses["fps"])
    frames = poses["frames"]
    every = np.array([f["values"] for f in frames], float)
    values = every[:, 2:]
    chains = list(project["paws"])
    paws = {c: np.array([f["paws"][c] for f in frames], float) for c in chains}
    warnings = []
    if brief["loop"]:
        period = period_frames(values, fps)
        n = int(round(period))
        first, last = int(np.ceil(LEAD_IN * len(frames))), len(frames) - 1 - period
        origin = project.get("phaseOrigin", chains[0])

        def cycle_paws(begin):
            times = begin + np.arange(n) * period / n
            return {c: resample(paws[c], times) for c in chains}

        start = float(steadiest_start(values, period))
        cycle = cycle_paws(start)
        masks = {c: contact_mask(cycle[c][:, 1]) for c in chains}
        touchdown = int(round(contact_window(masks[origin])[0][0] * n))
        if touchdown:
            steadiest = start
            start += touchdown * period / n
            if start > last:
                start -= float(np.ceil((start - last) / period)) * period
                if start >= first:
                    # Moving back whole periods must not land in the warm-up: the moved cycle must be about as
                    # steady as the steadiest one.
                    moved, best = steadiness(values, period, start), steadiness(values, period, steadiest)
                    if moved is not None and best is not None and moved > best + WRAP_TOL:
                        raise FactsError(
                            f"No steady cycle that starts at the {origin} touchdown fits in the clip: the steadiest "
                            f"cycle starts at frame {steadiest}, its touchdown falls too late, and the cycle a period "
                            f"earlier (from frame {start:.1f}) is not steady (score {moved:.2f} vs {best:.2f}); "
                            f"generate a longer video")
            if start < first:
                raise FactsError(f"No full cycle that starts at the {origin} touchdown fits after the lead-in "
                                 f"(the first {first} frames); generate a longer video")
            cycle = cycle_paws(start)
            masks = {c: contact_mask(cycle[c][:, 1]) for c in chains}
        rate = n * fps / period
        speed = ground_speed({c: cycle[c][:, 0] for c in chains}, masks, rate)
        contacts, phase, duty = {}, {}, {}
        for c in chains:
            window, fraction = contact_window(masks[c])
            bone = project["paws"][c]
            contacts[bone], phase[bone], duty[bone] = window, window[0], round(fraction, 4)
        entry = {"duration_s": period / fps, "loop": True, "authored_speed_mps": round(speed, 3),
                 "stride_m": round(speed * period / fps, 3), "contacts": contacts, "frames": n, "fps": rate,
                 "footfall_phase": phase, "duty_factor": duty}
        keys = resample(every, start + np.arange(n + 1) * period / n)
        # The root's travel over one period (the resampled dx, dz at s + P less those at s), taken before the closing
        # key is overwritten: the bake removes drift * k / n, so a periodic sway is left untouched.
        drift = {"dx": float(keys[-1][0] - keys[0][0]), "dz": float(keys[-1][1] - keys[0][1])}
        keys[-1] = keys[0]
        cycle_out = {"frames": keys.tolist(), "fps": rate, "periodFrames": period, "start": start, "drift": drift}
    else:
        start, period = 0, len(frames) - 1
        masks = {c: contact_mask(paws[c][:, 1]) for c in chains}
        speed = ground_speed({c: paws[c][:, 0] for c in chains}, masks, fps)
        all_windows = {project["paws"][c]: windows(masks[c]) for c in chains}
        entry = {"duration_s": period / fps, "loop": False, "authored_speed_mps": round(speed, 3),
                 "stride_m": 0.0, "contacts": {b: w[0] if w else [0.0, 0.0] for b, w in all_windows.items()},
                 "frames": period, "fps": fps, "contact_windows": all_windows}
        cycle_out = {"frames": every.tolist(), "fps": fps, "periodFrames": period, "start": start,
                     "drift": {"dx": 0.0, "dz": 0.0}}
    target = brief.get("targetSpeedMps")
    if target and abs(speed - target) / target > 0.25:
        warnings.append(f"Measured ground speed {speed:.2f} m/s is more than 25% from the brief's {target} m/s")
    return {"start": start, "period": period, "fps": fps, "speedMps": speed, "strideM": entry["stride_m"],
            "entry": entry, "warnings": warnings, "cycle": cycle_out}
