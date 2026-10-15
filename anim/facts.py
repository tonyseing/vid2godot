"""Clip facts from the fitted poses: one clean loop, paw contacts, ground speed and stride, the sidecar entry."""
import numpy as np

from anim.ledger import StageError

LEAD_IN = 0.10
PERIOD_MAX_SCORE = 0.2


class FactsError(StageError):
    pass


def period_frames(values, fps, min_s=0.25, max_s=2.5):
    """The fundamental period in frames, at sub-frame precision.

    The lag score (mean squared difference between the clip and itself shifted by the lag) is normalised by twice the
    mean total variance, so 0 is a perfect repeat and about 1 is no relation. A parabola through each whole-frame local
    minimum and its neighbours gives the sub-frame period and the score there; the first local minimum in range whose
    refined score is below PERIOD_MAX_SCORE is the fundamental (its multiples come later). Gating on the refined score,
    not the whole-frame one, keeps a short period that falls between frames from losing to its double."""
    x = values - values.mean(axis=0)
    low, high = max(2, int(np.ceil(min_s * fps))), min(int(max_s * fps), len(x) // 2)
    if high < low + 1:
        raise FactsError(f"The clip ({len(x)} frames) is too short to hold two cycles")
    variance = float(np.mean(np.sum(x ** 2, axis=1)))
    scores = {lag: float(np.mean(np.sum((x[:-lag] - x[lag:]) ** 2, axis=1))) / (2 * variance) if variance else np.inf
              for lag in range(low - 1, high + 2) if 0 < lag < len(x)}
    for lag in range(low, high + 1):
        before, here, after = scores.get(lag - 1, np.inf), scores[lag], scores.get(lag + 1, np.inf)
        if here <= before and here <= after:
            curve = before - 2 * here + after
            fitted = np.isfinite(curve) and curve > 0
            vertex = here - (before - after) ** 2 / (8 * curve) if fitted else here
            if vertex < PERIOD_MAX_SCORE:
                offset = 0.5 * (before - after) / curve if fitted else 0.0
                return float(lag + min(0.5, max(-0.5, offset)))
    raise FactsError(f"no full cycle found between {min_s} s and {max_s} s; the clip must hold at least two cycles")


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

        start = float(best_start(values, period))
        cycle = cycle_paws(start)
        masks = {c: contact_mask(cycle[c][:, 1]) for c in chains}
        touchdown = int(round(contact_window(masks[origin])[0][0] * n))
        if touchdown:
            start += touchdown * period / n
            if start > last:
                start -= float(np.ceil((start - last) / period)) * period
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
        keys[-1] = keys[0]
        cycle_out = {"frames": keys.tolist(), "fps": rate, "periodFrames": period, "start": start}
    else:
        start, period = 0, len(frames) - 1
        masks = {c: contact_mask(paws[c][:, 1]) for c in chains}
        speed = ground_speed({c: paws[c][:, 0] for c in chains}, masks, fps)
        all_windows = {project["paws"][c]: windows(masks[c]) for c in chains}
        entry = {"duration_s": period / fps, "loop": False, "authored_speed_mps": round(speed, 3),
                 "stride_m": 0.0, "contacts": {b: w[0] if w else [0.0, 0.0] for b, w in all_windows.items()},
                 "frames": period, "fps": fps, "contact_windows": all_windows}
        cycle_out = {"frames": every.tolist(), "fps": fps, "periodFrames": period, "start": start}
    target = brief.get("targetSpeedMps")
    if target and abs(speed - target) / target > 0.25:
        warnings.append(f"Measured ground speed {speed:.2f} m/s is more than 25% from the brief's {target} m/s")
    return {"start": start, "period": period, "fps": fps, "speedMps": speed, "strideM": entry["stride_m"],
            "entry": entry, "warnings": warnings, "cycle": cycle_out}
