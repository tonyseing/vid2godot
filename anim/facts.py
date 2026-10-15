"""Clip facts from the fitted poses: one clean loop, paw contacts, ground speed and stride, the sidecar entry."""
import numpy as np

from anim.ledger import StageError

LEAD_IN = 0.10


class FactsError(StageError):
    pass


def period_frames(values, fps, min_s=0.25, max_s=2.5):
    x = values - values.mean(axis=0)
    low, high = max(2, int(np.ceil(min_s * fps))), min(int(max_s * fps), len(x) // 2)
    if high < low + 1:
        raise FactsError(f"The clip ({len(x)} frames) is too short to hold two cycles")
    scores = {lag: float(np.mean(np.sum((x[:-lag] - x[lag:]) ** 2, axis=1))) for lag in range(low - 1, high + 2)
              if 0 < lag < len(x)}
    best = min(scores[lag] for lag in range(low, high + 1))
    minima = [lag for lag in range(low, high + 1)
              if scores[lag] <= scores.get(lag - 1, np.inf) and scores[lag] <= scores.get(lag + 1, np.inf)]
    return next(lag for lag in minima + [min(scores, key=scores.get)] if scores[lag] <= 1.15 * best + 1e-12)


def best_start(values, period):
    first = int(np.ceil(LEAD_IN * len(values)))
    candidates = range(first, len(values) - period)
    if not candidates:
        raise FactsError("No full cycle after the lead-in; generate a longer video")
    return min(candidates, key=lambda s: float(np.sum((values[s] - values[s + period]) ** 2)))


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
    runs, start = [], None
    for i, on in enumerate(list(mask) + [False]):
        if on and start is None:
            start = i
        if not on and start is not None:
            runs.append([round(start / len(mask), 4), round(i / len(mask), 4)])
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
    values = np.array([f["values"][2:] for f in frames])
    chains = list(project["paws"])
    xs = {c: np.array([f["paws"][c][0] for f in frames]) for c in chains}
    zs = {c: np.array([f["paws"][c][1] for f in frames]) for c in chains}
    warnings = []
    if brief["loop"]:
        period = period_frames(values, fps)
        start = best_start(values, period)
        span = slice(start, start + period)
        masks = {c: contact_mask(zs[c][span]) for c in chains}
        origin = project.get("phaseOrigin", chains[0])
        shift = contact_window(masks[origin])[0][0]
        if shift:
            start = start + int(round(shift * period))
            if start + period >= len(frames):
                start -= period
            span = slice(start, start + period)
            masks = {c: contact_mask(zs[c][span]) for c in chains}
        speed = ground_speed({c: xs[c][span] for c in chains}, masks, fps)
        contacts, phase, duty = {}, {}, {}
        for c in chains:
            window, fraction = contact_window(masks[c])
            bone = project["paws"][c]
            contacts[bone], phase[bone], duty[bone] = window, window[0], round(fraction, 4)
        entry = {"duration_s": round(period / fps, 4), "loop": True, "authored_speed_mps": round(speed, 3),
                 "stride_m": round(speed * period / fps, 3), "contacts": contacts, "frames": period, "fps": fps,
                 "footfall_phase": phase, "duty_factor": duty}
    else:
        start, period = 0, len(frames) - 1
        span = slice(0, len(frames))
        masks = {c: contact_mask(zs[c]) for c in chains}
        speed = ground_speed(xs, masks, fps)
        all_windows = {project["paws"][c]: windows(masks[c]) for c in chains}
        entry = {"duration_s": round(period / fps, 4), "loop": False, "authored_speed_mps": round(speed, 3),
                 "stride_m": 0.0, "contacts": {b: w[0] if w else [0.0, 0.0] for b, w in all_windows.items()},
                 "frames": period, "fps": fps, "contact_windows": all_windows}
    target = brief.get("targetSpeedMps")
    if target and abs(speed - target) / target > 0.25:
        warnings.append(f"Measured ground speed {speed:.2f} m/s is more than 25% from the brief's {target} m/s")
    return {"start": start, "period": period, "fps": fps, "speedMps": speed, "strideM": entry["stride_m"],
            "entry": entry, "warnings": warnings}
