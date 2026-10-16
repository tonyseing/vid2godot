"""Clip facts from the fitted poses: one clean loop, paw contacts, ground speed and stride, the sidecar entry.

A loop's period comes from the video (the registered masks' self-similarity, anim/cycle.py), not from the fitted
angles: joints the silhouette barely constrains trade angle for the same outline from frame to frame. The cycle is
the phase-aligned average over the steady cycles (warm-up left out, leg identity aligned per cycle, angles by their
circular mean), rolled to the phase origin's touchdown, and it is rendered at every frame's phase of those cycles and
checked against the masks (mean IoU at least CYCLE_MIN_IOU) before it is accepted."""
import numpy as np

from anim import cycle
from anim.ledger import StageError

LEAD_IN = 0.10
# The fitted angles' period score (`period_search`) gates nothing since Task 17b fix round 2; it is reported, on the
# well-observed leg joints, against this threshold.
PERIOD_MAX_SCORE = 0.2
# The averaged cycle, rendered at each frame's phase, must match the video masks at least this well (mean IoU).
CYCLE_MIN_IOU = 0.85


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


def clip_facts(poses, project, brief, masks=None, render=None):
    """The clip's facts. A loop needs the video: `masks` (the ingest's working masks, (T, H, W) bool) and `render`
    (value vectors -> the rig's silhouettes at the masks' size) for the averaged cycle's check."""
    fps = float(poses["fps"])
    frames = poses["frames"]
    every = np.array([f["values"] for f in frames], float)
    values = every[:, 2:]
    chains = list(project["paws"])
    paws = {c: np.array([f["paws"][c] for f in frames], float) for c in chains}
    warnings = []
    if brief["loop"]:
        found = loop_cycle(poses, project, every, paws, chains, masks, render)
        start, period, entry, cycle_out, extra = found
        speed = entry["authored_speed_mps"]
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
        extra = {}
    target = brief.get("targetSpeedMps")
    if target and abs(speed - target) / target > 0.25:
        warnings.append(f"Measured ground speed {speed:.2f} m/s is more than 25% from the brief's {target} m/s")
    out = {"start": start, "period": period, "fps": fps, "speedMps": speed, "strideM": entry["stride_m"],
           "entry": entry, "warnings": warnings, "cycle": cycle_out}
    if brief["loop"]:
        out.update(extra)
    return out


def leg_period(values, names, project, fps, period):
    """The secondary report on the fitted angles: over the well-observed leg joints (`cycle.leg_joints`), the
    normalised lag score at the video's period (interpolated between whole frames) and the angle-only period
    search."""
    joints = cycle.leg_joints(names, project)
    if not joints:
        return {"joints": [], "scoreAtPeriod": None, "search": None, "gate": PERIOD_MAX_SCORE}
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
    return {"joints": joints, "scoreAtPeriod": None if score is None else round(score, 4),
            "search": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.items()},
            "gate": PERIOD_MAX_SCORE}


def loop_cycle(poses, project, every, paws, chains, masks, render):
    """The loop's cycle from the video: period, steady cycles, phase average, touchdown roll, the averaged cycle's
    check against the masks. Returns (start, period, entry, cycle block, extra facts)."""
    if masks is None or render is None:
        raise FactsError("facts finds a loop's period in the video: it needs the ingest masks and the rig's renderer")
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
    origin = project.get("phaseOrigin", chains[0])
    contacts_of = {c: contact_mask(avg[:n, paw_cols[c][1]]) for c in chains}
    touchdown = int(round(contact_window(contacts_of[origin])[0][0] * n))
    rolled = cycle.roll(avg, touchdown, kinds)
    start = float(kept[0] + touchdown * period / n)
    rate = n * fps / period
    cycle_paws = {c: rolled[:n][:, list(paw_cols[c])] for c in chains}
    masks_of = {c: contact_mask(cycle_paws[c][:, 1]) for c in chains}
    speed = ground_speed({c: cycle_paws[c][:, 0] for c in chains}, masks_of, rate)
    contacts, phase, duty = {}, {}, {}
    for c in chains:
        window, fraction = contact_window(masks_of[c])
        bone = project["paws"][c]
        contacts[bone], phase[bone], duty[bone] = window, window[0], round(fraction, 4)
    entry = {"duration_s": period / fps, "loop": True, "authored_speed_mps": round(speed, 3),
             "stride_m": round(speed * period / fps, 3), "contacts": contacts, "frames": n, "fps": rate,
             "footfall_phase": phase, "duty_factor": duty}
    keys = rolled[:, :width].copy()
    # The root's travel over one period (averaged over the cycles), taken before the closing key is overwritten:
    # the bake removes drift * k / n, so a periodic sway is left untouched.
    drift = {"dx": float(keys[-1][0] - keys[0][0]), "dz": float(keys[-1][1] - keys[0][1])}
    keys[-1] = keys[0]
    cycle_out = {"frames": keys.tolist(), "fps": rate, "periodFrames": period, "start": start, "drift": drift}
    check = check_cycle(avg, samples, exchanged, swaps, kept, period, n, kinds, width, masks, render)
    if check["meanIoU"] < CYCLE_MIN_IOU:
        phases = ", ".join(f"{v:.2f}" for v in check["phaseIoU"])
        raise FactsError(f"The averaged cycle ({len(kept)} cycles of {period:.2f} frames) does not match the video: "
                         f"rendered at each frame's phase it scores mean IoU {check['meanIoU']:.3f} against the "
                         f"masks, under the gate {CYCLE_MIN_IOU}; per phase: {phases}. The fitted poses differ "
                         f"from cycle to cycle where the video repeats; refit (marks, spans) or regenerate")
    extra = {"periodSource": "video masks", "maskPeriod": found,
             "legPeriod": leg_period(every, names, project, fps, period),
             "cycleStarts": [round(s, 4) for s in kept], "cyclesAveraged": len(kept),
             "cycleScores": {f"{s:.4f}": round(v, 4) for s, v in steady["scores"].items()},
             "identityExchanged": exchanged, "averagedCycleIoU": check["meanIoU"], "cycleCheck": check}
    return start, period, entry, cycle_out, extra


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
