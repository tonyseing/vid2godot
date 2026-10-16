"""Approach-C paw keypoints: paws found in a side-view mask, tracked over the clip, and the fitter's soft paw term.

numpy only: the fitter runs inside Blender's Python, which has no scipy.

A leg is a branch of the silhouette below the body line (the belly). Its paw is the far end of that branch, measured
along the silhouette (geodesic distance) from the body line: the far end of a straight or bent leg alike. Left and
right legs look the same from the side, so the paw term is identity-free: each fitted paw should be near some
detected paw and each detected paw near some fitted paw (a robust two-way chamfer), and a fitted paw drawn inside the
torso while a detected paw below the body has no leg costs its depth above the body line. Identity stays with the
fitter's relabelling.
"""
import numpy as np

BODY_ROW_SHARE = 0.5       # a body row is at least this share of the widest row
MIN_BRANCH_SHARE = 0.3     # a second paw's branch is at least this share of the longest leg
BLOB_SHARE = 0.12          # a paw's blob: silhouette within this share of the leg's length of its far end
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

TRACK_PX = 60.0            # a detection with no detection this close in the frame before or after is dropped
PAW_DEADZONE = 8.0         # working px: this close, a fitted and a detected paw agree (the term is zero)
PAW_HUBER = 4.0            # px beyond the dead zone where the cost turns from quadratic to linear
PAW_CAP = 100.0            # px beyond the dead zone after which the cost stops growing: a wrong detection is bounded
W_PAW = 0.01               # per px (the linear part), per fitted and per detected paw
PAW_CLAIM_PX = 15.0        # a detected paw with no fitted paw this close is unclaimed
W_TORSO = 0.02             # per px a fitted paw sits inside the torso above the body line, when a paw is unclaimed
TORSO_CAP = 60.0


def body_line(mask):
    """The lowest row of the body: the last row at least BODY_ROW_SHARE as wide as the widest."""
    widths = mask.sum(axis=1)
    return int(np.nonzero(widths >= BODY_ROW_SHARE * widths.max())[0].max())


def _shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    h, w = a.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h + min(-dy, 0),
                                                                  max(-dx, 0):w + min(-dx, 0)]
    return out


def geodesic(region, seeds):
    """8-connected step distance from `seeds` inside `region` (-1 where unreachable). Each step grows the frontier
    only within its bounding box, so a thin wavefront costs little."""
    h, w = region.shape
    dist = np.where(seeds, 0, -1).astype(np.int32)
    frontier, step = seeds & region, 0
    rows, cols = np.nonzero(frontier)
    while len(rows):
        step += 1
        r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 2, h)
        c0, c1 = max(cols.min() - 1, 0), min(cols.max() + 2, w)
        sub = frontier[r0:r1, c0:c1]
        grown = np.zeros_like(sub)
        for dy, dx in NEIGHBOURS:
            grown |= _shift(sub, dy, dx, False)
        grown &= region[r0:r1, c0:c1] & (dist[r0:r1, c0:c1] < 0)
        dist[r0:r1, c0:c1][grown] = step
        frontier = np.zeros_like(frontier)
        frontier[r0:r1, c0:c1] = grown
        rows, cols = np.nonzero(grown)
        rows, cols = rows + r0, cols + c0
    return dist


def find_paws(mask, max_paws=6):
    """Paws in one mask: [{"tip": [u, v], "blob": [u, v]}], strongest branch first, in the mask's pixels. tip is the
    far end of the leg branch below the body line; blob the centroid of the silhouette near that end (the paw)."""
    if not mask.any():
        return []
    line = body_line(mask)
    rows, cols = np.nonzero(mask)
    top, bottom, left, right = line, rows.max() + 1, cols.min(), cols.max() + 1
    if bottom - top < 3:
        return []
    region = mask[top:bottom, left:right].copy()
    seeds = np.zeros_like(region)
    seeds[0] = region[0]
    d0 = geodesic(region, seeds)
    if d0.max() <= 0:
        return []
    leg = d0.max()
    big = 10 * (region.shape[0] + region.shape[1])
    allowed = d0 >= MIN_BRANCH_SHARE * leg
    score = np.where(allowed, d0.astype(float) + big, -np.inf)
    found, from_tips = [], []
    while len(from_tips) < max_paws:
        flat = int(np.argmax(score))
        if not np.isfinite(score.flat[flat]):
            break
        branch = score.flat[flat] - (big if not from_tips else 0)
        if from_tips and branch < MIN_BRANCH_SHARE * leg:
            break
        v, u = divmod(flat, region.shape[1])
        tip = np.zeros_like(region)
        tip[v, u] = True
        dt = geodesic(d0 >= 0, tip)
        near = (dt >= 0) & (dt <= max(2, BLOB_SHARE * leg))
        bv, bu = np.nonzero(near)
        found.append({"tip": [float(u + left), float(v + top)],
                      "blob": [float(bu.mean() + left), float(bv.mean() + top)]})
        from_tips.append((dt, d0[v, u]))
        candidate = np.full(region.shape, np.inf)
        for dist_t, d0_t in from_tips:
            candidate = np.minimum(candidate, np.where(dist_t >= 0, dist_t + d0 - d0_t, big + d0))
        score = np.where(allowed, candidate, -np.inf)
    return found


def points(found):
    """Every tip and blob point of `find_paws`, flattened: [tip, blob, tip, blob, ...]."""
    return [p for paw in found for p in (paw["tip"], paw["blob"])]


def track(per_frame, radius=TRACK_PX):
    """Keep, per frame, the detections that persist: those with a detection within `radius` in the frame before or
    after (a paw moves a little between frames; a one-frame blip does not). A clip of one frame keeps everything."""
    per_frame = [np.asarray(p, float).reshape(-1, 2) for p in per_frame]
    if len(per_frame) < 2:
        return per_frame
    out = []
    for f, here in enumerate(per_frame):
        near = [per_frame[g] for g in (f - 1, f + 1) if 0 <= g < len(per_frame) and len(per_frame[g])]
        if not len(here) or not near:
            out.append(here)
            continue
        others = np.concatenate(near)
        d = np.hypot(here[:, None, 0] - others[None, :, 0], here[:, None, 1] - others[None, :, 1]).min(axis=1)
        out.append(here[d <= radius])
    return out


def robust(d):
    """The cost of a distance d (px): zero inside the dead zone, then Huber (quadratic to PAW_HUBER, linear after),
    flat beyond PAW_CAP."""
    e = np.clip(np.asarray(d, float) - PAW_DEADZONE, 0.0, PAW_CAP)
    return np.where(e <= PAW_HUBER, 0.5 * e ** 2 / PAW_HUBER, e - 0.5 * PAW_HUBER)


def penalty(fitted, detected, body_row, mask=None, roots=None, reach=None, others=None):
    """The soft paw term for one frame (loss units).

    fitted: (F, 2) fitted paw points; detected: (D, 2) tracked detections; body_row: the mask's body line.
    roots, reach: when given, each fitted paw is compared only with the detections within reach[i] of roots[i] (where
    its leg leaves the body): a foreleg is not pulled to a hind paw it cannot reach. others: further fitted points
    (the tail tip, the nose) that may claim a detection (a drooping tail is not a paw) but cost nothing themselves.
    mask: when given, a fitted paw above the body line and inside the mask (drawn inside the torso) costs W_TORSO
    per px of depth while some detected paw is unclaimed (no fitted point within PAW_CLAIM_PX)."""
    fitted, detected = np.asarray(fitted, float).reshape(-1, 2), np.asarray(detected, float).reshape(-1, 2)
    if not len(fitted) or not len(detected):
        return 0.0
    d = np.hypot(fitted[:, None, 0] - detected[None, :, 0], fitted[:, None, 1] - detected[None, :, 1])
    near = d
    if roots is not None:
        roots = np.asarray(roots, float).reshape(-1, 2)
        span = np.hypot(roots[:, None, 0] - detected[None, :, 0], roots[:, None, 1] - detected[None, :, 1])
        near = np.where(span <= np.asarray(reach, float)[:, None], d, np.inf)
    each = near.min(axis=1)
    cost = float(robust(each[np.isfinite(each)]).sum())
    claim = d
    if others is not None and len(others):
        others = np.asarray(others, float).reshape(-1, 2)
        claim = np.vstack([d, np.hypot(others[:, None, 0] - detected[None, :, 0],
                                       others[:, None, 1] - detected[None, :, 1])])
    cost = W_PAW * (cost + float(robust(claim.min(axis=0)).sum()))
    if mask is not None and (claim.min(axis=0) > PAW_CLAIM_PX).any():
        h, w = mask.shape
        for u, v in fitted:
            r, c = int(v), int(u)
            if v < body_row and 0 <= r < h and 0 <= c < w and mask[r, c]:
                cost += W_TORSO * min(body_row - v, TORSO_CAP)
    return cost
