"""The fitter's per-frame search, apart from Blender so it can be tested: leave the tracked pose's local minimum
without trading a full-resolution fit for a coarse-level one.

numpy only (the fitter runs inside Blender's Python)."""

EPS = 1e-6


def search(start, candidates, solve, coarse, fine, polish):
    """The best of `start` and the candidates, compared like with like.

    candidates: [(make, idx)]: make(best) gives a candidate pose from the best pose so far (a fixed alternative, or
    a random restart about it); solve(pose, idx) solves it over the joints idx. Every solved candidate is scored by
    `coarse` (the objective at one pyramid level), as the start is, and replaces the best when it is lower. A winner
    is then polished (`polish`, full resolution) and kept only if its `fine` objective (full resolution, prior
    included) beats the start's; otherwise the start is returned unchanged."""
    best_x, best = start, coarse(start)
    for make, idx in candidates:
        y = solve(make(best_x), idx)
        loss = coarse(y)
        if loss < best - EPS:
            best_x, best = y, loss
    if best_x is start:
        return start
    polished = polish(best_x)
    if fine(polished) < fine(start) - EPS:
        return polished
    return start
