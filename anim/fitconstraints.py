"""NumPy-only coupled steps for the fitter's original source-observation disks.

This is a local search, not a certificate that unsuccessful constraints are impossible.
Parameter scales condition numerical derivatives; they do not change source uncertainty.
"""
import numpy as np


TOL = 1e-12  # identical normalized boundary tolerance to independent articulation.assess


def _quality(residual):
    norms = np.linalg.norm(residual, axis=1)
    if not np.isfinite(norms).all():
        return False, float('inf')
    excess = np.maximum(norms - 1., 0.)
    return bool(np.all(norms <= 1. + TOL)), float(excess @ excess)


def prefer(candidate, incumbent, objective, residuals):
    """Feasibility first, then the unchanged objective; no observations means legacy score."""
    a, b = residuals(candidate), residuals(incumbent)
    if len(a):
        fa, _ = _quality(a)
        fb, _ = _quality(b)
        if fa != fb:
            return fa
    return objective(candidate) < objective(incumbent) - 1e-6


def _box_step(matrix, target, lower, upper):
    """Damped quadratic step with first-hit bounds and releasable active limits."""
    damping = 1e-5
    delta = np.clip(np.zeros(len(lower)), lower, upper)
    movable = lower < upper
    free = movable.copy()
    reason = 'active-set-limit'
    for iteration in range(4 * len(lower) + 1):
        if free.any():
            columns = matrix[:, free]
            rhs = target - matrix[:, ~free] @ delta[~free]
            goal = delta.copy()
            goal[free] = np.linalg.lstsq(
                columns.T @ columns + damping * np.eye(int(free.sum())),
                columns.T @ rhs, rcond=None)[0]
            direction = goal - delta
            alpha, hit = 1.0, None
            for index in np.flatnonzero(free):
                if direction[index] > 0. and goal[index] > upper[index]:
                    distance = (upper[index] - delta[index]) / direction[index]
                elif direction[index] < 0. and goal[index] < lower[index]:
                    distance = (lower[index] - delta[index]) / direction[index]
                else:
                    continue
                if distance < alpha:
                    alpha, hit = max(0., float(distance)), index
            delta = np.clip(delta + alpha * direction, lower, upper)
            if hit is not None:
                # Stop at just the first bound. Another initially outward
                # parameter may need to reverse once this parameter is held.
                free[hit] = False
                continue
        gradient = matrix.T @ (matrix @ delta - target) + damping * delta
        release = (~free & movable
                   & (((delta <= lower) & (gradient < 0.))
                      | ((delta >= upper) & (gradient > 0.))))
        if not release.any():
            reason = 'quadratic-stationary'
            break
        indices = np.flatnonzero(release)
        free[indices[np.argmax(np.abs(gradient[indices]))]] = True
    return delta, {'status': reason, 'iterations': iteration + 1,
                   'heldParameters': np.flatnonzero(~free).tolist()}


def refine(x, objective, residuals, clamp, scales, *, max_iterations=32):
    """Restore disks and descend the original objective with coupled projected moves.

    Residual rows are actual projected landmark displacement / source uncertainty.
    Zero scales freeze parameters, including camera scale in fixed-camera passes.
    Every accepted feasible iterate remains feasible, and the best one is retained.
    """
    x = np.asarray(x, float).copy()
    initial = np.asarray(residuals(x), float)
    if not len(initial):
        return x, {'status': 'no-observations', 'feasible': True, 'iterations': 0}
    scales = np.asarray(scales, float)
    active = np.flatnonzero(scales > 0)
    evaluations = 0
    initial_restoration = {'rounds': [], 'activeParameterIndices': active.tolist()}

    def bounded(y, origin):
        y = np.asarray(clamp(y.copy()), float)
        y[scales == 0] = origin[scales == 0]
        return y

    def jacobian(y, r):
        jac = np.zeros((r.size, len(active)))
        for column, index in enumerate(active):
            h = scales[index] * .01
            plus, minus = y.copy(), y.copy()
            plus[index] += h
            minus[index] -= h
            plus, minus = bounded(plus, y), bounded(minus, y)
            distance = (plus[index] - minus[index]) / scales[index]
            if distance:
                jac[:, column] = (residuals(plus).ravel() - residuals(minus).ravel()) / distance
        return jac

    def restore(y, rounds=12, record=False):
        nonlocal evaluations
        r = np.asarray(residuals(y), float)
        feasible, violation = _quality(r)
        stop = 'round-limit'
        for _ in range(rounds):
            if feasible:
                stop = 'feasible'
                break
            if not len(active) or not np.isfinite(r).all():
                stop = 'no-active-parameters' if not len(active) else 'nonfinite-residual'
                break
            jac = jacobian(y, r)
            norms = np.linalg.norm(r, axis=1)
            # Half squared distance to the ORIGINAL product of disks. Its residual-
            # space Hessian has radial eigenvalue 1 and tangential eigenvalue
            # 1 - 1/norm outside each disk, and is zero inside. Keeping this
            # Cartesian curvature avoids the rank loss of radial equalities
            # without imposing invented targets on already satisfied disks.
            violated = norms > 1.0
            unit = r[violated] / norms[violated, None]
            tangent = np.sqrt(1.0 - 1.0 / norms[violated])
            outer = unit[:, :, None] * unit[:, None, :]
            square_root = (tangent[:, None, None] * np.eye(2)
                           + (1.0 - tangent[:, None, None]) * outer)
            cartesian = np.einsum('mij,mjk->mik', square_root,
                                   jac.reshape(len(r), 2, -1)[violated]).reshape(-1, len(active))
            # The disk-excess gradient is radial, so multiplication by the
            # inverse square root leaves it unchanged.
            correction = -((norms[violated] - 1.0)[:, None] * unit).ravel()
            low, high = y.copy(), y.copy()
            low[active] -= 4. * scales[active]
            high[active] += 4. * scales[active]
            lower = (bounded(low, y)[active] - y[active]) / scales[active]
            upper = (bounded(high, y)[active] - y[active]) / scales[active]
            delta, box_diagnostic = _box_step(cartesian, correction, lower, upper)
            trace = {'normalizedErrors': norms.tolist(), 'violation': violation,
                     'linearizedSingularValues': np.linalg.svd(cartesian, compute_uv=False).tolist(),
                     'model': 'original-disk-distance-vector-curvature',
                     'boxStep': box_diagnostic, 'step': delta.tolist(), 'trials': []} if record else None
            delta /= max(1., np.linalg.norm(delta) / 4.)
            accepted = False
            for alpha in (1., .5, .25, .125, .0625):
                trial = y.copy()
                trial[active] += alpha * scales[active] * delta
                trial = bounded(trial, y)
                rr = np.asarray(residuals(trial), float)
                ff, vv = _quality(rr)
                evaluations += 1
                if record:
                    trace['trials'].append({'alpha': alpha, 'violation': vv, 'feasible': ff})
                if ff or vv < violation * (1.0 - 1e-8):
                    y, r, feasible, violation = trial, rr, ff, vv
                    accepted = True
                    if record:
                        trace['acceptedAlpha'] = alpha
                    break
            if record:
                initial_restoration['rounds'].append(trace)
            if not accepted:
                stop = 'line-search-no-decrease'
                break
        if record:
            initial_restoration['stoppingReason'] = 'feasible' if feasible else stop
        return y, r, feasible, violation

    x, r, feasible, violation = restore(x, record=True)
    value = float(objective(x))
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        if not feasible or not len(active) or not np.isfinite(value):
            break
        jac = jacobian(x, r)
        gradient = np.zeros(len(active))
        for column, index in enumerate(active):
            plus, minus = x.copy(), x.copy()
            plus[index] += scales[index]
            minus[index] -= scales[index]
            plus, minus = bounded(plus, x), bounded(minus, x)
            distance = (plus[index] - minus[index]) / scales[index]
            if distance:
                gradient[column] = (objective(plus) - objective(minus)) / distance
        if not np.isfinite(gradient).all():
            break
        # Remove motion normal to currently tight disks. Interior constraints are
        # handled by restoration after each trial; all acceptance uses actual FK.
        tight = np.linalg.norm(r, axis=1) > .8
        normals = np.array([r[i] @ jac.reshape(len(r), 2, -1)[i] for i in np.flatnonzero(tight)])
        direction = -gradient
        if len(normals):
            direction -= normals.T @ np.linalg.lstsq(
                normals @ normals.T + 1e-8 * np.eye(len(normals)), normals @ direction, rcond=None)[0]
        directions = [direction, -gradient]
        # Coordinate proposals followed by coupled restoration work with the
        # fitter's nonsmooth raster objective, whose tiny numerical gradient is zero.
        directions.extend(sign * np.eye(len(active))[i] for i in range(len(active)) for sign in (1., -1.))
        best, best_r, best_value = x, r, value
        for direction in directions:
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / max(1., norm)
            for alpha in (2., 1., .5, .25):
                trial = x.copy()
                trial[active] += alpha * scales[active] * direction
                trial, rr, ff, _ = restore(bounded(trial, x))
                if not ff:
                    continue
                loss = float(objective(trial))
                if np.isfinite(loss) and loss < best_value - 1e-6:
                    best, best_r, best_value = trial, rr, loss
                    break
        if best is x:
            break
        x, r, value = best, best_r, best_value
    feasible, violation = _quality(r)
    return x, {'status': 'feasible' if feasible else 'unresolved', 'feasible': feasible,
               'iterations': iterations, 'projectionEvaluations': evaluations,
               'maxNormalizedError': float(np.linalg.norm(r, axis=1).max()),
               'violation': violation, 'objective': value,
               'initialRestoration': initial_restoration}
