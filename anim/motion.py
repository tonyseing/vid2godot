"""The baked cycle's motion in numpy: the rig's side-plane forward kinematics, a smooth periodic fit of each joint's
cycle, and two-bone leg IK that pins each planted paw for the 60 Hz keys facts hands to the bake.

Every fitted joint turns about its bone's side axis, which is world +Y at rest (rigutil.side_axis_local), and so do
its parents: a bone's posed transform is a rotation about world +Y and a shift in x/z, and the side view is exact in
two dimensions. `PlanarRig` poses the contract's rig.json (bone heads, tails, parents) that way; the armature's
dx/dz moves every bone (the bake keys it on the pelvis, which carries every paw).

The smooth fit is a truncated Fourier series per column of the averaged cycle. The trunk, neck, tail and root columns
take the lowest order that keeps RANGE_KEEP of their sampled range (the video's back, neck and tail swing, without
the 24 Hz steps); when that series' jerk is over JERK_MAX times a sine's of the same range (a neck nodding at twice the
stride rate), its overtones are tapered until it is not, and the series is scaled about its mean back to that range
(the jerk ratio does not change with scale). Leg columns keep every harmonic the samples hold (trigonometric interpolation: smooth and through
every averaged sample), since their stance and swing shapes are what the contacts are read from and the IK reshapes
them in stance anyway.

numpy only."""
import numpy as np

KEY_RATE = 60.0          # the bake's key rate (Hz of the played clip)
RANGE_KEEP = 0.95        # a smoothed trunk, neck, tail or root column keeps this share of its sampled range
JERK_MAX = 3.0           # ... with its jerk at most this many times a sine's of the same range (the bar's criterion 4)
BLEND = 0.05             # cycle fraction over which a paw eases from free to pinned before touchdown, and back after
SUBSAMPLES = 8           # re-detection samples per key interval (the keys are interpolated linearly, as played)


def rotate(phi, points):
    """Rotate x/z points (..., 2) about world +Y by phi (per sample: phi (S,), points (S, 2) or (2,))."""
    c, s = np.cos(phi), np.sin(phi)
    points = np.asarray(points, float)
    x, z = points[..., 0], points[..., 1]
    return np.stack([c * x + s * z, -s * x + c * z], axis=-1)


def polar(v):
    """The angle of x/z vectors in the side plane, atan2(z, x). rotate(phi) takes it to polar - phi."""
    v = np.asarray(v, float)
    return np.arctan2(v[..., 1], v[..., 0])


def wrap(angle):
    return (np.asarray(angle, float) + np.pi) % (2 * np.pi) - np.pi


class PlanarRig:
    """The contract's rig.json posed in the side plane. Values are poses.json vectors (`names`: dx, dz, joints)."""

    def __init__(self, rig):
        bones = {b["name"]: b for b in (rig or {}).get("bones") or [] if isinstance(b, dict)}
        if not bones:
            raise ValueError("the rig.json lists no bones")
        order, seen = [], set()
        while len(order) < len(bones):
            added = False
            for name, bone in bones.items():
                if name not in seen and (bone.get("parent") is None or bone["parent"] in seen):
                    order.append(name)
                    seen.add(name)
                    added = True
            if not added:
                raise ValueError("the rig.json's bone parents do not form a tree")
        self.order = order
        self.parent = {n: bones[n].get("parent") for n in order}
        self.head = {n: np.array([bones[n]["head"][0], bones[n]["head"][2]], float) for n in order}
        self.tail = {n: np.array([bones[n]["tail"][0], bones[n]["tail"][2]], float) for n in order}

    def pose(self, values, names):
        """Every bone's world rotation phi (S,) and shift t (S, 2) for values (S, V) in `names` order."""
        values = np.atleast_2d(np.asarray(values, float))
        index = {n: i for i, n in enumerate(names)}
        count = len(values)
        root_t = values[:, [index["dx"], index["dz"]]]
        phi, shift = {}, {}
        for name in self.order:
            theta = values[:, index[name]] if name in index else np.zeros(count)
            parent = self.parent[name]
            p_phi, p_t = (phi[parent], shift[parent]) if parent else (np.zeros(count), root_t)
            head = self.head[name]
            phi[name] = p_phi + theta
            shift[name] = rotate(p_phi, head - rotate(theta, head)) + p_t
        return phi, shift

    def point(self, posed, bone, end="tail"):
        phi, shift = posed
        rest = self.tail[bone] if end == "tail" else self.head[bone]
        return rotate(phi[bone], rest) + shift[bone]

    def paws(self, values, names, project):
        posed = self.pose(values, names)
        return {c: self.point(posed, bone) for c, bone in project["paws"].items()}


# ------------------------------------------------------------------------------------------- the smooth cycle
def coefficients(samples):
    """The Fourier coefficients of n uniform samples of one cycle (rfft / n)."""
    samples = np.asarray(samples, float)
    return np.fft.rfft(samples) / len(samples)


def evaluate(coeffs, phases, order, taper=1.0, gain=1.0, derivative=0):
    """The series truncated at `order` at cycle phases in [0, 1): its mean plus `gain` times the fundamental and the
    overtones (harmonics 2..order) scaled by `taper`; `derivative` > 0 differentiates it in phase that many times."""
    phases = np.asarray(phases, float)
    out = np.full(phases.shape, float(np.real(coeffs[0])) if derivative == 0 else 0.0)
    for k in range(1, order + 1):
        weight = gain * (1.0 if k == 1 else taper) * (2j * np.pi * k) ** derivative
        out = out + 2.0 * np.real(weight * coeffs[k] * np.exp(2j * np.pi * k * phases))
    return out


def jerk_ratio(coeffs, order, taper=1.0, gain=1.0, dense=600):
    """The series' largest jerk over a sine's of the same range (1.0 for a sine; independent of scale and time)."""
    grid = np.arange(dense) / dense
    span = float(np.ptp(evaluate(coeffs, grid, order, taper, gain)))
    if span <= 1e-12:
        return 1.0
    return float(np.abs(evaluate(coeffs, grid, order, taper, gain, 3)).max() / (span / 2 * (2 * np.pi) ** 3))


def smoothest(coeffs, order, jerk_max=JERK_MAX, dense=600):
    """(taper, gain) for a series of `order`: (1, 1) when its jerk ratio is within `jerk_max`; else the least taper
    of the overtones (in steps of 0.01) that brings it there, and the gain that scales the series back to its range."""
    if order < 2 or jerk_ratio(coeffs, order) <= jerk_max:
        return 1.0, 1.0
    grid = np.arange(dense) / dense
    span = float(np.ptp(evaluate(coeffs, grid, order)))
    taper = 1.0
    while taper > 0 and jerk_ratio(coeffs, order, taper) > jerk_max:
        taper = round(taper - 0.01, 2)
    taper = max(taper, 0.0)
    return taper, span / float(np.ptp(evaluate(coeffs, grid, order, taper)))


def max_order(n):
    """The highest harmonic n samples hold without the (ambiguous) Nyquist term."""
    return max(0, (n - 1) // 2)


def range_order(samples, keep=RANGE_KEEP, dense=600):
    """The lowest order whose series keeps `keep` of the samples' range (0 for a constant column)."""
    samples = np.asarray(samples, float)
    span = float(np.ptp(samples))
    if span <= 1e-12:
        return 0
    coeffs = coefficients(samples)
    grid = np.arange(dense) / dense
    top = max_order(len(samples))
    for order in range(1, top + 1):
        if np.ptp(evaluate(coeffs, grid, order)) >= keep * span:
            return order
    return top


def leg_bones(project):
    return {b for pair in project.get("legPairs", []) for c in pair for b in project["chains"].get(c, [])}


class SmoothCycle:
    """The periodic smooth fit of an averaged cycle. `samples` (n, V): one cycle in `names` order with the root's
    drift already removed (periodic); angle columns are unwrapped before fitting."""

    def __init__(self, samples, names, project):
        samples = np.asarray(samples, float)
        self.names, self.n = list(names), len(samples)
        legs = leg_bones(project)
        self.coeffs, self.orders, self.ranges, self.taper, self.gain = [], {}, {}, {}, {}
        for i, name in enumerate(self.names):
            column = samples[:, i] if name in ("dx", "dz") else np.unwrap(samples[:, i])
            coeffs = coefficients(column)
            order = max_order(self.n) if name in legs else range_order(column)
            self.taper[name], self.gain[name] = (1.0, 1.0) if name in legs else smoothest(coeffs, order)
            self.coeffs.append(coeffs)
            self.orders[name] = order
            self.ranges[name] = float(np.ptp(column))

    def at(self, phases):
        """Value vectors (S, V) at cycle phases (any real numbers; one cycle is 1)."""
        phases = np.asarray(phases, float) % 1.0
        return np.stack([evaluate(c, phases, self.orders[n], self.taper[n], self.gain[n])
                         for c, n in zip(self.coeffs, self.names)], axis=1)

    def jerk_ratio(self):
        return {n: round(jerk_ratio(c, self.orders[n], self.taper[n], self.gain[n]), 4)
                for c, n in zip(self.coeffs, self.names)}

    def roll(self, phase):
        """The same motion started `phase` later: at(u) of the result is at(u + phase) of this one."""
        out = SmoothCycle.__new__(SmoothCycle)
        out.names, out.n, out.orders, out.ranges = self.names, self.n, self.orders, self.ranges
        out.taper, out.gain = self.taper, self.gain
        k = np.arange(len(self.coeffs[0]))
        out.coeffs = [c * np.exp(2j * np.pi * k * phase) for c in self.coeffs]
        return out

    def kept_range(self, dense=600):
        """Per column, the smoothed range over the sampled range (1.0 for a constant column)."""
        values = self.at(np.arange(dense) / dense)
        return {n: (round(float(np.ptp(values[:, i]) / self.ranges[n]), 4) if self.ranges[n] > 1e-12 else 1.0)
                for i, n in enumerate(self.names)}


# ------------------------------------------------------------------------------------------------ two-bone IK
def ik_bones(chain):
    """(the IK pair, the foot) of a leg chain: the two bones above the last two (the foot keeps its world
    orientation); a three-bone leg's pair is its first two."""
    if len(chain) >= 4:
        return chain[-4:-2], chain[-2:]
    if len(chain) == 3:
        return chain[:2], chain[2:]
    raise ValueError(f"two-bone IK needs a leg chain of at least three bones, not {chain}")


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def solve_leg(rig, values, names, chain, paw_bone, target, pivot=True):
    """Two-bone IK in the side plane: new values (S, V) with the chain's pair and first foot bone turned so the paw
    bone's tail reaches `target` (S, 2) while the foot keeps its world orientation and the knee bends the way it
    bent. Where the pair cannot reach, `pivot` turns the foot about the pinned toe (the heel lifts, as at the end of a
    stance) by the least angle that brings the ankle within reach. Returns (values, reached paw positions)."""
    (upper, lower), foot = ik_bones(chain)
    index = {n: i for i, n in enumerate(names)}
    missing = [b for b in (upper, lower, foot[0]) if b not in index]
    if missing:
        raise ValueError(f"leg IK turns {missing}, which the fit did not solve")
    posed = rig.pose(values, names)
    phi, _ = posed
    parent = rig.parent[upper]
    a = rig.point(posed, upper, "head")
    knee = rig.point(posed, lower, "head")
    end = rig.point(posed, foot[0], "head")
    offset = rig.point(posed, paw_bone) - end
    target = np.asarray(target, float)
    l1 = float(np.linalg.norm(rig.head[lower] - rig.head[upper]))
    l2 = float(np.linalg.norm(rig.head[foot[0]] - rig.head[lower]))
    turn = np.zeros(len(values))
    if pivot:
        turn = toe_pivot(target - a, offset, (l1 + l2) * (1 - 1e-6))
        offset = rotate(turn, offset)
    q = target - offset
    reach = q - a
    d = np.clip(np.linalg.norm(reach, axis=1), abs(l1 - l2) + 1e-9, l1 + l2 - 1e-9)
    beta = np.arccos(np.clip((l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d), -1.0, 1.0))
    side = np.sign(cross(end - a, knee - a))
    side[side == 0] = 1.0
    toward = polar(reach)
    new_knee = a + l1 * np.stack([np.cos(toward + side * beta), np.sin(toward + side * beta)], axis=1)
    new_end = a + d[:, None] * np.stack([np.cos(toward), np.sin(toward)], axis=1)
    phi_upper = polar(rig.head[lower] - rig.head[upper]) - polar(new_knee - a)
    phi_lower = polar(rig.head[foot[0]] - rig.head[lower]) - polar(new_end - new_knee)
    p_phi = phi[parent] if parent else np.zeros(len(values))
    out = np.array(values, float, copy=True)
    for bone, angle in ((upper, phi_upper - p_phi), (lower, phi_lower - phi_upper),
                        (foot[0], phi[foot[0]] + turn - phi_lower)):
        free = out[:, index[bone]]
        out[:, index[bone]] = free + wrap(angle - free)
    return out, new_end + offset


def toe_pivot(toe, offset, length):
    """Per sample, the least turn gamma of the foot (rotate(gamma, offset): ankle-to-toe) about the toe that puts the
    ankle within `length` of the hip: 0 where it already is, and where no turn can (the closest turn instead)."""
    toe, offset = np.atleast_2d(toe), np.atleast_2d(offset)
    gamma = np.zeros(len(toe))
    far = np.linalg.norm(toe - offset, axis=1) > length
    v, w = np.linalg.norm(toe, axis=1), np.linalg.norm(offset, axis=1)
    cos = np.clip((v ** 2 + w ** 2 - length ** 2) / np.maximum(2 * v * w, 1e-12), -1.0, 1.0)
    beta = np.arccos(cos)
    # rotate(gamma) takes an angle to angle - gamma: the turned offset points at polar(toe) +- beta.
    base = polar(offset) - polar(toe)
    options = np.stack([wrap(base - beta), wrap(base + beta)], axis=1)
    pick = options[np.arange(len(toe)), np.argmin(np.abs(options), axis=1)]
    gamma[far] = pick[far]
    return gamma


def stance_weight(phases, window, blend=BLEND):
    """1 inside the contact window [on, off) (cycle phases, wrapping), easing to 0 over `blend` outside it
    (smoothstep), 0 beyond. Also returns each phase unwrapped next to the window."""
    on, off = window
    off = off if off > on else off + 1.0
    phases = np.asarray(phases, float)
    best = np.full(phases.shape, np.inf)
    near = phases.copy()
    for shift in (-1.0, 0.0, 1.0):
        u = phases + shift
        dist = np.maximum(0.0, np.maximum(on - u, u - off))
        better = dist < best
        best, near = np.where(better, dist, best), np.where(better, u, near)
    x = np.clip(1.0 - best / blend, 0.0, 1.0) if blend > 0 else (best <= 0).astype(float)
    return np.where(best <= 0, 1.0, x * x * (3 - 2 * x)), near


def pin_lines(rig, smooth, names, project, windows, stride, rest, lift, dx_shift=0.0, dense=240):
    """Per planted paw, the world-still line it is pinned to: in the in-place clip a planted paw slides back one
    stride per cycle (x = x_mid - stride * (u - u_mid)) on the ground (z = its rest tail height once the shared
    ground is lifted to the rest ground). x_mid is the least-squares fit to the free paw over the window."""
    grid = np.arange(dense) / dense
    values = smooth.at(grid)
    values[:, names.index("dz")] += lift
    values[:, names.index("dx")] -= dx_shift
    paws = rig.paws(values, names, project)
    lines = {}
    for chain, window in windows.items():
        on, off = window
        off = off if off > on else off + 1.0
        mid = 0.5 * (on + off)
        weight, near = stance_weight(grid, window, 0.0)
        inside = weight >= 1.0
        x = paws[chain][inside, 0] + stride * (near[inside] - mid)
        lines[chain] = {"mid": mid, "x": float(np.mean(x)), "z": float(rest[chain])}
    return lines


def plan_keys(rig, smooth, names, project, windows, stride, rest, lift, n_keys, blend=BLEND):
    """The bake's keys: n_keys + 1 value vectors at phases k / n_keys (the last repeats the first), the smooth cycle
    with its root lifted by `lift` and every planted paw pinned to its world-still line (`pin_lines`) over its
    contact window, easing in and out over `blend`. Returns (keys, predicted paws per key, IK report)."""
    phases = np.arange(n_keys) / n_keys
    keys = smooth.at(phases)
    keys[:, names.index("dz")] += lift
    shift = float(keys[:, names.index("dx")].mean())       # in place: the bake would remove it; the pins move with it
    keys[:, names.index("dx")] -= shift
    lines = pin_lines(rig, smooth, names, project, windows, stride, rest, lift, shift)
    report = {}
    for chain, window in windows.items():
        weight, near = stance_weight(phases, window, blend)
        line = lines[chain]
        live = weight > 0
        if not live.any():
            continue
        paw_bone = project["paws"][chain]
        free = rig.paws(keys, names, project)[chain]
        pin = np.stack([line["x"] - stride * (near - line["mid"]), np.full(len(phases), line["z"])], axis=1)
        target = weight[:, None] * pin + (1 - weight[:, None]) * free
        solved, reached = solve_leg(rig, keys[live], names, project["chains"][chain], paw_bone, target[live])
        keys[live] = solved
        residual = np.linalg.norm(reached - target[live], axis=1)
        report[chain] = {"pinnedKeys": int(np.count_nonzero(weight >= 1.0)), "easedKeys": int(np.count_nonzero(
            (weight > 0) & (weight < 1.0))), "maxResidualM": round(float(residual.max()), 5),
            "pinX": round(line["x"], 4), "pinZ": round(line["z"], 4)}
    keys = np.vstack([keys, keys[:1]])
    paws = rig.paws(keys, names, project)
    return keys, {c: p for c, p in paws.items()}, report


def played(keys, subsamples=SUBSAMPLES):
    """The keys as the clip plays them: linear interpolation of every column, `subsamples` per key interval, over
    one cycle (phases k / (n * subsamples), the last key excluded)."""
    keys = np.asarray(keys, float)
    n = len(keys) - 1
    t = np.arange(n * subsamples) / subsamples
    lo = np.floor(t).astype(int)
    w = (t - lo)[:, None]
    return (1 - w) * keys[lo] + w * keys[lo + 1], t / n


def key_count(duration_s, rate=KEY_RATE):
    return max(8, int(round(rate * duration_s)))


def heglund_taylor_hz(mass_kg):
    """Stride frequency at the preferred galloping speed, 4.44 M^-0.156 Hz (Heglund & Taylor 1988, J Exp Biol
    138:301-318)."""
    return 4.44 * float(mass_kg) ** -0.156

