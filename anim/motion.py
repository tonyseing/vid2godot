"""The baked cycle's motion in numpy: the rig's side-plane forward kinematics, a smooth periodic fit of each joint's
cycle, and two-bone leg IK that pins each planted paw for the keys (60 Hz, at least MIN_KEYS_PER_STRIDE per stride)
facts hands to the bake.

Every fitted joint turns about its bone's side axis, which is world +Y at rest (rigutil.side_axis_local), and so do
its parents: a bone's posed transform is a rotation about world +Y and a shift in x/z, and the side view is exact in
two dimensions. `PlanarRig` poses the contract's rig.json (bone heads, tails, parents) that way; the armature's
dx/dz moves every bone (the bake keys it on the pelvis, which carries every paw).

The smooth fit is a truncated Fourier series per column of the averaged cycle. The trunk, neck, tail and root columns
take the lowest order that keeps RANGE_KEEP of their sampled range (the video's back, neck and tail swing, without
the 24 Hz steps); when that series' jerk is over JERK_MAX times a sine's of the same range at its dominant harmonic (a
neck nodding at twice the stride rate; a root that bobs twice per stride is all second harmonic), the harmonics above
the dominant one are tapered until it is not. The series is then scaled about its mean up to the sampled range when it
falls short of it (fix round 1: the bar compares each joint's range with the video fit's, at the keys; the jerk ratio
does not change with scale). Leg columns keep every harmonic the samples hold (trigonometric interpolation: smooth and
through every averaged sample), since their stance and swing shapes are what the contacts are read from and the IK
reshapes them in stance anyway.

The keys pin each planted paw world-still over its stance less `PIN_RELEASE.inset_keys` (half a key) at each end
(`release_windows`); outside, the correction from the free paw to the pin fades to nothing over
max(BLEND, PIN_RELEASE.span_stride) of the cycle by a quintic. The correction matches the pin's derivatives up to
`PIN_RELEASE`'s orders per axis and edge: by default position, velocity and acceleration in x and position and
velocity in z at both edges, so the paw's position and velocity are continuous at liftoff and touchdown (Task 19
round 3; `edge_jumps` measures the velocity change across each approach and release span, round 4). `start_key`
picks the loop's first key outside every approach and release (round 4).

Vole round 5 (a 5.3 Hz, 46 mm-leg body): every stride gets at least MIN_KEYS_PER_STRIDE keys (`key_count`; the key
rate rises above KEY_RATE for a fast stride), the ease is a share of the stride (`PinRelease.span_stride`), an eased
paw never goes below the ground it is pinned to, and where a pin lies beyond the leg's reach from the fit's pose the
front leg's girdle (scapula) swings about its head (`solve_leg`) and, past the whole straight leg's reach, the body
comes down by the least smooth amount that reaches (`reach_drop`). None of it changes the wolf lope's keys.

numpy only."""
from typing import NamedTuple

import numpy as np

KEY_RATE = 60.0          # the bake's key rate (Hz of the played clip) ...
# ... raised for a fast stride so every stride gets at least MIN_KEYS_PER_STRIDE keys (vole round 5: the 5.3 Hz vole
# had 11 keys at 60 Hz). The floor is the wolf lope's own count (2.51 Hz at 60 Hz: 23.9, rounded to 24), so the wolf
# keys exactly as before; the vole gets 24 keys per stride at 127 Hz. Godot plays any key rate.
MIN_KEYS_PER_STRIDE = 24
REFERENCE_KEYS = 24      # the wolf lope's keys per stride, at which the pin ease's span_stride is 3 key intervals
RANGE_KEEP = 0.95        # a smoothed trunk, neck, tail or root column keeps this share of its sampled range
JERK_MAX = 3.0           # ... with its jerk at most this many times a sine's of the same range (the bar's criterion 4)
BLEND = 0.05             # least cycle fraction over which a paw eases from free to pinned before touchdown, and back
EDGE_SAMPLES = 3600      # samples per cycle the free paw's position, velocity and acceleration are read at


class PinRelease(NamedTuple):
    """How a pinned paw takes up its world-still line at touchdown and lets go of it at liftoff (Task 19 round 3).

    span_stride: the share of the stride the correction from the free paw to the pin fades over (and at least
    `min_span` of it); vole round 5: a share of the stride, not a count of keys, so a short stride's stance is not
    eaten by the eases. The default is the round-4 3 key intervals at the wolf lope's 24 keys per stride. touchdown_orders, liftoff_orders: per axis (x, z), the derivatives of the pin the correction matches at
    that edge: 0 position, 1 also velocity, 2 also acceleration. inset_keys: where the pin starts and ends, in key
    intervals inside the snapped contact window at each end (0 at its edge, negative outside it)."""
    span_stride: float = 3.0 / REFERENCE_KEYS
    touchdown_orders: tuple = (2, 1)
    liftoff_orders: tuple = (2, 1)
    inset_keys: float = 0.5
    min_span: float = BLEND

    def span(self, n_keys=None):
        """The ease as a share of the cycle (the same at every key count; `n_keys` is accepted and unused)."""
        return max(self.min_span, self.span_stride)

    def edge_orders(self, touchdown):
        return np.asarray(self.touchdown_orders if touchdown else self.liftoff_orders)


# The default (Task 19 round 4), chosen by a sweep of the wolf lope measured with chase3d's tools/anim_compare.py
# (task-19-round3-report.md in the chase3d plan folder, fix round 4). It matches the pin's position, velocity and
# acceleration in x and its position and velocity in z at both edges, over 3 key intervals (round 3: 2.6). With the
# loop's start outside the eases (start_key) the literature score's seam row passes at every span; of the settings
# that pass bar criteria 1-6, facts' enforced gates and the real fixture, this one pops least at the contact edges.
# None passes the per-paw edge criterion 8 (every front lift-off is over 2.3 m/s against the old clip's 1.4) or the
# span-wide edge gate (EDGE_GATE_SPAN in facts).
PIN_RELEASE = PinRelease()
REACH_MARGIN = 0.02        # a pinned paw lies within (1 - this) of its leg's whole straight reach (reach_drop)
REACH_OVERSAMPLE = 16    # samples per key interval the body drop is smoothed at
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


def evaluate(coeffs, phases, order, taper=1.0, gain=1.0, derivative=0, keep=1):
    """The series truncated at `order` at cycle phases in [0, 1): its mean plus `gain` times the harmonics up to
    `keep` and the harmonics above it scaled by `taper`; `derivative` > 0 differentiates it in phase that many times."""
    phases = np.asarray(phases, float)
    out = np.full(phases.shape, float(np.real(coeffs[0])) if derivative == 0 else 0.0)
    for k in range(1, order + 1):
        weight = gain * (1.0 if k <= keep else taper) * (2j * np.pi * k) ** derivative
        out = out + 2.0 * np.real(weight * coeffs[k] * np.exp(2j * np.pi * k * phases))
    return out


def jerk_ratio(coeffs, order, taper=1.0, gain=1.0, dense=600, keep=1):
    """The series' largest jerk over a sine's of the same range at the cycle rate (1.0 for a sine; independent of
    scale and time)."""
    grid = np.arange(dense) / dense
    span = float(np.ptp(evaluate(coeffs, grid, order, taper, gain, keep=keep)))
    if span <= 1e-12:
        return 1.0
    return float(np.abs(evaluate(coeffs, grid, order, taper, gain, 3, keep)).max() / (span / 2 * (2 * np.pi) ** 3))


def dominant(coeffs, order):
    """The harmonic (1..order) with the largest amplitude; 1 for a constant column."""
    if order < 1:
        return 1
    return 1 + int(np.argmax(np.abs(np.asarray(coeffs)[1:order + 1])))


def smoothest(coeffs, order, jerk_max=JERK_MAX, dense=600):
    """(taper, gain) for a series of `order`: (1, 1) when its jerk is within `jerk_max` times a sine's at its
    dominant harmonic d (jerk_ratio / d^3); else the least taper of the harmonics above d (in steps of 0.01) that
    brings it there, and the gain that scales the series back to its range. The dominant harmonic is never tapered,
    so a column that is mostly an overtone (a root bobbing twice per stride) is not divided by its near-zero
    fundamental."""
    d = dominant(coeffs, order)
    bound = jerk_max * d ** 3
    if order <= d or jerk_ratio(coeffs, order, dense=dense, keep=d) <= bound:
        return 1.0, 1.0
    grid = np.arange(dense) / dense
    span = float(np.ptp(evaluate(coeffs, grid, order)))
    taper = 1.0
    while taper > 0 and jerk_ratio(coeffs, order, taper, dense=dense, keep=d) > bound:
        taper = round(taper - 0.01, 2)
    taper = max(taper, 0.0)
    tapered = float(np.ptp(evaluate(coeffs, grid, order, taper, keep=d)))
    return taper, (span / tapered if tapered > 1e-12 else 1.0)


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

    def __init__(self, samples, names, project, dense=600):
        samples = np.asarray(samples, float)
        self.names, self.n = list(names), len(samples)
        legs = leg_bones(project)
        grid = np.arange(dense) / dense
        self.coeffs, self.orders, self.ranges, self.taper, self.gain, self.dominant = [], {}, {}, {}, {}, {}
        for i, name in enumerate(self.names):
            column = samples[:, i] if name in ("dx", "dz") else np.unwrap(samples[:, i])
            coeffs = coefficients(column)
            order = max_order(self.n) if name in legs else range_order(column)
            self.dominant[name] = dominant(coeffs, order)
            taper, gain = (1.0, 1.0) if name in legs else smoothest(coeffs, order)
            span = float(np.ptp(column))
            if name not in legs and span > 1e-12:
                # Fix round 1: up to the sampled range itself when the series falls short of it.
                now = float(np.ptp(evaluate(coeffs, grid, order, taper, gain, keep=self.dominant[name])))
                if 1e-12 < now < span:
                    gain *= span / now
            self.taper[name], self.gain[name] = taper, gain
            self.coeffs.append(coeffs)
            self.orders[name] = order
            self.ranges[name] = span

    def at(self, phases):
        """Value vectors (S, V) at cycle phases (any real numbers; one cycle is 1)."""
        phases = np.asarray(phases, float) % 1.0
        return np.stack([evaluate(c, phases, self.orders[n], self.taper[n], self.gain[n], keep=self.dominant[n])
                         for c, n in zip(self.coeffs, self.names)], axis=1)

    def jerk_ratio(self):
        return {n: round(jerk_ratio(c, self.orders[n], self.taper[n], self.gain[n], keep=self.dominant[n]), 4)
                for c, n in zip(self.coeffs, self.names)}

    def roll(self, phase):
        """The same motion started `phase` later: at(u) of the result is at(u + phase) of this one."""
        out = SmoothCycle.__new__(SmoothCycle)
        out.names, out.n, out.orders, out.ranges = self.names, self.n, self.orders, self.ranges
        out.taper, out.gain, out.dominant = self.taper, self.gain, self.dominant
        k = np.arange(len(self.coeffs[0]))
        out.coeffs = [c * np.exp(2j * np.pi * k * phase) for c in self.coeffs]
        return out

    def capped(self, order):
        """The same cycle with every column truncated at `order` harmonics (round 4: facts reads stillness from it).
        A column already within `order` is unchanged."""
        out = SmoothCycle.__new__(SmoothCycle)
        out.names, out.n, out.ranges = self.names, self.n, self.ranges
        out.taper, out.gain, out.dominant, out.coeffs = self.taper, self.gain, self.dominant, self.coeffs
        out.orders = {name: min(o, int(order)) for name, o in self.orders.items()}
        return out

    def kept_range(self, dense=600):
        """Per column, the smoothed range over the sampled range (1.0 for a constant column)."""
        values = self.at(np.arange(dense) / dense)
        return {n: (round(float(np.ptp(values[:, i]) / self.ranges[n]), 4) if self.ranges[n] > 1e-12 else 1.0)
                for i, n in enumerate(self.names)}


def world_pitch_kept(rig, samples, smooth, names, bones, dense=600):
    """Per bone, the smooth cycle's world pitch range (the bone's rotation about +Y with its parents') over the
    sampled cycle's (fix round 1, ruling 7: reported, and a warning under RANGE_KEEP)."""
    sampled, _ = rig.pose(np.asarray(samples, float), names)
    smoothed, _ = rig.pose(smooth.at(np.arange(dense) / dense), names)
    out = {}
    for bone in bones:
        span = float(np.ptp(np.unwrap(sampled[bone])))
        out[bone] = round(float(np.ptp(np.unwrap(smoothed[bone]))) / span, 4) if span > 1e-9 else 1.0
    return out


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
    stance) by the least angle that brings the ankle within reach, and first, where even the straight pair and foot
    cannot reach from the shoulder, swings the chain's girdle bone (`girdle_bone`: a front leg's scapula) about its
    head by the least angle that brings the shoulder within that reach (vole round 5); the foot keeps its world
    orientation either way. Returns (values, reached paw positions)."""
    (upper, lower), foot = ik_bones(chain)
    index = {n: i for i, n in enumerate(names)}
    missing = [b for b in (upper, lower, foot[0]) if b not in index]
    if missing:
        raise ValueError(f"leg IK turns {missing}, which the fit did not solve")
    posed = rig.pose(values, names)
    phi, _ = posed
    foot_phi = phi[foot[0]]                      # the foot keeps this world orientation (before any girdle swing)
    offset = rig.point(posed, paw_bone) - rig.point(posed, foot[0], "head")
    target = np.asarray(target, float)
    l1 = float(np.linalg.norm(rig.head[lower] - rig.head[upper]))
    l2 = float(np.linalg.norm(rig.head[foot[0]] - rig.head[lower]))
    girdle = girdle_bone(rig, chain, upper)
    swing = np.zeros(len(values))
    if pivot and girdle is not None and girdle in index:
        # Vole round 5: where the pair and a pivoting foot cannot reach the pin from the shoulder, the bone above
        # the pair (the scapula) swings about its own head by the least angle that brings the shoulder within
        # reach, as a quadruped's scapula does through a stride; elsewhere nothing changes.
        hinge = rig.point(posed, girdle, "head")
        reach = (l1 + l2 + np.linalg.norm(offset, axis=1)) * (1 - 1e-6)
        swing = toe_pivot(target - hinge, rig.point(posed, upper, "head") - hinge, reach)
        if np.any(swing):
            values = np.array(values, float, copy=True)
            values[:, index[girdle]] += swing
            posed = rig.pose(values, names)
            phi, _ = posed
    parent = rig.parent[upper]
    a = rig.point(posed, upper, "head")
    knee = rig.point(posed, lower, "head")
    end = rig.point(posed, foot[0], "head")
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
                        (foot[0], foot_phi + turn - phi_lower)):
        free = out[:, index[bone]]
        out[:, index[bone]] = free + wrap(angle - free)
    return out, new_end + offset


def girdle_bone(rig, chain, upper):
    """The leg chain's bone above its IK pair, when the pair hangs from it (a front leg's scapula): the bone the
    shoulder swings about. None for a chain that starts at the pair (a hind leg from the hip)."""
    i = chain.index(upper)
    if i == 0 or rig.parent.get(upper) != chain[i - 1]:
        return None
    return chain[i - 1]


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


def release_windows(windows, n_keys, inset_keys=None):
    """Each contact window less `inset_keys` key intervals at each end (PIN_RELEASE's, half a key, by default; a
    negative inset widens it): the phases over which the pin holds the paw (a window that would close is its
    midpoint)."""
    inset = PIN_RELEASE.inset_keys if inset_keys is None else inset_keys
    out = {}
    for chain, (on, off) in windows.items():
        if (on, off) == (0.0, 1.0):
            out[chain] = (0.0, 1.0)
            continue
        end = off if off > on else off + 1.0
        half = inset / n_keys
        a, b = on + half, end - half
        if a > b:
            a = b = 0.5 * (on + end)
        out[chain] = (a % 1.0, b % 1.0) if (a < 0.0 or b >= 1.0) else (a, b)
    return out


def quintic(s):
    """The quintic Hermite basis on [0, 1] for a value, a slope and a curvature at 0 that all reach 0 at 1 with zero
    slope and curvature: (H0, H1, H2)."""
    s = np.asarray(s, float)
    return (1 - 10 * s ** 3 + 15 * s ** 4 - 6 * s ** 5, s - 6 * s ** 3 + 8 * s ** 4 - 3 * s ** 5,
            0.5 * s ** 2 - 1.5 * s ** 3 + 1.5 * s ** 4 - 0.5 * s ** 5)


def plan_keys(rig, smooth, names, project, windows, stride, rest, lift, n_keys, blend=None, release=None,
              reach=True):
    """The bake's keys: n_keys + 1 value vectors at phases k / n_keys (the last repeats the first), the smooth cycle
    with its root lifted by `lift` and every planted paw pinned to its world-still line (`pin_lines`) over its
    contact window less `release.inset_keys` at each end (`release_windows`). Outside, the paw's target is the free
    paw plus the correction where the pin lets go (pin minus free in position, and in velocity and acceleration on the
    axes whose `release.edge_orders` at that edge reach them) faded to nothing by a quintic over `blend` (by default `release.span`:
    max(BLEND, span_stride) of the cycle). `release` is PIN_RELEASE unless given. An eased paw is kept on or above
    its pin's ground. With `reach` (the default) the body comes down where a live target is beyond its leg's whole
    reach (`reach_drop`) and the targets are planned again. Returns (keys, predicted paws per key, IK report; a
    chain's report gains girdleSwingMaxDeg and bodyDropMaxM when they are not zero)."""
    release_setting = PIN_RELEASE if release is None else release
    span = release_setting.span(n_keys) if blend is None else blend
    phases = np.arange(n_keys) / n_keys
    keys = smooth.at(phases)
    keys[:, names.index("dz")] += lift
    shift = float(keys[:, names.index("dx")].mean())       # in place: the bake would remove it; the pins move with it
    keys[:, names.index("dx")] -= shift
    lines = pin_lines(rig, smooth, names, project, windows, stride, rest, lift, shift)
    release = release_windows(windows, n_keys, release_setting.inset_keys)
    grid = np.arange(EDGE_SAMPLES) / EDGE_SAMPLES
    dense = smooth.at(grid)
    dense[:, names.index("dz")] += lift
    dense[:, names.index("dx")] -= shift
    def targets(keys, dense):
        dense_paws = rig.paws(dense, names, project)
        free_paws = rig.paws(keys, names, project)
        out = {}
        for chain in windows:
            line, (on, off) = lines[chain], release[chain]
            end = off if off >= on else off + 1.0
            if (on, off) == (0.0, 1.0):
                end = 1.0

            def pin(u):
                return np.stack([line["x"] - stride * (np.asarray(u, float) - line["mid"]),
                                 np.full(np.shape(u), line["z"])], axis=-1)
            target = free_paws[chain].copy()
            pinned = np.zeros(n_keys, bool)
            for shift_u in (-1.0, 0.0, 1.0):
                u = phases + shift_u
                inside = (u >= on - 1e-9) & (u <= end + 1e-9) & ~pinned
                target[inside] = pin(u[inside])
                pinned |= inside
            eased = np.zeros(n_keys, bool)
            if (on, off) != (0.0, 1.0):
                p = dense_paws[chain]
                velocity = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) * EDGE_SAMPLES / 2
                accel = (np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0)) * EDGE_SAMPLES ** 2
                for edge, sign in ((on, -1.0), (end, 1.0)):
                    order = release_setting.edge_orders(touchdown=sign < 0)
                    i = int(round((edge % 1.0) * EDGE_SAMPLES)) % EDGE_SAMPLES
                    value = pin(edge) - p[i]
                    slope = (np.array([-stride, 0.0]) - velocity[i]) * sign * span * (order >= 1)
                    curve = -accel[i] * span ** 2 * (order >= 2)
                    for shift_u in (-1.0, 0.0, 1.0):
                        s = (phases + shift_u - edge) * sign / span
                        near = (s > 0) & (s < 1) & ~pinned
                        if near.any():
                            h0, h1, h2 = quintic(s[near])
                            target[near] += np.outer(h0, value) + np.outer(h1, slope) + np.outer(h2, curve)
                            eased |= near
                # Vole round 5: an eased paw never goes below the ground it is pinned to (the quintic's matched
                # velocity can carry a fast-descending paw through it; never at the wolf).
                target[eased, 1] = np.maximum(target[eased, 1], line["z"])
            # ... nor a swinging one (a noisy fit's mid-swing dip: the vole's hinds 3 mm into the ground): such a key
            # is lifted to the ground and solved with the rest (the "eased" keys of the report count it).
            under = ~pinned & ~eased & (target[:, 1] < line["z"] - 1e-9)
            target[under, 1] = line["z"]
            eased |= under
            out[chain] = (target, pinned, eased, line)
        return out

    plan = targets(keys, dense)
    drop = reach_drop(rig, keys, names, project, plan, span) if reach else np.zeros(n_keys)
    if np.any(drop > 0):
        # Vole round 5: where a pinned paw is out of the whole leg's reach (pair, pivoting foot and swinging
        # girdle all straight), the body comes down by the least smooth amount that brings it within reach.
        keys[:, names.index("dz")] -= drop
        dense[:, names.index("dz")] -= np.interp(grid, np.append(phases, 1.0), np.append(drop, drop[0]))
        plan = targets(keys, dense)
    report = {}
    for chain, (target, pinned, eased, line) in plan.items():
        live = pinned | eased
        if not live.any():
            continue
        paw_bone = project["paws"][chain]
        chain_bones = project["chains"][chain]
        girdle = girdle_bone(rig, chain_bones, ik_bones(chain_bones)[0][0])
        before = keys[live, names.index(girdle)].copy() if girdle in names else None
        solved, reached = solve_leg(rig, keys[live], names, chain_bones, paw_bone, target[live])
        keys[live] = solved
        residual = np.linalg.norm(reached - target[live], axis=1)
        report[chain] = {"pinnedKeys": int(np.count_nonzero(pinned)), "easedKeys": int(np.count_nonzero(eased)),
                         "edgeKeys": round(span * n_keys, 3), "maxResidualM": round(float(residual.max()), 5),
                         "pinX": round(line["x"], 4), "pinZ": round(line["z"], 4)}
        if before is not None and np.any(solved[:, names.index(girdle)] != before):
            swing = np.abs(wrap(solved[:, names.index(girdle)] - before))
            report[chain]["girdleSwingMaxDeg"] = round(float(np.degrees(swing.max())), 3)
        if np.any(drop > 0):
            report[chain]["bodyDropMaxM"] = round(float(drop.max()), 6)
    keys = np.vstack([keys, keys[:1]])
    paws = rig.paws(keys, names, project)
    return keys, {c: p for c, p in paws.items()}, report


def leg_reach(rig, chain, paw_bone):
    """A leg chain's hinge (the girdle bone's head when the IK pair hangs from one, else the pair's own head) and its
    whole reach from there with every bone straight: girdle, the pair, and the foot from its head to the paw tail."""
    (upper, lower), foot = ik_bones(chain)
    girdle = girdle_bone(rig, chain, upper)
    length = float(np.linalg.norm(rig.head[lower] - rig.head[upper]) + np.linalg.norm(rig.head[foot[0]] - rig.head[lower])
                   + np.linalg.norm(rig.tail[paw_bone] - rig.head[foot[0]]))
    if girdle is not None:
        length += float(np.linalg.norm(rig.head[upper] - rig.head[girdle]))
    return (girdle or upper), length


def reach_drop(rig, keys, names, project, plan, span, margin=REACH_MARGIN, oversample=REACH_OVERSAMPLE):
    """Per key, how far the body must come down (m, >= 0) for every live paw target (pinned or eased) to lie within
    (1 - margin) of its leg's whole reach (`leg_reach`) from the hinge, smoothed: the needed drop at each key,
    max-filtered and then averaged by a Hann window over half the pin's ease `span` of the cycle, so it is at least
    the need at every key and eases in and out. Zero for a body whose legs reach every pin (the wolf)."""
    n_keys = len(keys)
    need = np.zeros(n_keys)
    posed = rig.pose(keys, names)
    for chain, (target, pinned, eased, _) in plan.items():
        live = pinned | eased
        if not live.any():
            continue
        paw_bone = project["paws"][chain]
        hinge_bone, length = leg_reach(rig, project["chains"][chain], paw_bone)
        hinge = rig.point(posed, hinge_bone, "head")
        r = length * (1 - margin)
        dx = target[:, 0] - hinge[:, 0]
        dz = hinge[:, 1] - target[:, 1]
        far = live & (np.hypot(dx, dz) > r)
        if far.any():
            room = np.sqrt(np.maximum(r * r - dx[far] ** 2, 0.0))
            need[far] = np.maximum(need[far], dz[far] - room)
    need = np.maximum(need, 0.0)
    if not need.any():
        return need
    count = n_keys * oversample
    radius = max(1, int(round(0.5 * span * count)))
    fine = np.zeros(count)
    fine[::oversample] = need
    idx = (np.arange(count)[:, None] + np.arange(-radius, radius + 1)[None, :]) % count
    held = fine[idx].max(axis=1)
    window = np.hanning(2 * radius + 3)[1:-1]
    window /= window.sum()
    smooth = (held[idx] * window[None, :]).sum(axis=1)
    return np.maximum(smooth[::oversample], need)


def _pinned_mask(release_window, n_keys):
    on, off = release_window
    end = off if off >= on else off + 1.0
    phases = np.arange(n_keys) / n_keys
    pinned = np.zeros(n_keys, bool)
    for shift_u in (-1.0, 0.0, 1.0):
        u = phases + shift_u
        pinned |= (u >= on - 1e-9) & (u <= end + 1e-9)
    return pinned, on, end


def edge_jumps(paws, release, n_keys, dt, span=None):
    """Per paw with a pinned key, the velocity change at its contact edges against its swing (fix round 1, ruling 4;
    round 4: measured across the whole approach and release span). Over key interval k (key k to k + 1) the paw
    moves at v_k (x and z; the side view's y is 0), and at key k its velocity changes by |v_k - v_(k-1)|; a constant
    body speed adds the same velocity to every key, so the in-place change is the world change the bar's harness
    measures. Touchdown is the largest change from the key before the first eased key of the approach (the keys
    within `span`, a cycle fraction, before the pin: by default PIN_RELEASE.span) to the first pinned key; liftoff,
    from the last pinned key to the key after the last eased key of the release. The swing median over the keys
    that are not pinned. m/s, for `dt` seconds per key."""
    span = PIN_RELEASE.span(n_keys) if span is None else span
    out = {}
    phases = np.arange(n_keys) / n_keys
    for chain, window in release.items():
        p = np.asarray(paws[chain], float)[:, :2]
        pinned, on, end = _pinned_mask(window, n_keys)
        if not pinned.any() or pinned.all():
            continue
        v = (p[1:n_keys + 1] - p[:n_keys]) / dt
        change = np.linalg.norm(v - np.roll(v, 1, axis=0), axis=1)
        first = next(k for k in range(n_keys) if pinned[k] and not pinned[k - 1])
        last = next(k for k in range(n_keys) if pinned[k] and not pinned[(k + 1) % n_keys])

        def eased(edge, sign):
            # keys with 0 < (u - edge) * sign / span < 1, as plan_keys eases them
            d = ((phases - edge) * sign) % 1.0
            return int(np.count_nonzero((d > 1e-9) & (d < span - 1e-9) & ~pinned)) if span > 0 else 0
        before, after = eased(on, -1.0), eased(end, 1.0)
        td_keys = [(first - before - 1 + i) % n_keys for i in range(before + 2)]
        lo_keys = [(last + i) % n_keys for i in range(after + 2)]
        td = max(td_keys, key=lambda k: change[k])
        lo = max(lo_keys, key=lambda k: change[k])
        out[chain] = {"touchdownMps": round(float(change[td]), 4), "liftoffMps": round(float(change[lo]), 4),
                      "touchdownPinnedMps": round(float(change[first]), 4),
                      "liftoffPinnedMps": round(float(change[last]), 4),
                      "swingMedianMps": round(float(np.median(change[~pinned])), 4),
                      "touchdownKey": int(td), "liftoffKey": int(lo), "touchdownKeys": td_keys,
                      "liftoffKeys": lo_keys, "firstPinnedKey": int(first), "lastPinnedKey": int(last)}
    return out


def ease_spans(windows, n_keys, release=None):
    """Per paw, the phase intervals (a, b) over which its pin's correction eases (plan_keys): the approach
    (r_on - span, r_on) and the release (r_end, r_end + span) of its release window (`release_windows`); b may
    pass 1. A whole-cycle stance has none."""
    setting = PIN_RELEASE if release is None else release
    span = setting.span(n_keys)
    out = {}
    for chain, (on, off) in release_windows(windows, n_keys, setting.inset_keys).items():
        if (on, off) == (0.0, 1.0):
            out[chain] = []
            continue
        end = off if off >= on else off + 1.0
        out[chain] = [(on - span, on), (end, end + span)]
    return out


def ease_clearance(u, spans):
    """How far phase u lies outside every ease interval (cyclic, in cycle fractions): the distance to the nearest
    one, or minus the depth inside the deepest one that holds it."""
    best = np.inf
    for intervals in spans.values():
        for a, b in intervals:
            for shift in (-1.0, 0.0, 1.0):
                x = u + shift
                d = -min(x - a, b - x) if a < x < b else min(abs(x - a), abs(x - b))
                best = min(best, d)
    return float(best)


def start_key(windows, n_keys, release=None):
    """The loop's first key (round 4 ruling): of the n_keys keys of a cycle that starts at the phase origin's
    touchdown, the one with the most room outside every paw's contact ease (`ease_spans`), the nearest to the
    origin on a tie. A cyclic clip is the same motion from any key; this puts the seam where no paw eases into or out
    of its pin (mid-suspension or a mid-stance key). Returns (key, {key, outsideEases, clearanceKeys,
    originClearanceKeys})."""
    spans = ease_spans(windows, n_keys, release)
    if not any(spans.values()):
        return 0, {"key": 0, "outsideEases": True, "clearanceKeys": None, "originClearanceKeys": None}
    room = [ease_clearance(k / n_keys, spans) for k in range(n_keys)]
    best = max(range(n_keys), key=lambda k: (round(room[k] * n_keys, 9), -min(k, n_keys - k)))
    return best, {"key": int(best), "outsideEases": bool(room[best] > 1e-9),
                  "clearanceKeys": round(room[best] * n_keys, 4), "originClearanceKeys": round(room[0] * n_keys, 4)}


def played(keys, subsamples=SUBSAMPLES):
    """The keys as the clip plays them: linear interpolation of every column, `subsamples` per key interval, over
    one cycle (phases k / (n * subsamples), the last key excluded)."""
    keys = np.asarray(keys, float)
    n = len(keys) - 1
    t = np.arange(n * subsamples) / subsamples
    lo = np.floor(t).astype(int)
    w = (t - lo)[:, None]
    return (1 - w) * keys[lo] + w * keys[lo + 1], t / n


def key_count(duration_s, rate=KEY_RATE, floor=MIN_KEYS_PER_STRIDE):
    """Keys per stride: `rate` Hz over the stride's `duration_s`, at least `floor` (vole round 5)."""
    return max(int(floor), int(round(rate * duration_s)))


def key_rate(n_keys, duration_s):
    """The played clip's key rate for n_keys over one stride (Hz): KEY_RATE, or more where the floor raised it."""
    return n_keys / duration_s


def heglund_taylor_hz(mass_kg):
    """Stride frequency at the preferred galloping speed, 4.44 M^-0.156 Hz (Heglund & Taylor 1988, J Exp Biol
    138:301-318)."""
    return 4.44 * float(mass_kg) ** -0.156

