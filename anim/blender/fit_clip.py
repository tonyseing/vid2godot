"""Fit the rig to the video frame by frame: pose, rasterise the projected silhouette, score it against the video mask,
nudge, repeat.

The silhouette loss is area-based. The posed model is rasterised in numpy by splatting dense barycentric samples of
its projected triangles onto the pixel grid of one pyramid level, then scored as
    W_IOU * (1 - IoU)  +  W_CHAMFER * symmetric boundary distance (working px)
where the boundary distance is the mean distance from the model's outline pixels to the video outline (a precomputed
distance field) averaged with the mean distance from the video outline to the model's outline. The pose is solved
coarse to fine over the pyramid (1/4, 1/2, 1 of the working resolution). Added to it: marked-point error and a
constant-velocity prior against the previous frames.

Posing runs in numpy (forward kinematics plus linear-blend skinning, checked against Blender's own deform at start;
the script falls back to Blender's depsgraph if they disagree). Ingest registers scale from the first frame's height
alone, which a bent paw throws off by a few percent, so a new fit also solves the camera's scale (about the rest
model's bottom centre, where ingest pins the video): free on every frame of the forward pass, then one shared camera
for refinement. Without source disks this is the legacy median; with disks it is restored jointly with the observed
poses. The refined camera goes to poses.json as "camera" ("cameraScale" and "registrationCamera"
record the change); dx/dz stay in model metres. The first frame has no prior, so it is solved from several camera
scales with coarse grid searches over the legs, and the best is polished with local grids at the middle level.

The prior only steadies the tracking (W_SMOOTH is small next to the silhouette term), and a tracked frame does not
keep the first local minimum it finds: each leg pair is tried exchanged, the spine and neck from rest and every chain
from random restarts, and a frame still below RESTART_IOU is solved again from scratch like the first frame; when
that global solution wins, tracking continues from it without the old velocity. Candidates are compared with the
tracked pose at one pyramid level, and a winner is kept only if, polished, it also wins at full resolution
(anim/fitsearch.py). Approach-C paw keypoints (anim/paws.py) add a soft, capped paw term: a leg folded into the body
costs nothing in IoU but leaves a paw in the mask unclaimed.

Left and right legs give the same side silhouette, so after the forward pass each leg pair's identity is relabelled
per frame by dynamic programming (marks plus smooth velocity; a span refit also takes up to two fixed frames on
each side, pinned as fitted, so the span keeps its neighbours' identity), and the clip is then refined with a
two-sided prior.

A new full fit then runs a periodic pass (`periodic_pass`, PERIODIC_PASS; args "periodic": false turns it off): the
video's period comes from the masks (anim/cycle.py), and each frame is solved again with a weak term toward the first
pass's phase average, weighted per joint by how much the joint varies from cycle to cycle (the outline pins the main
leg joints, which get ~0; a paw, the neck, the chest or a tail tip get the full W_PERIODIC).
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix  # noqa: E402
import rigutil  # noqa: E402
from anim import articulation, cycle, fitconstraints, fitsearch, marks as marks_mod, paws, silhouette_regions  # noqa: E402
from anim.camera import SideCamera  # noqa: E402

LEVELS = (0.25, 0.5, 1.0)                       # pyramid levels, as fractions of the working resolution
LEVEL_STEPS = {0.25: (8.0, 4.0), 0.5: (2.0, 1.0), 1.0: (0.5, 0.25)}   # degrees; working px for the offset
SWEEPS = 4
MAX_REPEAT = 6
SPACING = 0.85                                  # sample spacing in level pixels: no holes when splatting
W_IOU, W_CHAMFER, W_MARKS = 40.0, 1.0, 0.5
W_SMOOTH, W_OFFSET = 2.0, 0.01                  # per rad^2 of prior error; per working px^2 of offset error
W_SWAP_MARKS, W_SWAP_ACCEL = 1.0, 400.0         # identity relabelling: mark px vs rad^2 of acceleration
TIE = 1e-9                                      # relabelling costs this close (relative) go to label 0
VELOCITY = 0.8                                  # share of the last frame's velocity the prior carries forward
SCALE_STEP = 0.0025                             # camera scale change per "degree" of step
REFINE_PASSES = 2
W_SCALE = 1.0                                   # per (1% camera scale change)^2 between frames
OUT_OF_FRAME = 50.0
CROSSING_PX = 4.0
MAX_BOUNDARY = 800
MAX_VIDEO_POINTS = 300
FIRST_SCALES = (0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06)   # starts for the first frame's camera scale
RESTARTS = 4                                    # random restarts per chain on every tracked frame
RESTART_SIGMA = math.radians(25.0)              # their spread about the tracked pose
RESTART_IOU = 0.85                              # a tracked frame below this IoU is also solved from scratch
RESTART_SCALES = (-0.02, 0.0, 0.02)             # that global search's camera-scale starts, about the tracked one
PERIODIC_PASS = True                            # the second pass with a weak periodic prior (a new full fit only)
W_PERIODIC = 5.0                                # per rad^2 from the first pass's phase average, times the joint's weight
PERIODIC_ROUNDS = 3                             # each round's average comes from the round before

args = rigutil.load_args()
base_camera = SideCamera.from_json(args["camera"])
project, brief = args["project"], args["brief"]
arm, skinned, _ = rigutil.load_glb(args["model"])
rigutil.rest_pose(arm)
data = np.load(args["masks"])
outside, masks = data["outside"], data["masks"]
frame_count, height, width = masks.shape
work_scale = args["workScale"]
paw_frames, joint_frames = marks_mod.validate(
    {"frames": args["marks"], "jointFrames": args.get("jointFrames", {})}, project, brief, frame_count)
marks = {int(f): {b: np.array(p, float) * work_scale for b, p in points.items()} for f, points in paw_frames.items()}
joint_marks = {int(f): {b: np.array(p, float) * work_scale for b, p in points.items()}
               for f, points in joint_frames.items()}
source_observations = (articulation.validate(args["sourceObservations"], project, brief, frame_count)
                       if args.get("sourceObservations") is not None else None)
observation_frames = {int(f): frame["observations"] for f, frame in
                      (source_observations or {}).get("frames", {}).items()}
usable_observations = {f: [item for item in items if articulation.usable(item)]
                       for f, items in observation_frames.items()}
constraint_search = {}
region_document = args.get("silhouetteRegions")
region_frames = {int(f): items for f, items in (region_document or {}).get("frames", {}).items()}
region_cache = {}
if region_frames:
    try:
        from anim import raster
    except ModuleNotFoundError as exc:
        if exc.name != "PIL":
            raise
        raise RuntimeError(
            "Region-guided fitting requires Pillow in Blender's Python. Set "
            "VID2GODOT_BLENDER_PYTHON_SITE to a dependency directory containing "
            "Pillow compatible with Blender's Python version and ABI."
        ) from exc

bones = list(dict.fromkeys(b for chain in brief["fitChains"] for b in project["chains"][chain]
                           if b != project["rootBone"]))
axes = [rigutil.side_axis_local(arm, b) for b in bones]
limits = [tuple(math.radians(x) for x in brief["limitsDeg"].get(b, [-60, 60])) for b in bones]
names = ["dx", "dz"] + bones
N = len(names)            # the internal vector carries one more entry: the camera scale
SCALE = N


# ---------------------------------------------------------------------------------------------- posing in numpy
class Rig:
    """Forward kinematics and linear-blend skinning in numpy, for the x/z world coordinates of every vertex."""

    def __init__(self):
        ordered, seen = [], set()
        while len(ordered) < len(arm.data.bones):               # parents before children
            for b in arm.data.bones:
                if b.name not in seen and (b.parent is None or b.parent.name in seen):
                    ordered.append(b)
                    seen.add(b.name)
        self.order = [b.name for b in ordered]
        self.index = {n: i for i, n in enumerate(self.order)}
        self.parent = [self.index[b.parent.name] if b.parent else -1 for b in ordered]
        rest = [np.array(b.matrix_local) for b in ordered]
        self.rest_inv = [np.linalg.inv(m) for m in rest]
        self.rel = [rest[i] if p < 0 else self.rest_inv[p] @ rest[i] for i, p in enumerate(self.parent)]
        self.length = [b.length for b in ordered]
        self.arm_world = np.array(arm.matrix_world)
        self.arm_inv = np.linalg.inv(self.arm_world)
        self.fit = [self.index[b] for b in bones]
        self.axes = [np.array(a) for a in axes]
        coords, groups = [], []
        for obj in skinned:
            co = np.empty(len(obj.data.vertices) * 3)
            obj.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3) @ np.array(obj.matrix_world)[:3, :3].T + np.array(obj.matrix_world)[:3, 3]
            coords.append(co @ self.arm_inv[:3, :3].T + self.arm_inv[:3, 3])
            names_by_group = {g.index: g.name for g in obj.vertex_groups}
            for v in obj.data.vertices:
                groups.append([(self.index[names_by_group[g.group]], g.weight) for g in v.groups
                               if g.group in names_by_group and names_by_group[g.group] in self.index
                               and arm.data.bones[names_by_group[g.group]].use_deform and g.weight > 0])
        self.rest_co = np.concatenate(coords)
        k = max(1, max(len(g) for g in groups))
        self.inf_bone = np.zeros((len(groups), k), int)
        self.inf_w = np.zeros((len(groups), k))
        for v, g in enumerate(groups):
            total = sum(w for _, w in g)
            for j, (b, w) in enumerate(g):
                self.inf_bone[v, j], self.inf_w[v, j] = b, w / total if total > 1e-4 else 0.0
        self.unweighted = self.inf_w.sum(axis=1) < 1e-6
        self.rest_h = np.concatenate([self.rest_co, np.ones((len(self.rest_co), 1))], axis=1)
        self.rest_t = self.rest_h.T.copy()
        self.vertex_ids = np.arange(len(self.rest_h))

    def pose(self, angles):
        basis = [None] * len(self.order)
        for i, axis, angle in zip(self.fit, self.axes, angles):
            basis[i] = np.array(Matrix.Rotation(float(angle), 4, axis))
        mats = []
        for i, p in enumerate(self.parent):
            m = self.rel[i] if p < 0 else mats[p] @ self.rel[i]
            mats.append(m if basis[i] is None else m @ basis[i])
        return mats

    def vertices_xz(self, angles, dx, dz):
        mats = self.pose(angles)
        deform = np.stack([self.arm_world @ m @ r for m, r in zip(mats, self.rest_inv)])[:, [0, 2], :]
        every = (deform.reshape(-1, 4) @ self.rest_t).reshape(len(mats), 2, -1)   # each bone moves every vertex
        out = np.zeros((len(self.rest_h), 2))
        for j in range(self.inf_bone.shape[1]):
            out += self.inf_w[:, j:j + 1] * every[self.inf_bone[:, j], :, self.vertex_ids]
        if self.unweighted.any():
            rest_world = self.rest_h @ self.arm_world.T
            out[self.unweighted] = rest_world[self.unweighted][:, [0, 2]]
        out[:, 0] += dx
        out[:, 1] += dz
        return out

    def tails_xz(self, angles, dx, dz, names_):
        return {n: p for (n, _), p in zip([(n, "tail") for n in names_],
                                          self.points_xz(angles, dx, dz, [(n, "tail") for n in names_]))}

    def points_xz(self, angles, dx, dz, wanted):
        """The x/z world points of [(bone, "head" | "tail")], from one forward-kinematics pass."""
        mats = self.pose(angles)
        out = []
        for n, end in wanted:
            i = self.index[n]
            p = self.arm_world @ mats[i] @ np.array([0.0, self.length[i] if end == "tail" else 0.0, 0.0, 1.0])
            out.append(np.array([p[0] + dx, p[2] + dz]))
        return out


def apply(values):
    arm.location = (float(values[0]), 0.0, float(values[1]))
    for name, axis, angle in zip(bones, axes, values[2:N]):
        rigutil.set_angle(arm, name, axis, float(angle))
    bpy.context.view_layer.update()


rig = Rig()
rng = np.random.default_rng(0)
check = np.zeros(N)
check[:2] = rng.uniform(-0.05, 0.05, 2)
check[2:] = rng.uniform(-0.5, 0.5, N - 2)
if "seed" in args:  # Opt-in restart seed for seed-variance studies; absent keeps the default stream.
    rng = np.random.default_rng(int(args["seed"]))
apply(check)
blender_xz = rigutil.evaluated_vertices(skinned)[:, [0, 2]]
mark_bones = sorted({b for points in marks.values() for b in points})
joint_bones = sorted({b for points in joint_marks.values() for b in points})
if blender_xz.shape == (len(rig.rest_co), 2):
    deform_error_px = max([float(np.abs(blender_xz - rig.vertices_xz(check[2:], check[0], check[1])).max())]
                          + [float(np.abs(np.array(rigutil.bone_point(arm, b))
                                          - rig.tails_xz(check[2:], check[0], check[1], [b])[b]).max())
                             for b in mark_bones]
                          + [float(np.abs(np.array(rigutil.bone_point(arm, b, "head"))
                                          - rig.points_xz(check[2:], check[0], check[1], [(b, "head")])[0]).max())
                             for b in joint_bones]) * base_camera.px_per_m
else:
    deform_error_px = float("inf")
FAST = deform_error_px < 0.25
DEFORM_MODEL = "numpy" if FAST else "depsgraph"
if FAST:
    print(f"VID2GODOT_FIT numpy deform matches Blender (max error {deform_error_px:.4f} px)", flush=True)
else:
    print(f"VID2GODOT_FIT WARNING: numpy deform differs from Blender by {deform_error_px:.3f} px (limit 0.25): "
          f"every pose is evaluated through Blender's depsgraph instead, which is many times slower "
          f"(expect tens of seconds per frame)", flush=True)
rigutil.rest_pose(arm)


PIVOT = None    # the rest model's bottom centre, where ingest pins the video: the camera scales about it


def camera_for(s):
    """The working camera zoomed by s about PIVOT (in working pixels)."""
    c = base_camera
    if PIVOT is None or s == 1.0:
        return SideCamera(c.width, c.height, c.px_per_m * s, c.center_x, c.center_z)
    pu, pv = PIVOT
    return SideCamera(c.width, c.height, c.px_per_m * s,
                      c.center_x + (1 - s) * (c.width / 2 - pu) / (s * c.px_per_m),
                      c.center_z - (1 - s) * (c.height / 2 - pv) / (s * c.px_per_m))


def posed_uv(x):
    cam = camera_for(x[SCALE])
    if FAST:
        return cam.project(rig.vertices_xz(x[2:N], x[0], x[1])), cam
    apply(x)
    return cam.project(rigutil.evaluated_vertices(skinned)[:, [0, 2]]), cam


def mark_points(x, cam, names_, end="tail"):
    if FAST:
        points = rig.points_xz(x[2:N], x[0], x[1], [(b, end) for b in names_])
        return {b: cam.project(p) for b, p in zip(names_, points)}
    apply(x)
    return {b: cam.project(np.array(rigutil.bone_point(arm, b, end))) for b in names_}


def mark_errors(x, f, cam, paw_names=None, joint_names=None):
    """Distances in working pixels; paw tails and articulated joint heads share one manual-point mean."""
    errors = []
    for frames_, wanted, end in ((marks, paw_names, "tail"), (joint_marks, joint_names, "head")):
        targets = frames_.get(f, {})
        selected = list(targets) if wanted is None else [b for b in wanted if b in targets]
        if selected:
            points = mark_points(x, cam, selected, end)
            errors.extend(np.hypot(*(points[b] - targets[b])) for b in selected)
    return errors


def observation_term(x, f, cam, selected=None):
    """Squared source-pixel residual / recorded uncertainty; hidden identity has no invented target."""
    errors = []
    for item in observation_frames.get(f, []):
        if not articulation.usable(item) or (selected is not None and item["bone"] not in selected):
            continue
        point = mark_points(x, cam, [item["bone"]], item["end"])[item["bone"]]
        errors.append(float(np.linalg.norm(point / work_scale - item["point"])) / item["uncertaintyPx"])
    return float(np.mean(np.square(errors))) if errors else 0.0


def observation_residuals(x, f, actual=False):
    """Source-pixel residuals; `actual` uses the same Blender bones as output landmarks."""
    items = usable_observations.get(f, [])
    if not items:
        return np.empty((0, 2))
    wanted = [(item["bone"], item["end"]) for item in items]
    if FAST and not actual:
        points = rig.points_xz(x[2:N], x[0], x[1], wanted)
    else:
        apply(x)
        points = [rigutil.bone_point(arm, bone, end) for bone, end in wanted]
    projected = camera_for(x[SCALE]).project(np.array(points)) / work_scale
    return np.array([(point - item["point"]) / item["uncertaintyPx"]
                     for point, item in zip(projected, items)])


def region_metrics(x, f):
    """Exact all-face working-mask feasibility, not an independent Blender projection certificate."""
    key = (f, x.tobytes())
    if key not in region_cache:
        uv, _ = posed_uv(x)
        level = levels[1.0]
        source = level.masks[f].reshape(level.h, level.w)
        candidate = raster.rasterize(uv.astype(np.float32), faces, level.w, level.h)
        iou = raster.iou(source, candidate)
        local = silhouette_regions.evaluate(source, candidate, region_frames.get(f, []), work_scale)
        if len(region_cache) >= 256:
            region_cache.clear()
        region_cache[key] = (iou, local["pass"])
    return region_cache[key]


def region_key(x, f, loss):
    iou, local_pass = region_metrics(x, f)
    deficit = max(0.0, brief["iouThreshold"] - iou)
    return (deficit > 0, deficit, not local_pass, loss)


def candidate_key(x, f, loss):
    """Original joint disks first, then regional body/local feasibility, then the augmented objective."""
    prefix = ()
    if usable_observations.get(f):
        norms = np.linalg.norm(observation_residuals(x, f), axis=1)
        prefix = (not bool(np.all(norms <= 1.0 + fitconstraints.TOL)),)
    return prefix + (region_key(x, f, loss) if region_frames.get(f) else (loss,))


def prefer_candidate(trial, x, f, prior, k, w_marks=W_MARKS):
    objective = lambda y: score(y, f, prior, k, w_marks)
    if region_frames.get(f):
        return candidate_key(trial, f, objective(trial)) < candidate_key(x, f, objective(x) - 1e-6)
    return fitconstraints.prefer(trial, x, objective, lambda y: observation_residuals(y, f))


# ------------------------------------------------------------------------------------- the rasteriser and pyramid
faces = rigutil.triangles(skinned)
rest_uv = base_camera.project(rig.vertices_xz(np.zeros(N - 2), 0.0, 0.0)) if FAST else None
if rest_uv is None:
    rigutil.rest_pose(arm)
    rest_uv = base_camera.project(rigutil.evaluated_vertices(skinned)[:, [0, 2]])
PIVOT = (0.5 * float(rest_uv[:, 0].min() + rest_uv[:, 0].max()), float(rest_uv[:, 1].max()))
tri = rest_uv[faces]
edge = np.max(np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1), np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)]), axis=0)
signed = 0.5 * ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
# A closed surface's camera-facing triangles cover its whole silhouette: keep the facing with the larger total area.
facing = 1.0 if signed[signed > 0].sum() >= -signed[signed < 0].sum() else -1.0
area = np.maximum(signed * facing, 0.0)


def barycentric_grid(m):
    i, j = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
    keep = i + j <= m
    a, b = i[keep] / m, j[keep] / m
    return np.stack([1 - a - b, a, b], axis=1)


def samples_for(k):
    """Barycentric samples of every camera-facing triangle, spaced at most SPACING level pixels apart (the rest pose
    projects the same areas: every joint turns about the camera's axis). Triangles that need the same subdivision
    share one barycentric grid, so a pose's samples are one small matrix product per group."""
    keep = np.nonzero(area * k * k > 1e-3)[0]
    m = np.clip(np.ceil(edge[keep] * k / SPACING).astype(int), 1, 64)
    groups = []
    for value in np.unique(m):
        group = faces[keep[m == value]]
        groups.append((group.T.copy(), barycentric_grid(int(value)).astype(np.float32)))
    return groups


def erode(mask):
    out = mask.copy()
    out[1:] &= mask[:-1]
    out[:-1] &= mask[1:]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def inside_distance(mask, edges):
    """Distance from every inside pixel to the nearest outline pixel (brute force, chunked)."""
    out = np.zeros(mask.shape, np.float32)
    rows, cols = np.nonzero(mask)
    er, ec = np.nonzero(edges)
    if len(er) == 0:
        return out
    pts = np.stack([er, ec], axis=1).astype(np.float32)
    for start in range(0, len(rows), 4096):
        q = np.stack([rows[start:start + 4096], cols[start:start + 4096]], axis=1).astype(np.float32)
        d2 = (q[:, None, 0] - pts[None, :, 0]) ** 2 + (q[:, None, 1] - pts[None, :, 1]) ** 2
        out[rows[start:start + 4096], cols[start:start + 4096]] = np.sqrt(d2.min(axis=1))
    return out


class Level:
    def __init__(self, k):
        self.k = k
        self.h, self.w = int(round(height * k)), int(round(width * k))
        block = int(round(1 / k))
        self.masks, self.dist, self.points, self.areas = [], [], [], []
        for f in range(frame_count):
            m = masks[f]
            edges = m & ~erode(m)
            d = np.where(m, inside_distance(m, edges), outside[f]).astype(np.float32)
            if block > 1:
                m = m[:self.h * block, :self.w * block].reshape(self.h, block, self.w, block).mean(axis=(1, 3)) >= 0.5
                d = d[:self.h * block, :self.w * block].reshape(self.h, block, self.w, block).min(axis=(1, 3)) * k
                edges = m & ~erode(m)
            r, c = np.nonzero(edges)
            pick = np.linspace(0, len(r) - 1, min(MAX_VIDEO_POINTS, len(r))).astype(int) if len(r) else np.zeros(0, int)
            self.points.append(np.stack([r[pick], c[pick]], axis=1).astype(np.float32))
            self.masks.append(m.ravel())
            self.dist.append(d)
            self.areas.append(int(np.count_nonzero(m)))
        self.groups = samples_for(k)
        self.samples = sum(c.shape[1] * len(g) for c, g in self.groups)
        self.points2 = [(p ** 2).sum(axis=1) for p in self.points]

    def occupancy(self, uv):
        # The scorer (raster.rasterize, PIL) truncates vertex coordinates, so a point at u lands in pixel floor(u):
        # the samples follow the same convention, so the loss optimises what the gate measures.
        q = (uv * self.k).astype(np.float32)
        qu, qv = q[:, 0].copy(), q[:, 1].copy()
        occ = np.zeros(self.h * self.w, bool)
        lost = 0
        for corners, grid in self.groups:
            u = (grid @ qu[corners].reshape(3, -1)).ravel()
            v = (grid @ qv[corners].reshape(3, -1)).ravel()
            ok = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
            if not ok.all():
                lost += int(np.count_nonzero(~ok))
                u, v = u[ok], v[ok]
            occ[v.astype(np.int32) * self.w + u.astype(np.int32)] = True     # truncation is floor here: >= 0
        return occ, lost

    def loss(self, uv, f, include_regions=False):
        occ, lost = self.occupancy(uv)
        inter = np.count_nonzero(occ & self.masks[f])
        model_area = np.count_nonzero(occ) + lost
        union = model_area + self.areas[f] - inter
        iou = inter / union if union else 1.0
        grid = occ.reshape(self.h, self.w)
        regional = None
        if include_regions:
            source = self.masks[f].reshape(self.h, self.w)
            candidate = grid
            items = region_frames.get(f, [])
            if self.k < 1.0 and items:
                # Ingest and pyramid dimensions each round once. Preserve the
                # source-coordinate crop by extending a missing coarse edge with
                # identical zeros, not by changing either crop or inventing overlap.
                need_w = max(self.w, max(math.ceil(item["bounds"][2] * work_scale * self.k) for item in items))
                need_h = max(self.h, max(math.ceil(item["bounds"][3] * work_scale * self.k) for item in items))
                if need_w > self.w or need_h > self.h:
                    padding = ((0, need_h - self.h), (0, need_w - self.w))
                    source, candidate = np.pad(source, padding), np.pad(candidate, padding)
            regional = silhouette_regions.evaluate(source, candidate, items, work_scale * self.k)
        edges = grid & ~erode(grid)
        r, c = np.nonzero(edges)
        if len(r) == 0:
            result = (W_IOU + W_CHAMFER * OUT_OF_FRAME, 0.0)
            return (result[0], iou, regional["loss"], regional["pass"]) if include_regions else result
        a = float(self.dist[f][r, c].mean())
        if len(r) > MAX_BOUNDARY:
            pick = np.linspace(0, len(r) - 1, MAX_BOUNDARY).astype(int)
            r, c = r[pick], c[pick]
        pts = self.points[f]
        if len(pts):
            q = np.stack([r, c]).astype(np.float32)                          # (2, M)
            d2 = self.points2[f][:, None] + (q ** 2).sum(axis=0)[None, :] - 2.0 * (pts @ q)
            b = float(np.sqrt(np.maximum(d2.min(axis=1), 0.0)).mean())
        else:
            b = 0.0
        chamfer = 0.5 * (a + b) / self.k + OUT_OF_FRAME * lost / max(model_area, 1)
        result = (W_IOU * (1.0 - iou) + W_CHAMFER * chamfer, iou)
        return (*result, regional["loss"], regional["pass"]) if include_regions else result


t0 = time.time()
levels = {k: Level(k) for k in LEVELS}
print(f"VID2GODOT_FIT pyramid ready in {time.time() - t0:.1f}s; samples "
      + ", ".join(f"{k}: {levels[k].samples}" for k in LEVELS), flush=True)


# ----------------------------------------------------------------------------------------- automatic paw keypoints
# Approach C as soft marks (anim/paws.py): the paws found in each mask, kept where they persist from frame to frame;
# each fitted paw (the middle of its paw bone) should be near a detected paw its leg can reach, each detected paw near
# a fitted paw (or the tail tip or the head, which may hang below the body line), and a paw drawn inside the torso
# costs its depth while a detected paw has no leg. The term is zero within PAW_DEADZONE, so a clean fit is unchanged.
paw_chains = [c for pair in project["legPairs"] for c in pair if c in brief["fitChains"] and c in project["paws"]]
paw_bones = [project["paws"][c] for c in paw_chains]
root_bones = [project["chains"][c][0] for c in paw_chains]
other_bones = [b for b in ([project["chains"]["tail"][-1]] if "tail" in project["chains"] else [])
               + ([project["headBone"]] if project.get("headBone") else []) if b in rig.index]
KEY_POINTS = ([(b, "head") for b in paw_bones] + [(b, "tail") for b in paw_bones]
              + [(b, "head") for b in root_bones] + [(b, "tail") for b in other_bones])
REACH_SLACK = 1.2               # a leg reaches at most this share more than its rest length (camera scale, stretch)


def key_points(x, cam):
    """(fitted paws, leg roots, other extremities) in working px."""
    if FAST:
        pts = rig.points_xz(x[2:N], x[0], x[1], KEY_POINTS)
    else:
        apply(x)
        pts = [np.array(rigutil.bone_point(arm, b, end)) for b, end in KEY_POINTS]
    uv = cam.project(np.array(pts)) if pts else np.zeros((0, 2))
    n = len(paw_bones)
    return 0.5 * (uv[:n] + uv[n:2 * n]), uv[2 * n:3 * n], uv[3 * n:]


t0 = time.time()
detected = paws.track([np.array([p["blob"] for p in paws.find_paws(m, max_paws=max(2, len(paw_bones)))])
                       for m in masks]) if paw_bones else [np.zeros((0, 2))] * frame_count
body_rows = [paws.body_line(m) if m.any() else 0 for m in masks]
if paw_bones:
    rest_paws, rest_roots, _ = key_points(np.concatenate([np.zeros(N), [1.0]]), base_camera)
    reach = REACH_SLACK * np.hypot(*(rest_paws - rest_roots).T)
print(f"VID2GODOT_FIT paw keypoints in {time.time() - t0:.1f}s: "
      f"{sum(len(d) for d in detected)} over {frame_count} frames", flush=True)


def paw_term(x, f, cam):
    if not paw_bones or not len(detected[f]):
        return 0.0
    fitted, roots, others = key_points(x, cam)
    return paws.penalty(fitted, detected[f], body_rows[f], masks[f], roots, reach, others)


# ------------------------------------------------------------------------------------------------ the objective
PERIODIC = {}       # frame -> (target values, per-value weights) while the periodic pass runs


def score(x, f, prior, k, w_marks=W_MARKS):
    uv, cam = posed_uv(x)
    if region_frames.get(f):
        loss, _, regional, _ = levels[k].loss(uv, f, include_regions=True)
        loss += regional
    else:
        loss, _ = levels[k].loss(uv, f)
    loss += paw_term(x, f, cam)
    loss += observation_term(x, f, cam)
    if w_marks and (f in marks or f in joint_marks):
        errors = mark_errors(x, f, cam)
        if errors:
            loss += w_marks * float(np.mean(errors))
    if prior is not None:
        target, weight = prior
        loss += weight * W_SMOOTH * float(np.sum((x[2:N] - target[2:N]) ** 2))
        loss += weight * W_OFFSET * float(np.sum(((x[:2] - target[:2]) * base_camera.px_per_m) ** 2))
        loss += weight * W_SCALE * float(((x[SCALE] - target[SCALE]) / 0.01) ** 2)
    if f in PERIODIC:
        target, weights = PERIODIC[f]
        loss += W_PERIODIC * float(np.sum(weights * (x[:N] - target) ** 2))
    return float(loss)


def clamp(x):
    for i, (low, high) in enumerate(limits):
        x[i + 2] = min(max(x[i + 2], low), high)
    x[SCALE] = min(max(x[SCALE], 0.8), 1.25)
    return x


def delta_for(i, step):
    if i < 2:
        return step / base_camera.px_per_m
    if i == SCALE:
        return step * SCALE_STEP
    return math.radians(step)


def child_index(i):
    """The fitted bone that follows bone i in its chain, if any."""
    for chain in brief["fitChains"]:
        members = [b for b in project["chains"][chain] if b in names]
        if names[i] in members and members.index(names[i]) + 1 < len(members):
            return names.index(members[members.index(names[i]) + 1])
    return None


# Moves: each parameter alone, and each joint turned with its child turned back (the child keeps its world angle and
# only travels), so a knee can follow a hip without first making the fit worse.
MOVES = [(i, None) for i in range(N)] + [(i, child_index(i)) for i in range(2, N) if child_index(i) is not None]


def descend(x, f, prior, free_scale=False, schedule=LEVELS, w_marks=W_MARKS, only=None):
    """Greedy coordinate descent, coarse to fine. `only` limits the moves to those indices."""
    moves = MOVES + ([(SCALE, None)] if free_scale else [])
    if only is not None:
        moves = [(i, c) for i, c in moves if i in only and (c is None or c in only)]
    best = None
    for k in schedule:
        best = score(x, f, prior, k, w_marks)
        for step in LEVEL_STEPS[k]:
            for _ in range(SWEEPS):
                improved = False
                for i, child in moves:
                    for sign in (1.0, -1.0):
                        moved = False
                        for _ in range(MAX_REPEAT):          # keep going while the same move still helps
                            trial = x.copy()
                            trial[i] += sign * delta_for(i, step)
                            if child is not None:
                                trial[child] -= sign * delta_for(i, step)
                            trial = clamp(trial)
                            if trial[i] == x[i]:
                                break
                            loss = score(trial, f, prior, k, w_marks)
                            wins = (candidate_key(trial, f, loss) < candidate_key(x, f, best - 1e-6)
                                    if usable_observations.get(f) or region_frames.get(f) else loss < best - 1e-6)
                            if not wins:
                                break
                            x, best, improved, moved = trial, loss, True, True
                        if moved:
                            break
                if not improved:
                    break
        if usable_observations.get(f) and only is None:
            # One coupled pass per pyramid level, never per grid/restart candidate.
            scales = np.array([delta_for(i, max(1.0, LEVEL_STEPS[k][0])) for i in range(N + 1)])
            if not free_scale:
                scales[SCALE] = 0.0
            refined, diagnostic = fitconstraints.refine(
                x, lambda y: score(y, f, prior, k, w_marks),
                lambda y: observation_residuals(y, f), clamp, scales, max_iterations=12)
            if not region_frames.get(f) or prefer_candidate(refined, x, f, prior, k, w_marks):
                x = refined
            diagnostic["level"] = k
            constraint_search.setdefault(f, []).append(diagnostic)
            best = score(x, f, prior, k, w_marks)
    return x, best


def initial_offset(f, s):
    """Align the rest model's projected centre and lowest point with the video mask of frame f."""
    x = np.zeros(N + 1)
    x[SCALE] = s
    uv, _ = posed_uv(x)
    rows, cols = np.nonzero(masks[f])
    x[0] = (cols.mean() - uv[:, 0].mean()) / (base_camera.px_per_m * s)
    x[1] = (uv[:, 1].max() - rows.max()) / (base_camera.px_per_m * s)
    return x


leg_chains = [c for pair in project["legPairs"] for c in pair if c in brief["fitChains"]]

pairs = []
for a, b in project["legPairs"]:
    ia = [names.index(n) for n in project["chains"][a] if n in names]
    ib = [names.index(n) for n in project["chains"][b] if n in names]
    if ia and len(ia) == len(ib):
        pairs.append((a, b, ia, ib))




def grid_search(x, f, idx, grid, k=LEVELS[0], around=False):
    best, keep = score(x, f, None, k), x[idx].copy()
    best_key = candidate_key(x, f, best)
    for combo in np.array(np.meshgrid(*[grid] * len(idx), indexing="ij")).reshape(len(idx), -1).T:
        trial = x.copy()
        trial[idx] = combo + (x[idx] if around else 0.0)
        loss = score(clamp(trial), f, None, k)
        key = candidate_key(trial, f, loss)
        if key < best_key:
            best, keep, best_key = loss, trial[idx].copy(), key
    x[idx] = keep
    return x


def chain_search(x, f):
    """Coarse grids: each leg pair's first two joints together (the pair overlaps in the silhouette), then each leg
    a chain at a time (its first two joints together, then each further joint): a global start for an unknown first
    pose."""
    for a, b, ia, ib in pairs:
        if len(ia) >= 2:
            x = grid_search(x, f, ia[:2] + ib[:2], np.radians(np.arange(-40, 41, 20)))
    grid = np.radians(np.arange(-45, 46, 15))
    for _ in range(2):
        for chain in leg_chains:
            idx = [names.index(b) for b in project["chains"][chain] if b in names]
            if len(idx) >= 2:
                x = grid_search(x, f, idx[:2], grid)
            for i in idx[2:]:
                x = grid_search(x, f, [i], grid)
    return x


def first_frame(f, known_scale, scales=FIRST_SCALES):
    """No previous frame: a global start (each start scale, then a coarse grid per leg), solved coarse first; the
    three best coarse solutions are refined to the middle level and the best of them to full resolution."""
    starts = [known_scale] if known_scale else scales
    coarse = []
    for s in starts:
        x = chain_search(initial_offset(f, s), f)
        coarse.append(descend(x, f, None, free_scale=known_scale is None, schedule=LEVELS[:1]))
    coarse.sort(key=lambda item: candidate_key(item[0], f, item[1]))
    mid = [descend(x, f, None, free_scale=known_scale is None, schedule=LEVELS[1:2]) for x, _ in coarse[:3]]
    if usable_observations.get(f) or region_frames.get(f):
        # Compare a bounded shortlist at the actual full-resolution objective.
        fine = [descend(chain_polish(x, f), f, None, free_scale=known_scale is None,
                        schedule=LEVELS[1:]) for x, _ in mid]
        return min(fine, key=lambda item: candidate_key(item[0], f, item[1]))
    x, _ = min(mid, key=lambda item: candidate_key(item[0], f, item[1]))
    x = chain_polish(x, f)
    return descend(x, f, None, free_scale=known_scale is None, schedule=LEVELS[1:])


def chain_polish(x, f):
    """Local grids around each leg at the middle level: a knee and the paw below it often have to move together
    to leave a fine-level local minimum."""
    near = np.radians(np.arange(-12, 13, 4))
    wide = np.radians(np.arange(-45, 46, 15))
    for chain in leg_chains:
        idx = [names.index(b) for b in project["chains"][chain] if b in names]
        if len(idx) < 2:
            continue
        best, keep = score(x, f, None, LEVELS[1]), x.copy()
        best_key = candidate_key(x, f, best)
        for a in near:
            for b in near:
                trial = x.copy()
                trial[idx[0]] += a
                trial[idx[1]] += b
                for i in idx[2:]:
                    trial = grid_search(clamp(trial), f, [i], wide, LEVELS[1])
                loss = score(clamp(trial), f, None, LEVELS[1])
                key = candidate_key(trial, f, loss)
                if key < best_key:
                    best, keep, best_key = loss, trial, key
        x = keep
    return x


def predict(history):
    if not history:
        return None
    if len(history) == 1:
        return history[-1].copy()
    out = history[-1] + VELOCITY * (history[-1] - history[-2])
    out[SCALE] = history[-1][SCALE]
    return out


chain_indices = {chain: [names.index(b) for b in project["chains"][chain] if b in names]
                 for chain in brief["fitChains"]}
trunk = [i for chain in ("spine", "neck") if chain in chain_indices for i in chain_indices[chain]]


def frame_iou(x, f):
    uv, _ = posed_uv(x)
    return levels[1.0].loss(uv, f)[1]


def search(x, f, prior, free_scale):
    """Leave the tracked pose's local minimum: each leg pair exchanged, the spine and neck from rest, and RESTARTS
    random restarts of every chain (about the best pose so far), each solved over its own joints at the middle level.
    The tracked pose and every candidate are scored there alike (prior included); a winner is polished at full
    resolution and kept only if its full-resolution objective beats the tracked pose's (fitsearch.search)."""
    k = LEVELS[1]
    candidates = [(lambda best, y=swapped(x, pair): y, pair[2] + pair[3]) for pair in pairs]
    if trunk:
        rest = x.copy()
        rest[trunk] = 0.0
        candidates.append((lambda best, y=rest: y, trunk + [0, 1]))

    def perturb(best, idx):
        y = best.copy()
        y[idx] += rng.normal(0.0, RESTART_SIGMA, len(idx))
        return y
    for chain, idx in chain_indices.items():
        candidates += [(lambda best, idx=idx: perturb(best, idx), idx)] * RESTARTS
    if usable_observations.get(f) or region_frames.get(f):
        best = x
        for make, idx in candidates:
            trial = descend(clamp(make(best)), f, prior, schedule=(k,), only=set(idx))[0]
            if prefer_candidate(trial, best, f, prior, k):
                best = trial
        if best is x:
            return x
        polished = descend(best, f, prior, free_scale=free_scale, schedule=LEVELS[2:])[0]
        return polished if prefer_candidate(polished, x, f, prior, 1.0) else x
    return fitsearch.search(
        x, candidates,
        solve=lambda y, idx: descend(clamp(y), f, prior, schedule=(k,), only=set(idx))[0],
        coarse=lambda y: score(y, f, prior, k),
        fine=lambda y: score(y, f, prior, 1.0),
        polish=lambda y: descend(y, f, prior, free_scale=free_scale, schedule=LEVELS[2:])[0])


def restart(x, f, free_scale):
    """A tracked frame still below RESTART_IOU: solve it again from scratch (the first frame's global search, about
    the tracked camera scale) and keep whichever has the lower silhouette loss. Returns the pose and whether the
    global solution won (tracking then restarts from it, without the old velocity)."""
    if region_frames.get(f):
        iou, local_pass = region_metrics(x, f)
        if iou >= brief["iouThreshold"] and local_pass:
            return x, False
    elif frame_iou(x, f) >= RESTART_IOU:
        return x, False
    if free_scale:
        g, _ = first_frame(f, None, tuple(x[SCALE] + d for d in RESTART_SCALES))
    else:
        g, _ = first_frame(f, x[SCALE])
    wins = (prefer_candidate(g, x, f, None, 1.0) if usable_observations.get(f) or region_frames.get(f)
            else score(g, f, None, 1.0) < score(x, f, None, 1.0) - 1e-6)
    if wins:
        return g, True
    return x, False


def shared_camera(xs, frames):
    """Choose one camera with the observed poses jointly, retaining feasible shared candidates."""
    median = float(np.median([xs[f][SCALE] for f in frames]))
    regional = any(region_frames.get(f) for f in frames)
    has_disks = any(usable_observations.get(f) for f in frames)
    observed = [f for f in frames if usable_observations.get(f) or region_frames.get(f)]
    if not observed:
        for f in frames:
            xs[f][SCALE] = median
        return median

    # Only observed poses need coupled restoration. The last coordinate is the
    # single camera shared by every frame, never an independently fitted scale.
    slots = {f: slice(i * N, (i + 1) * N) for i, f in enumerate(observed)}
    start = np.concatenate([xs[f][:N] for f in observed] + [[median]])

    def pose(y, f):
        return np.append(y[slots[f]] if f in slots else xs[f][:N], y[-1])

    def residuals(y):
        return np.concatenate([observation_residuals(pose(y, f), f) for f in observed])

    def objective(y):
        return sum(score(pose(y, f), f, None, 1.0) for f in frames)

    def bounded(y):
        for f in observed:
            value = clamp(pose(y, f))
            y[slots[f]], y[-1] = value[:N], value[SCALE]
        return y

    def key(y):
        norms = np.linalg.norm(residuals(y), axis=1)
        if regional:
            value = objective(y)
            metrics = {f: region_metrics(pose(y, f), f) for f in frames}
            threshold = brief["iouThreshold"]
            deficit = max(0.0, threshold - float(np.mean([row[0] for row in metrics.values()])))
            deficit += sum(max(0.0, threshold - metrics[f][0]) for f in observed if region_frames.get(f))
            return (not bool(np.all(norms <= 1.0 + fitconstraints.TOL)), deficit > 0, deficit,
                    any(not metrics[f][1] for f in observed if region_frames.get(f)), value)
        return (not bool(np.all(norms <= 1.0 + fitconstraints.TOL)), objective(y))

    step = max(1.0, LEVEL_STEPS[LEVELS[0]][0])
    scales = np.concatenate([np.tile([delta_for(i, step) for i in range(N)], len(observed)),
                             [delta_for(SCALE, step)]])
    best = start.copy()
    # The median is only a seed: an observed frame may already supply the only
    # viable shared scale. Judge all disks together before replacing that seed.
    seeds = dict.fromkeys([median] + [float(xs[f][SCALE]) for f in observed])
    for scale in seeds:
        trial = start.copy()
        trial[-1] = scale
        if key(trial) < key(best):
            best = trial.copy()
        if has_disks:
            restored, _ = fitconstraints.refine(trial, objective, residuals, bounded, scales)
            if key(restored) < key(best):
                best = restored
        if regional:
            # Refine only existing observed-pose parameters at one shared candidate
            # scale. Each accepted update must retain the aggregate body/local gate.
            for f in observed:
                refined, _ = descend(pose(trial, f), f, None, schedule=(1.0,))
                candidate = trial.copy()
                candidate[slots[f]] = refined[:N]
                if key(candidate) < key(trial):
                    trial = candidate
            if key(trial) < key(best):
                best = trial
    for f in frames:
        xs[f] = pose(best, f)
    return float(best[-1])




# ------------------------------------------------------------------------------------------- identity relabelling
def swapped(x, pair):
    _, _, ia, ib = pair
    y = x.copy()
    y[ia], y[ib] = x[ib], x[ia]
    return y


def better(c, best):
    """c beats best by more than a near-tie (TIE, relative): a near-tie keeps the earlier candidate, label 0 first."""
    if not math.isfinite(best):
        return c < best
    return c < best - TIE * max(abs(best), abs(c))


def fixed_neighbours(xs, frames):
    """Up to two fixed frames immediately before and after the frames being relabelled (a span refit has them)."""
    inside = set(frames)
    before, after = [], []
    if not frames:
        return before, after
    for g in (frames[0] - 1, frames[0] - 2):
        if g not in xs or g in inside:
            break
        before.insert(0, g)
    for g in (frames[-1] + 1, frames[-1] + 2):
        if g not in xs or g in inside:
            break
        after.append(g)
    return before, after


def relabel(xs, frames):
    """Left and right legs of a pair cast the same silhouette: choose, per frame, which fitted chain is which, by
    dynamic programming: source-disk feasibility first, then the original marks and acceleration costs.

    A span refit adds up to two fixed frames on each side with their label pinned to 0 (as fitted), so acceleration
    ties the span's identity to its neighbours': without them, keeping or exchanging a pair for the whole span costs
    the same. Near-ties (TIE, relative) go to label 0."""
    before, after = fixed_neighbours(xs, frames)
    seq, pinned = before + list(frames) + after, set(before + after)
    inf = float("inf")
    changed = 0
    for pair in pairs:
        a, b, ia, ib = pair
        paw_a, paw_b = project["paws"].get(a), project["paws"].get(b)
        joint_names = [bone for chain in (a, b) for bone in project["chains"][chain] if bone in bones]
        options = []
        for f in seq:
            opts = []
            source_error = (np.linalg.norm(observation_residuals(xs[f], f), axis=1)
                            if usable_observations.get(f) else np.empty(0))
            for label in (0, 1):
                y = swapped(xs[f], pair) if label else xs[f]
                cost = inf if label and f in pinned else 0.0
                if f not in pinned and (f in marks or f in joint_marks):
                    cam = camera_for(y[SCALE])
                    errors = mark_errors(y, f, cam, [p for p in (paw_a, paw_b) if p], joint_names)
                    if errors:
                        cost = W_SWAP_MARKS * float(np.mean(errors))
                if f not in pinned and f in observation_frames:
                    cost += observation_term(y, f, camera_for(y[SCALE]),
                                             joint_names + [p for p in (paw_a, paw_b) if p])
                if label and region_frames.get(f) and candidate_key(y, f, 0.0) > candidate_key(xs[f], f, 0.0):
                    cost = inf
                error = (np.linalg.norm(observation_residuals(y, f), axis=1)
                         if len(source_error) else source_error)
                if label and len(source_error):
                    # Preserve every established disk, even when another disk is
                    # still unresolved. A swap cannot trade away source evidence.
                    if np.any(error > np.maximum(1.0, source_error) + fitconstraints.TOL):
                        cost = inf
                infeasible = (int(np.any(error > 1.0 + fitconstraints.TOL))
                              if math.isfinite(cost) else inf)
                opts.append((y[ia + ib], cost, infeasible))
            options.append(opts)
        n = len(seq)
        if n < 3:
            # There is no acceleration term with fewer than three samples.
            # Source-observed frames still need identity selection; leave the
            # historical no-observation short-clip behavior untouched.
            for f, opts in zip(seq, options):
                if f in pinned or not usable_observations.get(f):
                    continue
                if (opts[1][2] < opts[0][2]
                        or (opts[1][2] == opts[0][2] and better(opts[1][1], opts[0][1]))):
                    xs[f] = swapped(xs[f], pair)
                    changed += 1
            continue
        # state = (label at t-1, label at t); cost[t][s]
        states = [(0, 0), (0, 1), (1, 0), (1, 1)]
        cost = {(l1, l2): options[0][l1][1] + options[1][l2][1] for l1, l2 in states}
        violations = {(l1, l2): options[0][l1][2] + options[1][l2][2] for l1, l2 in states}
        back = []
        for t in range(2, n):
            new, new_violations, ptr = {}, {}, {}
            for l1, l2 in states:
                best, best_violation, arg = inf, inf, 0
                for l0 in (0, 1):
                    accel = options[t][l2][0] - 2 * options[t - 1][l1][0] + options[t - 2][l0][0]
                    c = cost[(l0, l1)] + W_SWAP_ACCEL * float(np.sum(accel ** 2))
                    v = violations[(l0, l1)]
                    if v < best_violation or (v == best_violation and better(c, best)):
                        best, best_violation, arg = c, v, l0
                new[(l1, l2)], ptr[(l1, l2)] = best + options[t][l2][1], arg
                new_violations[(l1, l2)] = best_violation + options[t][l2][2]
            cost = new
            violations = new_violations
            back.append(ptr)
        state = states[0]
        for s in states[1:]:
            if (violations[s] < violations[state]
                    or (violations[s] == violations[state] and better(cost[s], cost[state]))):
                state = s
        labels = [state[0], state[1]]
        for ptr in reversed(back):
            labels.insert(0, ptr[(labels[0], labels[1])])
        for f, label in zip(seq, labels):
            if label and f not in pinned:
                xs[f] = swapped(xs[f], pair)
                changed += 1
    return changed


# ------------------------------------------------------------------------------------------------------ the clip
span = args.get("span") or [0, frame_count]
previous = json.loads(Path(args["init"]).read_text()) if args.get("init") else None
cam_scale = float(previous.get("cameraScale", 1.0)) if previous else None
entries = list(previous["frames"]) if previous else [None] * frame_count
projection = np.load(args["projection"])["uv"] if previous else np.zeros((frame_count, len(rest_uv), 2), np.float32)
xs = {}
for f in range(frame_count):
    if entries[f] and (f < span[0] or f >= span[1]):
        xs[f] = np.array(list(entries[f]["values"]) + [cam_scale])
history = [xs[f] for f in (span[0] - 2, span[0] - 1) if f in xs]
frames = args.get("frameIndices", list(range(span[0], span[1])))
if (not isinstance(frames, list) or len(set(frames)) != len(frames)
        or any(isinstance(f, bool) or not isinstance(f, int) or not 0 <= f < frame_count for f in frames)):
    raise ValueError("frameIndices must be distinct ingested frame indices")
frames = sorted(frames)
if args.get("frameIndices") is not None and (not frames or previous is not None or args.get("span")):
    raise ValueError("Sparse preflight needs frames and cannot initialize or overwrite a canonical span fit")
t_start = time.time()
# A new fit tracks scale freely, then restores one shared camera when source disks are present.
scale_free = cam_scale is None
for f in frames:
    t_frame = time.time()
    if args.get("frameIndices") is not None:
        history = []  # Representative frames are independent solves, not adjacent motion samples.
    prior = predict(history)
    if prior is None:
        x, loss = first_frame(f, cam_scale)
    else:
        x, loss = descend(clamp(prior.copy()), f, (prior, 1.0), free_scale=scale_free)
        x = search(x, f, (prior, 1.0), scale_free)
        x, jumped = restart(x, f, scale_free)
        loss = score(x, f, None if jumped else (prior, 1.0), 1.0)
        if jumped:
            history = []
    xs[f] = x
    history = (history + [x])[-2:]
    print(f"VID2GODOT_FIT frame {f + 1}/{frame_count} loss {loss:.3f} scale {x[SCALE]:.4f} "
          f"({time.time() - t_frame:.1f}s)", flush=True)

if scale_free:
    cam_scale = shared_camera(xs, frames)
    print(f"VID2GODOT_FIT camera scale {cam_scale:.4f}", flush=True)
changed = relabel(xs, frames) if args.get("frameIndices") is None else 0
print(f"VID2GODOT_FIT relabelled {changed} leg-pair frames", flush=True)
for p in range(REFINE_PASSES):
    for f in frames:
        neighbours = ([xs[g] for g in (f - 1, f + 1) if g in xs]
                      if args.get("frameIndices") is None else [])
        target = np.mean(neighbours, axis=0) if neighbours else None
        xs[f], _ = descend(xs[f], f, (target, 1.0) if target is not None else None,
                           schedule=LEVELS[1:] if p == 0 and scale_free else LEVELS[2:])
    changed = relabel(xs, frames) if args.get("frameIndices") is None else 0
    print(f"VID2GODOT_FIT refine pass {p + 1}: relabelled {changed}", flush=True)
print(f"VID2GODOT_FIT solved {len(frames)} frames in {time.time() - t_start:.1f}s", flush=True)


def periodic_pass(xs, frames):
    """The second pass: the video repeats (anim/cycle.py finds its period in the masks), but joints the silhouette
    barely constrains (a paw, the neck, the chest, a tail tip) come out of the first pass different from cycle to
    cycle. Each frame is solved again with an extra term pulling its joints toward the first pass's phase average,
    weighted per joint by that joint's cycle-to-cycle inconsistency (~0 for the main leg joints, which the outline
    pins; 1 for the loose ones), from its first-pass pose and from the phase average itself; the lower objective
    (silhouette, paws, the two-sided prior and the periodic term) wins. Returns whether it ran."""
    values = np.array([xs[f][:N] for f in range(frame_count)])
    prior = cycle.periodic_prior(values, masks, float(args["fps"]), [(ia, ib) for _, _, ia, ib in pairs],
                                 int(math.ceil(0.1 * frame_count)))
    if prior is None:
        print("VID2GODOT_FIT periodic pass skipped: the video shows no steady cycle", flush=True)
        return False
    print(f"VID2GODOT_FIT periodic pass: period {prior['period']:.3f} frames over {len(prior['kept'])} cycles; "
          "weights " + ", ".join(f"{names[i]} {w:.2f}" for i, w in enumerate(prior["weights"]) if w > 0), flush=True)
    t_pass = time.time()
    for f in frames:
        PERIODIC[f] = (prior["targets"][f], prior["weights"])
    for f in frames:
        neighbours = [xs[g] for g in (f - 1, f + 1) if g in xs]
        smooth = (np.mean(neighbours, axis=0), 1.0) if neighbours else None
        start = xs[f].copy()
        start[2:N] = prior["targets"][f][2:]
        kept, _ = descend(xs[f], f, smooth, schedule=LEVELS[2:])
        moved, _ = descend(clamp(start), f, smooth, schedule=LEVELS[1:])
        xs[f] = min((kept, moved), key=lambda y: candidate_key(y, f, score(y, f, smooth, 1.0)))
    PERIODIC.clear()
    changed = relabel(xs, frames)
    print(f"VID2GODOT_FIT periodic pass in {time.time() - t_pass:.1f}s: relabelled {changed}", flush=True)
    return True


periodic = bool(PERIODIC_PASS and scale_free and frames == list(range(frame_count)) and args.get("periodic", True))
periodic_ran = 0
for _ in range(PERIODIC_ROUNDS if periodic else 0):
    if not periodic_pass(xs, frames):
        break
    periodic_ran += 1

camera = camera_for(cam_scale)


def describe(values, previous_entry):
    apply(values)
    flags = []
    for a, b in project["legPairs"]:
        ua = camera.project(np.array(rigutil.bone_point(arm, project["paws"][a])))[0]
        ub = camera.project(np.array(rigutil.bone_point(arm, project["paws"][b])))[0]
        before = previous_entry["paws_u"].get(f"{a}/{b}") if previous_entry else None
        if abs(ua - ub) < CROSSING_PX or (before is not None and np.sign(ua - ub) != np.sign(before)):
            flags.append(f"leg_crossing:{a}/{b}")
    joints = {}
    for chain, chain_bones in project["chains"].items():
        present = [b for b in chain_bones if b in arm.pose.bones]
        if present:
            points = [rigutil.bone_point(arm, present[0], "head")] + [rigutil.bone_point(arm, b) for b in present]
            joints[chain] = camera.project(np.array(points)).tolist()
    paws = {chain: rigutil.bone_point(arm, bone) for chain, bone in project["paws"].items()}
    paws_u = {f"{a}/{b}": float(camera.project(np.array(paws[a]))[0] - camera.project(np.array(paws[b]))[0])
              for a, b in project["legPairs"]}
    uv = camera.project(rigutil.evaluated_vertices(skinned)[:, [0, 2]]).astype(np.float32)
    landmarks = {bone: {end: camera.project(np.array(rigutil.bone_point(arm, bone, end))).tolist()
                        for end in ("head", "tail")}
                 for bone in dict.fromkeys(bones + list(project["paws"].values()))}
    return {"flags": flags, "joints": joints, "paws": paws, "paws_u": paws_u, "landmarks": landmarks}, uv


prev_entry = entries[span[0] - 1] if span[0] > 0 else None
for f in frames:
    x = xs[f]
    if usable_observations.get(f):
        exact = lambda y: observation_residuals(y, f, actual=True)
        before = np.linalg.norm(exact(x), axis=1)
        if np.any(before > 1.0 + fitconstraints.TOL):
            # NumPy FK can differ by a few floating-point units at a disk's
            # boundary. Restore against the actual output oracle, not a larger
            # disk or a mandatory inward target. The shared camera stays fixed.
            center = x.copy()
            scales = np.array([delta_for(i, LEVEL_STEPS[1.0][-1]) for i in range(N + 1)])
            scales[SCALE] = 0.0

            def final_clamp(y):
                return clamp(np.clip(y, center - scales, center + scales))

            original = score(x, f, None, 1.0)
            x, diagnostic = fitconstraints.refine(
                x, lambda y: score(y, f, None, 1.0), exact, final_clamp, scales,
                max_iterations=0)
            diagnostic.update({"phase": "final-actual-projection", "oracle": "Blender pose bones",
                               "objectiveBefore": original,
                               "maxNormalizedErrorBefore": float(before.max())})
            constraint_search.setdefault(f, []).append(diagnostic)
            xs[f] = x
    loss = score(x, f, None, 1.0)
    info, uv = describe(x, prev_entry)
    entries[f] = {"index": f, "values": [float(v) for v in x[:N]], "loss": loss, **info}
    scored_uv, cam = posed_uv(x)
    silhouette, _ = levels[1.0].loss(scored_uv, f)
    manual = mark_errors(x, f, cam)
    entries[f]["objective"] = {"silhouette": float(silhouette), "paws": float(paw_term(x, f, cam)),
                                "marks": W_MARKS * float(np.mean(manual)) if manual else 0.0,
                                "observations": observation_term(x, f, cam), "total": loss,
                                "context": "same full-resolution objective, no temporal/periodic prior"}
    if region_frames.get(f):
        _, _, regional, _ = levels[1.0].loss(scored_uv, f, include_regions=True)
        entries[f]["objective"]["regions"] = regional
    if usable_observations.get(f):
        # Recompute after the shared-camera, relabel and all refinement passes.
        normalized = np.linalg.norm(observation_residuals(x, f, actual=True), axis=1)
        feasible = bool(np.all(normalized <= 1.0 + fitconstraints.TOL))
        entries[f]["constraintSearch"] = {
            "status": "feasible" if feasible else "unresolved", "feasible": feasible,
            "maxNormalizedError": float(normalized.max()), "cameraScale": float(cam_scale),
            "passes": constraint_search.get(f, []),
            "scope": "Local search diagnostics; independent assessment remains authoritative"}
    projection[f] = uv
    prev_entry = entries[f]

poses = {"bones": bones, "names": names, "axes": [list(a) for a in axes], "fps": args["fps"],
         "camera": camera.to_json(), "cameraScale": cam_scale, "registrationCamera": base_camera.to_json(),
         "deformModel": DEFORM_MODEL, "periodicPass": periodic_ran, "restartSeed": args.get("seed"),
         "frames": entries}
if args.get("witness"):
    witness = json.loads(Path(args["witness"]).read_text())
    if witness.get("names") != names or witness.get("registrationCamera") != base_camera.to_json():
        raise ValueError("Diagnostic witness must use the same rig parameter names and registered camera")
    witness_entries = {e["index"]: e for e in witness["frames"] if e is not None}
    witness_frames, comparisons = [], []
    for f in frames:
        if f not in witness_entries:
            continue
        values = witness_entries[f]["values"]
        valid = (len(values) == N and all(math.isfinite(v) for v in values)
                 and all(low <= values[i + 2] <= high for i, (low, high) in enumerate(limits))
                 and math.isclose(float(witness.get("cameraScale", 1)), cam_scale, rel_tol=1e-9, abs_tol=1e-9))
        if not valid:
            comparisons.append({"index": f, "validWitness": False})
            continue
        x = np.array(list(values) + [cam_scale], float)
        info, _ = describe(x, None)
        witness_frames.append({"index": f, "values": values, **info})
        comparisons.append({"index": f, "validWitness": True,
                            "fittedObjective": score(xs[f], f, None, 1.0),
                            "witnessObjective": score(x, f, None, 1.0)})
    poses["witness"] = {"frames": witness_frames}
    poses["comparisons"] = comparisons
if args.get("frameIndices") is not None:
    poses["frames"] = [entries[f] for f in frames]
Path(args["out"]).write_text(json.dumps(poses))
np.savez_compressed(args["projection"], uv=projection, faces=rigutil.triangles(skinned))
rigutil.write_result(args, {"frames": len(frames), "meanLoss": float(np.mean([entries[f]["loss"] for f in frames])),
                            "cameraScale": cam_scale, "deformModel": DEFORM_MODEL, "periodicPass": periodic_ran,
                            "seconds": time.time() - t_start})
