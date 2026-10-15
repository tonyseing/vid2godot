"""Fit the rig to the video frame by frame: pose, project the silhouette, score it against the video, nudge, repeat.

Loss = mean distance of projected vertices outside the video mask + mean distance from the mask's outline to the
nearest projected vertex (chamfer) + marked-point error + a smoothness penalty against the previous frame."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import kdtree  # noqa: E402
import rigutil  # noqa: E402
from anim.camera import SideCamera  # noqa: E402

STEPS = (8.0, 4.0, 2.0, 1.0, 0.5)     # degrees for angles; working pixels for the body offset
SWEEPS = 3
W_MARKS, W_SMOOTH, W_OFFSET = 1.0, 40.0, 0.01
OUT_OF_FRAME = 50.0
CROSSING_PX = 4.0

args = rigutil.load_args()
camera = SideCamera.from_json(args["camera"])
project, brief = args["project"], args["brief"]
arm, skinned, _ = rigutil.load_glb(args["model"])
rigutil.rest_pose(arm)
data = np.load(args["masks"])
outside, boundary, offsets, masks = data["outside"], data["boundary"], data["offsets"], data["masks"]
frame_count = outside.shape[0]
scale = args["workScale"]
marks = {int(f): {b: np.array(p) * scale for b, p in points.items()} for f, points in args["marks"].items()}

bones = list(dict.fromkeys(b for chain in brief["fitChains"] for b in project["chains"][chain]
                           if b != project["rootBone"]))
axes = [rigutil.side_axis_local(arm, b) for b in bones]
limits = [tuple(math.radians(x) for x in brief["limitsDeg"].get(b, [-60, 60])) for b in bones]
names = ["dx", "dz"] + bones
total = len(rigutil.evaluated_vertices(skinned))
sample = np.arange(0, total, max(1, total // 3000))


def apply(values):
    arm.location = (float(values[0]), 0.0, float(values[1]))
    for name, axis, angle in zip(bones, axes, values[2:]):
        rigutil.set_angle(arm, name, axis, float(angle))
    bpy.context.view_layer.update()


def score(values, f, prev):
    apply(values)
    uv = camera.project(rigutil.evaluated_vertices(skinned)[sample][:, [0, 2]])
    u, v = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    dist = np.full(len(uv), OUT_OF_FRAME)
    dist[inside] = outside[f, v[inside], u[inside]]
    tree = kdtree.KDTree(len(uv))
    for i, (x, y) in enumerate(uv):
        tree.insert((x, y, 0.0), i)
    tree.balance()
    points = boundary[offsets[f]:offsets[f + 1]]
    loss = dist.mean() + np.mean([tree.find((float(x), float(y), 0.0))[2] for x, y in points])
    if f in marks:
        loss += W_MARKS * np.mean([np.hypot(*(camera.project(np.array(rigutil.bone_point(arm, b))) - p))
                                   for b, p in marks[f].items()])
    if prev is not None:
        loss += W_SMOOTH * np.sum((values[2:] - prev[2:]) ** 2)
        loss += W_OFFSET * np.sum(((values[:2] - prev[:2]) * camera.px_per_m) ** 2)
    return float(loss)


def clamp(values):
    for i, (low, high) in enumerate(limits):
        values[i + 2] = min(max(values[i + 2], low), high)
    return values


def descend(values, f, prev):
    best = score(values, f, prev)
    for step in STEPS:
        for _ in range(SWEEPS):
            improved = False
            for i in range(len(values)):
                delta = step / camera.px_per_m if i < 2 else math.radians(step)
                for sign in (1.0, -1.0):
                    trial = clamp(values.copy())
                    trial[i] += sign * delta
                    trial = clamp(trial)
                    if trial[i] == values[i]:
                        continue
                    loss = score(trial, f, prev)
                    if loss < best - 1e-6:
                        values, best, improved = trial, loss, True
                        break
            if not improved:
                break
    return values, best


def initial_offset(f):
    """Align the rest model's projected centre and lowest point with the video mask of frame f."""
    apply(np.zeros(len(names)))
    uv = camera.project(rigutil.evaluated_vertices(skinned)[sample][:, [0, 2]])
    rows, cols = np.nonzero(masks[f])
    values = np.zeros(len(names))
    values[0] = (cols.mean() - uv[:, 0].mean()) / camera.px_per_m
    values[1] = (uv[:, 1].max() - rows.max()) / camera.px_per_m
    return values


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
    return {"flags": flags, "joints": joints, "paws": paws, "paws_u": paws_u}, uv


span = args.get("span") or [0, frame_count]
previous = json.loads(Path(args["init"]).read_text()) if args.get("init") else None
entries = list(previous["frames"]) if previous else [None] * frame_count
projection = np.load(args["projection"])["uv"] if previous else np.zeros((frame_count, total, 2), np.float32)
prev_values = np.array(entries[span[0] - 1]["values"]) if span[0] > 0 and entries[span[0] - 1] else None
prev_entry = entries[span[0] - 1] if span[0] > 0 else None
for f in range(span[0], span[1]):
    start = prev_values.copy() if prev_values is not None else initial_offset(f)
    values, loss = descend(clamp(start), f, prev_values)
    info, uv = describe(values, prev_entry)
    entries[f] = {"index": f, "values": [float(x) for x in values], "loss": loss, **info}
    projection[f] = uv
    prev_values, prev_entry = values, entries[f]
    print(f"VID2GODOT_FIT frame {f + 1}/{frame_count} loss {loss:.3f} flags {info['flags']}", flush=True)

poses = {"bones": bones, "names": names, "axes": [list(a) for a in axes], "fps": args["fps"],
         "camera": args["camera"], "frames": entries}
Path(args["out"]).write_text(json.dumps(poses))
np.savez_compressed(args["projection"], uv=projection, faces=rigutil.triangles(skinned))
rigutil.write_result(args, {"frames": span[1] - span[0], "meanLoss": float(np.mean([e["loss"] for e in entries]))})
