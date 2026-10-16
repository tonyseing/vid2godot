"""A toy side-plane quadruped rig (a contract rig.json stand-in) and gaits posed on it by two-bone IK from known paw
paths: each paw is world-still on the ground over its stance window (in the in-place clip it slides back one stride
per cycle) and swings forward on an arc between. The fitted values and the paws are exact for the rig, so facts'
contacts from the motion, its IK pins and its re-detection can be checked against the paths that made them."""
import math

import numpy as np

from anim import motion

HIPS = {"hind_L": -0.30, "hind_R": -0.28, "front_L": 0.30, "front_R": 0.32}
SHORT = {"hind_L": "hl", "hind_R": "hr", "front_L": "fl", "front_R": "fr"}
PROJECT = {"chains": {"spine": ["pelvis", "chest"], "tail": ["tail"],
                      **{c: [f"upper_{s}", f"lower_{s}", f"paw_{s}"] for c, s in SHORT.items()}},
           "legPairs": [["front_L", "front_R"], ["hind_L", "hind_R"]],
           "paws": {c: f"paw_{s}" for c, s in SHORT.items()}, "phaseOrigin": "hind_L",
           "pelvisBone": "pelvis"}
LOPE_WINDOWS = {"hind_L": (0.0, 0.3), "hind_R": (0.3, 0.55), "front_L": (0.3, 0.55), "front_R": (0.55, 0.8)}
REST = {c: 0.0 for c in SHORT}


def bone(name, parent, head, tail):
    return {"name": name, "parent": parent, "head": [head[0], 0.0, head[1]], "tail": [tail[0], 0.0, tail[1]]}


def toy_rig():
    bones = [bone("root", None, (0.0, 0.0), (0.0, 0.5)), bone("pelvis", "root", (-0.3, 0.6), (0.0, 0.6)),
             bone("chest", "pelvis", (0.0, 0.6), (0.3, 0.6)), bone("tail", "pelvis", (-0.3, 0.6), (-0.6, 0.5))]
    for chain, short in SHORT.items():
        x = HIPS[chain]
        parent = "pelvis" if chain.startswith("hind") else "chest"
        bones += [bone(f"upper_{short}", parent, (x, 0.5), (x + 0.08, 0.28)),
                  bone(f"lower_{short}", f"upper_{short}", (x + 0.08, 0.28), (x, 0.05)),
                  bone(f"paw_{short}", f"lower_{short}", (x, 0.05), (x + 0.06, 0.0))]
    return {"bones": bones}


NAMES = ["dx", "dz", "pelvis", "chest", "tail"] + [f"{p}_{s}" for s in SHORT.values() for p in ("upper", "lower",
                                                                                                "paw")]
# toy_gait.toy_mask draws values[2:6] as legs; put the four upper-leg joints there so the video shows the gait.
DRAWN = ["upper_hl", "upper_hr", "upper_fl", "upper_fr"]
NAMES = ["dx", "dz"] + DRAWN + [n for n in NAMES[2:] if n not in DRAWN]


def paw_path(phase, window, stride, lift, hip):
    """The in-place paw (x, z) at a cycle phase: on the ground over [on, off), sliding back one stride per cycle;
    between, a forward swing on a sine arc `lift` high."""
    on, off = window
    stance = (off - on) % 1.0
    u = (phase - on) % 1.0
    back = stride * stance
    if u < stance:
        return hip + 0.06 + back / 2 - stride * u, 0.0
    s = (u - stance) / (1.0 - stance)
    return hip + 0.06 - back / 2 + back * s, lift * math.sin(math.pi * s)


def toy_gait(frames=120, period=20.0, fps=24.0, speed=1.0, windows=None, lift=0.08, trunk=0.02, float_paw=None):
    """poses.json for the toy rig: every leg posed by IK onto its paw path, a pelvis and tail sway of `trunk`
    radians. `float_paw` = (chain, height): that paw's stance is lifted `height` off the ground (a fit error)."""
    windows = windows or LOPE_WINDOWS
    rig = motion.PlanarRig(toy_rig())
    stride = speed * period / fps
    entries = []
    for f in range(frames):
        phase = (f / period) % 1.0
        values = np.zeros((1, len(NAMES)))
        values[0, NAMES.index("pelvis")] = trunk * math.sin(2 * math.pi * phase)
        values[0, NAMES.index("tail")] = 2 * trunk * math.sin(2 * math.pi * phase - 1.0)
        for chain, window in windows.items():
            x, z = paw_path(phase, window, stride, lift, HIPS[chain])
            if float_paw and float_paw[0] == chain:
                z += float_paw[1] if (phase - window[0]) % 1.0 < (window[1] - window[0]) % 1.0 else 0.0
            values, _ = motion.solve_leg(rig, values, NAMES, PROJECT["chains"][chain], PROJECT["paws"][chain],
                                         np.array([[x, z]]))
        paws = rig.paws(values, NAMES, PROJECT)
        entries.append({"index": f, "values": values[0].tolist(), "flags": [],
                        "paws": {c: paws[c][0].tolist() for c in SHORT}})
    return {"fps": fps, "names": NAMES, "frames": entries}
