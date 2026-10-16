"""The motion brief: which clip to make, how it moves and how to judge the fit."""
import json
import math
import re
from pathlib import Path

from anim.ledger import StageError

REQUIRED = ("clip", "subject", "motion", "gait", "loop", "planar", "videoSeconds")
DEFAULTS = {"resolution": "720p", "aspectRatio": "16:9", "iouThreshold": 0.90,
            "backgrounds": ["#c8c8c8", "#7f7f7f", "#00b140"], "limitsDeg": {}, "fitChains": None,
            "targetSpeedMps": None, "massKg": None, "slowMotion": None}
# Task 20: a slowMotion factor asks the video model for the motion that many times slower than real time (a small,
# fast animal's steps undersampled at 24 fps); facts expects that retime.
SLOW_MOTION = (1.0, 8.0)
TWO_VIEW = ("This motion leaves the side plane. One side camera cannot see depth, so the fit would match the "
            "outline while the 3D pose is wrong. It needs the two-view method, which this version does not support.")


class BriefError(StageError):
    pass


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_brief(path, project, clip):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise BriefError("The brief must be a JSON object")
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise BriefError(f"The brief lacks {missing}")
    brief = {**DEFAULTS, **data}
    if brief["clip"] != clip:
        raise BriefError(f"The brief is for {brief['clip']!r}, the ledger for {clip!r}")
    if brief["planar"] is not True:
        raise BriefError(TWO_VIEW)
    if not isinstance(brief["loop"], bool):
        raise BriefError("loop must be true or false")
    if not isinstance(brief["videoSeconds"], int) or not 4 <= brief["videoSeconds"] <= 30:
        raise BriefError("videoSeconds must be an integer from 4 to 30")
    if not is_number(brief["iouThreshold"]) or not 0 < brief["iouThreshold"] <= 1:
        raise BriefError("iouThreshold must be a number in (0, 1]")
    if not isinstance(brief["resolution"], str) or not re.fullmatch(r"\d{3,4}p", brief["resolution"]):
        raise BriefError("resolution must look like '720p'")
    if not isinstance(brief["aspectRatio"], str) or not re.fullmatch(r"\d+:\d+", brief["aspectRatio"]):
        raise BriefError("aspectRatio must look like '16:9'")
    speed = brief["targetSpeedMps"]
    if speed is not None and (not is_number(speed) or not math.isfinite(speed) or speed <= 0):
        raise BriefError("targetSpeedMps must be null or a finite number above 0")
    mass = brief["massKg"]
    if mass is not None:
        pair = isinstance(mass, list) and len(mass) == 2 and all(is_number(m) and math.isfinite(m) for m in mass)
        if not (pair and 0 < mass[0] < mass[1]) and not (is_number(mass) and math.isfinite(mass) and mass > 0):
            raise BriefError("massKg must be null, a body mass above 0 kg, or [lightest, heaviest] kg")
    slow = brief["slowMotion"]
    low, high = SLOW_MOTION
    if slow is not None and not (is_number(slow) and math.isfinite(slow) and low < slow <= high):
        raise BriefError(f"slowMotion must be null or a factor over {low:g} and at most {high:g} (the video "
                         f"that many times slower than real time)")
    chains = brief["fitChains"]
    if chains is None:
        chains = list(project["chains"])
    elif not isinstance(chains, list) or not chains or not all(isinstance(c, str) for c in chains):
        raise BriefError("fitChains must be null (all chains) or a non-empty list of chain names")
    unknown = [chain for chain in chains if chain not in project["chains"]]
    if unknown:
        raise BriefError(f"fitChains names unknown chains {unknown}")
    brief["fitChains"] = chains
    if not isinstance(brief["limitsDeg"], dict):
        raise BriefError("limitsDeg must be an object of bone: [low, high]")
    for bone, limits in brief["limitsDeg"].items():
        if not isinstance(limits, list) or len(limits) != 2 or not all(is_number(v) for v in limits):
            raise BriefError(f"limitsDeg[{bone}] must be a list of two numbers [low, high]")
        if not -180 <= limits[0] < limits[1] <= 180:
            raise BriefError(f"limitsDeg[{bone}] must be [low, high] degrees with low < high")
    if not isinstance(brief["backgrounds"], list) or not brief["backgrounds"] or not all(
            isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in brief["backgrounds"]):
        raise BriefError("backgrounds must be a non-empty list of #rrggbb colours")
    return brief
