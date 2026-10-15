"""The motion brief: which clip to make, how it moves and how to judge the fit."""
import json
import re
from pathlib import Path

from anim.ledger import StageError

REQUIRED = ("clip", "subject", "motion", "gait", "loop", "planar", "videoSeconds")
DEFAULTS = {"resolution": "720p", "aspectRatio": "16:9", "iouThreshold": 0.90,
            "backgrounds": ["#c8c8c8", "#7f7f7f", "#00b140"], "limitsDeg": {}, "fitChains": None,
            "targetSpeedMps": None}
TWO_VIEW = ("This motion leaves the side plane. One side camera cannot see depth, so the fit would match the "
            "outline while the 3D pose is wrong. It needs the two-view method, which this version does not support.")


class BriefError(StageError):
    pass


def load_brief(path, project, clip):
    data = json.loads(Path(path).read_text())
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
    if not 0 < brief["iouThreshold"] <= 1:
        raise BriefError("iouThreshold must be in (0, 1]")
    chains = brief["fitChains"] or list(project["chains"])
    unknown = [chain for chain in chains if chain not in project["chains"]]
    if unknown:
        raise BriefError(f"fitChains names unknown chains {unknown}")
    brief["fitChains"] = chains
    for bone, limits in brief["limitsDeg"].items():
        if len(limits) != 2 or not -180 <= limits[0] < limits[1] <= 180:
            raise BriefError(f"limitsDeg[{bone}] must be [low, high] degrees with low < high")
    if not brief["backgrounds"] or not all(re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in brief["backgrounds"]):
        raise BriefError("backgrounds must be a non-empty list of #rrggbb colours")
    return brief
