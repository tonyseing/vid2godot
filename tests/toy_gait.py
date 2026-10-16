"""A toy side-view 'video' for the cycle and facts tests: a body block and four legs drawn as thick lines, each leg
turned by one joint value. Values beyond the four leg joints (such as a tail the silhouette does not show) are not
drawn, like the joints a real silhouette barely constrains."""
import math

import numpy as np
from PIL import Image, ImageDraw

HIPS = (22, 34, 66, 78)          # the hips of the joints a, b, c, d (values 2..5)
SHAPE = (64, 100)
LEG = 24.0


def toy_mask(values, gain=1.0, hips=HIPS, shift=0):
    """The silhouette of one value vector [dx, dz, a, b, c, d, ...]. `gain` scales the leg angles (a pose with tiny
    angles can stand for a visible gait); `shift` moves the whole animal right, in pixels."""
    image = Image.new("1", (SHAPE[1], SHAPE[0]), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle([12 + shift, 14, 88 + shift, 28], fill=1)
    for hip, angle in zip(hips, values[2:6]):
        a = gain * float(angle)
        draw.line([(hip + shift, 26), (hip + shift + LEG * math.sin(a), 26 + LEG * math.cos(a))], fill=1, width=5)
    return np.array(image, bool)


def toy_video(poses, gain=1.0, hips=HIPS, masks_from=None):
    """(masks, render): the clip's masks (from `masks_from` value vectors when given, else the poses' own values)
    and a renderer for facts' check of the averaged cycle (value vectors -> masks)."""
    source = masks_from if masks_from is not None else [f["values"] for f in poses["frames"]]
    masks = np.array([toy_mask(v, gain, hips) for v in source])

    def render(values):
        return [toy_mask(v, gain, hips) for v in values]
    return masks, render
