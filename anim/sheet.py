"""The contract's review sheet: poses across the cycle at game distance on the palette background."""
from pathlib import Path

import numpy as np
from PIL import Image

from anim.camera import SideCamera
from anim.ledger import StageError


def game_camera(bounds, px_per_m, width=1280, height=800):
    center_x = (bounds["min"][0] + bounds["max"][0]) / 2
    return SideCamera(width, height, float(px_per_m), center_x, (0.78 * height - height / 2) / px_per_m)


def compose(paths, background_hex, out, margin=8):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(np.array(image.convert("RGB")))
    target = np.array([int(background_hex[i:i + 2], 16) for i in (1, 3, 5)])
    union = np.zeros(images[0].shape[:2], bool)
    for image in images:
        union |= np.abs(image.astype(int) - target).sum(axis=2) > 12
    if not union.any():
        raise StageError("The sheet render shows nothing but background; check the model is in frame and visible")
    rows, cols = np.nonzero(union.any(axis=1))[0], np.nonzero(union.any(axis=0))[0]
    v0, v1 = max(rows.min() - margin, 0), min(rows.max() + 1 + margin, union.shape[0])
    u0, u1 = max(cols.min() - margin, 0), min(cols.max() + 1 + margin, union.shape[1])
    tiles = [Image.fromarray(image[v0:v1, u0:u1]) for image in images]
    sheet = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), tuple(int(c) for c in target))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * tile.width, 0))
    sheet.save(out)
    return Path(out)
