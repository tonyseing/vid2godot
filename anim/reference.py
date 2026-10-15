"""Pick the reference background the coat stands out from, so the video's silhouette can be cut cleanly."""
import numpy as np
from PIL import Image
from scipy import ndimage

from anim.ledger import StageError

MIN_CONTRAST = 48.0


def model_mask(png):
    return np.array(Image.open(png).convert("L")) < 128


def contrast(render_png, mask, colour):
    rgb = np.array(Image.open(render_png).convert("RGB"), dtype=float)
    inner = ndimage.binary_erosion(mask, iterations=2)
    if not inner.any():
        raise StageError("The model mask is empty or too thin to judge contrast; check that the model rendered")
    target = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return float(np.percentile(np.linalg.norm(rgb[inner] - target, axis=1), 2))


def choose_background(renders, mask):
    report = {colour: contrast(path, mask, colour) for colour, path in renders.items()}
    for colour in renders:
        if report[colour] >= MIN_CONTRAST:
            return colour, report
    raise StageError(f"No background separates the model (2nd-percentile colour distance {report}, need "
                     f">= {MIN_CONTRAST}); add a contrasting colour to the brief's backgrounds")
