import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from anim.blender_run import run_blender
from anim.reference import choose_background, model_mask
from anim.ledger import StageError
from tests.blender_case import BLENDER, fixture_glb, requires_blender


def image(path, background, square):
    pixels = np.zeros((90, 160, 3), dtype=np.uint8)
    pixels[:] = background
    pixels[30:70, 50:110] = square
    Image.fromarray(pixels).save(path)


class BackgroundChoiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        mask = np.full((90, 160), 255, dtype=np.uint8)
        mask[30:70, 50:110] = 0
        Image.fromarray(mask).save(self.tmp / "mask.png")
        self.mask = model_mask(self.tmp / "mask.png")

    def test_low_contrast_background_rejected(self):
        image(self.tmp / "grey.png", (127, 127, 127), (128, 128, 128))
        image(self.tmp / "green.png", (0, 177, 64), (128, 128, 128))
        colour, report = choose_background({"#7f7f7f": self.tmp / "grey.png", "#00b140": self.tmp / "green.png"},
                                           self.mask)
        self.assertEqual(colour, "#00b140")
        self.assertLess(report["#7f7f7f"], 48.0)

    def test_no_background_passes(self):
        image(self.tmp / "grey.png", (127, 127, 127), (128, 128, 128))
        with self.assertRaises(StageError):
            choose_background({"#7f7f7f": self.tmp / "grey.png"}, self.mask)

    def test_empty_mask_is_a_clean_error(self):
        image(self.tmp / "grey.png", (127, 127, 127), (128, 128, 128))
        blank = np.full((90, 160), 255, dtype=np.uint8)
        Image.fromarray(blank).save(self.tmp / "blank.png")
        with self.assertRaises(StageError) as caught:
            choose_background({"#7f7f7f": self.tmp / "grey.png"}, model_mask(self.tmp / "blank.png"))
        self.assertIn("empty or too thin", str(caught.exception))


@requires_blender
class ReferenceRenderTests(unittest.TestCase):
    def test_fixture_reference(self):
        glb, _ = fixture_glb()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_blender("render_reference", {"model": str(glb), "outDir": tmp,
                                                      "backgrounds": ["#c8c8c8"]}, Path(tmp), BLENDER)
            mask = model_mask(result["mask"])
            self.assertEqual(mask.shape, (720, 1280))
            rows = np.nonzero(mask.any(axis=1))[0]
            self.assertAlmostEqual(rows.max(), 0.78 * 720, delta=12)   # paws on the ground line
            cols = np.nonzero(mask.any(axis=0))[0]
            self.assertAlmostEqual(cols.max() - cols.min(), 640, delta=20)
            self.assertTrue(Path(result["renders"]["#c8c8c8"]).is_file())
