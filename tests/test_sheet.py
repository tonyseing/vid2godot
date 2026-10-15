import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from anim import sheet
from anim.ledger import StageError


class SheetTests(unittest.TestCase):
    def test_compose_crops_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                pixels = np.zeros((100, 200, 3), np.uint8)
                pixels[:] = (0x6b, 0x5b, 0x9f)
                pixels[40:60, 50 + 10 * i:90 + 10 * i] = (20, 20, 20)
                paths.append(Path(tmp) / f"{i}.png")
                Image.fromarray(pixels).save(paths[-1])
            out = sheet.compose(paths, "#6b5b9f", Path(tmp) / "sheet.png")
            with Image.open(out) as image:
                self.assertEqual(image.height, 20 + 16)
                self.assertEqual(image.width, 3 * (60 + 16))

    def test_compose_of_only_background_is_a_stage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            pixels = np.zeros((100, 200, 3), np.uint8)
            pixels[:] = (0x6b, 0x5b, 0x9f)
            path = Path(tmp) / "0.png"
            Image.fromarray(pixels).save(path)
            with self.assertRaisesRegex(StageError, "nothing but background"):
                sheet.compose([path], "#6b5b9f", Path(tmp) / "sheet.png")

    def test_game_camera_puts_ground_at_78_percent(self):
        cam = sheet.game_camera({"min": [-0.8, -0.2, 0.0], "max": [0.8, 0.2, 1.0]}, 75.0)
        self.assertAlmostEqual(cam.project(np.array([0.0, 0.0]))[1], 0.78 * 800)
        self.assertEqual(cam.px_per_m, 75.0)
