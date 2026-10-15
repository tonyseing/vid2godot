import tempfile
import unittest
from pathlib import Path

import numpy as np

from anim import raster


class RasterTests(unittest.TestCase):
    def test_rasterize_square(self):
        uv = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mask = raster.rasterize(uv, faces, 40, 40)
        self.assertAlmostEqual(mask.sum(), 21 * 21, delta=25)

    def test_iou(self):
        a = np.zeros((10, 10), bool)
        a[:, :5] = True
        b = np.zeros((10, 10), bool)
        b[:, :10] = True
        self.assertAlmostEqual(raster.iou(a, b), 0.5)
        self.assertEqual(raster.iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)), 1.0)

    def test_overlay_marks_both_outlines(self):
        frame = np.full((40, 40, 3), 200, np.uint8)
        video = np.zeros((40, 40), bool)
        video[10:30, 10:30] = True
        fitted = np.zeros((40, 40), bool)
        fitted[12:32, 12:32] = True
        image = np.array(raster.overlay(frame, fitted, video, [([(0, 0), (39, 39)], raster.LEFT)]))
        self.assertEqual(tuple(image[10, 20]), (0, 220, 220))
        self.assertEqual(tuple(image[31, 20]), (230, 0, 160))
        self.assertEqual(tuple(image[5, 5]), raster.LEFT)

    def test_score_fit_flags_low_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            square = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
            shifted = square + np.array([15, 0], dtype=np.float32)
            np.savez(tmp / "proj.npz", uv=np.stack([square, shifted]), faces=np.array([[0, 1, 2], [0, 2, 3]]))
            masks = np.zeros((2, 40, 60), bool)
            masks[:, 10:31, 10:31] = True
            np.savez(tmp / "masks.npz", masks=masks)
            poses = {"frames": [{"index": 0, "flags": []}, {"index": 1, "flags": ["leg_crossing:a/b"]}]}
            report = raster.score_fit(poses, tmp / "proj.npz", tmp / "masks.npz", 0.9, {"0": ["border_touch"]})
            self.assertEqual(report["lowFrames"], [0, 1])
            self.assertGreater(report["frames"][0]["iou"], 0.9)
            self.assertIn("border_touch", report["frames"][0]["flags"])
            self.assertLess(report["frames"][1]["iou"], 0.9)
