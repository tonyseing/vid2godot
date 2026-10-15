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

    def test_outline(self):
        mask = np.zeros((9, 9), bool)
        mask[2:7, 2:7] = True
        ring = raster.outline(mask)
        self.assertEqual(ring.dtype, bool)
        self.assertEqual(int(ring.sum()), 16)
        self.assertFalse(ring[3:6, 3:6].any())
        expected = mask.copy()
        expected[3:6, 3:6] = False
        np.testing.assert_array_equal(ring, expected)
        empty = raster.outline(np.zeros((9, 9), bool))
        self.assertFalse(empty.any())

    def test_overlay_marks_both_outlines(self):
        frame = np.full((40, 40, 3), 200, np.uint8)
        original = frame.copy()
        video = np.zeros((40, 40), bool)
        video[10:30, 10:30] = True
        fitted = np.zeros((40, 40), bool)
        fitted[12:32, 12:32] = True
        image = np.array(raster.overlay(frame, fitted, video, [([(0, 0), (39, 39)], raster.LEFT)]))
        self.assertEqual(tuple(image[10, 20]), (0, 220, 220))
        self.assertEqual(tuple(image[31, 20]), (230, 0, 160))
        self.assertEqual(tuple(image[5, 5]), raster.LEFT)
        # Inside both silhouettes, on neither outline and off the skeleton: the frame shows through.
        self.assertEqual(tuple(image[20, 25]), (200, 200, 200))
        # The video's right edge (column 29) crosses the fitted top edge (row 12): fitted wins.
        self.assertTrue(raster.outline(video)[12, 29] and raster.outline(fitted)[12, 29])
        self.assertEqual(tuple(image[12, 29]), (230, 0, 160))
        # The skeleton diagonal passes the video outline's corner and the fitted outline's corner: skeleton wins.
        self.assertTrue(raster.outline(video)[10, 10])
        self.assertEqual(tuple(image[10, 10]), raster.LEFT)
        self.assertTrue(raster.outline(fitted)[12, 12])
        self.assertEqual(tuple(image[12, 12]), raster.LEFT)
        np.testing.assert_array_equal(frame, original)

    def test_score_fit_flags_low_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # The square fills rows and columns 10..30 inclusive (Pillow fills the boundary): 21 x 21 = 441 px.
            square = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
            shifted = square + np.array([15, 0], dtype=np.float32)
            np.savez(tmp / "proj.npz", uv=np.stack([square, shifted, square]),
                     faces=np.array([[0, 1, 2], [0, 2, 3]]))
            masks = np.zeros((3, 40, 60), bool)
            masks[:, 10:31, 10:31] = True
            np.savez(tmp / "masks.npz", masks=masks)
            poses = {"frames": [
                {"index": 0, "flags": []},                      # (a) high IoU, no flags
                {"index": 1, "flags": []},                      # (b) low IoU, no flags
                {"index": 2, "flags": ["leg_crossing:a/b"]},    # (c) high IoU, pose + ingest flags
            ]}
            report = raster.score_fit(poses, tmp / "proj.npz", tmp / "masks.npz", 0.9, {"2": ["border_touch"]})
            self.assertEqual(report["lowFrames"], [1, 2])
            self.assertEqual([item["index"] for item in report["frames"]], [0, 1, 2])
            self.assertEqual(report["frames"][0]["iou"], 1.0)
            self.assertEqual(report["frames"][0]["flags"], [])
            # Shifted 15 px: overlap columns 25..30 = 6 x 21 = 126; union 441 + 441 - 126 = 756.
            self.assertAlmostEqual(report["frames"][1]["iou"], round(126 / 756, 4))
            self.assertEqual(report["frames"][1]["flags"], [])
            self.assertEqual(report["frames"][2]["iou"], 1.0)
            self.assertEqual(sorted(report["frames"][2]["flags"]), ["border_touch", "leg_crossing:a/b"])
            self.assertAlmostEqual(report["meanIoU"], (1.0 + round(126 / 756, 4) + 1.0) / 3, places=4)
