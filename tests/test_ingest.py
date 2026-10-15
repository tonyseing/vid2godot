import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from anim import ingest
from anim.camera import SideCamera

GREY = (200, 200, 200)


def frame(u, w=320, h=180, size=(60, 40), colour=(40, 30, 25), shape=None):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = GREY
    v1 = int(h * 0.78)
    img[v1 - size[1]:v1, u:u + size[0]] = colour
    return img


class IngestUnitTests(unittest.TestCase):
    def test_duplicates_dropped_and_rate_found(self):
        unique = [frame(40 + 3 * i) for i in range(24)]
        sixty = []
        for i, img in enumerate(unique):
            sixty += [img] * (2 if i % 2 == 0 else 3)
        sixty = sixty[:60]
        keep = ingest.unique_indices(sixty)
        self.assertEqual(len(keep), 24)
        self.assertEqual(ingest.true_rate(len(keep), len(sixty), 60.0), 24.0)

    def test_rate_not_snapped_when_far(self):
        self.assertAlmostEqual(ingest.true_rate(20, 60, 60.0), 20.0)

    def test_silhouette_keeps_largest_component(self):
        img = frame(100)
        img[5:8, 5:8] = (0, 0, 0)            # speck
        mask = ingest.silhouette(img, ingest.background_colour(img))
        self.assertEqual(int(mask.sum()), 60 * 40)

    def test_border_touch_flagged(self):
        mask = ingest.silhouette(frame(290), np.array(GREY, dtype=float))
        self.assertTrue(ingest.touches_border(mask))
        self.assertFalse(ingest.touches_border(ingest.silhouette(frame(100), np.array(GREY, dtype=float))))

    def test_registration_rescales_other_resolution(self):
        ref = ingest.silhouette(frame(100, 1280, 720, (300, 200)), np.array(GREY, dtype=float))
        video = ingest.silhouette(frame(460, 1248, 704, (293, 195)), np.array(GREY, dtype=float))
        reg = ingest.registration(ref, video)
        warped = ingest.warp(video, reg, ref.shape) > 0.5
        iou = (warped & ref).sum() / (warped | ref).sum()
        self.assertGreater(iou, 0.95)
        half = ingest.warp(video, reg, (360, 640), extra=0.5) > 0.5
        self.assertAlmostEqual(half.sum(), ref.sum() / 4, delta=ref.sum() * 0.03)

    def test_boundary_points(self):
        mask = ingest.silhouette(frame(100), np.array(GREY, dtype=float))
        points = ingest.boundary_points(mask, max_points=50)
        self.assertEqual(points.shape, (50, 2))
        self.assertTrue(np.all(points[:, 0] >= 100) and np.all(points[:, 0] < 160))


class IngestVideoTests(unittest.TestCase):
    def test_run_ingest_on_encoded_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "src").mkdir()
            for i in range(30):
                Image.fromarray(frame(500 + 4 * i, 1280, 720, (300, 200))).save(tmp / "src" / f"{i:03d}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-framerate", "24", "-i", str(tmp / "src" / "%03d.png"),
                            "-c:v", "mpeg4", "-q:v", "2", str(tmp / "clip.mp4")], check=True)
            ref = np.full((720, 1280), 255, dtype=np.uint8)
            ref[561 - 200:561, 500:800] = 0
            Image.fromarray(ref).save(tmp / "mask.png")
            camera = SideCamera(1280, 720, 400.0, 0.0, 0.5)
            meta = ingest.run_ingest(tmp / "clip.mp4", tmp / "mask.png", camera, tmp / "ingest")
            self.assertEqual(meta["frames"], 30)
            self.assertEqual(meta["fps"], 24.0)
            data = np.load(tmp / "ingest" / "masks.npz")
            self.assertEqual(data["masks"].shape, (30, 360, 640))
            self.assertEqual(len(data["offsets"]), 31)
            self.assertEqual(meta["camera"]["px_per_m"], 200.0)
            self.assertTrue((tmp / "ingest" / "registered" / "00029.png").is_file())
            self.assertEqual(json.loads((tmp / "ingest" / "ingest.json").read_text())["frames"], 30)

    def test_rerun_with_shorter_video_drops_stale_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "src").mkdir()
            for i in range(12):
                Image.fromarray(frame(100 + 4 * i)).save(tmp / "src" / f"{i:03d}.png")
            for n in (12, 8):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", str(tmp / "src" / "%03d.png"),
                                "-frames:v", str(n), "-c:v", "mpeg4", "-q:v", "2", str(tmp / f"clip{n}.mp4")],
                               check=True)
            ref = np.full((180, 320), 255, dtype=np.uint8)
            ref[100:140, 100:160] = 0
            Image.fromarray(ref).save(tmp / "mask.png")
            camera = SideCamera(320, 180, 100.0, 0.0, 0.5)
            ingest.run_ingest(tmp / "clip12.mp4", tmp / "mask.png", camera, tmp / "ingest")
            meta = ingest.run_ingest(tmp / "clip8.mp4", tmp / "mask.png", camera, tmp / "ingest")
            self.assertEqual(meta["frames"], 8)
            self.assertEqual(len(list((tmp / "ingest" / "registered").glob("*.png"))), 8)
