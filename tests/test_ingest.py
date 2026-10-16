import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

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
        unique = ingest.unique_indices(sixty)
        self.assertEqual(len(unique), 24)
        self.assertEqual(ingest.true_rate(len(unique), len(sixty), 60.0), 24.0)
        keep, fps, times = ingest.cadence(unique, len(sixty), 60.0)
        self.assertEqual(keep, unique)
        self.assertEqual(fps, 24.0)
        self.assertEqual(times, [i / 60.0 for i in unique])

    def test_held_frames_kept_at_container_rate(self):
        clip = [frame(40)] * 6 + [frame(40 + 3 * i) for i in range(1, 19)]
        unique = ingest.unique_indices(clip)
        self.assertEqual(len(unique), 19)
        keep, fps, times = ingest.cadence(unique, len(clip), 24.0)
        self.assertEqual(keep, list(range(24)))
        self.assertEqual(fps, 24.0)
        self.assertEqual(times, [i / 24.0 for i in range(24)])

    def test_pulldown_rate_ignores_a_frozen_run(self):
        unique = [frame(40 + 3 * i) for i in range(24)]
        sixty = []
        for i, img in enumerate(unique):
            sixty += [img] * (2 if i % 2 == 0 else 3)
            if i == 10:
                sixty += [img] * 15
        keep, fps, _ = ingest.cadence(ingest.unique_indices(sixty), len(sixty), 60.0)
        self.assertEqual(len(keep), 24)
        self.assertEqual(fps, 24.0)

    def pulldown(self, content_fps, container_fps, seconds):
        """A clip at container_fps whose frame i shows content frame floor(i * content / container)."""
        n = container_fps * seconds
        return [frame(40 + 3 * (i * content_fps // container_fps), w=640) for i in range(n)]

    def assert_pulldown(self, content_fps, container_fps, seconds=2):
        clip = self.pulldown(content_fps, container_fps, seconds)
        unique = ingest.unique_indices(clip)
        keep, fps, times = ingest.cadence(unique, len(clip), float(container_fps))
        self.assertEqual(len(keep), content_fps * seconds, f"{content_fps}-in-{container_fps}: kept per second")
        self.assertEqual(keep, unique, "pulldown keeps the first frame of each run")
        self.assertEqual(fps, float(content_fps))
        self.assertEqual(times, [i / container_fps for i in keep])

    def test_24_in_30_pulldown(self):
        self.assert_pulldown(24, 30)

    def test_25_in_30_pulldown(self):
        self.assert_pulldown(25, 30)

    def test_48_in_60_pulldown(self):
        self.assert_pulldown(48, 60)

    def test_24_in_60_pulldown(self):
        self.assert_pulldown(24, 60)

    # Clean pulldown at a non-standard content rate keeps that rate, as computed.
    def test_20_in_60_pulldown(self):
        self.assert_pulldown(20, 60)

    def test_16_in_24_pulldown(self):
        self.assert_pulldown(16, 24)

    def test_10_in_30_pulldown(self):
        self.assert_pulldown(10, 30)

    def test_8_in_24_pulldown(self):
        self.assert_pulldown(8, 24)

    def test_native_30_no_repeats_kept_whole(self):
        clip = [frame(40 + 3 * i) for i in range(30)]
        keep, fps, times = ingest.cadence(ingest.unique_indices(clip), len(clip), 30.0)
        self.assertEqual(keep, list(range(30)))
        self.assertEqual(fps, 30.0)
        self.assertEqual(times, [i / 30.0 for i in range(30)])

    def native_with_repeats(self, n, repeats):
        """A native clip of n frames, every frame moving, except that each frame in repeats repeats the one before."""
        content, c = [], 0
        for i in range(n):
            if i and i not in repeats:
                c += 1
            content.append(c)
        return [frame(40 + 2 * c, w=640) for c in content]

    def assert_native(self, clip, fps):
        keep, out_fps, times = ingest.cadence(ingest.unique_indices(clip), len(clip), fps)
        self.assertEqual(keep, list(range(len(clip))), "native keeps every frame")
        self.assertEqual(out_fps, fps)
        self.assertEqual(times, [i / fps for i in range(len(clip))])

    def test_native_24_one_stray_repeat_per_second_kept_whole(self):
        clip = self.native_with_repeats(48, {12, 36})
        self.assertEqual(len(ingest.unique_indices(clip)), 46)
        self.assert_native(clip, 24.0)

    def test_native_24_three_scattered_repeats_kept_whole(self):
        clip = self.native_with_repeats(96, {17, 50, 81})
        self.assertEqual(len(ingest.unique_indices(clip)), 93)
        self.assert_native(clip, 24.0)

    def test_native_24_irregular_repeats_kept_whole(self):
        # 12.5% duplicates, clustered at the start: no regular cadence, so native.
        clip = self.native_with_repeats(48, {3, 5, 7, 9, 11, 13})
        self.assertEqual(len(ingest.unique_indices(clip)), 42)
        self.assert_native(clip, 24.0)
        # The ruling's example: paired repeats make runs of 3, holds, so no steady duplicates at all.
        clip = self.native_with_repeats(48, {3, 4, 11, 12, 30, 31})
        self.assertEqual(len(ingest.unique_indices(clip)), 42)
        self.assert_native(clip, 24.0)

    def test_two_pose_clip_kept_whole(self):
        clip = [frame(40, w=640)] * 12 + [frame(100, w=640)] * 12
        self.assertEqual(ingest.unique_indices(clip), [0, 12])
        self.assert_native(clip, 24.0)

    def test_native_30_ten_percent_even_repeats_reads_as_pulldown(self):
        # Exactly 10% duplicates (1 - 54/60 rounds to 0.0999...), evenly placed: the boundary reads as pulldown.
        clip = self.native_with_repeats(60, {5, 15, 25, 35, 45, 55})
        unique = ingest.unique_indices(clip)
        self.assertEqual(len(unique), 54)
        keep, fps, times = ingest.cadence(unique, len(clip), 30.0)
        self.assertEqual(keep, unique)
        self.assertEqual(fps, 27.0)
        self.assertEqual(times, [i / 30.0 for i in keep])

    def test_all_identical_frames_kept_at_container_rate(self):
        self.assert_native([frame(40)] * 24, 24.0)
        self.assert_native([frame(40)] * 30, 30.0)
        # One run of 5 at 60 fps would read as 12 fps, a standard rate: one run is never enough for pulldown.
        self.assert_native([frame(40)] * 5, 60.0)

    def test_rate_not_snapped_when_far(self):
        self.assertAlmostEqual(ingest.true_rate(20, 60, 60.0), 20.0)

    def test_silhouette_keeps_largest_component(self):
        img = frame(100)
        img[5:8, 5:8] = (0, 0, 0)            # speck
        mask = ingest.silhouette(img, ingest.background_colour(img))
        self.assertEqual(int(mask.sum()), 60 * 40)

    def test_silhouette_keeps_the_gap_between_touching_paws(self):
        """Two legs whose paws touch close a pocket of background under the body: it stays background."""
        img = np.zeros((180, 320, 3), dtype=np.uint8)
        img[:] = GREY
        dark = (40, 30, 25)
        img[60:100, 100:220] = dark               # body
        img[100:150, 110:122] = dark              # back leg
        img[100:150, 190:202] = dark              # front leg
        img[140:150, 110:202] = dark              # the two paws touching along the ground
        gap = np.zeros((180, 320), dtype=bool)
        gap[100:140, 122:190] = True
        mask = ingest.silhouette(img, ingest.background_colour(img))
        self.assertFalse(mask[gap].any(), "background enclosed by the legs was filled")
        self.assertEqual(int(mask.sum()), int(np.any(img != GREY, axis=2).sum()))

    def test_silhouette_fills_enclosed_subject_coloured_pixels(self):
        """A thin subject-coloured line inside an enclosed pocket is lost to the opening; it is subject, so it is
        filled back, while the pocket's background pixels around it stay out."""
        img = np.zeros((180, 320, 3), dtype=np.uint8)
        img[:] = GREY
        dark = (40, 30, 25)
        img[60:140, 100:220] = dark
        img[80:120, 130:190] = GREY               # an enclosed background pocket
        img[100, 140:180] = dark                  # a 1 px subject line in it (the 3x3 opening removes it)
        mask = ingest.silhouette(img, ingest.background_colour(img))
        self.assertTrue(mask[100, 140:180].all())
        self.assertFalse(mask[85:95, 135:185].any())

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
        mask = np.zeros((180, 320), dtype=bool)
        mask[100:140, 200:260] = True        # rows 100-139, columns 200-259: not square in index space
        points = ingest.boundary_points(mask, max_points=50)
        self.assertEqual(points.shape, (50, 2))
        u, v = points[:, 0].astype(int), points[:, 1].astype(int)
        self.assertTrue(np.all((u >= 200) & (u < 260)), "column 0 must be u (the image column)")
        self.assertTrue(np.all((v >= 100) & (v < 140)), "column 1 must be v (the image row)")
        interior = ndimage.binary_erosion(mask)
        self.assertTrue(np.all(mask[v, u]) and not np.any(interior[v, u]), "every point must be on the boundary")


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
            self.assertEqual(meta["frameTimes"], [i / 24.0 for i in range(30)])
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
