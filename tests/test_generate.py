import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from anim import generate
from anim.ledger import StageError
from anim.openrouter import OpenRouterError

INFO = {"id": "bytedance/seedance-2.5", "supported_resolutions": ["480p", "720p"],
        "supported_aspect_ratios": ["16:9"], "supported_sizes": ["854x480", "1280x720"],
        "supported_durations": list(range(4, 31)), "supported_frame_images": ["first_frame", "last_frame"],
        "pricing_skus": {"video_tokens": "0.0000107"}}
BRIEF = {"clip": "lope", "subject": "grey wolf", "motion": "loping", "gait": "3-beat lope with a lead",
         "videoSeconds": 5, "resolution": "720p", "aspectRatio": "16:9"}


class FakeClient:
    def __init__(self, remaining=100.0, fail_second=False):
        self.remaining, self.fail_second, self.submitted = remaining, fail_second, []

    def video_model(self, model_id):
        return INFO

    def limit_remaining(self):
        return self.remaining

    def submit(self, payload):
        self.submitted.append(payload)
        return {"id": f"j{len(self.submitted)}", "polling_url": "x"}

    def wait(self, job, **kwargs):
        if self.fail_second and job["id"] == "j2":
            raise OpenRouterError("Video job j2 failed: provider error")
        return {**job, "status": "completed", "generation_id": f"g{job['id']}", "usage": {"cost": 1.1},
                "unsigned_urls": ["u"]}

    def download(self, job, path, index=0):
        Path(path).write_bytes(b"video-" + job["id"].encode())
        return Path(path)

    def generation(self, generation_id):
        return {"model": "bytedance/seedance-2.5-20260807"}


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.png = self.tmp / "ref.png"
        Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8)).save(self.png)
        self.strip = mock.patch.object(generate, "strip", lambda video, out, count=8: Path(out).write_bytes(b"png"))
        self.strip.start()

    def tearDown(self):
        self.strip.stop()

    def test_size_and_cost(self):
        self.assertEqual(generate.size_for(INFO, "720p", "16:9"), (1280, 720))
        self.assertAlmostEqual(generate.estimate_cost(INFO, 1280, 720, 5), 1280 * 720 * 24 * 5 / 1024 * 0.0000107)

    def test_prompt_is_strict(self):
        prompt = generate.build_prompt(BRIEF, "#c8c8c8")
        for phrase in ("No cuts, no zoom", "Exactly four legs", "light grey (#c8c8c8)", "grey wolf, loping"):
            self.assertIn(phrase, prompt)

    def test_two_candidates_recorded(self):
        client = FakeClient()
        record = generate.run_generate(client, self.tmp / "gen", "bytedance/seedance-2.5", BRIEF, self.png, "#c8c8c8")
        self.assertEqual(len(record["candidates"]), 2)
        self.assertEqual(record["candidates"][1]["modelReturned"], "bytedance/seedance-2.5-20260807")
        payload = client.submitted[0]
        self.assertEqual(payload["frame_images"][0]["frame_type"], "first_frame")
        self.assertTrue(payload["frame_images"][0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual((payload["duration"], payload["resolution"], payload["aspect_ratio"]), (5, "720p", "16:9"))
        self.assertNotEqual(client.submitted[0]["seed"], client.submitted[1]["seed"])

    def test_cost_guard_refuses(self):
        with self.assertRaises(StageError):
            generate.run_generate(FakeClient(remaining=2.0), self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8")
        with self.assertRaises(StageError):
            generate.run_generate(FakeClient(), self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", max_cost=1.0)

    def test_unsupported_duration_refused(self):
        with self.assertRaises(StageError):
            generate.run_generate(FakeClient(), self.tmp / "gen", "m", {**BRIEF, "videoSeconds": 40}, self.png,
                                  "#c8c8c8")

    def test_failed_job_records_nothing(self):
        with self.assertRaises(StageError) as caught:
            generate.run_generate(FakeClient(fail_second=True), self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8")
        self.assertIn("spent so far: $1.10", str(caught.exception))
        self.assertFalse((self.tmp / "gen" / "record.json").exists())

    def test_accept_needs_every_check(self):
        record = {"candidates": [{"index": 1, "video": "a.mp4", "sha256": "s"}]}
        checks = {name: True for name in generate.CHECKS} | {"reviewer": "agent", "notes": "clean"}
        self.assertEqual(generate.accept(record, 1, checks)["candidate"]["index"], 1)
        with self.assertRaises(StageError):
            generate.accept(record, 1, {**checks, "noCuts": False})
        with self.assertRaises(StageError):
            generate.accept(record, 2, checks)
