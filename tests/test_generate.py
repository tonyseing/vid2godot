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
    def __init__(self, remaining=100.0, fail_second=False, info=INFO, cost=1.1, fail_download_at=None,
                 generation_model="bytedance/seedance-2.5-20260807", job_model=None):
        self.remaining, self.fail_second, self.submitted = remaining, fail_second, []
        self.info, self.cost, self.fail_download_at = info, cost, fail_download_at
        self.generation_model, self.job_model, self.generation_calls = generation_model, job_model, 0

    def video_model(self, model_id):
        return self.info

    def limit_remaining(self):
        return self.remaining

    def submit(self, payload):
        self.submitted.append(payload)
        return {"id": f"j{len(self.submitted)}", "polling_url": "x"}

    def wait(self, job, **kwargs):
        if self.fail_second and job["id"] == "j2":
            raise OpenRouterError("Video job j2 failed: provider error")
        return {**job, "status": "completed", "generation_id": f"g{job['id']}", "usage": {"cost": self.cost},
                "unsigned_urls": ["u"], "model": self.job_model}

    def download(self, job, path, index=0):
        if job["id"] == self.fail_download_at:
            raise OpenRouterError("download dropped")
        Path(path).write_bytes(b"video-" + job["id"].encode())
        return Path(path)

    def generation(self, generation_id):
        self.generation_calls += 1
        return None if self.generation_model is None else {"model": self.generation_model}


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.png = self.tmp / "ref.png"
        Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8)).save(self.png)
        self.strip = mock.patch.object(generate, "strip", lambda video, out, count=8: Path(out).write_bytes(b"png"))
        self.strip.start()
        self.which = mock.patch.object(generate.shutil, "which", lambda name: "/usr/bin/" + name)
        self.which.start()

    def tearDown(self):
        self.which.stop()
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
        for client, kwargs in ((FakeClient(remaining=2.0), {}), (FakeClient(), {"max_cost": 1.0})):
            with self.assertRaises(StageError):
                generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", **kwargs)
            self.assertEqual(client.submitted, [])

    def test_unsupported_duration_refused(self):
        client = FakeClient()
        with self.assertRaises(StageError):
            generate.run_generate(client, self.tmp / "gen", "m", {**BRIEF, "videoSeconds": 40}, self.png,
                                  "#c8c8c8")
        self.assertEqual(client.submitted, [])

    def test_failed_job_records_nothing_final(self):
        with self.assertRaises(StageError) as caught:
            generate.run_generate(FakeClient(fail_second=True), self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8")
        self.assertIn("spent so far: $1.10", str(caught.exception))
        self.assertFalse((self.tmp / "gen" / "record.json").exists())
        partial = json.loads((self.tmp / "gen" / "record.partial.json").read_text())
        self.assertEqual([c["jobId"] for c in partial["candidates"]], ["j1"])
        self.assertEqual([j["jobId"] for j in partial["jobs"]], ["j1", "j2"])

    def test_download_failure_keeps_the_paid_job(self):
        client = FakeClient(fail_download_at="j2")
        with self.assertRaises(StageError) as caught:
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8")
        self.assertIn("spent so far: $2.20", str(caught.exception))
        self.assertIn("record.partial.json", str(caught.exception))
        partial = json.loads((self.tmp / "gen" / "record.partial.json").read_text())
        self.assertEqual([c["jobId"] for c in partial["candidates"]], ["j1"])
        self.assertEqual([j["jobId"] for j in partial["jobs"]], ["j1", "j2"])
        self.assertEqual(partial["jobs"][1]["pollingUrl"], "x")
        self.assertFalse((self.tmp / "gen" / "record.json").exists())

    def test_non_stage_error_is_wrapped_too(self):
        client = FakeClient()
        client.generation = mock.Mock(side_effect=KeyError("model"))
        with self.assertRaises(StageError) as caught:
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", sleep=lambda s: None)
        self.assertIn("spent so far: $1.10", str(caught.exception))
        self.assertTrue((self.tmp / "gen" / "record.partial.json").exists())

    def test_missing_ffmpeg_refused_before_any_submit(self):
        client = FakeClient()
        with mock.patch.object(generate.shutil, "which", lambda name: None):
            with self.assertRaises(StageError) as caught:
                generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8")
        self.assertIn("ffmpeg", str(caught.exception))
        self.assertEqual(client.submitted, [])

    def test_max_cost_stops_second_candidate_after_real_cost(self):
        client = FakeClient(cost=2.0)
        with self.assertRaises(StageError) as caught:
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", max_cost=2.4)
        self.assertEqual(len(client.submitted), 1)
        self.assertIn("spent so far: $2.00", str(caught.exception))
        partial = json.loads((self.tmp / "gen" / "record.partial.json").read_text())
        self.assertEqual([c["jobId"] for c in partial["candidates"]], ["j1"])

    def test_remaining_is_rechecked_before_the_second_candidate(self):
        client = FakeClient(remaining=100.0)
        reads = iter([100.0, 1.0])
        client.limit_remaining = lambda: next(reads)
        with self.assertRaises(StageError):
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8")
        self.assertEqual(len(client.submitted), 1)

    def test_unknown_price_with_max_cost_warns_and_caps_by_spend(self):
        info = {k: v for k, v in INFO.items() if k != "pricing_skus"}
        client = FakeClient(info=info, cost=3.0)
        with self.assertRaises(StageError):
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", max_cost=2.0)
        self.assertEqual(len(client.submitted), 1)
        partial = json.loads((self.tmp / "gen" / "record.partial.json").read_text())
        self.assertTrue(any("unbounded" in w for w in partial["warnings"]))
        client = FakeClient(info=info, cost=0.5)
        record = generate.run_generate(client, self.tmp / "gen2", "m", BRIEF, self.png, "#c8c8c8", max_cost=5.0)
        self.assertTrue(any("unbounded (price unknown)" in w for w in record["warnings"]))
        self.assertEqual(len(record["candidates"]), 2)

    def test_model_returned_retries_then_falls_back_to_job_model(self):
        client = FakeClient(generation_model=None, job_model="bytedance/seedance-2.5-x")
        record = generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", sleep=lambda s: None)
        self.assertEqual(client.generation_calls, 6)
        self.assertEqual(record["candidates"][0]["modelReturned"], "bytedance/seedance-2.5-x")
        self.assertEqual(record["warnings"], [])

    def test_model_returned_unknown_is_null_with_warning(self):
        client = FakeClient(generation_model=None)
        record = generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", sleep=lambda s: None)
        self.assertIsNone(record["candidates"][0]["modelReturned"])
        self.assertEqual(len(record["warnings"]), 2)

    def test_accept_needs_every_check(self):
        record = {"candidates": [{"index": 1, "video": "a.mp4", "sha256": "s"}]}
        checks = {name: True for name in generate.CHECKS} | {"reviewer": "agent", "notes": "clean"}
        self.assertEqual(generate.accept(record, 1, checks)["candidate"]["index"], 1)
        with self.assertRaises(StageError):
            generate.accept(record, 1, {**checks, "noCuts": False})
        with self.assertRaises(StageError):
            generate.accept(record, 2, checks)
