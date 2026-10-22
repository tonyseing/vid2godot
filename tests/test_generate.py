import json
import shutil
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
        return {**job, "status": "completed", "generation_id": f"g{job['id']}", "usage": {} if self.cost is None else {"cost": self.cost},
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
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
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

    def test_slow_motion_prompt(self):
        """Task 20: a brief's slowMotion asks for the motion in slow motion with every step clearly visible, the
        factor stated; the strict lines stay. Without it the prompt is unchanged."""
        plain = generate.build_prompt(BRIEF, "#c8c8c8")
        self.assertNotIn("slow motion", plain.lower())
        self.assertEqual(generate.build_prompt({**BRIEF, "slowMotion": None}, "#c8c8c8"), plain)
        prompt = generate.build_prompt({**BRIEF, "slowMotion": 3}, "#c8c8c8")
        for phrase in ("slow motion", "3 times slower than real time", "every step clearly visible",
                       "No cuts, no zoom", "Exactly four legs", "The same animal", "Strict side profile",
                       "camera tracks alongside", "grey wolf, loping"):
            self.assertIn(phrase, prompt)
        self.assertIn("2.5 times slower", generate.build_prompt({**BRIEF, "slowMotion": 2.5}, "#c8c8c8"))
        for line in plain.splitlines():
            self.assertIn(line, prompt)
        self.assertTrue(prompt.splitlines()[0].startswith(plain.splitlines()[0] + " Filmed in slow motion"))

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

    def test_known_price_bounds_later_candidates_by_observed_cost_when_higher(self):
        each = generate.estimate_cost(INFO, 1280, 720, 5)
        client = FakeClient(cost=2.0)
        max_cost = 2.0 + (each + 2.0) / 2
        self.assertLess(2.0 + each, max_cost)
        self.assertLess(max_cost, 2.0 + 2.0)
        with self.assertRaises(StageError) as caught:
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", max_cost=max_cost)
        self.assertEqual(len(client.submitted), 1)
        self.assertIn("about $2.00", str(caught.exception))

    def test_known_price_rechecks_remaining_with_observed_cost_when_higher(self):
        client = FakeClient(cost=2.0)
        reads = iter([100.0, 2.5])
        client.limit_remaining = lambda: next(reads)
        with self.assertRaises(StageError):
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8")
        self.assertEqual(len(client.submitted), 1)

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

    def test_unknown_price_bounds_later_candidates_by_observed_cost(self):
        info = {k: v for k, v in INFO.items() if k != "pricing_skus"}
        client = FakeClient(info=info, cost=1.0)
        with self.assertRaises(StageError) as caught:
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", max_cost=1.5)
        self.assertEqual(len(client.submitted), 1)
        self.assertIn("spent so far: $1.00", str(caught.exception))

    def test_unknown_price_rechecks_remaining_with_observed_cost(self):
        info = {k: v for k, v in INFO.items() if k != "pricing_skus"}
        client = FakeClient(info=info, cost=1.0)
        reads = iter([100.0, 1.2])
        client.limit_remaining = lambda: next(reads)
        with self.assertRaises(StageError):
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", max_cost=50.0)
        self.assertEqual(len(client.submitted), 1)

    def test_unknown_price_and_missing_cost_stops_before_next_submit(self):
        info = {k: v for k, v in INFO.items() if k != "pricing_skus"}
        client = FakeClient(info=info, cost=None)
        with self.assertRaises(StageError) as caught:
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8", max_cost=50.0)
        self.assertEqual(len(client.submitted), 1)
        self.assertIn("bound", str(caught.exception))
        partial = json.loads((self.tmp / "gen" / "record.partial.json").read_text())
        self.assertIn("job j1 reported no cost", partial["warnings"])

    def test_known_price_and_missing_cost_warns(self):
        client = FakeClient(cost=None)
        record = generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8")
        self.assertEqual(record["warnings"], ["job j1 reported no cost", "job j2 reported no cost"])


FIXTURES = Path(__file__).resolve().parent / "fixtures"
LISTING = {m["id"]: m for m in json.loads((FIXTURES / "openrouter_video_models_2026-10-15.json").read_text())["data"]}
COST_TABLE = json.loads((FIXTURES / "bakeoff_cost_table_2026-10-15.json").read_text())


class FixtureTests(unittest.TestCase):
    def test_cost_table_holds_no_account_numbers(self):
        text = (FIXTURES / "bakeoff_cost_table_2026-10-15.json").read_text().lower()
        for word in ("keylimit", "remaining", "balance", "credit"):
            self.assertFalse(word in text, f"cost-table fixture mentions {word!r}")


class SubmitCalled(BaseException):
    """Escapes run_generate's `except Exception`, so a submit cannot be swallowed into a StageError."""


class NoSubmitClient(FakeClient):
    submit_called = False

    def submit(self, payload):
        self.submit_called = True
        raise SubmitCalled("submit must not be called")


class PricingTests(unittest.TestCase):
    def test_every_listed_model_matches_the_bakeoff_table(self):
        self.assertEqual(len(COST_TABLE["rows"]), len(LISTING))
        for row in COST_TABLE["rows"]:
            with self.subTest(model=row["model"]):
                dollars, basis = generate.price(LISTING[row["model"]], 5, "720p", "16:9")
                if row["estimateUsd"] is None:
                    self.assertIsNone(dollars)
                else:
                    self.assertAlmostEqual(dollars, row["estimateUsd"], places=6)
                self.assertEqual(basis, row["pricingBasis"])

    def test_estimate_cost_prices_per_second_skus(self):
        kling = LISTING["kwaivgi/kling-v3.0-pro"]
        self.assertAlmostEqual(generate.estimate_cost(kling, 1280, 720, 5), 0.84)
        self.assertAlmostEqual(generate.estimate_cost(kling, 1280, 720, 5, audio=False), 0.56)
        self.assertAlmostEqual(generate.estimate_cost(INFO, 1280, 720, 5), 1280 * 720 * 24 * 5 / 1024 * 0.0000107)

    def test_unknown_pricing_is_none(self):
        dollars, basis = generate.price(LISTING["black-forest-labs/flux-video-upscale"], 5, "720p", "16:9")
        self.assertIsNone(dollars)
        self.assertTrue(basis.startswith("unknown"))
        self.assertIsNone(generate.estimate_cost({"id": "m"}, 1280, 720, 5))

    def test_null_sizes_derived_from_resolution_and_aspect(self):
        self.assertEqual(generate.size_for(LISTING["alibaba/wan-3.0"], "720p", "16:9"), (1280, 720))
        self.assertEqual(generate.size_for(LISTING["alibaba/wan-3.0"], "720p", "9:16"), (720, 1280))
        self.assertEqual(generate.size_for(LISTING["alibaba/wan-3.0"], "480p", "16:9"), (854, 480))
        absent = {k: v for k, v in LISTING["alibaba/wan-3.0"].items() if k != "supported_sizes"}
        self.assertEqual(generate.size_for(absent, "720p", "16:9"), (1280, 720))

    def test_listed_sizes_without_a_match_refused(self):
        info = {**INFO, "supported_sizes": ["854x480"]}
        with self.assertRaises(StageError):
            generate.size_for(info, "720p", "16:9")

    def test_null_sizes_with_unlisted_resolution_refused(self):
        with self.assertRaises(StageError):
            generate.size_for(LISTING["minimax/hailuo-3-max"], "720p", "16:9")
        with self.assertRaises(StageError):
            generate.size_for(LISTING["black-forest-labs/flux-video-edit"], "720p", "16:9")

    def test_audio_with_only_without_audio_rates_is_unknown(self):
        info = {"id": "m", "pricing_skus": {"duration_seconds_without_audio": "0.1"}}
        dollars, basis = generate.price(info, 5, "720p", "16:9", audio=True)
        self.assertIsNone(dollars)
        self.assertTrue(basis.startswith("unknown"))
        self.assertIsNone(generate.estimate_cost(info, 1280, 720, 5))
        self.assertAlmostEqual(generate.price(info, 5, "720p", "16:9", audio=False)[0], 0.5)
        resolved = {"id": "m", "pricing_skus": {"duration_seconds_without_audio_720p": "0.1"}}
        self.assertIsNone(generate.price(resolved, 5, "720p", "16:9", audio=True)[0])

    def test_audio_with_plain_or_with_audio_rate_is_priced(self):
        plain = {"id": "m", "pricing_skus": {"duration_seconds": "0.1", "duration_seconds_without_audio": "0.05"}}
        self.assertAlmostEqual(generate.price(plain, 5, "720p", "16:9", audio=True)[0], 0.5)
        with_audio = {"id": "m", "pricing_skus": {"duration_seconds_with_audio": "0.2",
                                                  "duration_seconds_without_audio": "0.05"}}
        self.assertAlmostEqual(generate.price(with_audio, 5, "720p", "16:9", audio=True)[0], 1.0)
        self.assertAlmostEqual(generate.price(with_audio, 5, "720p", "16:9", audio=False)[0], 0.25)

    def test_other_resolution_per_second_skus_are_skipped(self):
        for key in ("duration_seconds_540p", "duration_seconds_360p", "cents_per_second_output_2k",
                    "duration_seconds_with_audio_4k"):
            with self.subTest(key=key):
                info = {"id": "m", "pricing_skus": {key: "0.1"}}
                dollars, basis = generate.price(info, 5, "720p", "16:9", audio=True)
                self.assertIsNone(dollars)
                self.assertTrue(basis.startswith("unknown"))
        info = {"id": "m", "pricing_skus": {"duration_seconds_540p": "0.5", "duration_seconds": "0.1"}}
        dollars, basis = generate.price(info, 5, "720p", "16:9", audio=True)
        self.assertAlmostEqual(dollars, 0.5)
        self.assertIn("duration_seconds =", basis)

    def test_video_tokens_prefers_the_resolution_sku(self):
        seedance = LISTING["bytedance/seedance-2.0"]
        dollars, basis = generate.price(seedance, 5, "1080p", "16:9")
        self.assertAlmostEqual(dollars, 1920 * 1080 * 24 * 5 / 1024 * 0.0000077)
        self.assertEqual(basis, "video_tokens_1080p: 1920x1080 x 24 fps x 5 s / 1024 x $0.0000077")
        self.assertAlmostEqual(generate.estimate_cost(seedance, 1920, 1080, 5),
                               1920 * 1080 * 24 * 5 / 1024 * 0.0000077)
        dollars, basis = generate.price(seedance, 5, "720p", "16:9")
        self.assertAlmostEqual(dollars, 1280 * 720 * 24 * 5 / 1024 * 0.000007)
        self.assertEqual(basis, "video_tokens: 1280x720 x 24 fps x 5 s / 1024 x $0.000007")

    def test_video_tokens_adds_image_fee_and_minimum(self):
        tokens = 1280 * 720 * 24 * 5 / 1024 * 0.000001
        info = {**INFO, "pricing_skus": {"video_tokens": "0.000001", "cents_per_image_input": "10"}}
        dollars, basis = generate.price(info, 5, "720p", "16:9")
        self.assertAlmostEqual(dollars, tokens + 0.1)
        self.assertIn("+ cents_per_image_input $0.1 (one first frame)", basis)
        info = {**INFO, "pricing_skus": {"video_tokens": "0.000001", "minimum_cents_per_generation": "500"}}
        dollars, basis = generate.price(info, 5, "720p", "16:9")
        self.assertAlmostEqual(dollars, 5.0)
        self.assertIn("raised to minimum_cents_per_generation $5", basis)

    def test_sends_audio_unless_the_listing_says_it_cannot(self):
        self.assertFalse(generate.sends_audio(LISTING["runway/gen-4.5"]))
        self.assertTrue(generate.sends_audio(LISTING["kwaivgi/kling-v3.0-pro"]))
        self.assertTrue(generate.sends_audio(LISTING["x-ai/grok-imagine-video"]))
        self.assertTrue(generate.sends_audio(INFO))


class PerSecondGenerateTests(unittest.TestCase):
    setUp, tearDown = GenerateTests.setUp, GenerateTests.tearDown

    def test_per_second_max_cost_refused_before_any_submit(self):
        client = NoSubmitClient(info=LISTING["kwaivgi/kling-v3.0-pro"])
        with self.assertRaises(StageError) as caught:
            generate.run_generate(client, self.tmp / "gen", "kwaivgi/kling-v3.0-pro", BRIEF, self.png, "#c8c8c8",
                                  max_cost=1.0)
        self.assertIn("$1.68", str(caught.exception))
        self.assertFalse(client.submit_called)

    def test_per_second_estimate_checked_against_remaining(self):
        client = NoSubmitClient(info=LISTING["kwaivgi/kling-v3.0-pro"], remaining=2.0)
        with self.assertRaises(StageError) as caught:
            generate.run_generate(client, self.tmp / "gen", "kwaivgi/kling-v3.0-pro", BRIEF, self.png, "#c8c8c8")
        self.assertIn("exceeds the key's remaining $2.00", str(caught.exception))
        self.assertFalse(client.submit_called)

    def test_audio_without_only_rate_needs_max_cost_and_warns(self):
        info = {**LISTING["kwaivgi/kling-v3.0-pro"], "pricing_skus": {"duration_seconds_without_audio": "0.1"}}
        client = NoSubmitClient(info=info)
        with self.assertRaises(StageError) as caught:
            generate.run_generate(client, self.tmp / "gen", "m", BRIEF, self.png, "#c8c8c8")
        self.assertIn("pass --max-cost", str(caught.exception))
        self.assertFalse(client.submit_called)
        client = FakeClient(info=info, cost=0.5)
        record = generate.run_generate(client, self.tmp / "gen2", "m", BRIEF, self.png, "#c8c8c8", max_cost=5.0)
        self.assertIsNone(record["estimateUsd"])
        self.assertTrue(record["pricingBasis"].startswith("unknown"))
        self.assertIn("first candidate's cost is unbounded (price unknown)", record["warnings"])

    def test_runway_omits_generate_audio_and_prices_plain(self):
        client = FakeClient(info=LISTING["runway/gen-4.5"], cost=0.6)
        record = generate.run_generate(client, self.tmp / "gen", "runway/gen-4.5", BRIEF, self.png, "#c8c8c8")
        for payload in client.submitted:
            self.assertNotIn("generate_audio", payload)
        self.assertAlmostEqual(record["estimateUsd"], 1.2)
        self.assertIn("cents_per_second_output", record["pricingBasis"])
        self.assertEqual(record["warnings"], [])
        saved = json.loads((self.tmp / "gen" / "record.json").read_text())
        self.assertIs(saved["generateAudio"], False)

    def test_audio_capable_model_sends_generate_audio(self):
        client = FakeClient(info=LISTING["kwaivgi/kling-v3.0-pro"], cost=0.84)
        record = generate.run_generate(client, self.tmp / "gen", "kwaivgi/kling-v3.0-pro", BRIEF, self.png, "#c8c8c8")
        self.assertEqual(len(client.submitted), 2)
        for payload in client.submitted:
            self.assertIs(payload["generate_audio"], True)
        self.assertEqual(client.submitted[1]["seed"], 1002)
        self.assertEqual((client.submitted[1]["duration"], client.submitted[1]["resolution"],
                          client.submitted[1]["aspect_ratio"]), (5, "720p", "16:9"))
        self.assertAlmostEqual(record["estimateUsd"], 1.68)
        saved = json.loads((self.tmp / "gen" / "record.json").read_text())
        self.assertIs(saved["generateAudio"], True)
        self.assertEqual(len(saved["candidates"]), 2)

    def test_audio_off_sends_explicit_false_to_an_audio_model(self):
        client = FakeClient()
        generate.run_generate(client, self.tmp / "gen", "bytedance/seedance-2.5", BRIEF, self.png, "#c8c8c8",
                              audio=False)
        self.assertEqual(len(client.submitted), 2)
        for payload in client.submitted:
            self.assertIs(payload["generate_audio"], False)
        saved = json.loads((self.tmp / "gen" / "record.json").read_text())
        self.assertIs(saved["generateAudio"], False)

    def test_audio_off_prices_without_audio(self):
        client = FakeClient(info=LISTING["kwaivgi/kling-v3.0-pro"], cost=0.56)
        record = generate.run_generate(client, self.tmp / "gen", "kwaivgi/kling-v3.0-pro", BRIEF, self.png, "#c8c8c8",
                                       audio=False)
        self.assertAlmostEqual(record["estimateUsd"], 1.12)

    def test_audio_off_on_a_silent_model_still_omits_generate_audio(self):
        client = FakeClient(info=LISTING["runway/gen-4.5"], cost=0.6)
        generate.run_generate(client, self.tmp / "gen", "runway/gen-4.5", BRIEF, self.png, "#c8c8c8", audio=False)
        for payload in client.submitted:
            self.assertNotIn("generate_audio", payload)

    def test_null_sizes_model_generates_at_derived_size(self):
        client = FakeClient(info=LISTING["alibaba/wan-3.0"], cost=0.5)
        record = generate.run_generate(client, self.tmp / "gen", "alibaba/wan-3.0", BRIEF, self.png, "#c8c8c8")
        self.assertEqual(record["size"], [1280, 720])
        self.assertAlmostEqual(record["estimateUsd"], 1.0)
