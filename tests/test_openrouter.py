import json
import tempfile
import unittest
from pathlib import Path

from anim.openrouter import Client, OpenRouterError


class FakeTransport:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, method, url, headers, data):
        self.calls.append((method, url, headers, json.loads(data) if data else None))
        for (m, fragment), responses in self.routes.items():
            if m == method and fragment in url:
                status, body = responses.pop(0) if len(responses) > 1 else responses[0]
                return status, body if isinstance(body, bytes) else json.dumps(body).encode()
        return 404, b"{}"


class ClientTests(unittest.TestCase):
    def client(self, routes):
        transport = FakeTransport(routes)
        return Client("key", transport), transport

    def test_requires_a_key(self):
        with self.assertRaises(OpenRouterError):
            Client(api_key="", transport=FakeTransport({}))

    def test_limit_remaining(self):
        client, transport = self.client({("GET", "/key"): [(200, {"data": {"limit_remaining": 139.07}})]})
        self.assertAlmostEqual(client.limit_remaining(), 139.07)
        self.assertEqual(transport.calls[0][2]["Authorization"], "Bearer key")

    def test_video_model_lookup(self):
        client, _ = self.client({("GET", "/videos/models"): [(200, {"data": [{"id": "a/b"}]})]})
        self.assertEqual(client.video_model("a/b")["id"], "a/b")
        with self.assertRaises(OpenRouterError):
            client.video_model("x/y")

    def test_submit_wait_download(self):
        job = {"id": "j1", "polling_url": "https://openrouter.ai/api/v1/videos/j1", "status": "pending"}
        done = {**job, "status": "completed", "generation_id": "g1",
                "unsigned_urls": ["/api/v1/videos/j1/content?index=0"], "usage": {"cost": 0.42}}
        client, transport = self.client({("POST", "/videos"): [(202, job)],
                                         ("GET", "/videos/j1/content"): [(200, b"MP4")],
                                         ("GET", "/videos/j1"): [(200, {**job, "status": "in_progress"}), (200, done)]})
        submitted = client.submit({"model": "a/b"})
        finished = client.wait(submitted, interval=0, sleep=lambda s: None)
        with tempfile.TemporaryDirectory() as tmp:
            path = client.download(finished, Path(tmp) / "v.mp4")
            self.assertEqual(path.read_bytes(), b"MP4")
        self.assertTrue(transport.calls[-1][1].startswith("https://openrouter.ai/api/v1/videos/j1/content"))

    def test_failed_job_raises(self):
        job = {"id": "j1", "polling_url": "https://openrouter.ai/api/v1/videos/j1", "status": "pending"}
        client, _ = self.client({("GET", "/videos/j1"): [(200, {**job, "status": "failed", "error": "nsfw"})]})
        with self.assertRaises(OpenRouterError) as caught:
            client.wait(job, interval=0, sleep=lambda s: None)
        self.assertIn("nsfw", str(caught.exception))

    def test_wait_times_out(self):
        job = {"id": "j1", "polling_url": "https://openrouter.ai/api/v1/videos/j1", "status": "pending"}
        ticks = iter(range(0, 10000, 100))
        client, _ = self.client({("GET", "/videos/j1"): [(200, job)]})
        with self.assertRaises(OpenRouterError):
            client.wait(job, interval=0, timeout=250, sleep=lambda s: None, clock=lambda: next(ticks))

    def test_malformed_responses_raise_openrouter_error(self):
        job = {"id": "j1", "polling_url": "https://openrouter.ai/api/v1/videos/j1"}
        client, _ = self.client({("GET", "/videos/models"): [(200, {})], ("GET", "/key"): [(200, {})],
                                 ("POST", "/videos"): [(202, {})], ("GET", "/videos/j1"): [(200, {})]})
        for call in (lambda: client.video_model("a/b"), client.limit_remaining, lambda: client.submit({}),
                     lambda: client.wait(job, interval=0, sleep=lambda s: None),
                     lambda: client.download({"id": "j1"}, "unused.mp4"),
                     lambda: client.download({"id": "j1", "unsigned_urls": []}, "unused.mp4")):
            with self.assertRaises(OpenRouterError):
                call()

    def test_non_object_and_non_json_bodies_raise(self):
        client, _ = self.client({("GET", "/key"): [(200, b"[1]")], ("GET", "/videos/models"): [(200, b"<html>")]})
        with self.assertRaises(OpenRouterError):
            client.limit_remaining()
        with self.assertRaises(OpenRouterError):
            client.video_model("a/b")

    def test_key_only_goes_to_the_base_host(self):
        done = {"id": "j1", "unsigned_urls": ["https://cdn.example.com/v.mp4", "http://openrouter.ai/v.mp4",
                                              "https://openrouter.ai/api/v1/videos/j1/content"]}
        client, transport = self.client({("GET", "cdn.example.com"): [(200, b"A")],
                                         ("GET", "http://openrouter.ai"): [(200, b"B")],
                                         ("GET", "https://openrouter.ai"): [(200, b"C")]})
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(3):
                client.download(done, Path(tmp) / "v.mp4", index=index)
        self.assertNotIn("Authorization", transport.calls[0][2])
        self.assertNotIn("Authorization", transport.calls[1][2])
        self.assertEqual(transport.calls[2][2]["Authorization"], "Bearer key")
