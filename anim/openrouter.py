"""OpenRouter video API: submit, poll, download. Paid; never called by tests."""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

from anim.ledger import StageError

BASE = "https://openrouter.ai/api/v1"


class OpenRouterError(StageError):
    pass


def urllib_transport(method, url, headers, data, timeout=120):
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def resolve(base, url):
    if url.startswith("http"):
        return url
    if url.startswith("/api/"):
        return urljoin(base, url)
    return f"{base}/{url.lstrip('/')}"


class Client:
    def __init__(self, api_key=None, transport=None, base=BASE):
        self.key = os.environ.get("OPENROUTER_API_KEY", "") if api_key is None else api_key
        if not self.key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        self.transport, self.base = transport or urllib_transport, base

    def _call(self, method, url, body=None, raw=False):
        url = resolve(self.base, url)
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        status, data = self.transport(method, url, headers, json.dumps(body).encode() if body is not None else None)
        if status >= 400:
            raise OpenRouterError(f"OpenRouter {method} {url} -> HTTP {status}: {data[:300]!r}")
        return data if raw else json.loads(data)

    def video_model(self, model_id):
        for model in self._call("GET", "videos/models")["data"]:
            if model["id"] == model_id:
                return model
        raise OpenRouterError(f"OpenRouter serves no video model {model_id!r}")

    def limit_remaining(self):
        return self._call("GET", "key")["data"].get("limit_remaining")

    def submit(self, payload):
        job = self._call("POST", "videos", payload)
        if "id" not in job or "polling_url" not in job:
            raise OpenRouterError(f"Unexpected submit response: {job}")
        return job

    def wait(self, job, interval=10, timeout=1800, sleep=time.sleep, clock=time.monotonic):
        start = clock()
        while True:
            state = self._call("GET", job["polling_url"])
            if state["status"] == "completed":
                return state
            if state["status"] == "failed":
                raise OpenRouterError(f"Video job {job['id']} failed: {state.get('error')}")
            if clock() - start > timeout:
                raise OpenRouterError(f"Video job {job['id']} did not finish in {timeout} s")
            sleep(interval)

    def download(self, job, path, index=0):
        path = Path(path)
        path.write_bytes(self._call("GET", job["unsigned_urls"][index], raw=True))
        return path

    def generation(self, generation_id):
        try:
            return self._call("GET", f"generation?id={generation_id}").get("data")
        except OpenRouterError:
            return None
