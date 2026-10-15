"""OpenRouter video API: submit, poll, download. Paid; never called by tests."""
import json
import os
import functools
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from anim.ledger import StageError

BASE = "https://openrouter.ai/api/v1"


class OpenRouterError(StageError):
    pass


def is_trusted(url, base):
    """True only for https on exactly the base host (no userinfo, same port): the only place the key may go."""
    try:
        target, home = urlsplit(url), urlsplit(base)
        return (target.scheme == "https" and home.scheme == "https" and bool(target.hostname)
                and target.hostname == home.hostname and "@" not in target.netloc and target.port == home.port)
    except ValueError:
        return False


class StripKeyOnRedirect(urllib.request.HTTPRedirectHandler):
    """urllib copies Authorization to any redirect target; keep it only for https on the base host."""

    def __init__(self, base=BASE):
        self.base = base

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not is_trusted(newurl, self.base):
            for table in (new.headers, new.unredirected_hdrs):
                for name in [k for k in table if k.lower() == "authorization"]:
                    del table[name]
        return new


def urllib_transport(method, url, headers, data, timeout=120, base=BASE):
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(StripKeyOnRedirect(base))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def resolve(base, url):
    if url.startswith(("http://", "https://")):
        return url
    if url.startswith("/api/"):
        return urljoin(base, url)
    return f"{base}/{url.lstrip('/')}"


class Client:
    def __init__(self, api_key=None, transport=None, base=BASE):
        self.key = os.environ.get("OPENROUTER_API_KEY", "") if api_key is None else api_key
        if not self.key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        self.transport = transport or functools.partial(urllib_transport, base=base)
        self.base = base

    def _call(self, method, url, body=None, raw=False):
        url = resolve(self.base, url)
        headers = {"Content-Type": "application/json"}
        if is_trusted(url, self.base):
            headers["Authorization"] = f"Bearer {self.key}"
        status, data = self.transport(method, url, headers, json.dumps(body).encode() if body is not None else None)
        if status >= 400:
            raise OpenRouterError(f"OpenRouter {method} {url} -> HTTP {status}: {data[:300]!r}")
        if raw:
            return data
        try:
            parsed = json.loads(data)
        except ValueError as error:
            raise OpenRouterError(f"OpenRouter {method} {url} returned a body that is not JSON: {data[:200]!r}") \
                from error
        if not isinstance(parsed, dict):
            raise OpenRouterError(f"OpenRouter {method} {url} returned JSON that is not an object: {data[:200]!r}")
        return parsed

    def video_model(self, model_id):
        models = self._call("GET", "videos/models").get("data")
        if not isinstance(models, list):
            raise OpenRouterError("OpenRouter's video model list has no 'data' list")
        for model in models:
            if isinstance(model, dict) and model.get("id") == model_id:
                return model
        raise OpenRouterError(f"OpenRouter serves no video model {model_id!r}")

    def limit_remaining(self):
        data = self._call("GET", "key").get("data")
        if not isinstance(data, dict):
            raise OpenRouterError("OpenRouter's key response has no 'data' object")
        remaining = data.get("limit_remaining")
        if remaining is not None and (isinstance(remaining, bool) or not isinstance(remaining, (int, float))):
            raise OpenRouterError(f"OpenRouter's limit_remaining is not a number: {remaining!r}")
        return remaining

    def submit(self, payload):
        job = self._call("POST", "videos", payload)
        if not job.get("id") or not job.get("polling_url"):
            raise OpenRouterError(f"Submit response lacks 'id' or 'polling_url': {job}")
        return job

    def wait(self, job, interval=10, timeout=1800, sleep=time.sleep, clock=time.monotonic):
        if not job.get("polling_url"):
            raise OpenRouterError(f"Video job {job.get('id')} has no 'polling_url' to poll")
        start = clock()
        while True:
            state = self._call("GET", job["polling_url"])
            status = state.get("status")
            if not isinstance(status, str):
                raise OpenRouterError(f"Poll response for video job {job.get('id')} has no 'status': {state}")
            if status == "completed":
                return state
            if status == "failed":
                raise OpenRouterError(f"Video job {job.get('id')} failed: {state.get('error')}")
            if clock() - start > timeout:
                raise OpenRouterError(f"Video job {job.get('id')} did not finish in {timeout} s")
            sleep(interval)

    def download(self, job, path, index=0):
        urls = job.get("unsigned_urls")
        if not isinstance(urls, list) or not 0 <= index < len(urls) or not isinstance(urls[index], str):
            raise OpenRouterError(f"Completed video job {job.get('id')} has no 'unsigned_urls'[{index}] to download")
        path = Path(path)
        path.write_bytes(self._call("GET", urls[index], raw=True))
        return path

    def generation(self, generation_id):
        try:
            return self._call("GET", f"generation?id={generation_id}").get("data")
        except OpenRouterError:
            return None
