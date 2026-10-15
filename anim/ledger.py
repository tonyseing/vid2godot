"""Hash-bound stage ledger: a stage records the files it read and wrote; any changed file makes it stale."""
import hashlib
import json
from pathlib import Path

STAGES = ("contract", "brief", "reference", "generate", "accept", "ingest", "fit", "facts", "bake", "review",
          "verify")


class StageError(Exception):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Ledger:
    def __init__(self, work):
        self.work = Path(work).absolute()
        self.path = self.work / "ledger.json"
        if not self.path.is_file():
            raise StageError(f"No ledger at {self.path}; run `init` first")
        self.data = json.loads(self.path.read_text())

    @classmethod
    def init(cls, work, model, clip):
        work, model = Path(work).absolute(), Path(model).absolute()
        if (work / "ledger.json").exists():
            raise StageError(f"A ledger already exists at {work / 'ledger.json'}")
        if not model.is_file():
            raise StageError(f"Model not found: {model}")
        work.mkdir(parents=True, exist_ok=True)
        data = {"schemaVersion": 1, "clip": clip, "model": {"path": str(model), "sha256": sha256_file(model)},
                "stages": {}}
        (work / "ledger.json").write_text(json.dumps(data, indent=2) + "\n")
        return cls(work)

    @property
    def clip(self):
        return self.data["clip"]

    @property
    def model(self):
        return Path(self.data["model"]["path"])

    def _entry(self, path):
        path = Path(path).absolute()
        if not path.is_file():
            raise StageError(f"Missing file for the ledger: {path}")
        return {"path": str(path), "sha256": sha256_file(path)}

    def record(self, stage, inputs, outputs, facts=None):
        if stage not in STAGES:
            raise StageError(f"Unknown stage {stage!r}")
        for earlier in STAGES[:STAGES.index(stage)]:
            if self.state(earlier) != "ok":
                raise StageError(f"`{stage}` needs `{earlier}` first (it is {self.state(earlier)})")
        self.data["stages"][stage] = {"inputs": {k: self._entry(v) for k, v in inputs.items()},
                                      "outputs": {k: self._entry(v) for k, v in outputs.items()},
                                      "facts": facts or {}}
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def _own_state(self, stage):
        entry = self.data["stages"].get(stage)
        if entry is None:
            return "missing"
        if not self.model.is_file() or sha256_file(self.model) != self.data["model"]["sha256"]:
            return "stale"
        for item in list(entry["inputs"].values()) + list(entry["outputs"].values()):
            if not Path(item["path"]).is_file() or sha256_file(item["path"]) != item["sha256"]:
                return "stale"
        return "ok"

    def state(self, stage):
        own = self._own_state(stage)
        if own != "ok":
            return own
        if any(self._own_state(earlier) != "ok" for earlier in STAGES[:STAGES.index(stage)]):
            return "stale"
        return "ok"

    def require(self, stage):
        state = self.state(stage)
        if state != "ok":
            raise StageError(f"Stage `{stage}` is {state}; run it (again) first")
        return self.data["stages"][stage]

    def output(self, stage, name):
        return Path(self.require(stage)["outputs"][name]["path"])

    def status(self):
        return {stage: self.state(stage) for stage in STAGES}

    def next(self):
        return next((stage for stage in STAGES if self.state(stage) != "ok"), None)
