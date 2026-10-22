"""`fit --seed`: restart noise is a draw the fit records, never a hidden constant."""
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from anim import pipeline
from anim.ledger import StageError
from tests import test_articulation
from tests.test_pipeline import run


class FitSeedTests(unittest.TestCase):
    def setUp(self):
        test_articulation.ArticulationPipelineTests.setUp(self)
        # A plain (unenrolled) fit: no source observations or regions, only the seed under test.
        self.ledger.data["stages"]["fit"]["inputs"].pop("observations")
        self.ledger.path.write_text(json.dumps(self.ledger.data))
        shutil.rmtree(self.work / "articulation")

    def blender_args(self, *extra):
        seen = {}

        def capture(script, args, folder, blender):
            seen.update(args)
            raise StageError("stop after capturing the fit arguments")
        with mock.patch.object(pipeline, "run_blender", side_effect=capture):
            code, _, error = run("fit", "--work", str(self.work), *extra)
        self.assertEqual(code, 1, error)
        self.assertIn("stop after capturing", error)
        return seen

    def test_seed_reaches_the_fitter_only_when_asked(self):
        self.assertEqual(self.blender_args("--seed", "7")["seed"], 7)
        self.assertNotIn("seed", self.blender_args())  # Absent keeps the historical seed-0 stream.

    def test_fit_script_reseeds_restarts_and_records_the_draw(self):
        source = (Path(pipeline.__file__).parent / "blender" / "fit_clip.py").read_text()
        self.assertIn('rng = np.random.default_rng(int(args["seed"]))', source)
        self.assertIn('"restartSeed": args.get("seed")', source)


if __name__ == "__main__":
    unittest.main()
