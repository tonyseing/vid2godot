import shutil
import tempfile
import unittest
from pathlib import Path

from anim.ledger import STAGES, Ledger, StageError


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.model = self.tmp / "wolf.glb"
        self.model.write_bytes(b"glb")
        self.ledger = Ledger.init(self.tmp / "work", self.model, "lope")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write(self, name, text):
        path = self.tmp / "work" / name
        path.write_text(text)
        return path

    def test_fresh_ledger_points_at_first_stage(self):
        self.assertEqual(self.ledger.next(), STAGES[0])
        self.assertEqual(self.ledger.status()["contract"], "missing")

    def test_init_refuses_existing_ledger(self):
        with self.assertRaises(StageError):
            Ledger.init(self.tmp / "work", self.model, "lope")

    def test_record_requires_earlier_stages(self):
        with self.assertRaises(StageError):
            self.ledger.record("brief", {}, {"brief": self.write("b.json", "{}")})

    def test_record_and_reload(self):
        self.ledger.record("contract", {}, {"rig": self.write("rig.json", "{}")}, {"bones": 35})
        again = Ledger(self.tmp / "work")
        self.assertEqual(again.state("contract"), "ok")
        self.assertEqual(again.require("contract")["facts"]["bones"], 35)
        self.assertEqual(again.next(), "brief")

    def test_changed_output_makes_stage_stale(self):
        rig = self.write("rig.json", "{}")
        self.ledger.record("contract", {}, {"rig": rig})
        rig.write_text('{"x": 1}')
        self.assertEqual(self.ledger.state("contract"), "stale")
        with self.assertRaises(StageError):
            self.ledger.require("contract")

    def test_changed_model_makes_everything_stale(self):
        self.ledger.record("contract", {}, {"rig": self.write("rig.json", "{}")})
        self.model.write_bytes(b"other")
        self.assertEqual(self.ledger.state("contract"), "stale")

    def test_upstream_change_makes_later_stages_stale(self):
        rig = self.write("rig.json", "{}")
        self.ledger.record("contract", {}, {"rig": rig})
        brief = self.write("brief.json", "{}")
        self.ledger.record("brief", {"rig": rig}, {"brief": brief})
        rig.write_text("changed")
        self.assertEqual(self.ledger.state("brief"), "stale")
        self.assertEqual(self.ledger.next(), "contract")
