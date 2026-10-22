"""Contact correction must use redundant leg articulation, not erase a feasible fold.

The target is an actual Blender pose with a 40-degree elbow and unchanged foot
orientation. With the source scapula frozen its ankle is beyond the two-link
reach, so a pair-only solve collapses. This oracle measures imported/evaluated
Blender bones independently; it does not accept the solver's FK as ground truth.
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from anim import motion
from anim.blender_run import run_blender
from tests.blender_case import BLENDER, FIXTURES, fixture_glb, requires_blender

# The solver's numerical target contract and Blender's float32 pose measurement
# budget are deliberately separate. Neither changes production contact gates.
TARGET_TOLERANCE_M = 1e-6
BLENDER_POSITION_EPS_M = 5e-7
BLENDER_DIRECTION_EPS = 2e-6


def point(points, bone, end="head"):
    return np.asarray(points[bone][end], dtype=float)


def direction(points, bone):
    vector = point(points, bone, "tail") - point(points, bone)
    return vector / np.linalg.norm(vector)


def elbow_degrees(points):
    upper = point(points, "lower_fl") - point(points, "upper_fl")
    lower = point(points, "foot_fl") - point(points, "lower_fl")
    cosine = np.dot(upper, lower) / (np.linalg.norm(upper) * np.linalg.norm(lower))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


@requires_blender
class ContactArticulationRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.work = Path(cls._tmp.name)
        cls.model, _ = fixture_glb()
        cls.oracle = run_blender(
            FIXTURES / "contact_articulation_fixture.py", {"model": str(cls.model)},
            cls.work / "constructive", BLENDER)
        cls.names = cls.oracle["names"]
        cls.chain = cls.oracle["chain"]
        cls.rig = motion.PlanarRig(cls.oracle["rig"])

    def assert_bounds(self, values):
        for bone in self.chain:
            self.assertLessEqual(abs(float(values[self.names.index(bone)])), math.radians(60) + 1e-9,
                                 f"{bone} exceeds the original +/-60 degree limit")

    def assert_frozen_body(self, source, corrected):
        for bone in source:
            if bone not in self.chain:
                for end in ("head", "tail"):
                    np.testing.assert_allclose(point(corrected, bone, end), point(source, bone, end),
                                               atol=BLENDER_POSITION_EPS_M, rtol=0,
                                               err_msg=f"contact correction moved {bone}.{end}")

    def assert_fixed_foot(self, source, corrected):
        for bone in ("foot_fl", "paw_fl"):
            np.testing.assert_allclose(direction(corrected, bone), direction(source, bone),
                                       atol=BLENDER_DIRECTION_EPS, rtol=0,
                                       err_msg=f"contact correction changed {bone}'s world direction")

    def assert_constructive_fixture(self):
        source, witness = (self.oracle["points"][label] for label in ("source", "witness"))
        for label in ("source", "witness"):
            values = self.oracle["values"][label]
            measured = self.oracle["points"][label]
            self.assert_bounds(values)
            self.assertAlmostEqual(elbow_degrees(measured), 40.0, delta=1e-3)
            # Crosscheck both constructed inputs against actual Bpy geometry;
            # especially important at the split foot head used by the IK pair.
            posed = self.rig.pose([values], self.names)
            for bone in measured:
                for end in ("head", "tail"):
                    np.testing.assert_allclose(self.rig.point(posed, bone, end)[0],
                                               point(measured, bone, end)[[0, 2]],
                                               atol=BLENDER_POSITION_EPS_M, rtol=0,
                                               err_msg=f"fixture FK disagrees with Blender: {label}/{bone}/{end}")
        self.assert_fixed_foot(source, witness)
        self.assert_frozen_body(source, witness)
        # A full-chain bent witness reaches the chosen target by construction,
        # while the old fixed-shoulder, fixed-foot pair cannot reach its ankle.
        upper = point(source, "upper_fl")
        knee = point(source, "lower_fl")
        ankle = point(source, "foot_fl")
        pair_reach = np.linalg.norm(knee - upper) + np.linalg.norm(ankle - knee)
        foot_offset = point(source, "paw_fl", "tail") - ankle
        target = point(witness, "paw_fl", "tail")
        self.assertGreater(np.linalg.norm(target - foot_offset - upper), pair_reach + 0.005,
                           "fixture must actually exhaust the fixed-shoulder two-bone solve")
        return source, witness, target

    def test_constructive_bent_witness_is_independently_feasible(self):
        self.assert_constructive_fixture()

    def test_contact_solve_reaches_without_collapsing_a_feasible_elbow(self):
        source, witness, target = self.assert_constructive_fixture()
        values = np.asarray([self.oracle["values"]["source"]], dtype=float)
        before = values.copy()
        # Use the original fit bounds explicitly, not expanded recovery bounds.
        # Disabling toe pivot must not disable redundant fixed-foot articulation.
        limits_deg = {bone: [-60, 60] for bone in self.chain}
        corrected, reached = motion.solve_leg(self.rig, values, self.names, self.chain,
                                               "paw_fl", target[[0, 2]][None], pivot=False,
                                               limits_deg=limits_deg)
        measured = run_blender(
            FIXTURES / "contact_articulation_fixture.py",
            {"model": str(self.model), "poses": {"corrected": dict(zip(self.names, corrected[0].tolist()))}},
            self.work / "corrected", BLENDER)["points"]["corrected"]
        np.testing.assert_array_equal(values, before, err_msg="solve_leg mutated the source pose")
        self.assert_bounds(corrected[0])
        self.assert_fixed_foot(source, measured)
        self.assert_frozen_body(source, measured)
        self.assertGreaterEqual(elbow_degrees(measured), 30.0,
                                "a feasible 40-degree source/witness elbow must not be straightened by contact IK")
        self.assertLessEqual(float(np.linalg.norm(np.asarray(reached)[0] - target[[0, 2]])),
                             TARGET_TOLERANCE_M, "solver must meet its numerical endpoint target")
        self.assertLessEqual(float(np.linalg.norm(point(measured, "paw_fl", "tail") - target)),
                             TARGET_TOLERANCE_M + BLENDER_POSITION_EPS_M,
                             "actual evaluated Blender paw must reach the constructive contact target")
