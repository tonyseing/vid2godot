"""Task 19: the baked cycle's motion in numpy (side-plane FK, the smooth periodic fit, two-bone leg IK, the keys)."""
import math
import unittest

import numpy as np

from anim import motion
from tests import toy_rig


def one_bone_rig():
    return {"bones": [{"name": "root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 0.5]},
                      {"name": "arm", "parent": "root", "head": [0.0, 0.0, 1.0], "tail": [1.0, 0.0, 1.0]},
                      {"name": "hand", "parent": "arm", "head": [1.0, 0.0, 1.0], "tail": [1.5, 0.0, 1.0]}]}


class PlanarRigTests(unittest.TestCase):
    def test_a_joint_turns_about_world_y_like_blender(self):
        """+90 degrees about +Y takes +X to -Z (the right-hand rule Blender's Quaternion(axis, angle) follows)."""
        rig = motion.PlanarRig(one_bone_rig())
        posed = rig.pose([[0.0, 0.0, math.pi / 2, 0.0]], ["dx", "dz", "arm", "hand"])
        np.testing.assert_allclose(rig.point(posed, "arm")[0], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(rig.point(posed, "hand")[0], [0.0, -0.5], atol=1e-12)
        posed = rig.pose([[0.1, -0.2, 0.0, -math.pi / 2]], ["dx", "dz", "arm", "hand"])
        np.testing.assert_allclose(rig.point(posed, "hand", "head")[0], [1.1, 0.8], atol=1e-12, err_msg="dx, dz")
        np.testing.assert_allclose(rig.point(posed, "hand")[0], [1.1, 1.3], atol=1e-12)

    def test_unfitted_bones_follow_their_parents(self):
        rig = motion.PlanarRig(one_bone_rig())
        posed = rig.pose([[0.0, 0.0, math.pi]], ["dx", "dz", "arm"])
        np.testing.assert_allclose(rig.point(posed, "hand")[0], [-1.5, 1.0], atol=1e-12)


class SmoothCycleTests(unittest.TestCase):
    def setUp(self):
        n = 15
        u = np.arange(n) / n
        self.trunk = 0.3 * np.sin(2 * np.pi * u) + 0.04 * np.sin(6 * np.pi * u + 1) + 0.01 * (-1) ** np.arange(n)
        self.leg = np.where(u < 0.4, -0.5 + 2.5 * u, 0.5 - 1.5 * (u - 0.4))
        self.samples = np.stack([0.002 * np.cos(2 * np.pi * u), 0.01 * np.sin(4 * np.pi * u), self.trunk, self.leg],
                                axis=1)
        self.names = ["dx", "dz", "pelvis", "upper_hl"]

    def test_trunk_takes_the_lowest_order_that_keeps_its_range_and_legs_keep_every_sample(self):
        smooth = motion.SmoothCycle(self.samples, self.names, toy_rig.PROJECT)
        self.assertLessEqual(smooth.orders["pelvis"], 3)
        kept = smooth.kept_range()
        self.assertGreaterEqual(kept["pelvis"], motion.RANGE_KEEP)
        self.assertEqual(smooth.orders["upper_hl"], 7)
        u = np.arange(15) / 15
        np.testing.assert_allclose(smooth.at(u)[:, 3], self.leg, atol=1e-9, err_msg="legs pass through the samples")
        lower = motion.SmoothCycle(self.samples, self.names, toy_rig.PROJECT)
        lower.orders["pelvis"] -= 1
        self.assertLess(lower.kept_range()["pelvis"], motion.RANGE_KEEP, "the order is the lowest that keeps it")

    def test_a_strong_overtone_is_tapered_to_the_jerk_bound_and_the_range_restored(self):
        """A neck that nods at twice the stride rate (the real lope's neck_02: second harmonic 0.54 of the first)
        needs order 2 to keep its range, and its jerk is then 4.2 times a sine's. The overtones are tapered until
        the jerk ratio is at most JERK_MAX and the series scaled about its mean back to the range."""
        n = 15
        u = np.arange(n) / n
        neck = 0.08 * np.sin(2 * np.pi * u) + 0.043 * np.sin(4 * np.pi * u + 0.7)
        samples = np.stack([np.zeros(n), np.zeros(n), neck], axis=1)
        smooth = motion.SmoothCycle(samples, ["dx", "dz", "neck_02"], toy_rig.PROJECT)
        self.assertEqual(smooth.orders["neck_02"], 2)
        self.assertLess(smooth.taper["neck_02"], 1.0)
        self.assertLessEqual(smooth.jerk_ratio()["neck_02"], motion.JERK_MAX + 1e-6)
        self.assertGreaterEqual(smooth.kept_range()["neck_02"], motion.RANGE_KEEP)
        values = smooth.at(np.arange(600) / 600)[:, 2]
        self.assertAlmostEqual(float(values.mean()), float(neck.mean()), places=9, msg="about its mean")
        plain = motion.SmoothCycle(samples, ["dx", "dz", "neck_02"], toy_rig.PROJECT)
        plain.taper["neck_02"], plain.gain["neck_02"] = 1.0, 1.0
        self.assertGreater(plain.jerk_ratio()["neck_02"], 4.0, "untapered, the fit would be rough")

    def test_roll_starts_the_motion_later(self):
        smooth = motion.SmoothCycle(self.samples, self.names, toy_rig.PROJECT)
        u = np.linspace(0, 1, 37)
        np.testing.assert_allclose(smooth.roll(0.3).at(u), smooth.at(u + 0.3), atol=1e-12)

    def test_the_series_is_periodic(self):
        """Fix round 1: at() wraps its phases, so comparing at(0) with at(1) proved nothing. The series itself, read
        without the wrap, closes at the seam in value, slope and curvature."""
        smooth = motion.SmoothCycle(self.samples, self.names, toy_rig.PROJECT)
        for coeffs, name in zip(smooth.coeffs, smooth.names):
            args = (smooth.orders[name], smooth.taper[name], smooth.gain[name])
            for derivative in (0, 1, 2):
                before = motion.evaluate(coeffs, np.array([1.0 - 1e-7]), *args, derivative=derivative,
                                         keep=smooth.dominant[name])
                at_zero = motion.evaluate(coeffs, np.array([0.0]), *args, derivative=derivative,
                                          keep=smooth.dominant[name])
                scale = 1.0 + float(np.abs(at_zero).max())
                self.assertLess(float(np.abs(before - at_zero).max()) / scale, 1e-4, f"{name} d{derivative}")
        self.assertNotEqual(float(smooth.at([0.25])[0, 2]), float(smooth.at([0.0])[0, 2]), "the column moves")

    def test_a_column_mostly_in_an_overtone_is_not_divided_by_its_fundamental(self):
        """Fix round 1, ruling 7: a root that bobs twice per stride is all second harmonic. Only harmonics above the
        dominant one are tapered and its jerk is measured against a sine at the dominant harmonic, so the series is
        neither tapered to nothing nor scaled by a near-zero fundamental."""
        n = 15
        u = np.arange(n) / n
        bob = 0.01 * np.sin(4 * np.pi * u + 0.3) + 1e-9 * np.sin(2 * np.pi * u)
        samples = np.stack([np.zeros(n), bob, 0.1 * np.sin(2 * np.pi * u)], axis=1)
        smooth = motion.SmoothCycle(samples, ["dx", "dz", "pelvis"], toy_rig.PROJECT)
        self.assertEqual(smooth.dominant["dz"], 2)
        self.assertEqual(smooth.taper["dz"], 1.0)
        self.assertTrue(math.isfinite(smooth.gain["dz"]) and 0.9 < smooth.gain["dz"] < 1.2, smooth.gain["dz"])
        values = smooth.at(np.arange(600) / 600)[:, 1]
        self.assertAlmostEqual(float(np.ptp(values)), float(np.ptp(bob)), delta=0.05 * float(np.ptp(bob)))
        np.testing.assert_allclose(smooth.at(u)[:, 1], bob, atol=0.002)
        taper, gain = motion.smoothest(motion.coefficients(bob), 2)
        self.assertEqual((taper, gain), (1.0, 1.0))
        overtone = 0.01 * np.sin(4 * np.pi * u) + 0.004 * np.sin(8 * np.pi * u + 1.0)
        coeffs = motion.coefficients(overtone)
        taper, gain = motion.smoothest(coeffs, 4)
        self.assertLess(taper, 1.0, "the 4th harmonic, above the dominant 2nd, is tapered")
        self.assertLessEqual(motion.jerk_ratio(coeffs, 4, taper, gain, keep=2) / 8.0, motion.JERK_MAX + 1e-6)
        self.assertTrue(math.isfinite(gain) and gain < 2.0)

    def test_trunk_columns_keep_their_sampled_range(self):
        """Fix round 1: the bar measures each trunk joint's range against the video fit's; a series of the lowest
        order that keeps RANGE_KEEP of it was then sampled at the 60 Hz keys and fell under (neck_01 0.945). The
        smoothed series is scaled about its mean up to the sampled range itself when it falls short of it."""
        smooth = motion.SmoothCycle(self.samples, self.names, toy_rig.PROJECT)
        self.assertGreaterEqual(smooth.kept_range()["pelvis"], 1.0 - 1e-4)
        short = np.stack([np.zeros(15), np.zeros(15), np.sign(np.sin(2 * np.pi * np.arange(15) / 15 + 0.1))], axis=1)
        square = motion.SmoothCycle(short, ["dx", "dz", "pelvis"], toy_rig.PROJECT)
        self.assertGreaterEqual(square.kept_range()["pelvis"], 1.0 - 1e-4, "a square wave's low order is scaled up")
        keys = smooth.at(np.arange(24) / 24)[:, 2]
        self.assertGreaterEqual(float(np.ptp(keys)) / float(np.ptp(self.trunk)), 0.98)


class LegIkTests(unittest.TestCase):
    def setUp(self):
        self.rig = motion.PlanarRig(toy_rig.toy_rig())
        self.names = toy_rig.NAMES
        rng = np.random.default_rng(3)
        self.values = rng.uniform(-0.15, 0.15, (5, len(self.names)))
        self.values[:, :2] = 0.0
        self.chain = toy_rig.PROJECT["chains"]["front_L"]

    def knee_side(self, values):
        posed = self.rig.pose(values, self.names)
        hip, knee, ankle = (self.rig.point(posed, b, "head") for b in ("upper_fl", "lower_fl", "paw_fl"))
        return np.sign(motion.cross(ankle - hip, knee - hip))

    def world(self, values, bone):
        return self.rig.pose(values, self.names)[0][bone]

    def test_reaches_the_target_keeping_the_foot_and_the_knee_side(self):
        free = self.rig.paws(self.values, self.names, toy_rig.PROJECT)["front_L"]
        target = free + np.array([0.03, 0.03])
        solved, reached = motion.solve_leg(self.rig, self.values, self.names, self.chain, "paw_fl", target)
        np.testing.assert_allclose(reached, target, atol=1e-9)
        np.testing.assert_allclose(self.rig.paws(solved, self.names, toy_rig.PROJECT)["front_L"], target, atol=1e-9)
        np.testing.assert_allclose(self.world(solved, "paw_fl"), self.world(self.values, "paw_fl"), atol=1e-9,
                                   err_msg="the foot keeps its world orientation")
        np.testing.assert_array_equal(self.knee_side(solved), self.knee_side(self.values), "the knee bends as it bent")
        others = [i for i, n in enumerate(self.names) if n not in ("upper_fl", "lower_fl", "paw_fl")]
        np.testing.assert_array_equal(solved[:, others], self.values[:, others])

    def test_a_target_just_out_of_reach_pivots_the_foot_about_the_toe(self):
        """When the pair cannot reach, the foot turns about the pinned toe (the heel lifts) just enough; the paw still
        lands on the target. Within reach, the foot keeps its world orientation (the test above)."""
        values = np.zeros((1, len(self.names)))
        free = self.rig.paws(values, self.names, toy_rig.PROJECT)["front_L"]
        target = free + np.array([-0.17, -0.02])            # far back and down: the leg alone is too short
        _, straight = motion.solve_leg(self.rig, values, self.names, self.chain, "paw_fl", target, pivot=False)
        self.assertGreater(float(np.linalg.norm(straight - target)), 0.003)
        solved, reached = motion.solve_leg(self.rig, values, self.names, self.chain, "paw_fl", target)
        np.testing.assert_allclose(reached, target, atol=1e-6)
        np.testing.assert_allclose(self.rig.paws(solved, self.names, toy_rig.PROJECT)["front_L"], target, atol=1e-6)
        turn = self.world(solved, "paw_fl") - self.world(values, "paw_fl")
        self.assertGreater(abs(float(turn[0])), 0.01, "the foot turned")
        self.assertLess(abs(float(turn[0])), 1.0)

    def test_an_unreachable_target_straightens_the_leg_short_of_it(self):
        free = self.rig.paws(self.values[:1], self.names, toy_rig.PROJECT)["front_L"]
        target = free + np.array([0.0, -1.0])
        _, reached = motion.solve_leg(self.rig, self.values[:1], self.names, self.chain, "paw_fl", target)
        self.assertGreater(float(np.linalg.norm(reached - target)), 0.4)

    def test_ik_pair_and_foot(self):
        self.assertEqual(motion.ik_bones(["scap", "up", "fore", "wrist", "paw"]), (["up", "fore"], ["wrist", "paw"]))
        self.assertEqual(motion.ik_bones(["thigh", "shin", "hock", "paw"]), (["thigh", "shin"], ["hock", "paw"]))
        self.assertEqual(motion.ik_bones(["up", "low", "paw"]), (["up", "low"], ["paw"]))
        with self.assertRaises(ValueError):
            motion.ik_bones(["up", "paw"])


class RedundantLegIkTests(unittest.TestCase):
    """A target constructed from a bent, fixed-foot pose is not a straight-leg mandate."""

    def setUp(self):
        points = [(-0.3, 0.0), (0.0, -0.4), (0.17, -0.87), (0.0, -1.34),
                  (0.05, -1.42), (0.12, -1.42)]
        self.chain = ["girdle", "upper", "lower", "wrist", "paw"]
        bones = [{"name": "body", "parent": None, "head": [-0.3, 0, 0], "tail": [0, 0, 0]}]
        for i, bone in enumerate(self.chain):
            a, b = points[i], points[i + 1]
            bones.append({"name": bone, "parent": "body" if i == 0 else self.chain[i - 1],
                          "head": [a[0], 0, a[1]], "tail": [b[0], 0, b[1]]})
        self.rig = motion.PlanarRig({"bones": bones})
        self.names = ["dx", "dz", "body"] + self.chain
        self.values = np.zeros((1, len(self.names)))
        self.witness = self.values.copy()
        self.witness[0, self.names.index("girdle")] = np.radians(10)
        self.witness[0, self.names.index("upper")] = np.radians(-10)
        self.target = self.rig.point(self.rig.pose(self.witness, self.names), "paw")

    def bend(self, values):
        posed = self.rig.pose(values, self.names)
        a, b, c = (self.rig.point(posed, n, "head")[0] for n in ("upper", "lower", "wrist"))
        return np.degrees(np.arccos(np.clip(np.dot(b - a, c - b) /
                                           np.linalg.norm(b - a) / np.linalg.norm(c - b), -1, 1)))

    def assert_pose_contract(self, solved, reached, limits=None):
        np.testing.assert_allclose(reached, self.target, atol=1e-6, rtol=0)
        posed = self.rig.pose(solved, self.names)
        np.testing.assert_allclose(self.rig.point(posed, "paw"), reached, atol=1e-12, rtol=0)
        np.testing.assert_allclose(posed[0]["wrist"], 0, atol=1e-9, rtol=0)
        frozen = [self.names.index(n) for n in ("dx", "dz", "body", "paw")]
        np.testing.assert_array_equal(solved[:, frozen], self.values[:, frozen])
        for bone in self.chain:
            low, high = np.radians((limits or {}).get(bone, [-60, 60]))
            self.assertGreaterEqual(solved[0, self.names.index(bone)], low - 1e-12)
            self.assertLessEqual(solved[0, self.names.index(bone)], high + 1e-12)

    def test_bent_fixed_foot_witness_uses_redundancy_before_straightening(self):
        self.assertGreater(self.bend(self.witness), 39)
        solved, reached = motion.solve_leg(self.rig, self.values, self.names, self.chain, "paw", self.target,
                                           pivot=False)
        self.assert_pose_contract(solved, reached)
        self.assertGreater(self.bend(solved), 30)
        self.assertLess(np.sum((solved - self.values) ** 2), np.sum((self.witness - self.values) ** 2))

    def test_shared_target_broadcast_matches_single_pose_solves(self):
        values = np.repeat(self.values, 2, axis=0)
        single, endpoint = motion.solve_leg(self.rig, self.values, self.names, self.chain, "paw", self.target,
                                            pivot=False)
        for target in (self.target, self.target[0]):
            with self.subTest(target_shape=target.shape):
                solved, reached = motion.solve_leg(self.rig, values, self.names, self.chain, "paw", target,
                                                   pivot=False)
                np.testing.assert_allclose(solved, np.repeat(single, 2, axis=0), atol=1e-12, rtol=0)
                np.testing.assert_allclose(reached, np.repeat(endpoint, 2, axis=0), atol=1e-12, rtol=0)
                np.testing.assert_allclose(reached, np.repeat(self.target, 2, axis=0), atol=1e-6, rtol=0)
                frozen = [self.names.index(n) for n in ("dx", "dz", "body", "paw")]
                np.testing.assert_array_equal(solved[:, frozen], values[:, frozen])

    def test_incompatible_target_batch_is_rejected_instead_of_partially_solving(self):
        values = np.repeat(self.values, 2, axis=0)
        with self.assertRaises(ValueError):
            motion.solve_leg(self.rig, values, self.names, self.chain, "paw", np.repeat(self.target, 3, axis=0))

    def test_integer_pose_arrays_and_lists_match_float_endpoint_and_foot_frame(self):
        expected, endpoint = motion.solve_leg(self.rig, self.values, self.names, self.chain, "paw", self.target,
                                              pivot=False)
        for values in (self.values.astype(np.int64), self.values.astype(np.int64).tolist()):
            with self.subTest(input_type=type(values).__name__):
                solved, reached = motion.solve_leg(self.rig, values, self.names, self.chain, "paw", self.target,
                                                   pivot=False)
                np.testing.assert_allclose(solved, expected, atol=1e-12, rtol=0)
                np.testing.assert_allclose(reached, endpoint, atol=1e-12, rtol=0)
                self.assert_pose_contract(solved, reached)

    def test_asymmetric_boxes_constrain_derived_upper_lower_and_wrist(self):
        limits = {"girdle": [8, 12], "upper": [-12, -8], "lower": [-2, 2], "wrist": [-2, 2]}
        solved, reached = motion.solve_leg(self.rig, self.values, self.names, self.chain, "paw", self.target,
                                           pivot=False, limits_deg=limits)
        self.assert_pose_contract(solved, reached, limits)

    def test_unresolved_preservation_search_retains_legacy_analytic_fallback(self):
        limits = {bone: [-0.01, 0.01] for bone in self.chain}
        solved, reached = motion.solve_leg(self.rig, self.values, self.names, self.chain, "paw", self.target,
                                           limits_deg=limits)
        frozen = [self.names.index(n) for n in ("dx", "dz", "body", "paw")]
        np.testing.assert_array_equal(solved[:, frozen], self.values[:, frozen])
        np.testing.assert_allclose(self.rig.pose(solved, self.names)[0]["wrist"],
                                   self.rig.pose(self.values, self.names)[0]["wrist"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(reached, self.target, atol=1e-6, rtol=0)
        np.testing.assert_allclose(reached, self.rig.point(self.rig.pose(solved, self.names), "paw"),
                                   atol=1e-12, rtol=0)

    def test_frozen_angle_outside_fit_box_skips_preservation_not_legacy_correction(self):
        values = self.values.copy()
        values[0, self.names.index("paw")] = np.radians(20)
        witness = self.witness.copy()
        witness[0, self.names.index("paw")] = values[0, self.names.index("paw")]
        target = self.rig.point(self.rig.pose(witness, self.names), "paw")
        solved, reached = motion.solve_leg(self.rig, values, self.names, self.chain, "paw", target,
                                           limits_deg={"paw": [-10, 10]})
        frozen = [self.names.index(n) for n in ("dx", "dz", "body", "paw")]
        np.testing.assert_array_equal(solved[:, frozen], values[:, frozen])
        np.testing.assert_allclose(self.rig.pose(solved, self.names)[0]["wrist"],
                                   self.rig.pose(values, self.names)[0]["wrist"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(reached, target, atol=1e-6, rtol=0)
        self.assertEqual(solved[0, self.names.index("paw")], values[0, self.names.index("paw")])

    def test_ground_clearance_keeps_corrected_support_while_lifting_reachable_paw(self):
        keys = np.repeat(self.witness, 4, axis=0)
        before = self.rig.pose(keys, self.names)
        ground = float(self.rig.point(before, "paw")[0, 1] + 0.005)
        project = {"chains": {"front": self.chain}, "paws": {"front": "paw"}}
        solved, lifted = motion.clear_ground(self.rig, keys, self.names, project, {"front": ground}, 0.002,
                                             rounds=1)
        after = self.rig.pose(solved, self.names)
        self.assertGreater(lifted["front"], 0.)
        heights = motion.played_heights(self.rig, solved, self.names, project, {"front": ground})["front"]
        self.assertGreaterEqual(float(heights.min()), -0.002)
        np.testing.assert_allclose(self.rig.point(after, "upper", "head"),
                                   self.rig.point(before, "upper", "head"), atol=1e-12, rtol=0,
                                   err_msg="secondary clearance must not redistribute already corrected support")
        np.testing.assert_allclose(after[0]["wrist"], before[0]["wrist"], atol=1e-9, rtol=0)
        frozen = [self.names.index(n) for n in ("dx", "dz", "body", "paw")]
        np.testing.assert_array_equal(solved[:, frozen], keys[:, frozen])

    def test_unreachable_fallback_returns_actual_endpoint_not_a_claimed_pin(self):
        target = self.target + np.array([0., -5.])
        solved, reached = motion.solve_leg(self.rig, self.values, self.names, self.chain, "paw", target)
        np.testing.assert_allclose(reached, self.rig.point(self.rig.pose(solved, self.names), "paw"),
                                   atol=1e-12, rtol=0)
        self.assertGreater(float(np.linalg.norm(reached - target)), 1.)
        frozen = [self.names.index(n) for n in ("dx", "dz", "body", "paw")]
        np.testing.assert_array_equal(solved[:, frozen], self.values[:, frozen])


class PlanKeysTests(unittest.TestCase):
    def test_planted_paws_are_world_still_on_the_ground(self):
        poses = toy_rig.toy_gait(frames=20, trunk=0.02)
        values = np.array([f["values"] for f in poses["frames"]])
        smooth = motion.SmoothCycle(values, poses["names"], toy_rig.PROJECT)
        rig = motion.PlanarRig(toy_rig.toy_rig())
        stride = 1.0 * 20 / 24
        keys, paws, report = motion.plan_keys(rig, smooth, poses["names"], toy_rig.PROJECT, toy_rig.LOPE_WINDOWS,
                                              stride, toy_rig.REST, -0.01, 40)
        self.assertEqual(keys.shape, (41, len(poses["names"])))
        np.testing.assert_array_equal(keys[-1], keys[0])
        self.assertAlmostEqual(float(keys[:40, 0].mean()), 0.0, places=12)
        np.testing.assert_allclose(keys[:40, 1], values[0, 1] - 0.01, atol=1e-9, err_msg="the root is lifted")
        for chain, (on, off) in toy_rig.LOPE_WINDOWS.items():
            # Changed with fix round 1: the pin holds the keys inside the window less half a key at each end.
            inside = [k for k in range(41) if on + 0.5 / 40 - 1e-9 <= k / 40 <= off - 0.5 / 40 + 1e-9]
            p = paws[chain][inside]
            np.testing.assert_allclose(p[:, 1], 0.0, atol=1e-9, err_msg=chain)
            np.testing.assert_allclose(np.diff(p[:, 0]), -stride / 40, atol=1e-9, err_msg=chain)
            self.assertLess(report[chain]["maxResidualM"], 1e-6)
            self.assertEqual(report[chain]["pinnedKeys"], len(inside))

    def test_the_pins_move_with_the_in_place_root(self):
        """The keys lose the root's mean dx (in place); the pins are made from the same shifted motion, so a root
        offset changes no joint (before this, the pins kept the offset and the legs reached for it)."""
        poses = toy_rig.toy_gait(frames=20)
        values = np.array([f["values"] for f in poses["frames"]])
        rig = motion.PlanarRig(toy_rig.toy_rig())
        planned = []
        for offset in (0.0, 0.05):
            moved = values.copy()
            moved[:, 0] += offset
            smooth = motion.SmoothCycle(moved, poses["names"], toy_rig.PROJECT)
            keys, _, _ = motion.plan_keys(rig, smooth, poses["names"], toy_rig.PROJECT, toy_rig.LOPE_WINDOWS,
                                          20 / 24, toy_rig.REST, 0.0, 40)
            planned.append(keys)
        np.testing.assert_allclose(planned[1], planned[0], atol=1e-9)

    def test_the_pin_releases_without_a_velocity_jump(self):
        """Fix round 1, ruling 4 (the setting changed in round 3): the pin holds the paw over the window less half a
        key at each end; outside, the correction to the free paw fades over PIN_RELEASE.span_keys key intervals, so
        at liftoff and touchdown the paw's velocity changes by no more than the swing's median per-key change. The
        old blend (a 0.05-cycle position ease, about one key) jumped there."""
        poses = toy_rig.toy_gait(frames=20, trunk=0.02)
        values = np.array([f["values"] for f in poses["frames"]])
        smooth = motion.SmoothCycle(values, poses["names"], toy_rig.PROJECT)
        rig = motion.PlanarRig(toy_rig.toy_rig())
        n = 24
        windows = {c: (round(on * n) / n, round(off * n) / n) for c, (on, off) in toy_rig.LOPE_WINDOWS.items()}
        keys, paws, report = motion.plan_keys(rig, smooth, poses["names"], toy_rig.PROJECT, windows, 20 / 24,
                                              toy_rig.REST, 0.0, n)
        jumps = motion.edge_jumps(paws, motion.release_windows(windows, n), n, 0.02)
        # Round 4: the round-3 reading at the first and last pinned key (the span-wide reading is
        # test_a_hard_release_breaks_the_edge_gate_and_the_default_does_not's).
        for chain, jump in jumps.items():
            self.assertLessEqual(max(jump["touchdownPinnedMps"], jump["liftoffPinnedMps"]), jump["swingMedianMps"],
                                 chain)
            self.assertGreaterEqual(report[chain]["edgeKeys"], motion.PIN_RELEASE.span(n) * n - 1e-9)
        old = self.old_blend(rig, smooth, poses["names"], windows, 20 / 24, n)
        jumped = motion.edge_jumps(old, windows, n, 0.02)                  # the old plan pinned the whole window
        self.assertTrue(any(max(j["touchdownPinnedMps"], j["liftoffPinnedMps"]) > j["swingMedianMps"]
                            for j in jumped.values()), jumped)

    @staticmethod
    def old_blend(rig, smooth, names, windows, stride, n):
        """The keys' paws as the old plan made them: the target eased from the free paw to the pin line by a
        smoothstep weight over 0.05 of a cycle outside the window."""
        phases = np.arange(n) / n
        keys = smooth.at(phases)
        shift = float(keys[:, names.index("dx")].mean())
        keys[:, names.index("dx")] -= shift
        lines = motion.pin_lines(rig, smooth, names, toy_rig.PROJECT, windows, stride, toy_rig.REST, 0.0, shift)
        for chain, window in windows.items():
            weight, near = motion.stance_weight(phases, window, 0.05)
            line, live = lines[chain], weight > 0
            free = rig.paws(keys, names, toy_rig.PROJECT)[chain]
            pin = np.stack([line["x"] - stride * (near - line["mid"]), np.full(n, line["z"])], axis=1)
            target = weight[:, None] * pin + (1 - weight[:, None]) * free
            keys[live], _ = motion.solve_leg(rig, keys[live], names, toy_rig.PROJECT["chains"][chain],
                                             toy_rig.PROJECT["paws"][chain], target[live])
        keys = np.vstack([keys, keys[:1]])
        return rig.paws(keys, names, toy_rig.PROJECT)

    def test_edge_jumps_measure_the_velocity_change_across_the_release_span(self):
        """dv at key k is |v(k -> k+1) - v(k-1 -> k)| (x and z: the side plane; a constant body speed adds the same
        velocity to both keys, so the in-place and world changes are equal). Round 4 ruling: each edge's jump is the
        largest over its whole approach or release span, where the bar's harness finds the kink: from the key before
        the first eased key to the first pinned key at touchdown, from the last pinned key to the key after the last
        eased key at liftoff. The swing median over the keys that are not pinned."""
        n, dt = 8, 0.1
        x = np.array([0.0, -0.1, -0.2, -0.3, 0.0, 0.3, 0.4, 0.2, 0.0])     # pinned keys 0..3 slide back 0.1 per key
        paws = {"a": np.stack([x, np.zeros(9)], axis=1)}
        v = np.diff(x) / dt                                               # -1 -1 -1 3 3 1 -2 -2
        dv = [abs(v[k] - v[k - 1]) for k in range(8)]                     # 1 0 0 4 0 2 3 0
        jumps = motion.edge_jumps(paws, {"a": (0.0, 3 / 8)}, n, dt, span=2 / 8)["a"]
        # a 2-key span eases key 7 before touchdown and key 4 after liftoff
        self.assertAlmostEqual(jumps["touchdownMps"], max(dv[6], dv[7], dv[0]))   # 3.0, at key 6
        self.assertAlmostEqual(jumps["liftoffMps"], max(dv[3], dv[4], dv[5]))     # 4.0, at key 3
        self.assertEqual((jumps["touchdownKey"], jumps["liftoffKey"]), (6, 3))
        self.assertEqual(jumps["touchdownKeys"], [6, 7, 0])
        self.assertEqual(jumps["liftoffKeys"], [3, 4, 5])
        self.assertAlmostEqual(jumps["swingMedianMps"], float(np.median([dv[k] for k in (4, 5, 6, 7)])))
        z = motion.edge_jumps({"a": np.stack([np.zeros(9), x], axis=1)}, {"a": (0.0, 3 / 8)}, n, dt, span=2 / 8)["a"]
        self.assertAlmostEqual(z["touchdownMps"], 3.0)                    # the vertical counts the same
        default = motion.edge_jumps(paws, {"a": (0.0, 3 / 8)}, n, dt)["a"]
        self.assertEqual(default, motion.edge_jumps(paws, {"a": (0.0, 3 / 8)}, n, dt, span=motion.PIN_RELEASE.span(n))["a"])
        self.assertEqual(motion.release_windows({"a": (0.25, 0.5)}, 8), {"a": (0.3125, 0.4375)})
        self.assertEqual(motion.release_windows({"a": (0.25, 0.3125)}, 16), {"a": (0.28125, 0.28125)})

    def test_stance_weight_eases_outside_the_window(self):
        w, near = motion.stance_weight(np.array([0.95, 0.99, 0.0, 0.1, 0.2, 0.22, 0.3]), (0.0, 0.2), 0.05)
        np.testing.assert_allclose(w[[2, 3, 4]], 1.0)
        self.assertEqual(w[0], 0.0)
        self.assertTrue(0 < w[1] < 1 and 0 < w[5] < 1)
        self.assertEqual(w[6], 0.0)
        self.assertAlmostEqual(near[1], -0.01)

    def test_played_keys_are_linear_between_keys(self):
        keys = np.array([[0.0, 1.0], [1.0, 3.0], [0.0, 1.0]])
        values, phases = motion.played(keys, 4)
        np.testing.assert_allclose(values[:, 0], [0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25])
        np.testing.assert_allclose(phases, np.arange(8) / 8)

    def test_key_count_is_sixty_hertz(self):
        self.assertEqual(motion.key_count(0.4), 24)
        self.assertEqual(motion.key_count(0.6293), 38)

    def test_a_fast_stride_gets_at_least_the_floor_of_keys(self):
        """Vole round 5: the 5.31 Hz vole stride had round(60 / 5.31) = 11 keys at 60 Hz; every stride gets at least
        MIN_KEYS_PER_STRIDE, the wolf lope's own count, so its key rate rises (Godot plays any key rate)."""
        self.assertEqual(motion.MIN_KEYS_PER_STRIDE, 24)
        self.assertEqual(motion.key_count(1 / 5.31), 24)
        self.assertEqual(motion.key_count(0.05), 24)
        self.assertAlmostEqual(motion.key_rate(24, 1 / 5.31), 24 * 5.31)
        # The wolf lope (2.51 Hz after its retime): 60 Hz gives 23.9 keys, which round to the floor itself, so the
        # floor changes neither its key count nor its keys.
        wolf = 1 / 2.5118
        self.assertEqual(int(round(motion.KEY_RATE * wolf)), 24)
        self.assertEqual(motion.key_count(wolf), 24)
        self.assertAlmostEqual(motion.key_rate(24, wolf), 60.28, places=2)


# The old process's worst contact-edge velocity jump: chase3d tools/anim_compare.py criterion 8 on the shipped wolf
# lope (game/assets/models/wolf.glb, its hind_R), m/s over adjacent 60 Hz keys (Task 19 round 3 baseline).
OLD_CLIP_EDGE_JUMP_MPS = 4.879


class PinReleaseTests(unittest.TestCase):
    """Task 19 round 3: the pin release is a setting (span, derivatives matched per axis, where it starts), and the
    default is the one the wolf-lope sweep chose."""

    def plan(self, release=None, n=24):
        poses = toy_rig.toy_gait(frames=20, trunk=0.02)
        values = np.array([f["values"] for f in poses["frames"]])
        smooth = motion.SmoothCycle(values, poses["names"], toy_rig.PROJECT)
        rig = motion.PlanarRig(toy_rig.toy_rig())
        windows = {c: (round(on * n) / n, round(off * n) / n) for c, (on, off) in toy_rig.LOPE_WINDOWS.items()}
        keys, paws, report = motion.plan_keys(rig, smooth, poses["names"], toy_rig.PROJECT, windows, 20 / 24,
                                              toy_rig.REST, 0.0, n, release=release)
        return windows, keys, paws, report

    @staticmethod
    def velocity(paw, n, dt):
        p = np.asarray(paw, float)[:, :2]
        return (p[1:n + 1] - p[:n]) / dt                     # v_k over key k -> k + 1

    def test_the_default_is_the_swept_setting(self):
        self.assertEqual(motion.PIN_RELEASE, motion.PinRelease(span_stride=3.0 / 24, touchdown_orders=(2, 1),
                                                               liftoff_orders=(2, 1), inset_keys=0.5,
                                                               min_span=motion.BLEND))
        self.assertEqual(motion.PIN_RELEASE.span(24), 3.0 / 24)          # the round-4 3 keys at the wolf's 24
        self.assertEqual(motion.PIN_RELEASE.span_stride, 0.125)
        self.assertEqual(motion.PinRelease(span_stride=0.01).span(200), motion.BLEND)   # at least BLEND of a cycle

    def test_the_ease_is_a_share_of_the_stride_not_a_count_of_keys(self):
        """Vole round 5: at 11 keys per stride three key intervals were 0.27 of the stride on each side of a ~0.55
        stance; the ease is now the same share of every stride."""
        for n in (11, 24, 30, 48, 127):
            self.assertEqual(motion.PIN_RELEASE.span(n), 0.125, n)
            self.assertAlmostEqual(motion.PIN_RELEASE.span(n) * n, 3.0 * n / 24)

    def test_a_hard_release_breaks_the_edge_gate_and_the_default_does_not(self):
        """Round 4 ruling (the old test compared the toy with the wolf's 4.879 m/s, which no toy speed reaches, so it
        could not fail). Facts' enforced gate, the edge's velocity change at the first and last pinned key at most the
        swing median: the default passes it and a hard release (a one-key span matching position only) breaks it.
        Across the whole approach and release span (the round-4 reading, where the bar's harness finds the kink)
        neither is under the toy's swing median on any paw, because the toy's free paw paths themselves turn by
        2.6-2.7 m/s in a key at touchdown and liftoff (the pin's correction fades to the free paw, so its kink
        stays)."""
        n, dt = 24, 0.4 / 24                                  # a 0.4 s stride at 60 Hz, as the wolf lope
        windows, _, paws, _ = self.plan(n=n)
        release = motion.release_windows(windows, n)
        jumps = motion.edge_jumps(paws, release, n, dt)
        self.assertEqual(sorted(jumps), sorted(windows))
        for chain, j in jumps.items():
            self.assertLessEqual(max(j["touchdownPinnedMps"], j["liftoffPinnedMps"]), j["swingMedianMps"], chain)
        hard = motion.PinRelease(span_stride=1.0 / n, touchdown_orders=(0, 0), liftoff_orders=(0, 0), min_span=0.0)
        _, _, hard_paws, _ = self.plan(hard, n)
        broken = motion.edge_jumps(hard_paws, release, n, dt, span=hard.span(n))
        self.assertTrue(any(max(j["touchdownPinnedMps"], j["liftoffPinnedMps"]) > j["swingMedianMps"]
                            for j in broken.values()), broken)
        for chain in jumps:
            for j in (jumps[chain], broken[chain]):
                self.assertGreater(max(j["touchdownMps"], j["liftoffMps"]), j["swingMedianMps"], chain)

    def test_orders_choose_the_matched_derivatives_per_axis_and_edge(self):
        """The liftoff orders change only the keys after liftoff and the touchdown orders only those before
        touchdown."""
        n, dt = 24, 0.4 / 24
        windows, base, _, _ = self.plan(n=n)
        _, lift, _, _ = self.plan(motion.PinRelease(liftoff_orders=(2, 0)), n)
        _, land, _, _ = self.plan(motion.PinRelease(touchdown_orders=(0, 0)), n)
        self.assertFalse(np.allclose(lift, base))
        self.assertFalse(np.allclose(land, base))
        release = motion.release_windows(windows, n)
        for keys, side in ((lift, "after liftoff"), (land, "before touchdown")):
            changed = np.where(np.any(np.abs(keys - base) > 1e-9, axis=1))[0] % n
            for k in changed:
                u = k / n
                near = []
                for chain, (on, off) in release.items():
                    edge = off if side == "after liftoff" else on
                    d = ((u - edge) if side == "after liftoff" else (edge - u)) % 1.0
                    near.append(0 < d < motion.PIN_RELEASE.span(n) + 1e-9)
                self.assertTrue(any(near), f"key {k} changed but is not {side}")
        self.assertEqual(motion.PinRelease(liftoff_orders=(2, 0)).edge_orders(True).tolist(), [2, 1])
        self.assertEqual(motion.PinRelease(liftoff_orders=(2, 0)).edge_orders(False).tolist(), [2, 0])

    def test_the_span_and_the_inset_are_settings(self):
        self.assertEqual(motion.release_windows({"a": (0.25, 0.5)}, 8, 0.0), {"a": (0.25, 0.5)})
        self.assertEqual(motion.release_windows({"a": (0.25, 0.5)}, 8, 1.0), {"a": (0.375, 0.375)})
        self.assertEqual(motion.release_windows({"a": (0.25, 0.5)}, 8, -0.5), {"a": (0.1875, 0.5625)})
        self.assertEqual(motion.release_windows({"a": (0.0, 0.25)}, 8, -0.5), {"a": (0.9375, 0.3125)})
        self.assertEqual(motion.release_windows({"a": (0.25, 0.5)}, 8), motion.release_windows({"a": (0.25, 0.5)}, 8,
                                                                                               0.5))
        _, _, _, wide = self.plan(motion.PinRelease(span_stride=6.0 / 24))
        _, _, _, deep = self.plan(motion.PinRelease(inset_keys=1.25))
        _, _, _, default = self.plan()
        for chain in wide:
            self.assertAlmostEqual(wide[chain]["edgeKeys"], 6.0)
            self.assertAlmostEqual(default[chain]["edgeKeys"], 3.0)
            self.assertGreater(wide[chain]["easedKeys"], default[chain]["easedKeys"], chain)
            self.assertLess(deep[chain]["pinnedKeys"], default[chain]["pinnedKeys"], chain)


class StartKeyTests(unittest.TestCase):
    """Round 4 ruling: the loop's first key lies outside every paw's contact ease (the pin's approach and release
    spans), so the seam is not where a paw decelerates into its pin."""

    def test_ease_spans_are_the_approach_and_release_of_each_pin(self):
        spans = motion.ease_spans({"a": (0.25, 0.5)}, 8, motion.PinRelease(span_stride=2.0 / 8, min_span=0.0))
        np.testing.assert_allclose(spans["a"], [(0.0625, 0.3125), (0.4375, 0.6875)])
        self.assertEqual(motion.ease_spans({"a": (0.0, 1.0)}, 8), {"a": []})

    def test_the_start_key_is_outside_every_ease_with_the_most_room(self):
        n = 50
        windows = {c: (round(on * n) / n, round(off * n) / n) for c, (on, off) in toy_rig.LOPE_WINDOWS.items()}
        spans = motion.ease_spans(windows, n)
        inside = lambda u: any(a < u + d < b for c in spans.values() for a, b in c for d in (-1.0, 0.0, 1.0))
        self.assertTrue(inside(0.0), "the phase origin's touchdown key is inside its own approach")
        key, info = motion.start_key(windows, n)
        self.assertFalse(inside(key / n), key)
        self.assertTrue(info["outsideEases"])
        self.assertGreater(info["clearanceKeys"], 0.0)
        self.assertLess(info["originClearanceKeys"], 0.0)
        for k in range(n):
            self.assertLessEqual(motion.ease_clearance(k / n, spans), info["clearanceKeys"] / n + 1e-12, k)
        self.assertEqual(motion.start_key({"a": (0.0, 1.0)}, n)[0], 0)


class HeglundTaylorTests(unittest.TestCase):
    def test_preferred_gallop_stride_frequency(self):
        self.assertAlmostEqual(motion.heglund_taylor_hz(30), 2.61, places=2)
        self.assertAlmostEqual(motion.heglund_taylor_hz(50), 2.41, places=2)


# ------------------------------------------------------------------ vole round 5: the bake scaled to the body
def derived(name):
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parent / "fixtures" / "vole_scurry_derived.json").read_text())[name]


class LegReachTests(unittest.TestCase):
    """Vole round 5: the round-4 front-leg miss (6.2 mm against a 1.6 mm contact tolerance) was not the rig's chain
    lengths (the vole's rest front leg reaches its toe from the withers with 14 % to spare, the wolf's with 21 %) nor
    the toe pivot (at the missed keys the pair and foot were already straight): the IK held the scapula at the fit's
    angle, and with the fit's body 10 mm up (its ground lift) the withers were up to 58 mm from a pin the straight leg
    reaches at 53 mm. The scapula now swings, and past the whole leg's reach the body comes down."""

    def setUp(self):
        self.block = derived("voleR3Cycle")
        self.rig = motion.PlanarRig(self.block["rig"])
        self.project = self.block["project"]
        self.names = self.block["names"]

    def test_the_front_legs_hang_from_a_girdle_and_the_hinds_do_not(self):
        for chain, bones in self.project["chains"].items():
            if chain not in self.project["paws"]:
                continue
            (upper, _), _ = motion.ik_bones(bones)
            want = bones[0] if chain.startswith("front") else None
            self.assertEqual(motion.girdle_bone(self.rig, bones, upper), want, chain)

    def test_the_rest_leg_reaches_its_own_paw(self):
        rest = np.zeros((1, len(self.names)))
        for chain in self.project["paws"]:
            bones = self.project["chains"][chain]
            hinge, length = motion.leg_reach(self.rig, bones, self.project["paws"][chain])
            posed = self.rig.pose(rest, self.names)
            gap = np.linalg.norm(self.rig.point(posed, self.project["paws"][chain]) - self.rig.point(posed, hinge, "head"))
            self.assertLess(gap, 0.9 * length, chain)

    def pose_and_target(self, reach_share):
        """The rest pose with the front_L pin straight below the withers at `reach_share` of the whole leg's reach."""
        values = np.zeros((1, len(self.names)))
        bones = self.project["chains"]["front_L"]
        hinge, length = motion.leg_reach(self.rig, bones, "paw_front_L")
        posed = self.rig.pose(values, self.names)
        top = self.rig.point(posed, hinge, "head")[0]
        way = np.array([-0.25, -1.0]) / np.hypot(0.25, 1.0)            # down and back, away from the scapula's line
        return values, bones, top + reach_share * length * way

    def test_the_scapula_swings_to_reach_a_pin_the_pair_cannot(self):
        values, bones, target = self.pose_and_target(0.98)
        (upper, lower), foot = motion.ik_bones(bones)
        posed = self.rig.pose(values, self.names)
        pair = (np.linalg.norm(self.rig.head[lower] - self.rig.head[upper])
                + np.linalg.norm(self.rig.head[foot[0]] - self.rig.head[lower])
                + np.linalg.norm(self.rig.tail["paw_front_L"] - self.rig.head[foot[0]]))
        self.assertGreater(np.linalg.norm(target - self.rig.point(posed, upper, "head")[0]), pair)
        solved, reached = motion.solve_leg(self.rig, values, self.names, bones, "paw_front_L", target[None])
        self.assertLess(float(np.linalg.norm(reached[0] - target)), 1e-6)
        self.assertNotEqual(solved[0, self.names.index("scapula_L")], 0.0)
        np.testing.assert_allclose(self.rig.paws(solved, self.names, self.project)["front_L"][0], target, atol=1e-6)
        active = {upper, lower, foot[0], "scapula_L"}
        frozen = [i for i, name in enumerate(self.names) if name not in active]
        np.testing.assert_array_equal(solved[:, frozen], values[:, frozen])

    def test_a_reachable_pin_preserves_the_fixed_foot_frame(self):
        values, bones, target = self.pose_and_target(0.6)
        solved, reached = motion.solve_leg(self.rig, values, self.names, bones, "paw_front_L", target[None])
        before = self.rig.pose(values, self.names)[0]["wrist_L"]
        after = self.rig.pose(solved, self.names)[0]["wrist_L"]
        np.testing.assert_allclose(after, before, atol=1e-9, rtol=0)
        self.assertLess(float(np.linalg.norm(reached[0] - target)), 1e-6)

    def test_reach_drop_lowers_the_body_only_where_a_pin_is_out_of_reach(self):
        n = 24
        keys = np.zeros((n, len(self.names)))
        values, bones, near = self.pose_and_target(0.6)
        _, _, far = self.pose_and_target(1.1)
        target = np.repeat(near[None], n, axis=0)
        target[10] = far
        live = np.zeros(n, bool)
        live[8:13] = True
        plan = {"front_L": (target, live, np.zeros(n, bool), None)}
        drop = motion.reach_drop(self.rig, keys, self.names, self.project, plan, 0.125)
        self.assertTrue(np.all(drop >= 0))
        self.assertGreater(drop[10], 0.0)
        lowered = keys.copy()
        lowered[:, self.names.index("dz")] -= drop
        hinge, length = motion.leg_reach(self.rig, bones, "paw_front_L")
        top = self.rig.point(self.rig.pose(lowered, self.names), hinge, "head")
        self.assertLessEqual(float(np.linalg.norm(target[10] - top[10])), length * (1 - motion.REACH_MARGIN) + 1e-9)
        self.assertEqual(float(drop[0]), 0.0)                      # far from the out-of-reach key: untouched
        self.assertTrue(np.all(np.diff(drop[:11]) >= -1e-12))       # it eases in
        plan = {"front_L": (np.repeat(near[None], n, axis=0), live, np.zeros(n, bool), None)}
        self.assertFalse(motion.reach_drop(self.rig, keys, self.names, self.project, plan, 0.125).any())


class VoleKeysTests(unittest.TestCase):
    """Derived cycles retain their contact endpoints and body-reach contracts."""

    def plan(self, name, n=24):
        from anim import facts
        from tests.test_facts import cycle_contacts
        block = derived(name)
        tol = facts.tolerances(block["rig"], block["project"])
        contacts = cycle_contacts(block, tol=tol)
        windows = {c: facts.snapped(m, n) for c, m in contacts["masks"].items() if m.any()}
        period_s = block["periodFrames"] / block["fps"]
        return motion.plan_keys(contacts["planar"], contacts["smooth"], block["names"], block["project"], windows,
                                contacts["speed"] * period_s, block["rest"], -contacts["groundHeight"], n), block

    def test_the_vole_reaches_every_pin_and_no_eased_paw_goes_underground(self):
        (keys, paws, report), block = self.plan("voleR3Cycle")
        for chain, entry in report.items():
            self.assertLess(entry["maxResidualM"], 1e-4, (chain, entry))
            self.assertGreater(entry["bodyDropMaxM"], 0.0)
            self.assertLess(entry["bodyDropMaxM"], 0.012)
            self.assertGreaterEqual(float(np.min(paws[chain][:, 1])), entry["pinZ"] - 1e-4, chain)
        self.assertIn("girdleSwingMaxDeg", report["front_L"])
        self.assertNotIn("girdleSwingMaxDeg", report["hind_L"])

    def test_the_wolf_reaches_its_pins_without_a_body_drop(self):
        (keys, paws, report), _ = self.plan("wolfLopeCycle")
        for chain, entry in report.items():
            self.assertNotIn("bodyDropMaxM", entry, chain)
            self.assertEqual(entry["maxResidualM"], 0.0)


# ------------------------------------------------------------- vole round 6: the cycle as the clip plays it
VOLE_S = 60.09637720555386 / 24.0 / 13.296      # the vole's played stride, s


class PlayedJerkTests(unittest.TestCase):
    """The bar's criterion 4 on the played keys: a sine at the stride rate scores about 1 (the 60 Hz difference
    shrinks it a little), the same range at twice the rate about 8 times that: no smoothing of a column whose range
    is a second harmonic keeps it under 3."""

    def keys(self, harmonic, n=24):
        u = np.arange(n + 1) / n
        return np.sin(2 * np.pi * harmonic * u)

    def test_a_sine_at_the_stride_rate(self):
        jerk, span = motion.played_jerk(self.keys(1), VOLE_S)
        self.assertAlmostEqual(jerk, 0.99, delta=0.05)
        self.assertAlmostEqual(span, 2.0, delta=0.02)

    def test_the_same_range_at_twice_the_rate(self):
        jerk, _ = motion.played_jerk(self.keys(2), VOLE_S)
        self.assertGreater(jerk, 6.0)

    def test_a_constant_column(self):
        self.assertEqual(motion.played_jerk(np.ones(25), VOLE_S), (0.0, 0.0))


class ForKeysTests(unittest.TestCase):
    def cycle(self, name):
        from anim import facts
        from tests.test_facts import cycle_contacts
        block = derived(name)
        contacts = cycle_contacts(block, tol=facts.tolerances(block["rig"], block["project"]))
        return block, contacts

    def test_the_wolf_cycle_is_returned_untouched(self):
        block, contacts = self.cycle("wolfLopeCycle")
        smooth = contacts["smooth"]
        out, report = motion.for_keys(smooth, 24, 15.104269770324972 / 24.0 / 1.580815,
                                      motion.leg_bones(block["project"]), contacts["planar"], contacts["closed"],
                                      contacts["trunk"])
        self.assertIs(out, smooth)
        self.assertEqual((report["columns"], report["worldPitch"]), ({}, {}))

    def test_the_vole_columns_are_capped_low_passed_and_scaled_back(self):
        block, contacts = self.cycle("voleR3Cycle")
        smooth = contacts["smooth"]
        out, report = motion.for_keys(smooth, 24, VOLE_S, motion.leg_bones(block["project"]), contacts["planar"],
                                      contacts["closed"], contacts["trunk"])
        columns = report["columns"]
        self.assertEqual(columns["chest"]["orderCapped"], [25, 11])           # 24 keys hold 11 harmonics
        self.assertGreater(columns["tail_03"]["playedJerkBefore"], 20.0)
        for name, entry in columns.items():
            self.assertLessEqual(entry["playedJerk"], motion.JERK_TARGET * motion.JERK_MAX + 1e-9, name)
            self.assertGreaterEqual(entry["keptRangeAtKeys"], motion.RANGE_KEEP, name)
        for leg in motion.leg_bones(block["project"]):
            self.assertNotIn(leg, columns)
            i = smooth.names.index(leg)
            self.assertIs(out.coeffs[i], smooth.coeffs[i])

    def test_effective_coefficients_reproduce_the_series(self):
        _, contacts = self.cycle("voleR3Cycle")
        smooth = contacts["smooth"]
        u = np.arange(97) / 97
        for name in ("neck_02", "tail_03", "pelvis", "thigh_L"):
            np.testing.assert_allclose(motion.evaluate(smooth.effective(name), u, smooth.orders[name]),
                                       smooth.at(u)[:, smooth.names.index(name)], atol=1e-12)


class KneeSideTests(unittest.TestCase):
    """Round 5's vole front legs stood nearly straight and the IK took each key's lean as the elbow's side: it flipped
    between keys (forearm -121 to +52 degrees) and the played paw swung 6.9 mm into the ground. The knee bends to
    the rest pose's side."""

    def test_a_nearly_straight_leg_bends_to_its_rest_side_at_every_key(self):
        block = derived("voleR3Cycle")
        rig, names, project = motion.PlanarRig(block["rig"]), block["names"], block["project"]
        bones = project["chains"]["front_L"]
        (upper, lower), foot = motion.ik_bones(bones)
        rest_side = np.sign(motion.cross(rig.head[foot[0]] - rig.head[upper], rig.head[lower] - rig.head[upper]))
        values = np.zeros((2, len(names)))
        values[0, names.index(lower)] = np.radians(-40.0)      # past straight: the elbow leans the other way
        values[1, names.index(lower)] = np.radians(10.0)
        posed = rig.pose(values, names)
        reach = rig.point(posed, "paw_front_L")
        top = rig.point(posed, upper, "head")
        target = top + 0.97 * (reach - top)
        solved, _ = motion.solve_leg(rig, values, names, bones, "paw_front_L", target)
        posed = rig.pose(solved, names)
        a, k, e = (rig.point(posed, b, "head") for b in (upper, lower, foot[0]))
        np.testing.assert_array_equal(np.sign(motion.cross(e - a, k - a)), [rest_side, rest_side])


class SeamTests(unittest.TestCase):
    def test_a_smooth_loop_has_no_seam_jump_and_a_kink_does(self):
        rig = motion.PlanarRig(one_bone_rig())
        names, project = ["dx", "dz", "arm"], {"paws": {}, "chains": {}, "legPairs": []}
        u = np.arange(25) / 24
        keys = np.zeros((25, 3))
        keys[:, 2] = np.sin(2 * np.pi * u)
        self.assertLess(motion.seam_jumps(keys, rig, names, project, ["arm"])["arm"], 1.0)
        kinked = keys.copy()
        kinked[:, 2] = np.abs(np.sin(np.pi * u))                # a cusp at the seam
        self.assertGreater(motion.seam_jumps(kinked, rig, names, project, ["arm"])["arm"], 1.0)
        shift = motion.seam_shift(kinked, rig, names, project)
        self.assertIsNotNone(shift)
        rolled = np.roll(kinked[:-1], -shift, axis=0)
        self.assertLessEqual(motion.seam_jumps(np.vstack([rolled, rolled[:1]]), rig, names, project, ["arm"])["arm"],
                             1.0)
