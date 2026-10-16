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
        smooth = motion.SmoothCycle(self.samples, self.names, toy_rig.PROJECT)
        np.testing.assert_allclose(smooth.at([0.0]), smooth.at([1.0]), atol=1e-12)


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
            inside = [k for k in range(41) if on <= k / 40 <= off]
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


class HeglundTaylorTests(unittest.TestCase):
    def test_preferred_gallop_stride_frequency(self):
        self.assertAlmostEqual(motion.heglund_taylor_hz(30), 2.61, places=2)
        self.assertAlmostEqual(motion.heglund_taylor_hz(50), 2.41, places=2)
