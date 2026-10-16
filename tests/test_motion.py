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
        for chain, jump in jumps.items():
            self.assertLessEqual(max(jump["touchdownMps"], jump["liftoffMps"]), jump["swingMedianMps"], chain)
            self.assertGreaterEqual(report[chain]["edgeKeys"], motion.PIN_RELEASE.span_keys - 1e-9)
        old = self.old_blend(rig, smooth, poses["names"], windows, 20 / 24, n)
        jumped = motion.edge_jumps(old, windows, n, 0.02)                  # the old plan pinned the whole window
        self.assertTrue(any(max(j["touchdownMps"], j["liftoffMps"]) > j["swingMedianMps"] for j in jumped.values()),
                        jumped)

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

    def test_edge_jumps_measure_the_velocity_change_at_the_pinned_keys(self):
        """dv at key k is |v(k -> k+1) - v(k-1 -> k)|; touchdown is read at the first pinned key, liftoff at the last;
        the swing median over the keys that are not pinned."""
        n, dt = 8, 0.1
        x = np.array([0.0, -0.1, -0.2, -0.3, 0.0, 0.3, 0.4, 0.2, 0.0])     # pinned keys 0..3 slide back 0.1 per key
        paws = {"a": np.stack([x, np.zeros(9)], axis=1)}
        jumps = motion.edge_jumps(paws, {"a": (0.0, 3 / 8)}, n, dt)["a"]
        v = np.diff(x) / dt                                               # -1 -1 -1 3 3 1 -2 -2
        self.assertAlmostEqual(jumps["touchdownMps"], abs(v[0] - v[7]))  # 1.0
        self.assertAlmostEqual(jumps["liftoffMps"], abs(v[3] - v[2]))     # 4.0
        self.assertAlmostEqual(jumps["swingMedianMps"], float(np.median([abs(v[k] - v[k - 1]) for k in (4, 5, 6, 7)])))
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
        self.assertEqual(motion.PIN_RELEASE, motion.PinRelease(span_keys=2.6, touchdown_orders=(2, 1),
                                                               liftoff_orders=(2, 1), inset_keys=0.5,
                                                               min_span=motion.BLEND))
        self.assertAlmostEqual(motion.PIN_RELEASE.span(24), 2.6 / 24)
        self.assertAlmostEqual(motion.PIN_RELEASE.span(12), 2.6 / 12)
        self.assertAlmostEqual(motion.PIN_RELEASE.span(200), motion.BLEND)   # a slow gait: at least BLEND of a cycle

    def test_the_default_has_no_edge_jump_above_the_old_clips_on_a_synthetic_stance(self):
        """Every per-key velocity change of every paw (the bar's criterion 8 measure at the keys, world = in place
        for a constant body speed) is under the old clip's worst, and at the pinned edges under the swing median."""
        n, dt = 24, 0.4 / 24                                  # a 0.4 s stride at 60 Hz, as the wolf lope
        windows, _, paws, _ = self.plan(n=n)
        for chain, paw in paws.items():
            v = self.velocity(paw, n, dt)
            change = np.linalg.norm(v - np.roll(v, 1, axis=0), axis=1)
            self.assertLessEqual(float(change.max()), OLD_CLIP_EDGE_JUMP_MPS, chain)
        jumps = motion.edge_jumps(paws, motion.release_windows(windows, n), n, dt)
        self.assertEqual(sorted(jumps), sorted(windows))
        for chain, j in jumps.items():
            self.assertLessEqual(max(j["touchdownMps"], j["liftoffMps"]), j["swingMedianMps"], chain)

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
        _, _, _, wide = self.plan(motion.PinRelease(span_keys=6.0))
        _, _, _, deep = self.plan(motion.PinRelease(inset_keys=1.25))
        _, _, _, default = self.plan()
        for chain in wide:
            self.assertAlmostEqual(wide[chain]["edgeKeys"], 6.0)
            self.assertAlmostEqual(default[chain]["edgeKeys"], 2.6)
            self.assertGreater(wide[chain]["easedKeys"], default[chain]["easedKeys"], chain)
            self.assertLess(deep[chain]["pinnedKeys"], default[chain]["pinnedKeys"], chain)


class HeglundTaylorTests(unittest.TestCase):
    def test_preferred_gallop_stride_frequency(self):
        self.assertAlmostEqual(motion.heglund_taylor_hz(30), 2.61, places=2)
        self.assertAlmostEqual(motion.heglund_taylor_hz(50), 2.41, places=2)
