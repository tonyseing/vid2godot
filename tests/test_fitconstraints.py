"""Geometric regressions for the existing fitter's coupled constraint steps."""
import unittest

import numpy as np

from anim import fitconstraints


class ConstraintRefinementTests(unittest.TestCase):
    def test_coupled_camera_root_and_joint_recover_observed_points(self):
        # A two-segment planar chain, translated at its root and camera-scaled.
        def points(x):
            root, scale, shoulder, elbow = x
            a = np.array([np.cos(shoulder), np.sin(shoulder)])
            b = a + np.array([np.cos(shoulder + elbow), np.sin(shoulder + elbow)])
            return scale * np.array([a, b]) + [root, 0.0]
        target = points(np.array([0.25, 0.85, 0.35, -0.7]))
        residuals = lambda x: (points(x) - target) / 0.025
        clamp = lambda x: np.clip(x, [-1, .8, -1, -1], [1, 1.25, 1, 1])
        start = np.array([0., 1.1, -.1, .1])
        result, diagnostic = fitconstraints.refine(
            start, lambda x: float(np.sum((points(x) - target) ** 2)),
            residuals, clamp, np.array([.1, .05, .1, .1]))
        self.assertTrue(diagnostic['feasible'])
        self.assertLessEqual(np.linalg.norm(residuals(result), axis=1).max(), 1 + 1e-7)
        self.assertGreater(np.count_nonzero(np.abs(result - start) > 1e-3), 2)
        np.testing.assert_array_equal(start, [0., 1.1, -.1, .1])

    def test_boundary_solution_makes_coupled_objective_progress(self):
        # Independent x/y steps leave a narrow observed disk; a tangent step can progress.
        residuals = lambda x: np.array([[(x[0] - x[1]) / .001, 0.]])
        objective = lambda x: float(np.sum((x - 1.) ** 2))
        result, diagnostic = fitconstraints.refine(
            np.array([.001, 0.]), objective, residuals, lambda x: np.clip(x, -2, 2), np.full(2, .1))
        self.assertTrue(diagnostic['feasible'])
        self.assertLess(objective(result), .25)
        self.assertLessEqual(np.linalg.norm(residuals(result), axis=1).max(), 1 + 1e-7)

    def test_incompatible_geometry_reports_unresolved_not_impossible(self):
        residuals = lambda x: np.array([[x[0] - 3., 0.], [x[0] + 3., 0.]])
        result, diagnostic = fitconstraints.refine(
            np.zeros(1), lambda x: float(x[0] ** 2), residuals,
            lambda x: np.clip(x, -1, 1), np.array([.1]))
        self.assertFalse(diagnostic['feasible'])
        self.assertEqual(diagnostic['status'], 'unresolved')
        self.assertTrue(np.isfinite(result).all())

    def test_steps_respect_parameter_bounds_and_frozen_scale(self):
        start = np.array([0., .9])
        clamp = lambda x: np.array([np.clip(x[0], -.2, .2), .9])
        result, diagnostic = fitconstraints.refine(
            start, lambda x: float((x[0] - 10.) ** 2),
            lambda x: np.array([[(x[0] - .15) / .1, 0.]]), clamp, np.array([.1, 0.]))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(result[0], -.2)
        self.assertLessEqual(result[0], .2)
        self.assertEqual(result[1], .9)
        self.assertTrue(diagnostic['feasible'])

    def test_absent_observations_leave_legacy_search_untouched(self):
        def unexpected(_):
            self.fail('no-observation refinement must not call objective or clamp')
        start = np.array([.1, .9])
        result, diagnostic = fitconstraints.refine(
            start, unexpected, lambda x: np.empty((0, 2)), unexpected, np.ones(2))
        np.testing.assert_array_equal(result, start)
        self.assertEqual(diagnostic['status'], 'no-observations')

    def test_feasible_candidate_beats_lower_soft_score_failure(self):
        residuals = lambda x: np.array([[x[0], 0.]])
        objective = lambda x: -float(x[0])
        self.assertTrue(fitconstraints.prefer(np.array([.9]), np.array([2.]), objective, residuals))
        self.assertFalse(fitconstraints.prefer(np.array([2.]), np.array([.9]), objective, residuals))

    def test_restoration_uses_disks_not_incompatible_interior_point_targets(self):
        # Both disks admit x=1, but their 0.95-radius targets do not intersect.
        # Treating them as equalities strands the old least-squares solve at .99.
        residuals = lambda x: np.array([[x[0] - 2., 0.], [2. * x[0] - 1., 0.]])
        result, diagnostic = fitconstraints.refine(
            np.zeros(1), lambda x: float(x[0] ** 2), residuals,
            lambda x: np.clip(x, -3., 3.), np.ones(1))
        self.assertTrue(diagnostic['feasible'])
        self.assertLessEqual(np.linalg.norm(residuals(result), axis=1).max(), 1. + 1e-12)
        self.assertAlmostEqual(result[0], 1., places=9)

    def test_restoration_redistributes_correction_from_saturated_parameter(self):
        # A root at its upper limit cannot supply the requested positive move;
        # the other joint can. The two disks meet only at root=0, joint=1.
        residuals = lambda x: np.array([[x[0] + x[1] - 2., 0.], [x[1], 0.]])
        clamp = lambda x: np.clip(x, [-1., -3.], [0., 3.])
        result, diagnostic = fitconstraints.refine(
            np.zeros(2), lambda x: float(np.sum(x ** 2)), residuals, clamp, np.ones(2))
        self.assertTrue(diagnostic['feasible'])
        self.assertLessEqual(np.linalg.norm(residuals(result), axis=1).max(), 1. + 1e-12)
        self.assertLessEqual(result[0], 0.)
        self.assertGreaterEqual(result[0], -1.)

    def test_restoration_reconsiders_bounds_when_joint_direction_must_reverse(self):
        residuals = lambda x: np.array([[2. - 2. * x[0] + x[1], 0.],
                                        [3. - 3. * x[0] + x[1], 0.]])
        clamp = lambda x: np.clip(x, [-4., -4.], [0., 0.])
        self.assertLessEqual(np.linalg.norm(residuals(np.array([0., -2.])), axis=1).max(), 1.)
        result, diagnostic = fitconstraints.refine(
            np.zeros(2), lambda x: 0., residuals, clamp, np.ones(2))
        self.assertTrue(diagnostic['feasible'])
        self.assertLessEqual(np.linalg.norm(residuals(result), axis=1).max(), 1. + 1e-12)
        np.testing.assert_array_equal(clamp(result), result)

    def test_restoration_detects_saturation_below_old_parameter_threshold(self):
        residuals = lambda x: np.array([[x[0] + x[1] - (1. + 2e-11), 0.],
                                        [2. * x[0] + x[1] - (1. + 5e-11), 0.]])
        clamp = lambda x: np.clip(x, [-1., -4.], [0., 4.])
        self.assertLessEqual(np.linalg.norm(residuals(np.array([0., 5e-11])), axis=1).max(), 1.)
        result, diagnostic = fitconstraints.refine(
            np.zeros(2), lambda x: 0., residuals, clamp, np.ones(2))
        self.assertTrue(diagnostic['feasible'])
        self.assertLessEqual(np.linalg.norm(residuals(result), axis=1).max(), 1. + 1e-12)
        np.testing.assert_array_equal(clamp(result), result)


if __name__ == '__main__':
    unittest.main()
