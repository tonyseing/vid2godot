"""Approach-C paw keypoints (anim/paws.py): paws found in a mask, tracked over the clip, and the fitter's soft paw
term."""
import unittest

import numpy as np

from anim import paws


def quadruped(spread=40, hind=(45, 55), front=(125, 135), length=40):
    """A side silhouette: a body block and two legs whose paws stand at the bottom."""
    mask = np.zeros((140, 200), bool)
    mask[30:60, 30:170] = True
    mask[60:60 + length, hind[0]:hind[1]] = True
    mask[60:60 + length, front[0]:front[1]] = True
    return mask


class FindPawsTests(unittest.TestCase):
    def test_finds_each_leg_end(self):
        found = paws.find_paws(quadruped())
        blobs = np.array([p["blob"] for p in found])
        for u in (50, 130):
            self.assertLess(float(np.min(np.hypot(blobs[:, 0] - u, blobs[:, 1] - 96))), 5.0)

    def test_body_line_is_the_belly(self):
        self.assertEqual(paws.body_line(quadruped()), 59)

    def test_points_flattens_tip_and_blob(self):
        found = paws.find_paws(quadruped())
        self.assertEqual(len(paws.points(found)), 2 * len(found))

    def test_empty_mask(self):
        self.assertEqual(paws.find_paws(np.zeros((10, 10), bool)), [])


class TrackTests(unittest.TestCase):
    def test_a_detection_with_no_neighbour_in_time_is_dropped(self):
        steady = [np.array([[50.0 + 3 * f, 96.0]]) for f in range(5)]
        steady[2] = np.array([[56.0, 96.0], [180.0, 20.0]])      # a one-frame blip far from everything
        out = paws.track(steady, radius=30.0)
        self.assertEqual(len(out[2]), 1)
        self.assertEqual(out[2][0].tolist(), [56.0, 96.0])
        self.assertEqual([len(p) for p in out], [1, 1, 1, 1, 1])

    def test_a_lone_frame_keeps_its_detections(self):
        out = paws.track([np.array([[1.0, 2.0]])], radius=30.0)
        self.assertEqual(out[0].tolist(), [[1.0, 2.0]])


class PenaltyTests(unittest.TestCase):
    def test_paws_within_the_dead_zone_cost_nothing(self):
        """A clean frame: every fitted paw is near a detected paw and every detection is claimed."""
        fitted = np.array([[50.0, 95.0], [131.0, 97.0]])
        detected = np.array([[52.0, 96.0], [130.0, 96.0]])
        self.assertEqual(paws.penalty(fitted, detected, body_row=59), 0.0)

    def test_cost_grows_then_is_capped(self):
        detected = np.array([[50.0, 96.0]])
        cost = [paws.penalty(np.array([[50.0 + d, 96.0]]), detected, body_row=59) for d in (10, 20, 40, 400, 800)]
        self.assertTrue(cost[0] < cost[1] < cost[2] < cost[3])
        self.assertEqual(cost[3], cost[4])
        self.assertLessEqual(cost[4], 2 * paws.W_PAW * paws.PAW_CAP)

    def test_huber_is_quadratic_near_the_dead_zone_and_linear_beyond(self):
        e = np.array([0.0, paws.PAW_HUBER / 2, paws.PAW_HUBER, 2 * paws.PAW_HUBER, 3 * paws.PAW_HUBER])
        r = paws.robust(e + paws.PAW_DEADZONE)
        self.assertAlmostEqual(r[0], 0.0)
        self.assertAlmostEqual(r[1], 0.25 * r[2], places=6)      # quadratic: (h/2)^2 / (2h) = h/8 = (h/2) / 4
        self.assertAlmostEqual(r[4] - r[3], r[3] - r[2])                       # linear beyond the Huber knee

    def test_an_unclaimed_detected_paw_costs_even_when_every_fitted_paw_is_near_one(self):
        """Both fitted forepaws sit on one detected paw; the other detected paw is unclaimed."""
        fitted = np.array([[50.0, 96.0], [51.0, 96.0]])
        detected = np.array([[50.0, 96.0], [130.0, 96.0]])
        self.assertGreater(paws.penalty(fitted, detected, body_row=59), 0.0)

    def test_a_paw_hidden_in_the_torso_costs_more_than_one_below_it(self):
        """One leg is drawn inside the body while the mask shows a second paw below the body: the paw inside the
        torso silhouette is penalised for its depth above the body line."""
        mask = quadruped()
        detected = np.array([[50.0, 96.0], [130.0, 96.0]])
        hidden = np.array([[50.0, 96.0], [100.0, 40.0]])          # inside the torso, 19 rows above the belly
        d_hidden = float(np.hypot(100 - 130, 40 - 96))
        below = np.array([[50.0, 96.0], [130.0 + d_hidden, 96.0]])   # outside the torso, as far from its paw
        cost_hidden = paws.penalty(hidden, detected, body_row=59, mask=mask)
        cost_below = paws.penalty(below, detected, body_row=59, mask=mask)
        self.assertGreater(cost_hidden, cost_below)
        self.assertAlmostEqual(cost_hidden - cost_below, paws.W_TORSO * 19.0, places=6)

    def test_no_torso_cost_when_every_detected_paw_is_claimed(self):
        """A tucked leg in the body is not penalised when the mask shows no paw it should be at."""
        mask = quadruped()
        fitted = np.array([[50.0, 96.0], [130.0, 96.0], [100.0, 40.0]])
        detected = np.array([[50.0, 96.0], [130.0, 96.0]])
        tucked = paws.penalty(fitted, detected, body_row=59, mask=mask)
        self.assertEqual(tucked, paws.penalty(fitted, detected, body_row=59))

    def test_a_paw_is_compared_only_with_detections_its_leg_can_reach(self):
        """A foreleg folded back to the hip is near the hind paw but cannot reach it from the shoulder: it is pulled
        to the front paw instead (the cost is its distance from the front paw, not from the hind paw)."""
        detected = np.array([[50.0, 96.0], [130.0, 96.0]])
        fitted = np.array([[50.0, 96.0], [130.0, 96.0], [60.0, 70.0]])      # hind, front, a foreleg at the hip
        roots = np.array([[50.0, 60.0], [130.0, 60.0], [130.0, 60.0]])
        reach = np.array([40.0, 40.0, 40.0])
        reached = paws.penalty(fitted, detected, body_row=59, roots=roots, reach=reach)
        anywhere = paws.penalty(fitted, detected, body_row=59)
        self.assertAlmostEqual(reached, paws.W_PAW * float(paws.robust(np.hypot(70.0, 26.0))))
        self.assertAlmostEqual(anywhere, paws.W_PAW * float(paws.robust(np.hypot(10.0, 26.0))))

    def test_a_paw_with_no_detection_in_reach_costs_nothing(self):
        fitted = np.array([[50.0, 96.0]])
        cost = paws.penalty(fitted, np.array([[50.0, 96.0]]), body_row=59, roots=[[300.0, 60.0]], reach=[40.0])
        self.assertEqual(cost, 0.0)

    def test_a_tail_tip_claims_a_drooping_tail(self):
        """A detection at the tail tip is claimed by the fitted tail, so it costs nothing and leaves no paw
        unclaimed (no torso cost for a leg inside the body)."""
        mask = quadruped()
        detected = np.array([[50.0, 96.0], [130.0, 96.0], [20.0, 80.0]])
        fitted = np.array([[50.0, 96.0], [130.0, 96.0], [100.0, 40.0]])
        tail = np.array([[21.0, 79.0]])
        claimed = paws.penalty(fitted, detected, body_row=59, mask=mask, others=tail)
        self.assertAlmostEqual(claimed, paws.penalty(fitted, detected[:2], body_row=59, mask=mask))
        self.assertGreater(paws.penalty(fitted, detected, body_row=59, mask=mask), claimed)

    def test_no_detections_cost_nothing(self):
        self.assertEqual(paws.penalty(np.array([[1.0, 2.0]]), np.zeros((0, 2)), body_row=59), 0.0)


if __name__ == "__main__":
    unittest.main()
