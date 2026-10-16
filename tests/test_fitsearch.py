"""The fitter's per-frame search (anim/fitsearch.py): candidates are compared with the start at one pyramid level,
and a winner is kept only if, after its full-resolution polish, it also beats the start at full resolution."""
import unittest

import numpy as np

from anim import fitsearch


class SearchTests(unittest.TestCase):
    def run_search(self, coarse, fine, candidates, seed=0, polish=lambda y: y):
        rng = np.random.default_rng(seed)
        made = [(lambda best, c=c: np.array([c]) if c is not None else best + rng.normal(0.0, 1.0, 1), [0])
                for c in candidates]
        return fitsearch.search(np.array([0.0]), made, solve=lambda y, idx: y, coarse=coarse, fine=fine,
                                polish=polish)

    def test_a_coarse_win_that_is_worse_at_full_resolution_is_rejected(self):
        """At the coarse level the candidate at 2 beats the start at 0; at full resolution it is worse, so the
        start is kept (seeded random restarts included, all of them also worse at full resolution)."""
        coarse = lambda y: float((y[0] - 2.0) ** 2)
        fine = lambda y: float(y[0] ** 2)
        out = self.run_search(coarse, fine, [2.0, None, None, None], seed=3)
        self.assertEqual(out.tolist(), [0.0])

    def test_a_win_at_both_levels_is_kept(self):
        coarse = lambda y: float((y[0] - 2.0) ** 2)
        fine = lambda y: float((y[0] - 1.9) ** 2)
        out = self.run_search(coarse, fine, [2.0])
        self.assertEqual(out.tolist(), [2.0])

    def test_the_polished_pose_is_what_is_judged_at_full_resolution(self):
        """The coarse winner at 2 is polished to 2.5; at full resolution 2.5 is worse than the start, though 2 is
        not: the polished pose is the one judged, so the start is kept."""
        coarse = lambda y: float((y[0] - 2.0) ** 2)
        fine = lambda y: float(min((y[0] - 2.0) ** 2, 1.0) if y[0] < 2.2 else 5.0)
        out = self.run_search(coarse, fine, [2.0], polish=lambda y: y + 0.5)
        self.assertEqual(out.tolist(), [0.0])

    def test_candidates_are_scored_at_the_coarse_level_not_by_the_solver(self):
        """The solver may report any number; the search scores the solved pose itself at the coarse level."""
        coarse = lambda y: float((y[0] - 1.0) ** 2)
        fine = coarse
        made = [(lambda best: np.array([5.0]), [0])]
        out = fitsearch.search(np.array([0.0]), made, solve=lambda y, idx: y, coarse=coarse, fine=fine,
                               polish=lambda y: y)
        self.assertEqual(out.tolist(), [0.0])

    def test_seeded_random_restarts_are_deterministic(self):
        coarse = lambda y: float((y[0] - 0.7) ** 2)
        fine = coarse
        a = self.run_search(coarse, fine, [None] * 6, seed=11)
        b = self.run_search(coarse, fine, [None] * 6, seed=11)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertNotEqual(a.tolist(), [0.0])


if __name__ == "__main__":
    unittest.main()
