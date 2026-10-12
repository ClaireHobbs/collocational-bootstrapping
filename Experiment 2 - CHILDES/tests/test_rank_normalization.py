"""Regression coverage for the empirical distribution used by the MSE fit."""

import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MODULES = ('analyze_complete_dataset_96mos', 'analyze_age_groups_96mos')


class RankNormalizationTest(unittest.TestCase):
    def test_missing_ranks_contribute_zero_to_the_mse_input(self):
        # run: (1/2, 1/2), jump: (1, 0). Average: (3/4, 1/4).
        # The old available-rank mean (3/4, 1/2) normalizes to (3/5, 2/5).
        for name in MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                averages, ranked = module.calculate_rank_proportions(
                    [('a', 'run'), ('b', 'run'), ('x', 'jump')],
                    ['run', 'jump'],
                )
                np.testing.assert_allclose(averages.average_proportion, [.75, .25])
                self.assertAlmostEqual(averages.average_proportion.sum(), 1)
                self.assertEqual(averages.num_verbs.tolist(), [2, 1])
                self.assertEqual(len(ranked), 3)
                alpha, _, _ = module.find_optimal_z(averages)
                # A two-rank Zipf has p(1)/p(2) = 2**alpha = 3.
                self.assertAlmostEqual(alpha, 1.58)

    def test_equal_subject_support_needs_no_correction(self):
        pairs = [('a', 'run'), ('a', 'run'), ('a', 'run'), ('b', 'run'),
                 ('x', 'jump'), ('y', 'jump')]
        for name in MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                averages, _ = module.calculate_rank_proportions(pairs, ['run', 'jump'])
                np.testing.assert_allclose(averages.average_proportion, [.625, .375])
                self.assertAlmostEqual(averages.average_proportion.sum(), 1)


if __name__ == '__main__':
    unittest.main()
