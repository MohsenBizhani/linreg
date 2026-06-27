"""
tests/test_math.py
==================
Unit tests for linreg._math primitives:
    dot, list_mean, list_std, transpose
"""

import math
import unittest

from linreg._math import dot, list_mean, list_std, transpose


class TestDot(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(dot([1, 2, 3], [4, 5, 6]), 32.0)

    def test_zeros(self):
        self.assertEqual(dot([0, 0, 0], [1, 2, 3]), 0.0)

    def test_single_element(self):
        self.assertAlmostEqual(dot([7.0], [3.0]), 21.0)

    def test_negative_values(self):
        self.assertAlmostEqual(dot([-1, 2], [3, -4]), -11.0)

    def test_floats(self):
        self.assertAlmostEqual(dot([0.5, 0.5], [2.0, 2.0]), 2.0)

    def test_empty_returns_zero(self):
        # zip stops at the shorter; sum of empty generator is 0
        self.assertEqual(dot([], []), 0.0)


class TestListMean(unittest.TestCase):
    def test_integers(self):
        self.assertAlmostEqual(list_mean([1, 2, 3, 4, 5]), 3.0)

    def test_single_element(self):
        self.assertAlmostEqual(list_mean([42.0]), 42.0)

    def test_negative_values(self):
        self.assertAlmostEqual(list_mean([-2, 0, 2]), 0.0)

    def test_floats(self):
        self.assertAlmostEqual(list_mean([1.5, 2.5]), 2.0)

    def test_all_same(self):
        self.assertAlmostEqual(list_mean([7, 7, 7, 7]), 7.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            list_mean([])


class TestListStd(unittest.TestCase):
    def test_known_std(self):
        # [2, 4, 4, 4, 5, 5, 7, 9] — population σ = 2.0
        self.assertAlmostEqual(list_std([2, 4, 4, 4, 5, 5, 7, 9]), 2.0)

    def test_single_element_returns_one(self):
        # σ = 0 → sentinel value 1.0 to avoid division-by-zero
        self.assertEqual(list_std([5.0]), 1.0)

    def test_constant_list_returns_one(self):
        self.assertEqual(list_std([3, 3, 3, 3]), 1.0)

    def test_symmetry(self):
        # [-1, 1] → μ = 0, variance = 1, σ = 1
        self.assertAlmostEqual(list_std([-1.0, 1.0]), 1.0)

    def test_always_non_negative(self):
        self.assertGreaterEqual(list_std([10, 20, 30, 40]), 0.0)

    def test_matches_manual_formula(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mu = sum(values) / len(values)
        expected = math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))
        self.assertAlmostEqual(list_std(values), expected)


class TestTranspose(unittest.TestCase):
    def test_2x3_to_3x2(self):
        matrix = [[1, 2, 3], [4, 5, 6]]
        result = transpose(matrix)
        self.assertEqual(result, [[1, 4], [2, 5], [3, 6]])

    def test_square_matrix(self):
        matrix = [[1, 2], [3, 4]]
        result = transpose(matrix)
        self.assertEqual(result, [[1, 3], [2, 4]])

    def test_single_row(self):
        matrix = [[10, 20, 30]]
        result = transpose(matrix)
        self.assertEqual(result, [[10], [20], [30]])

    def test_single_column(self):
        matrix = [[1], [2], [3]]
        result = transpose(matrix)
        self.assertEqual(result, [[1, 2, 3]])

    def test_double_transpose_is_identity(self):
        matrix = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(transpose(transpose(matrix)), matrix)

    def test_returns_list_of_lists(self):
        result = transpose([[1, 2], [3, 4]])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)


if __name__ == "__main__":
    unittest.main()
