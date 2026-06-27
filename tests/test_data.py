"""
tests/test_data.py
==================
Unit tests for linreg._data:
    train_test_split, generate_dataset
"""

import unittest

from linreg._data import generate_dataset, train_test_split


class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        self.X = [[float(i), float(i * 2)] for i in range(100)]
        self.y = [float(i) for i in range(100)]

    # ── Size correctness ──────────────────────────────────────────────

    def test_default_split_sizes(self):
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)
        self.assertEqual(len(y_tr), 80)
        self.assertEqual(len(y_te), 20)

    def test_custom_test_size(self):
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size=0.3)
        self.assertEqual(len(X_tr) + len(X_te), 100)
        self.assertEqual(len(X_te), 30)

    def test_no_samples_lost(self):
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size=0.25)
        self.assertEqual(len(X_tr) + len(X_te), 100)
        self.assertEqual(len(y_tr) + len(y_te), 100)

    # ── Reproducibility ───────────────────────────────────────────────

    def test_same_seed_produces_same_split(self):
        res1 = train_test_split(self.X, self.y, seed=0)
        res2 = train_test_split(self.X, self.y, seed=0)
        self.assertEqual(res1[0], res2[0])  # X_train identical
        self.assertEqual(res1[2], res2[2])  # y_train identical

    def test_different_seeds_produce_different_splits(self):
        X_tr1, _, _, _ = train_test_split(self.X, self.y, seed=1)
        X_tr2, _, _, _ = train_test_split(self.X, self.y, seed=2)
        self.assertNotEqual(X_tr1, X_tr2)

    # ── Alignment: X and y rows stay paired ───────────────────────────

    def test_xy_rows_remain_aligned(self):
        """Each y value must still correspond to its original X row."""
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, seed=7)
        # In our synthetic data, y[i] == i and X[i][0] == i
        for x_row, y_val in zip(X_tr, y_tr):
            self.assertAlmostEqual(x_row[0], y_val)
        for x_row, y_val in zip(X_te, y_te):
            self.assertAlmostEqual(x_row[0], y_val)

    # ── No overlap ────────────────────────────────────────────────────

    def test_train_test_no_overlap(self):
        X_tr, X_te, _, _ = train_test_split(self.X, self.y, seed=42)
        train_set = {tuple(row) for row in X_tr}
        test_set = {tuple(row) for row in X_te}
        self.assertEqual(len(train_set & test_set), 0)

    # ── Error paths ───────────────────────────────────────────────────

    def test_invalid_test_size_zero_raises(self):
        with self.assertRaises(ValueError):
            train_test_split(self.X, self.y, test_size=0.0)

    def test_invalid_test_size_one_raises(self):
        with self.assertRaises(ValueError):
            train_test_split(self.X, self.y, test_size=1.0)

    def test_mismatched_X_y_raises(self):
        with self.assertRaises(ValueError):
            train_test_split(self.X, self.y[:50])


class TestGenerateDataset(unittest.TestCase):
    # ── Shape ─────────────────────────────────────────────────────────

    def test_output_shapes(self):
        X, y = generate_dataset(n_samples=50)
        self.assertEqual(len(X), 50)
        self.assertEqual(len(y), 50)
        self.assertEqual(len(X[0]), 2)

    def test_default_n_samples(self):
        X, y = generate_dataset()
        self.assertEqual(len(X), 300)

    # ── Reproducibility ───────────────────────────────────────────────

    def test_same_seed_same_data(self):
        X1, y1 = generate_dataset(seed=99)
        X2, y2 = generate_dataset(seed=99)
        self.assertEqual(X1, X2)
        self.assertEqual(y1, y2)

    def test_different_seeds_different_data(self):
        _, y1 = generate_dataset(seed=1)
        _, y2 = generate_dataset(seed=2)
        self.assertNotEqual(y1, y2)

    # ── Value ranges ──────────────────────────────────────────────────

    def test_x1_in_range(self):
        X, _ = generate_dataset(n_samples=200, seed=0)
        for row in X:
            self.assertGreaterEqual(row[0], 0.0)
            self.assertLessEqual(row[0], 50.0)

    def test_x2_in_range(self):
        X, _ = generate_dataset(n_samples=200, seed=0)
        for row in X:
            self.assertGreaterEqual(row[1], 0.0)
            self.assertLessEqual(row[1], 30.0)

    def test_zero_noise_recovers_exact_relationship(self):
        """With noise=0, y should equal 3*x1 + 5*x2 + 10 exactly."""
        X, y = generate_dataset(n_samples=20, noise=0.0, seed=5)
        for row, target in zip(X, y):
            expected = 3 * row[0] + 5 * row[1] + 10
            self.assertAlmostEqual(target, expected, places=10)

    def test_noise_increases_variance(self):
        import math

        _, y_low = generate_dataset(n_samples=500, noise=1.0, seed=0)
        _, y_high = generate_dataset(n_samples=500, noise=50.0, seed=0)
        mean_low = sum(y_low) / len(y_low)
        mean_high = sum(y_high) / len(y_high)
        var_low = sum((v - mean_low) ** 2 for v in y_low) / len(y_low)
        var_high = sum((v - mean_high) ** 2 for v in y_high) / len(y_high)
        self.assertGreater(var_high, var_low)


if __name__ == "__main__":
    unittest.main()
