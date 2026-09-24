import unittest
import numpy as np
from pca_reference import bin_session, fit_pca, arm_labels


class PCATests(unittest.TestCase):
    def test_binning_conserves_counts_and_flags_transitions(self):
        t = np.arange(11) * .1
        xy = np.zeros((11, 2))
        xy[1:4, 0] = 8
        counts = np.arange(22).reshape(11, 2)
        rates, meta, info = bin_session(t, xy, counts, .2)
        np.testing.assert_allclose((rates*.2).sum(axis=0), counts[:10].sum(axis=0))
        self.assertEqual(info['discarded_samples'], 1)
        self.assertEqual(meta.loc[0, 'arm'], 'east_open')
        self.assertEqual(meta.loc[0, 'arm_fraction'], .5)
        self.assertTrue(meta.loc[0, 'is_transition'])
        self.assertFalse(meta.loc[1, 'is_transition'])

    def test_arm_categories_are_not_quadrants(self):
        xy = [[0, 0], [8, -1], [-8, 1], [-1, 8], [1, -8]]
        self.assertEqual(arm_labels(xy).tolist(),
                         ['center', 'east_open', 'west_open', 'north_closed', 'south_closed'])

    def test_pca_agrees_with_covariance_eigendecomposition(self):
        x = np.random.default_rng(42).normal(size=(80, 7)) * np.arange(1, 8)
        fit = fit_pca(x, standardize=False)
        expected = np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1]
        np.testing.assert_allclose(fit['variance'], expected, rtol=1e-12)
        np.testing.assert_allclose(fit['scores'] @ fit['components'] + fit['mean'], x, atol=1e-12)
        self.assertAlmostEqual(fit['ratio'].sum(), 1.)

    def test_standardization_is_invariant_to_positive_feature_rescaling(self):
        x = np.random.default_rng(3).normal(size=(100, 5))
        a, b = fit_pca(x), fit_pca(x*np.arange(1, 6) + 10)
        np.testing.assert_allclose(a['ratio'], b['ratio'], atol=1e-12)
        np.testing.assert_allclose(a['scores'], b['scores'], atol=1e-12)

    def test_constant_features_and_invalid_inputs(self):
        x = np.column_stack([np.arange(10), np.ones(10), np.arange(10)**2])
        fit = fit_pca(x)
        self.assertEqual(fit['constant_neurons'], [1])
        self.assertTrue(np.isfinite(fit['scores']).all())
        with self.assertRaises(ValueError):
            fit_pca(np.ones((5, 3)))
        with self.assertRaises(ValueError):
            bin_session(np.arange(20)*.1, np.zeros((20, 2)), np.zeros((20, 3)), .25)


if __name__ == '__main__':
    unittest.main(verbosity=2)
