"""Scientific invariants and failure controls; run with unittest."""
import unittest
from dataclasses import replace
import numpy as np
from model import (Config, simulate, behavior, inside, zone, firing_rate,
                   estimate_maps, decode, posterior)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(duration_s=120, n_cells=32)
        cls.data = simulate(cls.cfg)

    def test_maze_constraints_and_continuity(self):
        xy = self.data['xy']
        self.assertTrue(inside(xy).all())
        self.assertLess(np.linalg.norm(np.diff(xy, axis=0), axis=1).max(), .5)
        self.assertGreater(np.count_nonzero(zone(xy) == 1), 0)

    def test_spike_times_exactly_reconstruct_counts(self):
        d = self.data
        counts = np.zeros_like(d['counts'])
        samples = np.floor(d['spike_times'] / self.cfg.dt_s).astype(int)
        np.add.at(counts, (samples, d['spike_units']), 1)
        np.testing.assert_array_equal(counts, d['counts'])
        self.assertTrue(np.all(np.diff(d['spike_times']) >= 0))

    def test_occupancy_and_spike_conservation(self):
        d = self.data
        maps = estimate_maps(d['bins'], d['counts'], d['grid'], self.cfg)
        self.assertAlmostEqual(maps['occupancy'].sum(), self.cfg.duration_s)
        np.testing.assert_array_equal(maps['totals'].sum(axis=0), d['counts'].sum(axis=0))

    def test_modulation_changes_only_open_arm_rates(self):
        d = self.data
        on = firing_rate(d['xy'], d['cells'])
        off = firing_rate(d['xy'], d['cells'], modulation=False)
        closed = zone(d['xy']) != 1
        np.testing.assert_array_equal(on[closed], off[closed])
        np.testing.assert_allclose(on[~closed] / off[~closed],
            np.broadcast_to(d['cells']['gain'], on[~closed].shape), rtol=2e-7)

    def test_train_fit_cannot_see_test_spikes(self):
        original = decode(self.data, self.cfg)
        changed = dict(self.data, counts=self.data['counts'].copy())
        changed['counts'][original['split']:] = 0
        alternate = decode(changed, self.cfg)
        np.testing.assert_array_equal(original['train']['rates'], alternate['train']['rates'])
        np.testing.assert_array_equal(original['train']['occupancy'], alternate['train']['occupancy'])
        self.assertFalse(np.array_equal(original['posterior'], alternate['posterior']))
        np.testing.assert_allclose(original['posterior'].sum(axis=1), 1, atol=1e-12)

    def test_seed_reproducibility_and_separate_behavior_control(self):
        again = simulate(self.cfg)
        np.testing.assert_array_equal(self.data['spike_times'], again['spike_times'])
        altered = simulate(replace(self.cfg, avoidance=0))
        for key in self.data['cells']:
            np.testing.assert_array_equal(self.data['cells'][key], altered['cells'][key])

    def test_avoidance_reduces_open_time_across_seeds(self):
        fractions = []
        for avoidance in [0., 2.5]:
            values = []
            for seed in range(8):
                xy, _ = behavior(replace(self.cfg, avoidance=avoidance, duration_s=240), np.random.default_rng(seed))
                z = zone(xy)
                values.append((z == 1).sum() / (z != 0).sum())
            fractions.append(np.mean(values))
        self.assertGreater(fractions[0], .35)
        self.assertLess(fractions[1], .2)
        self.assertGreater(fractions[0] - fractions[1], .2)

    def test_poisson_decoder_matches_hand_calculation(self):
        rates = np.array([[1., 4.], [4., 1.]])
        p = posterior(np.array([[0, 2]]), rates, np.array([.5, .5]), 1.)
        np.testing.assert_allclose(p[0], [16/17, 1/17])

    def test_invalid_window_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.cfg, decode_window_s=.23).validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
