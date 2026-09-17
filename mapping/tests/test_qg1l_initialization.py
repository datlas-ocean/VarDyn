"""Regression coverage for QG initialization from per-variable BC fields."""
import unittest

import numpy as np

from src.mod import Model_qg1l
from src.state import State


class QgInitializationTests(unittest.TestCase):
    def make_model(self, names=('SSH',), init_from_bc=True):
        model = Model_qg1l.__new__(Model_qg1l)
        model.name_var = {name: name.lower() for name in names}
        model.init_from_bc = init_from_bc
        model.anomaly_from_bc = False
        model.bc = {name: {} for name in names}
        state = State.__new__(State)
        state.mask = np.zeros((4, 5), dtype=bool)
        state.mask[0, 0] = True
        state.preserve_device_arrays = False
        state.var = {name: np.full((4, 5), -1.)
                     for name in model.name_var.values()}
        return model, state

    def test_ssh_only_exact_and_nearest_time(self):
        for t0 in (10, 12):
            with self.subTest(t0=t0):
                model, state = self.make_model()
                model.bc['SSH'] = {10: np.full((4, 5), 2.),
                                   20: np.full((4, 5), 3.)}
                model.init(state, t0)
                np.testing.assert_equal(state.var['ssh'][~state.mask], 2.)
                self.assertTrue(np.isnan(state.var['ssh'][0, 0]))
                self.assertEqual(model.bc['SSH'][10][0, 0], 2.)

    def test_tracers_and_velocities_use_their_own_time_at_initialization(self):
        model, state = self.make_model(('SSH', 'U', 'V', 'SST'))
        for i, name in enumerate(model.name_var):
            model.bc[name] = {10 + i: np.full((4, 5), i + 1.),
                              100: np.full((4, 5), 99.)}
        model.init(state, 12)
        for i, name in enumerate(model.name_var.values()):
            np.testing.assert_equal(state.var[name][~state.mask], i + 1.)

    def test_dictionary_selection_uses_nearest_time(self):
        model, state = self.make_model(
            ('SSH', 'SST'), {'SSH': False, 'SST': True})
        model.bc['SSH'] = {10: np.full((4, 5), 2.)}
        model.bc['SST'] = {11: np.full((4, 5), 20.)}
        model.init(state, 12)
        np.testing.assert_equal(state.var['ssh'][~state.mask], -1.)
        np.testing.assert_equal(state.var['sst'][~state.mask], 20.)

    def test_missing_fields_preserve_existing_values(self):
        model, state = self.make_model(('SSH', 'SST'))
        model.bc['SSH'] = {0: np.full((4, 5), 2.)}
        model.init(state)
        np.testing.assert_equal(state.var['ssh'][~state.mask], 2.)
        np.testing.assert_equal(state.var['sst'][~state.mask], -1.)

    def test_disabled_anomaly_and_empty_boundary_conditions(self):
        for disabled, anomaly, empty in [(True, False, False),
                                         (False, True, False),
                                         (False, False, True)]:
            with self.subTest(disabled=disabled, anomaly=anomaly, empty=empty):
                model, state = self.make_model(init_from_bc=not disabled)
                model.anomaly_from_bc = anomaly
                if not empty:
                    model.bc['SSH'] = {0: np.full((4, 5), 2.)}
                model.init(state)
                np.testing.assert_equal(state.var['ssh'][~state.mask], -1.)
                self.assertTrue(np.isnan(state.var['ssh'][0, 0]))


if __name__ == '__main__':
    unittest.main()
