import unittest
from sensitivity import *


class ParameterSetTestCase(unittest.TestCase):

    def setUp(self):
        self.parameter_set = ParameterSet()

    def test_initialise(self):
        self.assertFalse(self.parameter_set.certain_parameters())
        self.assertFalse(self.parameter_set.uncertain_parameters())

    def test_add_certain_parameter(self):
        param = 'test'
        val = 1.1
        self.parameter_set.add_certain_parameter(param, val)
        self.assertItemsEqual(self.parameter_set.certain_parameters().keys(), [param])
        self.assertEqual(self.parameter_set.certain_parameters()[param], val)

    def test_add_uncertain_parameter_uniform_dist(self):
        param = 'test'
        min_val = 1.1
        max_val = 9.9
        self.parameter_set.add_uncertain_parameter_uniform_dist(param, min_val, max_val)
        self.assertItemsEqual(self.parameter_set.uncertain_parameters().keys(), [param])
        self.assertEqual(self.parameter_set.uncertain_parameters()[param],
                         (min_val, max_val, ParameterSet.DISTRIBUTION_UNIFORM))

    def test_create_latin_hypercube_stratifications(self):
        param_unc1 = 'unc1'
        param_unc2 = 'unc2'
        param_c1 = 'c1'
        param_c2 = 'c2'
        self.parameter_set.add_certain_parameter(param_c1, 1.1)
        self.parameter_set.add_certain_parameter(param_c2, 2.2)
        unc1_min = 1.1
        unc1_max = 11.2
        self.parameter_set.add_uncertain_parameter_uniform_dist(param_unc1, unc1_min, unc1_max)
        unc2_min = 5.5
        unc2_max = 5.7
        self.parameter_set.add_uncertain_parameter_uniform_dist(param_unc2, unc2_min, unc2_max)

        strat = 10
        lhs = self.parameter_set.create_latin_hypercube_stratifications(strat)
        for p, v in lhs.iteritems():
            self.assertEqual(len(v), strat)
        for k in range(strat):
            self.assertEqual(lhs[param_unc1][k], unc1_min + (k * ((unc1_max - unc1_min) / (strat - 1))))
            self.assertEqual(lhs[param_unc2][k], unc2_min + (k * ((unc2_max - unc2_min) / (strat - 1))))


if __name__ == '__main__':
    unittest.main()
