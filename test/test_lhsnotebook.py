import unittest
from epycsense import *
import numpy
import os
import json


class Model(epyc.Experiment):
    PARAM_X1 = 'x1'
    PARAM_X2 = 'x2'
    PARAM_X3 = 'x3'
    PARAM_FIX = 'fix'
    SA_PARAMS = [PARAM_X1, PARAM_X2, PARAM_X3, PARAM_FIX]

    RESULT_RANDOM = 'random'
    RESULT_RANDOM_X2 = 'random_x2'
    RESULT_FIX_X3 = 'fix_x3'
    SA_RESULTS = [RESULT_RANDOM,
                  RESULT_RANDOM_X2,
                  RESULT_FIX_X3]

    def do(self, params):
        x1 = params[self.PARAM_X1]
        x2 = params[self.PARAM_X2]
        x3 = params[self.PARAM_X3]
        fix = params[self.PARAM_FIX]

        results = {Model.RESULT_RANDOM: numpy.random.random(),
                   Model.RESULT_RANDOM_X2: numpy.random.random() * x2,
                   Model.RESULT_FIX_X3: fix * x3}
        return results


class Model2(epyc.Experiment):
    PARAM_X1 = 'x1'
    PARAM_X2 = 'x2'
    PARAM_X3 = 'x3'
    PARAM_FIX = 'fix'
    SA_PARAMS = [PARAM_X1, PARAM_X2, PARAM_X3, PARAM_FIX]

    RESULT_RANDOM = 'random'
    RESULT_RANDOM_X2 = 'random_x2'
    RESULT_RANDOM_X3_SQUARED = 'random_x3_squared'
    SA_RESULTS = [RESULT_RANDOM,
                  RESULT_RANDOM_X2,
                  RESULT_RANDOM_X3_SQUARED]

    def do(self, params):
        x1 = params[Model2.PARAM_X1]
        x2 = params[Model2.PARAM_X2]
        x3 = params[Model2.PARAM_X3]
        fix = params[Model2.PARAM_FIX]

        results = {Model2.RESULT_RANDOM: numpy.random.random(),
                   Model2.RESULT_RANDOM_X2: numpy.random.random() * x2,
                   Model2.RESULT_RANDOM_X3_SQUARED: numpy.random.random() * (x3**2)} # Non-linear for PRCC
        return results


class LatinHypercubeJSONNotebookTestCase(unittest.TestCase):
    def setUp(self):
        self.rep_filename = 'LHSlabtest_repetitions.json'
        self.rep_nb = LatinHypercubeJSONNotebook(self.rep_filename, True)

    def test_get_pearson_correlation_coefficient(self):
        model = Model()
        params = {Model.PARAM_X1: (0, 10),
                  Model.PARAM_X2: (10, 20),
                  Model.PARAM_X3: (20, 30)}
        stratifications = 100
        lab = LatinHypercubeLab(self.rep_nb)
        for k, v in params.iteritems():
            lab.set_parameter_stratifications(k, v, stratifications)
        lab[Model.PARAM_FIX] = 4
        lab.runExperiment(epyc.RepeatedExperiment(model, 20))

        # X1 is not well correlated
        for r in Model.SA_RESULTS:
            self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X1, r)[0] < 0.5)

        # X2 high correlation with random_x2
        self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X2, Model.RESULT_RANDOM)[0] < 0.5)
        self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X2, Model.RESULT_RANDOM_X2)[0] > 0.5)
        self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X2, Model.RESULT_FIX_X3)[0] < 0.5)

        # X3 perfect correlation with fix x3
        self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X3, Model.RESULT_RANDOM)[0] < 0.5)
        self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X3, Model.RESULT_RANDOM_X2)[0] < 0.5)
        self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X3, Model.RESULT_FIX_X3)[0] > 0.5)

    def test_get_all_pearson_correlation_coefficients(self):
        model = Model()
        params = {Model.PARAM_X1: (0, 10),
                  Model.PARAM_X2: (10, 20),
                  Model.PARAM_X3: (20, 30)}
        stratifications = 101
        lab = LatinHypercubeLab(self.rep_nb)
        for k, v in params.iteritems():
            lab.set_parameter_stratifications(k, v, stratifications)
        lab[Model.PARAM_FIX] = 4
        lab.runExperiment(epyc.RepeatedExperiment(model, 100))

        pccs = self.rep_nb.get_all_pearson_correlation_coefficients()
        expected = []
        for p in params.keys():
            for r in Model.SA_RESULTS:
                expected.append((p,r))
        self.assertItemsEqual(expected, pccs.keys())

    def test_calculate_prcc(self):
        model = Model2()
        params = {Model2.PARAM_X1: (0, 10),
                  Model2.PARAM_X2: (10, 20),
                  Model2.PARAM_X3: (20, 30)}
        stratifications = 101
        lab = LatinHypercubeLab(self.rep_nb)
        for k, v in params.iteritems():
            lab.set_parameter_stratifications(k, v, stratifications)
        lab[Model2.PARAM_FIX] = 4
        lab.runExperiment(epyc.RepeatedExperiment(model, 100))

        self.assertTrue(self.rep_nb.calculate_prcc(Model2.PARAM_X1, Model2.RESULT_RANDOM)[0] < 0.5)
        self.assertTrue(self.rep_nb.calculate_prcc(Model2.PARAM_X1, Model2.RESULT_RANDOM_X2)[0] < 0.5)
        self.assertTrue(self.rep_nb.calculate_prcc(Model2.PARAM_X1, Model2.RESULT_RANDOM_X3_SQUARED)[0] < 0.5)

        self.assertTrue(self.rep_nb.calculate_prcc(Model2.PARAM_X2, Model2.RESULT_RANDOM)[0] < 0.5)
        self.assertTrue(self.rep_nb.calculate_prcc(Model2.PARAM_X2, Model2.RESULT_RANDOM_X2)[0] > 0.5)
        self.assertTrue(self.rep_nb.calculate_prcc(Model2.PARAM_X2, Model2.RESULT_RANDOM_X3_SQUARED)[0] < 0.5)

        self.assertTrue(self.rep_nb.calculate_prcc(Model2.PARAM_X3, Model2.RESULT_RANDOM)[0] < 0.5)
        self.assertTrue(self.rep_nb.calculate_prcc(Model2.PARAM_X3, Model2.RESULT_RANDOM_X2)[0] < 0.5)
        self.assertTrue(self.rep_nb.calculate_prcc(Model2.PARAM_X3, Model2.RESULT_RANDOM_X3_SQUARED)[0] > 0.5)

    def test_get_all_prcc(self):
        model = Model2()
        params = {Model2.PARAM_X1: (0, 10),
                  Model2.PARAM_X2: (10, 20),
                  Model2.PARAM_X3: (20, 30)}
        stratifications = 101
        lab = LatinHypercubeLab(self.rep_nb)
        for k, v in params.iteritems():
            lab.set_parameter_stratifications(k, v, stratifications)
        lab[Model2.PARAM_FIX] = 4
        lab.runExperiment(epyc.RepeatedExperiment(model, 100))

        prccs = self.rep_nb.get_all_prcc()

if __name__ == '__main__':
    unittest.main()
