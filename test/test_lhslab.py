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


class LHSLabTestCase(unittest.TestCase):

    def setUp(self):
        self.filename = 'lhslabtest.json'
        self.nb = epyc.JSONLabNotebook(self.filename, True)
        self.lab = LatinHypercubeLab(self.nb)

    def tearDown(self):
        # # Get rid of json file
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_initialise(self):
        pass

    def test_set_parameter_stratifications(self):
        params = {'a': (0, 10), 'b': (10, 20), 'c': (20, 30)}
        stratifications = 11
        for k, v in params.iteritems():
            self.lab.set_parameter_stratifications(k, v, stratifications)
            self.assertItemsEqual(self.lab[k], numpy.linspace(v[0],v[1],stratifications))

    def test_parameter_space(self):
        # Set params
        params = {'a': (0,10), 'b': (10,20), 'c': (20,30)}
        stratifications = 11
        for k,v in params.iteritems():
            self.lab.set_parameter_stratifications(k,v,stratifications)

        # Fixed params
        fixed_params = {'d': 99, 'e': 98}
        for k, v in fixed_params.iteritems():
            self.lab[k] = v

        ps = self.lab.parameterSpace()

        self.assertEqual(len(ps), stratifications)
        values = {p: [q[p] for q in ps] for p in params.keys() + fixed_params.keys()}

        expected_params = {k: numpy.linspace(v[0],v[1], stratifications) for (k,v) in params.iteritems()}
        for p, v in expected_params.iteritems():
            self.assertItemsEqual(expected_params[p], values[p])

        for row in ps:
            for p in fixed_params:
                self.assertEqual(row[p], fixed_params[p])

    def test_run(self):
        model = Model()
        params = {Model.PARAM_X1: (0, 10),
                  Model.PARAM_X2: (10, 20),
                  Model.PARAM_X3: (20, 30)}
        stratifications = 21
        for k, v in params.iteritems():
            self.lab.set_parameter_stratifications(k, v, stratifications)
        self.lab[Model.PARAM_FIX] = 4
        self.lab.runExperiment(model)

        # Load data
        with open(self.filename, 'r') as file:
            data = json.load(file)['results'].values()

        self.assertEqual(len(data), stratifications)

        param_samples = [d[0]['parameters'] for d in data]

        values = {p: [q[p] for q in param_samples] for p in params.keys() + [Model.PARAM_FIX]}

        expected_params = {k: numpy.linspace(v[0], v[1], stratifications) for (k, v) in params.iteritems()}
        for k,v in expected_params.iteritems():
            self.assertItemsEqual(v, values[k])
        self.assertItemsEqual(values[Model.PARAM_FIX], [4,]*stratifications)


# TODO - testing cluster would require an ipcluster to be running

if __name__ == '__main__':
    unittest.main()
