import unittest
from epycsense import *
import numpy
import os
import json


class Model(epyc.Experiment):
    PARAM_X1 = 'x1'
    PARAM_X2 = 'x2'
    PARAM_X3 = 'x3'
    SA_PARAMS = [PARAM_X1, PARAM_X2, PARAM_X3]

    RESULT_2_X1 = '2_x1'
    RESULT_RANDOM_X2 = 'random_x2'
    RESULT_RANDOM_X3 = 'random_x3'
    SA_RESULTS = [RESULT_2_X1,
                  RESULT_RANDOM_X2,
                  RESULT_RANDOM_X3]

    def do(self, params):
        x1 = params[self.PARAM_X1]
        x2 = params[self.PARAM_X2]
        x3 = params[self.PARAM_X3]

        results = {Model.RESULT_2_X1: 2.0 * x1,
                   Model.RESULT_RANDOM_X2: numpy.random.random() * x2,
                   Model.RESULT_RANDOM_X3: numpy.random.random() * x3}

        return results


class ScatterLabTestCase(unittest.TestCase):

    def setUp(self):
        self.filename = 'scatterlabtest.json'
        self.nb = epyc.JSONLabNotebook(self.filename, True)
        self.lab = ScatterLab(self.nb)

    def tearDown(self):
        # # Get rid of json file
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_initialise(self):
        pass

    def test_parameter_space(self):
        # Set params
        params = {'a': [1,2,3], 'b': [4,5,6], 'c':[7,8,9]}
        for k, v in params.iteritems():
            self.lab[k] = v

        ps = self.lab.parameterSpace()

        for i in range(len(params)):
            param = params.keys()[i]
            other_params = params.keys()[:i] + params.keys()[i+1:]
            for v in params[param]:
                vals = {param: v}
                for q in other_params:
                    vals[q] = params[q][len(params[q])/2]
                self.assertTrue(vals in ps)

        # Check we have the right number of samples (i.e. only 1 instance where all params are baseline)
        self.assertEqual(len(ps), sum([len(v)-1 for v in params.values()])+1)

    def test_run(self):
        model = Model()
        params = {Model.PARAM_X1: range(0,10),
                  Model.PARAM_X2: range(0,10),
                  Model.PARAM_X3: range(0,10)}
        for k,v in params.iteritems():
            self.lab[k] = v
        self.lab.runExperiment(model)

        # Load data
        with open(self.filename, 'r') as file:
            data = json.load(file)['results'].values()

        self.assertEqual(len(data), sum([len(k)-1 for k in params.values()])+1)

        param_samples = [d[0]['parameters'] for d in data]

        for i in range(len(params)):
            param = params.keys()[i]
            other_params = params.keys()[:i] + params.keys()[i+1:]
            for v in params[param]:
                vals = {param: v}
                for q in other_params:
                    vals[q] = params[q][len(params[q])/2]
                self.assertTrue(vals in param_samples)

# TODO - testing cluster would require an ipcluster to be running
# class ScatterClusterLabTestCase(unittest.TestCase):
#
#     def setUp(self):
#         self.nb = epyc.JSONLabNotebook('scatterlabtest.json', True)
#         self.lab = ScatterClusterLab(self.nb)
#
#     def test_initialise(self):
#         pass
#
#     def test_parameter_space(self):
#         # Set params
#         params = {'a': [1,2,3], 'b': [4,5,6], 'c':[7,8,9]}
#         for k, v in params.iteritems():
#             self.lab[k] = v
#
#         ps = self.lab.parameterSpace()
#
#         for i in range(len(params)):
#             param = params.keys()[i]
#             other_params = params.keys()[:i] + params.keys()[i+1:]
#             print param, params[param]
#             for v in params[param]:
#                 vals = {param: v}
#                 for q in other_params:
#                     vals[q] = params[q][len(params[q])/2]
#                 self.assertTrue(vals in ps)
#
#         # Check we have the right number of samples (i.e. only 1 instance where all params are baseline)
#         self.assertEqual(len(ps), sum([len(v)-1 for v in params.values()])+1)

if __name__ == '__main__':
    unittest.main()
