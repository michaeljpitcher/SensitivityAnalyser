import unittest
from epycsense import *
import numpy
import os
import json

import pandas


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


class EFASTLabTestCase(unittest.TestCase):

    def setUp(self):
        self.filename = 'efastlabtest.json'
        self.nb = epyc.JSONLabNotebook(self.filename, True)
        self.lab = EFASTLab(self.nb)

    def tearDown(self):
        # # Get rid of json file
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_initialise(self):
        pass

    def test_parameter_space(self):
        # Set params
        params = {'a': (0, 10, 'unif'), 'b': (10, 20, 'unif'), 'c': (20, 30, 'unif'), 'd': (30, 40, 'unif'),
                  'e': (40, 50, 'unif'), 'f':  (50, 60, 'unif'), 'g':(60, 70, 'unif'), 'h': (70, 80, 'unif')}
        for k,v in params.iteritems():
            self.lab[k] = v

        # Fixed params
        fixed_params = {'i': 99, 'j': 98}
        for k, v in fixed_params.iteritems():
            self.lab[k] = v

        self.lab.set_sample_number(257)
        self.lab.set_interference_factor(4)

        ps = self.lab.parameterSpace()

        # All we can check is value falls between min and max since random numbers are involved
        for row in ps:
            for p, (min,max,_) in params.iteritems():
                self.assertTrue(min <= row[p] <= max)



# TODO - testing cluster would require an ipcluster to be running

if __name__ == '__main__':
    unittest.main()
