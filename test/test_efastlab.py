import unittest
from epycsense import *
import numpy
import os


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
        self.nb = EFASTJSONNotebook(self.filename, True)
        self.lab = EFASTLab(self.nb)

    def tearDown(self):
        # # Get rid of json file
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_initialise(self):
        pass

    def test_parameter_space(self):
        # Set params
        params = {'a': (0, 10, UNIFORM_DISTRIBUTION),
                  'b': (20, 0.5, NORMAL_DISTRIBUTION), 'c': (30, 0.6, LOGNORMAL_DISTRIBUTION),
                  'd': (40, 0.7, LOGNORMAL_DISTRIBUTION), 'e': (50, 0.8, NORMAL_DISTRIBUTION),
                  'f': (70, 80, UNIFORM_DISTRIBUTION)}

        for k,v in params.iteritems():
            self.lab[k] = v

        # Fixed params
        fixed_params = {'i': 99, 'j': 98}
        for k, v in fixed_params.iteritems():
            self.lab[k] = v

        self.lab.set_sample_number(257)
        self.lab.set_interference_factor(4)
        self.lab.set_resample_number(5)

        ps = self.lab.parameterSpace()

        # Has dummy parameter in so == len(params) + 1
        self.assertEqual(len(ps), (len(params) + 1) * 5 * 257)

        # All we can check is value falls within expected range since random numbers are involved
        for row in ps:
            for p, (a, b, dist) in params.iteritems():
                if dist == UNIFORM_DISTRIBUTION:
                    self.assertTrue(a <= row[p] <= b)
                if dist == NORMAL_DISTRIBUTION:
                    # TODO - testing a normal distribution
                    pass

    def test_parameter_space_filtered(self):
        # Test parameter space when we only want a few parameters
        # Set params
        params = {'a': (0, 10, UNIFORM_DISTRIBUTION),
                  'b': (20, 0.5, NORMAL_DISTRIBUTION), 'c': (30, 0.6, LOGNORMAL_DISTRIBUTION),
                  'd': (40, 0.7, LOGNORMAL_DISTRIBUTION), 'e': (50, 0.8, NORMAL_DISTRIBUTION),
                  'f': (70, 80, UNIFORM_DISTRIBUTION)}

        for k, v in params.iteritems():
            self.lab[k] = v

        # Fixed params
        fixed_params = {'i': 99, 'j': 98}
        for k, v in fixed_params.iteritems():
            self.lab[k] = v

        self.lab.set_sample_number(257)
        self.lab.set_interference_factor(4)
        self.lab.set_resample_number(5)

        req_params = ['a','b','c']
        self.lab.set_required_parameters(req_params)

        ps = self.lab.parameterSpace()
        self.assertEqual(len(ps), 257*5*3)
        for sample in ps:
            self.assertTrue(sample['parameter_of_interest'] in req_params)

# TODO - testing cluster would require an ipcluster to be running

if __name__ == '__main__':
    unittest.main()
