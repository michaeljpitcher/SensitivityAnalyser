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
        fix = params[self.PARAM_FIX]

        results = {Model.RESULT_2_X1: fix * x1,
                   Model.RESULT_RANDOM_X2: numpy.random.random() * x2,
                   Model.RESULT_RANDOM_X3: numpy.random.random() * x3}

        return results


class ScatterJSONNotebookTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = 'scatterlabtest.json'
        self.nb = ScatterJSONNotebook(self.filename, True)

        self.rep_filename = 'scatterlabtest_repetitions.json'
        self.rep_nb = ScatterJSONNotebook(self.filename, True)

    def tearDown(self):
        # # Get rid of json file
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.rep_filename):
            os.remove(self.rep_filename)

    def test_initialise(self):
        pass

    def test_plot_scatter_graphs(self):
        model = Model()
        params = {Model.PARAM_X1: range(0,10),
                  Model.PARAM_X2: range(0,10),
                  Model.PARAM_X3: range(0,10),
                  Model.PARAM_FIX: 4}
        lab = ScatterLab(self.nb)
        for k,v in params.iteritems():
            lab[k] = v
        lab.runExperiment(model)

        self.nb.plot_scatter_graphs(folder='scatter_no_reps')

    def test_plot_scatter_graphs_repetitions(self):
        model = Model()
        params = {Model.PARAM_X1: range(0, 10),
                  Model.PARAM_X2: range(0, 10),
                  Model.PARAM_X3: range(0, 10),
                  Model.PARAM_FIX: 4}
        lab = ScatterLab(self.rep_nb)
        for k, v in params.iteritems():
            lab[k] = v
        lab.runExperiment(epyc.RepeatedExperiment(model, 100))

        self.rep_nb.plot_scatter_graphs(folder='scatter_reps')


if __name__ == '__main__':
    unittest.main()
