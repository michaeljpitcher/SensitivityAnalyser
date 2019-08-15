import unittest
import numpy
from epycsense import *


class Model(epyc.Experiment):
    PARAM_X1 = 'x1'
    PARAM_X2 = 'x2'
    PARAM_X3 = 'x3'
    PARAM_X4 = 'x4'
    SA_PARAMS = [PARAM_X1, PARAM_X2, PARAM_X3, PARAM_X4]

    RESULT_1 = 'res1'
    RESULT_2 = 'res2'
    RESULT_3 = 'res3'
    RESULT_4 = 'res4'
    SA_RESULTS = [RESULT_1,
                  RESULT_2,
                  RESULT_3,
                  RESULT_4]

    def do(self, params):
        results = {Model.RESULT_1: params[self.PARAM_X1],
                   Model.RESULT_2: numpy.random.random() * params[self.PARAM_X2],
                   Model.RESULT_3: numpy.random.random() * params[self.PARAM_X3],
                   Model.RESULT_4: params[self.PARAM_X4]}
        return results


class AggregationJSONNotebookTestCase(unittest.TestCase):

    def setUp(self):
        self.filename = 'aggtest.json'

    def test_add_result(self):
        self.notebook = AggregationJSONNotebook(self.filename, create=True, description=None)
        lab = epyc.Lab(self.notebook)
        lab[Model.PARAM_X1] = 2
        lab[Model.PARAM_X2] = [2,3,4]
        lab[Model.PARAM_X3] = 5
        lab[Model.PARAM_X4] = [6,7,8]
        lab.runExperiment(RepeatedExperiment(Model(), 10))

        self.assertItemsEqual(self.notebook.dataframe_aggregated().columns, Model.SA_PARAMS + Model.SA_RESULTS)
        self.assertItemsEqual(self.notebook.dataframe_aggregated_parameters(), Model.SA_PARAMS)
        self.assertItemsEqual(self.notebook.dataframe_aggregated_results(), Model.SA_RESULTS)

        self.assertItemsEqual(self.notebook.uncertain_parameters(), [Model.PARAM_X2, Model.PARAM_X4])
        self.assertItemsEqual(self.notebook.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])
        self.assertItemsEqual(self.notebook.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])

    def test_load_from_file(self):
        nb = AggregationJSONNotebook(self.filename, False)

        self.assertItemsEqual(nb.dataframe_aggregated().columns, Model.SA_PARAMS + Model.SA_RESULTS)
        self.assertItemsEqual(nb.dataframe_aggregated_parameters(), Model.SA_PARAMS)
        self.assertItemsEqual(nb.dataframe_aggregated_results(), Model.SA_RESULTS)

        self.assertItemsEqual(nb.uncertain_parameters(), [Model.PARAM_X2, Model.PARAM_X4])
        self.assertItemsEqual(nb.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])
        self.assertItemsEqual(nb.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])


if __name__ == '__main__':
    unittest.main()
