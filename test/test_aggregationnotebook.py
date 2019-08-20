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

    def tearDown(self):
        # # Get rid of json file
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_single(self):
        # TODO - currently don't cater for a single run (may not need to)
        pass
        # # Original run
        # nb_pre = AggregationJSONNotebook(self.filename, create=True)
        # lab = epyc.Lab(nb_pre)
        # lab[Model.PARAM_X1] = 2
        # lab[Model.PARAM_X2] = [2, 3, 4]
        # lab[Model.PARAM_X3] = 5
        # lab[Model.PARAM_X4] = [6, 7, 8]
        # lab.runExperiment(Model())
        #
        # self.assertItemsEqual(nb_pre.dataframe_aggregated().columns, Model.SA_PARAMS + Model.SA_RESULTS)
        # self.assertItemsEqual(nb_pre.dataframe_aggregated_parameters(), Model.SA_PARAMS)
        # self.assertItemsEqual(nb_pre.dataframe_aggregated_results(), Model.SA_RESULTS)
        #
        # self.assertItemsEqual(nb_pre.uncertain_parameters(), [Model.PARAM_X2, Model.PARAM_X4])
        # self.assertItemsEqual(nb_pre.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])
        # self.assertItemsEqual(nb_pre.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])

        # Load from existing file
        # nb_post = AggregationJSONNotebook(self.filename, create=False)
        # self.assertItemsEqual(nb_post.dataframe_aggregated().columns, Model.SA_PARAMS + Model.SA_RESULTS)
        # self.assertItemsEqual(nb_post.dataframe_aggregated_parameters(), Model.SA_PARAMS)
        # self.assertItemsEqual(nb_post.dataframe_aggregated_results(), Model.SA_RESULTS)
        #
        # self.assertItemsEqual(nb_post.uncertain_parameters(), [Model.PARAM_X2, Model.PARAM_X4])
        # self.assertItemsEqual(nb_post.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])
        # self.assertItemsEqual(nb_post.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])
        #
        # for j in nb_pre.results_aggregated():
        #     params = j[Experiment.PARAMETERS]
        #     res = j[Experiment.RESULTS]
        #     res2 = [k[Experiment.RESULTS] for k in nb_post.results_aggregated() if k[Experiment.PARAMETERS] == params][
        #         0]
        #     for k, v in res.iteritems():
        #         self.assertAlmostEqual(v, res2[k])

    def test_repetition(self):
        # Original run
        nb_pre = AggregationJSONNotebook(self.filename, create=True)
        lab = epyc.Lab(nb_pre)
        lab[Model.PARAM_X1] = 2
        lab[Model.PARAM_X2] = [2,3,4]
        lab[Model.PARAM_X3] = 5
        lab[Model.PARAM_X4] = [6,7,8]
        lab.runExperiment(RepeatedExperiment(Model(), 10))

        self.assertItemsEqual(nb_pre.dataframe_aggregated().columns, Model.SA_PARAMS + Model.SA_RESULTS)

        print nb_pre.certain_parameters()

        self.assertItemsEqual(nb_pre.uncertain_parameters(), [Model.PARAM_X2, Model.PARAM_X4])
        self.assertItemsEqual(nb_pre.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])

        # Load from existing file
        nb_post = AggregationJSONNotebook(self.filename, create=False)
        self.assertItemsEqual(nb_post.dataframe_aggregated().columns, Model.SA_PARAMS + Model.SA_RESULTS)

        self.assertItemsEqual(nb_post.uncertain_parameters(), [Model.PARAM_X2, Model.PARAM_X4])
        self.assertItemsEqual(nb_post.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])
        self.assertItemsEqual(nb_post.certain_parameters(), [Model.PARAM_X1, Model.PARAM_X3])

        for j in nb_pre.results_aggregated():
            params = j[Experiment.PARAMETERS]
            res = j[Experiment.RESULTS]
            res2 = [k[Experiment.RESULTS] for k in nb_post.results_aggregated() if k[Experiment.PARAMETERS] == params][0]
            for k,v in res.iteritems():
                self.assertAlmostEqual(v,res2[k])


if __name__ == '__main__':
    unittest.main()
