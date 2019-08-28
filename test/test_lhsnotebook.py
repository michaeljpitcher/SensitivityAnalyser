import unittest
from epycsense import *
import numpy
import os
import json
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.integrate import odeint


class LotkaVolterraModel(epyc.Experiment):
    PARAM_ALPHA = 'alpha'
    PARAM_BETA = 'beta'
    PARAM_SIGMA = 'sigma'
    PARAM_DELTA = 'delta'

    Q = 'prey'
    P = 'predators'

    INIT_Q = 'initial_q'
    INIT_P = 'initial_p'
    INIT_CONDITS = [INIT_Q, INIT_P]

    def __init__(self):
        epyc.Experiment.__init__(self)
        self._max_time = 0
        self._time_points = 0

    def set_time_params(self, max_time, time_points):
        self._max_time = max_time
        self._time_points = time_points

    def do(self, params):

        t_max = self._max_time
        t_span = list(numpy.linspace(0, t_max, self._time_points))

        initial_conditions = [params[ic] for ic in LotkaVolterraModel.INIT_CONDITS]

        alpha = params[LotkaVolterraModel.PARAM_ALPHA]
        beta = params[LotkaVolterraModel.PARAM_BETA]
        sigma = params[LotkaVolterraModel.PARAM_SIGMA]
        delta = params[LotkaVolterraModel.PARAM_DELTA]

        def model_eqns(vals, t):
            Q, P = vals
            dQdt = alpha*Q - beta*Q*P
            dPdt = -sigma*P + delta*Q*P
            return [dQdt, dPdt]

        pde_results = odeint(model_eqns, initial_conditions, t_span)  # integrate

        observed_timesteps = [9]
        t_indexes = {x: t_span.index(x) for x in observed_timesteps}

        t = 9
        output = {LotkaVolterraModel.Q: pde_results[t_indexes[t]][0], LotkaVolterraModel.P: pde_results[t_indexes[t]][1]}
        return output


class LatinHypercubeJSONNotebookTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = 'prcc.json'

    # def tearDown(self):
    #     # # Get rid of json file
    #     if os.path.exists(self.filename):
    #         os.remove(self.filename)

    # def test_get_pearson_correlation_coefficient(self):
    #     model = Model()
    #     params = {Model.PARAM_X1: (0, 10),
    #               Model.PARAM_X2: (10, 20),
    #               Model.PARAM_X3: (20, 30)}
    #     stratifications = 100
    #     lab = LatinHypercubeLab(self.rep_nb)
    #     for k, v in params.iteritems():
    #         lab.set_parameter_stratifications(k, v, stratifications)
    #     lab[Model.PARAM_FIX] = 4
    #     lab.runExperiment(epyc.RepeatedExperiment(model, 20))
    #
    #     # X1 is not well correlated
    #     for r in Model.SA_RESULTS:
    #         self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X1, r)[0] < 0.5)
    #
    #     # X2 high correlation with random_x2
    #     self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X2, Model.RESULT_RANDOM)[0] < 0.5)
    #     self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X2, Model.RESULT_RANDOM_X2)[0] > 0.5)
    #     self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X2, Model.RESULT_FIX_X3)[0] < 0.5)
    #
    #     # X3 perfect correlation with fix x3
    #     self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X3, Model.RESULT_RANDOM)[0] < 0.5)
    #     self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X3, Model.RESULT_RANDOM_X2)[0] < 0.5)
    #     self.assertTrue(self.rep_nb.get_pearson_correlation_coefficient(Model.PARAM_X3, Model.RESULT_FIX_X3)[0] > 0.5)
    #
    # def test_get_all_pearson_correlation_coefficients(self):
    #     model = Model()
    #     params = {Model.PARAM_X1: (0, 10),
    #               Model.PARAM_X2: (10, 20),
    #               Model.PARAM_X3: (20, 30)}
    #     stratifications = 101
    #     lab = LatinHypercubeLab(self.rep_nb)
    #     for k, v in params.iteritems():
    #         lab.set_parameter_stratifications(k, v, stratifications)
    #     lab[Model.PARAM_FIX] = 4
    #     lab.runExperiment(epyc.RepeatedExperiment(model, 100))
    #
    #     pccs = self.rep_nb.get_all_pearson_correlation_coefficients()
    #     expected = []
    #     for p in params.keys():
    #         for r in Model.SA_RESULTS:
    #             expected.append((p,r))
    #     self.assertItemsEqual(expected, pccs.keys())

    def test_calculate_prcc(self):
        # nb_pre = LatinHypercubeJSONNotebook(self.filename, True)
        # self.lab = LatinHypercubeLab(nb_pre)
        #
        # model = LotkaVolterraModel()
        # self.lab[LotkaVolterraModel.PARAM_BETA] = [1, 0.2, NORMAL_DISTRIBUTION]
        # self.lab[LotkaVolterraModel.PARAM_ALPHA] = [1.5, 0.01, NORMAL_DISTRIBUTION]
        # self.lab[LotkaVolterraModel.PARAM_SIGMA] = [3, 0.2, NORMAL_DISTRIBUTION]
        # self.lab[LotkaVolterraModel.PARAM_DELTA] = [1, 0.01, NORMAL_DISTRIBUTION]
        #
        # self.lab[LotkaVolterraModel.INIT_Q] = 10
        # self.lab[LotkaVolterraModel.INIT_P] = 5
        #
        # model.set_time_params(10, 101)
        # self.lab.set_stratifications(1000)
        #
        # self.lab.runExperiment(RepeatedExperiment(model, 2))

        nb_post = LatinHypercubeJSONNotebook(self.filename, False)

        prccs = {}

        params = [LotkaVolterraModel.PARAM_ALPHA, LotkaVolterraModel.PARAM_BETA, LotkaVolterraModel.PARAM_SIGMA,
                  LotkaVolterraModel.PARAM_DELTA, 'dummy']

        for p in params:
            prccs[p] = nb_post.calculate_prcc(p, LotkaVolterraModel.Q, False)
            print p, prccs[p]

        matlab_vals = [-0.1425, 0.4931, -0.7219, 0.0145, 0.0144]

        plt.figure()
        plt.subplot(121)

        plt.text(-1.9, 1.1, 'A', fontsize=14)
        ind = np.arange(len(matlab_vals))
        p1 = plt.bar(ind, matlab_vals, 0.8, color='red')

        plt.axhline(color='k', linewidth=1)
        plt.ylabel('PRCC')
        plt.xticks(ind, (r'$\alpha$', r'$\beta$', r'$\sigma$', r'$\delta$', r'$dummy$'))
        plt.yticks(np.arange(-1, 1.1, 0.2))

        plt.subplot(122)
        plt.text(-1.3, 1.1, 'B', fontsize=14)
        p1 = plt.bar(range(len(prccs)), [prccs[p][0][0] for p in params], 0.8)
        plt.axhline(color='k', linewidth=1)
        plt.xticks(range(len(prccs)), (r'$\alpha$', r'$\beta$', r'$\sigma$', r'$\delta$', r'$dummy$'))
        plt.yticks(np.arange(-1, 1.1, 0.2), visible=False)
        plt.savefig("PRCC_LV.png")

        for k,v in nb_post.get_all_prcc().iteritems():
            print k, v

if __name__ == '__main__':
    unittest.main()
