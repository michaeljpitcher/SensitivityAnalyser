import unittest
from epycsense import *
import numpy
import os
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

# class SAExperiment(epyc.Experiment):
#     PARAM_S = 's'
#     PARAM_MUT = 'mu_T'
#     PARAM_R = 'r'
#     PARAM_K1 = 'k_1'
#     PARAM_K2 = 'k_2'
#     PARAM_MUB = 'mu_b'
#     PARAM_N = 'N_V'
#     PARAM_MUV = 'mu_V'
#     PARAM_DUMMY = 'dummy'
#     PARAM_T_MAX = 't_max'
#     SA_PARAMS = [PARAM_S, PARAM_MUT, PARAM_R, PARAM_K1, PARAM_K2, PARAM_MUB, PARAM_N, PARAM_MUV, PARAM_DUMMY,
#                  PARAM_T_MAX]
#
#     t0 = 'T'
#     t1 = 'T*'
#     t2 = 'T**'
#     v = 'V'
#
#     # Uninfected t-cells
#     INIT_T0 = 'initial_T0'
#     # Latently infected t-cells
#     INIT_T1 = 'initial_T1'
#     # Actively infected t-cells
#     INIT_T2 = 'initial_T2'
#     # Virus
#     INIT_V = 'initial_V'
#     INIT_CONDITS = [INIT_T0, INIT_T1, INIT_T2, INIT_V]
#
#     def __init__(self):
#         self._max_time = 0
#         self._time_points = 0
#         epyc.Experiment.__init__(self)
#
#     def set_time_params(self, max_time, time_points):
#         self._max_time = max_time
#         self._time_points = time_points
#
#     def do(self, params):
#         t_max = self._max_time
#         t_span = list(numpy.linspace(0, t_max, self._time_points))
#
#         initial_conditions = [params[ic] for ic in SAExperiment.INIT_CONDITS]
#
#         s = params[SAExperiment.PARAM_S]
#         mu_t = params[SAExperiment.PARAM_MUT]
#         r = params[SAExperiment.PARAM_R]
#         k1 = params[SAExperiment.PARAM_K1]
#         k2 = params[SAExperiment.PARAM_K2]
#         mu_b = params[SAExperiment.PARAM_MUB]
#         n = params[SAExperiment.PARAM_N]
#         mu_v = params[SAExperiment.PARAM_MUV]
#         t_max = params[SAExperiment.PARAM_T_MAX]
#         # dummy = params[SAExperiment.PARAM_DUMMY]
#
#         def model_eqns(vals, t):
#             t0, t1, t2, v = vals
#
#             dt0dt = s - mu_t * t0 + r * t0 * (1 - (float(t0 + t1 + t2)/t_max)) - k1*v*t0
#             dt1dt = k1*v*t0 - mu_t*t1 - k2*t1
#             dt2dt = k2*t1 - mu_b*t2
#             dvdt = n * mu_b*t2 - k1*v*t0 - mu_v*v
#
#             return [dt0dt, dt1dt, dt2dt, dvdt]
#
#         pde_results = odeint(model_eqns, initial_conditions, t_span)  # integrate
#
#         observed_timesteps = [2000]
#         t_indexes = {x: t_span.index(x) for x in observed_timesteps}
#
#         t = 2000
#
#         output = {SAExperiment.t0: pde_results[t_indexes[t]][0], SAExperiment.t1: pde_results[t_indexes[t]][1],
#                   SAExperiment.t2: pde_results[t_indexes[t]][2], SAExperiment.v: pde_results[t_indexes[t]][3]}
#         return output


class EFASTJSONNotebookTestCase(unittest.TestCase):

    def setUp(self):
        self.filename = 'efastlabtest.json'

    def tearDown(self):
        # # Get rid of json file
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_analyse(self):
        nb_pre = EFASTJSONNotebook(self.filename, True)
        self.lab = EFASTLab(nb_pre)

        model = LotkaVolterraModel()
        self.lab[LotkaVolterraModel.PARAM_ALPHA] = [1.5, 0.01, NORMAL_DISTRIBUTION]
        self.lab[LotkaVolterraModel.PARAM_BETA] = [1, 0.2, NORMAL_DISTRIBUTION]
        self.lab[LotkaVolterraModel.PARAM_SIGMA] = [3, 0.2, NORMAL_DISTRIBUTION]
        self.lab[LotkaVolterraModel.PARAM_DELTA] = [1, 0.01, NORMAL_DISTRIBUTION]

        self.lab[LotkaVolterraModel.INIT_Q] = 10
        self.lab[LotkaVolterraModel.INIT_P] = 5

        model.set_time_params(20, 21)

        resamples = 1  # NR
        runs = 257  # NS
        mi = 4.0

        self.lab.set_sample_number(runs)
        self.lab.set_interference_factor(mi)

        self.lab.runExperiment(RepeatedExperiment(model, 2))

        nb_post = EFASTJSONNotebook(self.filename, False)
        res_s1, res_st = nb_post.analyse()

        s1_vals = [res_s1[('prey',p)] for p in ['alpha','beta','sigma','delta']]
        st_vals = [res_st[('prey',p)] for p in ['alpha','beta','sigma','delta']]
        N = len(s1_vals)
        menStd = (0, 0, 0, 0)
        womenStd = (0, 0, 0, 0)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.8  # the width of the bars: can also be len(x) sequence

        p2 = plt.bar(ind, st_vals, width, bottom=[0, ] * N, yerr=womenStd)
        p1 = plt.bar(ind, s1_vals, width, yerr=menStd)

        plt.ylim((0,1))

        plt.ylabel('eFAST sensitivity')
        plt.title('Lotka-Volterra EFAST')
        plt.xticks(ind, (r'$\alpha$', r'$\beta$', r'$\sigma$', r'$\delta$'))
        # plt.yticks(np.arange(0, 81, 10))
        plt.legend((p1[0], p2[0]), (r'$S_1$', r'$S_T$'))

        plt.savefig("LV.png")

        # Values derived from Fig 4 of Marino S, Hogue IB, Ray CJ, Kirschner DE. A methodology for performing global
        # uncertainty and sensitivity analysis in systems biology. J Theor Biol 2008; 254: 178-96.
        self.assertTrue(0 < res_s1[('prey', 'alpha')] < 0.01)
        self.assertTrue(0 < res_st[('prey', 'alpha')] < 0.2)
        self.assertTrue(0.1 < res_s1[('prey', 'beta')] < 0.3)
        self.assertTrue(0.6 < res_st[('prey', 'beta')] < 0.8)
        self.assertTrue(0.2 < res_s1[('prey', 'sigma')] < 0.4)
        self.assertTrue(0.7 < res_st[('prey', 'sigma')] < 0.9)
        self.assertTrue(0 < res_s1[('prey', 'delta')] < 0.01)
        self.assertTrue(0 < res_st[('prey', 'delta')] < 0.2)


if __name__ == '__main__':
    unittest.main()
