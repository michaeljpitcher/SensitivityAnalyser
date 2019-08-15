import unittest
from epycsense import *
import numpy
import os

from scipy.integrate import odeint


class SAExperiment(epyc.Experiment):
    PARAM_S = 's'
    PARAM_MUT = 'mu_T'
    PARAM_R = 'r'
    PARAM_K1 = 'k_1'
    PARAM_K2 = 'k_2'
    PARAM_MUB = 'mu_b'
    PARAM_N = 'N_V'
    PARAM_MUV = 'mu_V'
    PARAM_DUMMY = 'dummy'
    PARAM_T_MAX = 't_max'
    SA_PARAMS = [PARAM_S, PARAM_MUT, PARAM_R, PARAM_K1, PARAM_K2, PARAM_MUB, PARAM_N, PARAM_MUV, PARAM_DUMMY,
                 PARAM_T_MAX]

    t0 = 'T'
    t1 = 'T*'
    t2 = 'T**'
    v = 'V'

    # Uninfected t-cells
    INIT_T0 = 'initial_T0'
    # Latently infected t-cells
    INIT_T1 = 'initial_T1'
    # Actively infected t-cells
    INIT_T2 = 'initial_T2'
    # Virus
    INIT_V = 'initial_V'
    INIT_CONDITS = [INIT_T0, INIT_T1, INIT_T2, INIT_V]

    def __init__(self):
        self._max_time = 0
        self._time_points = 0
        epyc.Experiment.__init__(self)

    def set_time_params(self, max_time, time_points):
        self._max_time = max_time
        self._time_points = time_points

    def do(self, params):
        t_max = self._max_time
        t_span = list(numpy.linspace(0, t_max, self._time_points))

        initial_conditions = [params[ic] for ic in SAExperiment.INIT_CONDITS]

        s = params[SAExperiment.PARAM_S]
        mu_t = params[SAExperiment.PARAM_MUT]
        r = params[SAExperiment.PARAM_R]
        k1 = params[SAExperiment.PARAM_K1]
        k2 = params[SAExperiment.PARAM_K2]
        mu_b = params[SAExperiment.PARAM_MUB]
        n = params[SAExperiment.PARAM_N]
        mu_v = params[SAExperiment.PARAM_MUV]
        t_max = params[SAExperiment.PARAM_T_MAX]
        # dummy = params[SAExperiment.PARAM_DUMMY]

        def model_eqns(vals, t):
            t0, t1, t2, v = vals

            dt0dt = s - mu_t * t0 + r * t0 * (1 - (float(t0 + t1 + t2)/t_max)) - k1*v*t0
            dt1dt = k1*v*t0 - mu_t*t1 - k2*t1
            dt2dt = k2*t1 - mu_b*t2
            dvdt = n * mu_b*t2 - k1*v*t0 - mu_v*v

            return [dt0dt, dt1dt, dt2dt, dvdt]

        pde_results = odeint(model_eqns, initial_conditions, t_span)  # integrate

        observed_timesteps = [2000,4000]

        t_indexes = {x: t_span.index(x) for x in observed_timesteps}

        output = {t: {SAExperiment.t0: pde_results[t_indexes[t]][0], SAExperiment.t1: pde_results[t_indexes[t]][1],
                      SAExperiment.t2: pde_results[t_indexes[t]][2], SAExperiment.v: pde_results[t_indexes[t]][3]}
                  for t in observed_timesteps}
        return output


class EFASTJSONNotebookTestCase(unittest.TestCase):

    def setUp(self):
        self.filename = 'efastlabtest.json'

    # def tearDown(self):
    #     # # Get rid of json file
    #     if os.path.exists(self.filename):
    #         os.remove(self.filename)

    def test_analyse(self):
        # self.nb = EFASTJSONNotebook(self.filename, True)
        # self.lab = EFASTLab(self.nb)
        # model = SAExperiment()
        # self.lab[SAExperiment.PARAM_S] = [1e-2, 50, UNIFORM_DISTRIBUTION]
        # self.lab[SAExperiment.PARAM_MUT] = [1e-4, 0.2, UNIFORM_DISTRIBUTION]
        # self.lab[SAExperiment.PARAM_R] = [1e-3, 50, UNIFORM_DISTRIBUTION]
        # self.lab[SAExperiment.PARAM_K1] = [1e-7, 1e-3, UNIFORM_DISTRIBUTION]
        # self.lab[SAExperiment.PARAM_K2] = [1e-5, 1e-2, UNIFORM_DISTRIBUTION]
        # self.lab[SAExperiment.PARAM_MUB] = [1e-1, 0.4, UNIFORM_DISTRIBUTION]
        # self.lab[SAExperiment.PARAM_N] = [1, 2000, UNIFORM_DISTRIBUTION]
        # self.lab[SAExperiment.PARAM_MUV] = [1e-1, 10, UNIFORM_DISTRIBUTION]
        # # self.lab[SAExperiment.PARAM_DUMMY] = [1, 10, UNIFORM_DISTRIBUTION]
        # self.lab[SAExperiment.PARAM_T_MAX] = 1500
        #
        # self.lab[SAExperiment.INIT_T0] = 1e3
        # self.lab[SAExperiment.INIT_T1] = 0
        # self.lab[SAExperiment.INIT_T2] = 0
        # self.lab[SAExperiment.INIT_V] = 1e-3
        #
        # model.set_time_params(4000, 4001)
        #
        # resamples = 1  # NR
        # runs = 65  # NS
        # mi = 4.0
        #
        # self.lab.set_sample_number(runs)
        # self.lab.set_interference_factor(mi)
        #
        # self.lab.runExperiment(model)

        self.nb = EFASTJSONNotebook(self.filename, False)
        self.nb.analyse()



if __name__ == '__main__':
    unittest.main()
