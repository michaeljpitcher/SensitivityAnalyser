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


class EFASTJSONNotebookTestCase(unittest.TestCase):

    def setUp(self):
        self.filename = 'efastlabtest.json'

    # def tearDown(self):
    #     # # Get rid of json file
    #     if os.path.exists(self.filename):
    #         os.remove(self.filename)

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

        resamples = 5  # NR
        runs = 257  # NS
        mi = 4.0

        self.lab.set_sample_number(runs)
        self.lab.set_resample_number(resamples)
        self.lab.set_interference_factor(mi)
        #
        # self.lab.set_required_parameters([LotkaVolterraModel.PARAM_ALPHA, LotkaVolterraModel.PARAM_BETA])

        print "Running Experiment"
        self.lab.runExperiment(RepeatedExperiment(model, 2))

        nb_post = EFASTJSONNotebook(self.filename, False)
        print "Analysing Experiment Results"
        res_s1, res_st = nb_post.generate_sensitivity_indices()

        s1_vals = [numpy.mean(res_s1[('prey',p)]) for p in ['alpha','beta','sigma','delta','dummy']]
        st_vals = [numpy.mean(res_st[('prey',p)]) for p in ['alpha','beta','sigma','delta','dummy']]
        N = len(s1_vals)
        s1_Std = [2*numpy.std(res_s1[('prey',p)]) for p in ['alpha','beta','sigma','delta','dummy']]
        st_Std = [2*numpy.std(res_s1[('prey',p)]) for p in ['alpha','beta','sigma','delta','dummy']]
        ind = np.arange(N)  # the x locations for the groups
        width = 0.8  # the width of the bars: can also be len(x) sequence

        print s1_vals
        print st_vals

        p2 = plt.bar(ind, st_vals, width, bottom=[0, ] * N, yerr=st_Std)
        p1 = plt.bar(ind, s1_vals, width, yerr=s1_Std)

        plt.ylim((0,1))

        plt.ylabel('eFAST sensitivity')
        plt.title('Lotka-Volterra EFAST')
        plt.xticks(ind, (r'$\alpha$', r'$\beta$', r'$\sigma$', r'$\delta$', r'$dummy$'))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.legend((p1[0], p2[0]), (r'$S_1$', r'$S_T$'))

        plt.savefig("EFAST_LV.png")

        print 'alpha s1', stats.ttest_ind(res_s1[('prey','alpha')], res_s1[('prey','dummy')])
        print 'beta s1',stats.ttest_ind(res_s1[('prey', 'beta')], res_s1[('prey', 'dummy')])
        print 'sigma s1',stats.ttest_ind(res_s1[('prey', 'sigma')], res_s1[('prey', 'dummy')])
        print 'delta s1',stats.ttest_ind(res_s1[('prey', 'delta')], res_s1[('prey', 'dummy')])

        print 'alpha st',stats.ttest_ind(res_st[('prey', 'alpha')], res_st[('prey', 'dummy')])
        print 'beta st',stats.ttest_ind(res_st[('prey', 'beta')], res_st[('prey', 'dummy')])
        print 'sigma st',stats.ttest_ind(res_st[('prey', 'sigma')], res_st[('prey', 'dummy')])
        print 'delta st',stats.ttest_ind(res_st[('prey', 'delta')], res_st[('prey', 'dummy')])
#

#
# class EFASTJSONNotebook2TestCase(unittest.TestCase):
#
#     def setUp(self):
#         self.filename = 'efastlabtest.json'
#
#     def tearDown(self):
#         # # Get rid of json file
#         if os.path.exists(self.filename):
#             os.remove(self.filename)
#
#     def test_analyse(self):
#         nb_pre = EFASTJSONNotebook(self.filename, True)
#         self.lab = EFASTClusterLab(nb_pre, 'efast_test')
#         print "{0} engines".format(self.lab.numberOfEngines())
#
#         model = LotkaVolterraModel()
#         self.lab[LotkaVolterraModel.PARAM_ALPHA] = [1.5, 0.01, NORMAL_DISTRIBUTION]
#         self.lab[LotkaVolterraModel.PARAM_BETA] = [1, 0.2, NORMAL_DISTRIBUTION]
#         self.lab[LotkaVolterraModel.PARAM_SIGMA] = [3, 0.2, NORMAL_DISTRIBUTION]
#         self.lab[LotkaVolterraModel.PARAM_DELTA] = [1, 0.01, NORMAL_DISTRIBUTION]
#
#         self.lab[LotkaVolterraModel.INIT_Q] = 10
#         self.lab[LotkaVolterraModel.INIT_P] = 5
#
#         model.set_time_params(20, 21)
#
#         resamples = 5  # NR
#         runs = 257  # NS
#         mi = 4.0
#
#         self.lab.set_sample_number(runs)
#         self.lab.set_resample_number(resamples)
#         self.lab.set_interference_factor(mi)
#
#         print "Running Experiment"
#         self.lab.runExperiment(RepeatedExperiment(model, 2))
#         print "JOBS SENT"
#
#         print self.lab.pendingResults()
#
#         nb_post = EFASTJSONNotebook(self.filename, False)
#         print "Analysing Experiment Results"
#         res_s1, res_st = nb_post.generate_sensitivity_indices()
#
#         s1_vals = [numpy.mean(res_s1[('prey',p)]) for p in ['alpha','beta','sigma','delta','dummy']]
#         st_vals = [numpy.mean(res_st[('prey',p)]) for p in ['alpha','beta','sigma','delta','dummy']]
#         N = len(s1_vals)
#         s1_Std = [2*numpy.std(res_s1[('prey',p)]) for p in ['alpha','beta','sigma','delta','dummy']]
#         st_Std = [2*numpy.std(res_s1[('prey',p)]) for p in ['alpha','beta','sigma','delta','dummy']]
#         ind = np.arange(N)  # the x locations for the groups
#         width = 0.8  # the width of the bars: can also be len(x) sequence
#
#         p2 = plt.bar(ind, st_vals, width, bottom=[0, ] * N, yerr=st_Std)
#         p1 = plt.bar(ind, s1_vals, width, yerr=s1_Std)
#
#         plt.ylim((0,1))
#
#         plt.ylabel('eFAST sensitivity')
#         plt.title('Lotka-Volterra EFAST')
#         plt.xticks(ind, (r'$\alpha$', r'$\beta$', r'$\sigma$', r'$\delta$', r'$dummy$'))
#         plt.yticks(np.arange(0, 1, 0.1))
#         plt.legend((p1[0], p2[0]), (r'$S_1$', r'$S_T$'))
#
#         plt.savefig("LV.png")
#
#         print res_st[('prey', 'alpha')], numpy.mean(res_st[('prey', 'alpha')]), numpy.std(res_st[('prey', 'alpha')])
#         print res_st[('prey', 'delta')], numpy.mean(res_st[('prey', 'delta')]), numpy.std(res_st[('prey', 'delta')])
#
#         print stats.ttest_ind(res_s1[('prey','alpha')], res_s1[('prey','dummy')])
#         print stats.ttest_ind(res_s1[('prey', 'beta')], res_s1[('prey', 'dummy')])
#         print stats.ttest_ind(res_s1[('prey', 'sigma')], res_s1[('prey', 'dummy')])
#         print stats.ttest_ind(res_s1[('prey', 'delta')], res_s1[('prey', 'dummy')])
#
#         print stats.ttest_ind(res_st[('prey', 'alpha')], res_s1[('prey', 'dummy')])
#         print stats.ttest_ind(res_st[('prey', 'beta')], res_s1[('prey', 'dummy')])
#         print stats.ttest_ind(res_st[('prey', 'sigma')], res_s1[('prey', 'dummy')])
#         print stats.ttest_ind(res_st[('prey', 'delta')], res_s1[('prey', 'dummy')])

if __name__ == '__main__':
    unittest.main()
