import epyc
from sensitivity import *


class SAExperiment(epyc.Experiment):

    PARAM_X = 'x'
    PARAM_Y = 'y'
    SA_PARAMS = [PARAM_X, PARAM_Y]

    RESULT_X_PLUS_Y = 'x_plus_y'
    RESULT_X_SQUARED = 'x_squared'
    SA_RESULTS = [RESULT_X_PLUS_Y, RESULT_X_SQUARED]

    def do(self, params):
        x = params[self.PARAM_X]
        y = params[self.PARAM_Y]
        r = x + y
        r2 = x ** 2
        return {self.RESULT_X_PLUS_Y: r, self.RESULT_X_SQUARED:r2}


model = SAExperiment()
sa = SensitivityAnalyser(model, "saanalysis")
sa.set_parameter_stratification({SAExperiment.PARAM_X: numpy.linspace(0.5, 99.5, 100),
                                 SAExperiment.PARAM_Y: numpy.linspace(0.5, 99.5, 100)})

sa.create_monotonicity_plots(10)

#
# sa.generate_output_matrix(10)
# sa.obtain_results()
# for p in SAExperiment.SA_PARAMS:
#     for r in SAExperiment.SA_RESULTS:
#         print p, r, sa.get_pearson_correlation_coefficient(p, r)

