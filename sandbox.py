import epyc
from sensitivity import *
import numpy


class SAExperiment(epyc.Experiment):
    PARAM_X = 'x'
    PARAM_Y = 'y'
    PARAM_Z = 'z'
    SA_PARAMS = [PARAM_X, PARAM_Y, PARAM_Z]

    RESULT_X_PLUS_Y = 'x_plus_y'
    RESULT_X_SQUARED = 'x_squared'
    RESULT_SIN_Y = 'sin_y'
    RESULT_RANDOM_Y = 'random_y'
    RESULT_RANDOM = 'random'
    SA_RESULTS = [RESULT_X_PLUS_Y, RESULT_X_SQUARED, RESULT_SIN_Y, RESULT_RANDOM_Y, RESULT_RANDOM]

    def do(self, params):
        x = params[self.PARAM_X]
        y = params[self.PARAM_Y]
        results = {SAExperiment.RESULT_X_PLUS_Y: x + y,
                   SAExperiment.RESULT_X_SQUARED: x ** 2,
                   SAExperiment.RESULT_SIN_Y: numpy.sin(y),
                   SAExperiment.RESULT_RANDOM_Y: numpy.random.random() * y,
                   SAExperiment.RESULT_RANDOM: numpy.random.random()}
        return results

model = SAExperiment()
parameter_set = ParameterSet()
parameter_set.add_parameter(SAExperiment.PARAM_Z, 4)
parameter_set.add_parameter(SAExperiment.PARAM_X, [0.5, 9.5, 'uniform'])
parameter_set.add_parameter(SAExperiment.PARAM_Y, [10.5, 20.5, 'uniform'])


# ----------------------------------
# CREATING SCATTER GRAPHS
# ----------------------------------
strats = 30
reps = 10
folder = 'experiments/scatter_new'
filename = create_correlation_scatter_data(model, parameter_set, strats, reps, folder)
# filename = folder + '/scatter.json'
rs = ResultSet(parameter_set)
rs.load_epyc_results_from_json(filename)
create_correlation_scatter_graphs(rs, folder)

# ----------------------------------
# SENSITIVITY ANALYSIS
# ----------------------------------

lhs_output_folder = 'experiments/lhs_new'
stratifications = 100
reps = 100
lhs_filename = run_lhs(model, parameter_set, stratifications, reps, lhs_output_folder)
result_set = ResultSet(parameter_set)
result_set.load_epyc_results_from_json(lhs_filename)
analyser = LHSAnalyser(parameter_set, result_set, lhs_output_folder)
print "PEARSON:"
for (param, res), (r, p) in analyser.get_all_pearson_correlation_coefficients().iteritems():
    print param, res, r, "p=", p
print
print "SPEARMAN:"
for (param, res), (r, p) in analyser.get_all_spearman_rank_correlation_coefficients().iteritems():
    print param, res, r, "p=", p
print
print "PRCC:"
for k, (r,p) in analyser.get_all_prcc(True):
    print k, r, "p=", p