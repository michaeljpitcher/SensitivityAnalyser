from sensitivity import *


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


class TimeExperiment(epyc.Experiment):
    PARAM_X = 'x'
    PARAM_Y = 'y'
    PARAM_Z = 'z'
    SA_PARAMS = [PARAM_X, PARAM_Y, PARAM_Z]

    RESULT_X_PLUS_Y = 'x_plus_y'
    RESULT_X_SQUARED = 'x_squared'
    RESULT_SIN_Y = 'sin_y'
    RESULT_RANDOM_Y = 'random_y'
    RESULT_RANDOM = 'random'
    TIME_RESULTS = [RESULT_X_PLUS_Y, RESULT_X_SQUARED, RESULT_SIN_Y, RESULT_RANDOM_Y, RESULT_RANDOM]

    def do(self, params):
        x = params[self.PARAM_X]
        y = params[self.PARAM_Y]
        z = params[self.PARAM_Z]
        results = {}
        for time in range(0, 4):
            results[time] = {SAExperiment.RESULT_X_PLUS_Y: x + y,
                             SAExperiment.RESULT_X_SQUARED: x ** 2,
                             SAExperiment.RESULT_SIN_Y: numpy.sin(y),
                             SAExperiment.RESULT_RANDOM_Y: numpy.random.random() * y,
                             SAExperiment.RESULT_RANDOM: numpy.random.random()}
        return results


model = SAExperiment()
params = ParameterSet()
params.add_certain_parameter(SAExperiment.PARAM_Z, 1)
params.add_uncertain_parameter_uniform_dist(SAExperiment.PARAM_X, 0.5, 10.5)
params.add_uncertain_parameter_uniform_dist(SAExperiment.PARAM_Y, 0.5, 10.5)
strats = 100
reps = 100
create_lhs_scatter_plots(model, params, strats, reps, 'scatter', False)

lhs_output_folder = 'lhs'
evaluate_lhs(model, params, strats, reps, lhs_output_folder)
analyser = LHSAnalyser(params, lhs_output_folder)
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

model = TimeExperiment()
params = ParameterSet()
params.add_certain_parameter(SAExperiment.PARAM_Z, 1)
params.add_uncertain_parameter_uniform_dist(SAExperiment.PARAM_X, 0.5, 10.5)
params.add_uncertain_parameter_uniform_dist(SAExperiment.PARAM_Y, 0.5, 10.5)
strats = 100
reps = 100
create_lhs_scatter_plots(model, params, strats, reps, 'scatter2', True)

lhs_output_folder = 'lhs2'
evaluate_lhs(model, params, strats, reps, lhs_output_folder)
analyser = LHSAnalyser(params, lhs_output_folder, True)
print "PEARSON:"
rs = []
for (t, param, res), (r, p) in analyser.get_all_pearson_correlation_coefficients().iteritems():
    rs.append((t, param, res, (r, p)))
rs.sort()
for (t, param, res, (r,p)) in rs:
    print t, param, res, r, "p=", p
print
print "SPEARMAN:"
rs = []
for (t, param, res), (r, p) in analyser.get_all_spearman_rank_correlation_coefficients().iteritems():
    rs.append((t, param, res, (r, p)))
rs.sort()
for (t, param, res, (r,p)) in rs:
    print t, param, res, r, "p=", p
print
print "PRCC:"
rs = []
for (t, param, res), (r, p) in analyser.get_all_prcc(True):
    rs.append((t, param, res, (r, p)))
rs.sort()
for (t, param, res, (r, p)) in rs:
    print t, param, res, r, "p=", p
print