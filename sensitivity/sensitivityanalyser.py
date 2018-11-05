import epyc
import json
import numpy
import scipy.stats
from lhslab import LatinHypercubeLab


# TODO: do we need extra labs or not (monotonic vs LHS)

class SensitivityAnalyser(object):
    def __init__(self, model, filename):
        assert isinstance(model, epyc.Experiment), "Model must be an epyc experiment"
        self.model = model
        self.filename = filename + ".json"
        self.notebook = epyc.JSONLabNotebook(self.filename, create=True, description="Sensitivity analysis")

        self.lab = LatinHypercubeLab(notebook=self.notebook)

        self._parameter_stratifications = None

        self.parameter_values = None
        self.result_values = None

    def set_parameter_stratification(self, stratifications):
        # TODO - assumes user has defined stratifications, allow automatic calculation (e.g. normal distribution)
        self._parameter_stratifications = stratifications

    def generate_output_matrix(self, repetitions):
        for p, r in self._parameter_stratifications.iteritems():
            self.lab[p] = r

        self.lab.runExperiment(epyc.RepeatedExperiment(self.model, repetitions))

    def obtain_results(self):
        with open(self.filename) as data_file:
            json_file = json.load(data_file)
        data = json_file[epyc.Experiment.RESULTS].values()
        number_of_param_variations = len(data)

        parameter_keys = data[0][0][epyc.Experiment.PARAMETERS].keys()
        result_keys = data[0][0][epyc.Experiment.RESULTS].keys()

        self.parameter_values = dict([(p, []) for p in parameter_keys])
        self.result_values = dict([(r, []) for r in result_keys])

        for n in range(number_of_param_variations):
            for p in parameter_keys:
                # Params are same for all repetitions, so just use first
                self.parameter_values[p].append(data[n][0][epyc.Experiment.PARAMETERS][p])
            for r in result_keys:
                # Output values may vary so take average
                self.result_values[r].append(numpy.average([repetition[epyc.Experiment.RESULTS][r] for repetition in
                                                            data[n]]))

    def create_monotonicity_plots(self, repetitions):
        for p, v in self._parameter_stratifications.iteritems():
            param_values = {p: v}
            for q, w in [(q,w) for (q,w) in self._parameter_stratifications.iteritems() if q != p]:
                # Choose the mid-point
                param_values[q] = w[len(w) / 2]
            for p, r in param_values.iteritems():
                self.lab[p] = r
            self.lab.runExperiment(epyc.RepeatedExperiment(self.model, repetitions))


    def get_pearson_correlation_coefficient(self, parameter, result):
        return scipy.stats.pearsonr(self.parameter_values[parameter], self.result_values[result])

    def get_spearman_rank_correlation_coefficient(self, parameter, result):
        return scipy.stats.spearmanr(self.parameter_values[parameter], self.result_values[result])