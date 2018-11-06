import os
import epyc
import json
import numpy
import collections
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from lhslab import LatinHypercubeLab


# TODO: do we need extra labs or not (monotonic vs LHS)

class SensitivityAnalyser(object):
    MONOTONICITY = 'monotonicity'
    MONOTONICITY_FOLDER = MONOTONICITY + '/'
    MONOTONICITY_FILENAME_SUFFIX = '_' + MONOTONICITY + '.json'
    LHS_FILENAME = "lhs_output.json"

    def __init__(self, model):
        assert isinstance(model, epyc.Experiment), "Model must be an epyc experiment"
        self.model = model

        # Uncertain parameters and their ranges
        self.uncertain_parameters = {}
        # Baseline values for all parameters (certain - value, uncertain - mid-range value)
        self.baseline_values = {}

        # Create an output folder for monotonicity data
        try:
            if not os.path.exists(SensitivityAnalyser.MONOTONICITY):
                os.makedirs(SensitivityAnalyser.MONOTONICITY)
        except OSError:
            print ('Error: Creating directory. ' + SensitivityAnalyser.MONOTONICITY)

    def add_certain_parameter(self, parameter, value):
        """
        Add a parameter whose value is known a priori
        :param parameter:
        :param value:
        :return:
        """
        self.baseline_values[parameter] = value

    def add_uncertain_parameter_uniform_distribution(self, parameter, start, stop, num):
        """
        Add a parameter whose value is not known a priori but is assumed to be uniform between two specified values
        :param parameter:
        :param start:
        :param stop:
        :param num:
        :return:
        """
        # Calculate range
        param_range = numpy.linspace(start, stop, num)
        # Baseline value is in the middle of the range
        self.baseline_values[parameter] = param_range[len(param_range)/2]
        self.uncertain_parameters[parameter] = param_range

    def read_data(self, filename):
        """
        Given a JSON filename for epyc data, reads the JSON file in and averages the data over the repetitions.
        :param filename:
        :return:
        """
        with open(filename) as data_file:
            data = json.load(data_file)[epyc.Experiment.RESULTS].values()
        averaged_data = []
        for param_variation in data:
            # Parameters will be the same for all repetitions
            param_sample = param_variation[0][epyc.Experiment.PARAMETERS]
            if len(param_variation) > 1:
                # Multiple repetitions so we need to average
                repetition_results = [p[epyc.Experiment.RESULTS] for p in param_variation]
                avg_for_param_sample = self.average_data_from_repetitions(repetition_results)
                averaged_data.append((param_sample, avg_for_param_sample))
            else:
                # Only 1 repetition (possibly a deterministic model) so just pass it through
                averaged_data.append((param_sample, param_variation[0][epyc.Experiment.RESULTS]))
        return averaged_data

    def average_data_from_repetitions(self, repetitions):
        """
        Calculate the average result values for the repetitions. Default assumes that each result is a simple numeric
        value, so averages them. May need to be sub-classed when the result is not so straightforward.
        :param repetitions:
        :return:
        """
        avg_data = {}
        result_keys = repetitions[0].keys()
        for k in result_keys:
            avg_data[k] = numpy.average([rep[k] for rep in repetitions])
        return avg_data

    def generate_monotonicity_data(self, repetitions):
        """
        Create monotonicity data. For each uncertain parameter, run simulations on all points in its range (other
        parameters have values set to their baseline)
        :param repetitions:
        :return:
        """
        for p in self.uncertain_parameters:
            notebook = epyc.JSONLabNotebook(SensitivityAnalyser.MONOTONICITY_FOLDER + p +
                                            SensitivityAnalyser.MONOTONICITY_FILENAME_SUFFIX,
                                            create=True, description=SensitivityAnalyser.MONOTONICITY + " " + p)
            lab = epyc.Lab(notebook)
            params = self.baseline_values.copy()
            params[p] = self.uncertain_parameters[p]
            for q, v in params.iteritems():
                lab[q] = v
            lab.runExperiment(epyc.RepeatedExperiment(self.model, repetitions))

    def create_monotonicity_plots(self, show=False):
        """
        For each uncertain parameter, read the monotonicity data and make a scatter graph comparing values of parameter
        against all recorded output results
        :param show: Whether graph is displayed to user (graph is always saved)
        :return:
        """
        for p in self.uncertain_parameters:
            # Read data from json file and take average if repetitions
            data = self.read_data(SensitivityAnalyser.MONOTONICITY_FOLDER + p +
                                  SensitivityAnalyser.MONOTONICITY_FILENAME_SUFFIX)
            # Determine which results were recorded
            rk = data[0][1].keys()
            # For each result, plot the parameter values against that result
            for r in rk:
                plt.scatter([d[0][p] for d in data], [d[1][r] for d in data])
                plt.title(SensitivityAnalyser.MONOTONICITY + " - " + p + " on " + r)
                plt.xlabel(p)
                plt.ylabel(r)
                plt.savefig(SensitivityAnalyser.MONOTONICITY_FOLDER + SensitivityAnalyser.MONOTONICITY + "_" + p + "_"
                            + r + ".png")
                if show:
                    plt.show()
                # Refresh figure window
                plt.close()

    # def generate_output_matrix(self, repetitions):
    #     notebook = epyc.JSONLabNotebook(SensitivityAnalyser.LHS_FILENAME, create=True,
    #                                     description="Sensitivity analysis")
    #     lab = LatinHypercubeLab(notebook=notebook)
    #     self._set_parameters(lab)
    #     lab.runExperiment(epyc.RepeatedExperiment(self.model, repetitions))
    #
    # def obtain_results(self):
    #     with open(SensitivityAnalyser.LHS_FILENAME) as data_file:
    #         json_file = json.load(data_file)
    #     data = json_file[epyc.Experiment.RESULTS].values()
    #     number_of_param_variations = len(data)
    #
    #     parameters = data[0][0][epyc.Experiment.PARAMETERS].keys()
    #     self.outcome_variables = data[0][0][epyc.Experiment.RESULTS].keys()
    #
    #     self.run_parameter_values = dict([(p, []) for p in parameters])
    #     self.run_outcome_variable_values = dict([(r, []) for r in self.outcome_variables])
    #
    #     for n in range(number_of_param_variations):
    #         for p in parameters:
    #             # Params are same for all repetitions, so just use first
    #             self.run_parameter_values[p].append(data[n][0][epyc.Experiment.PARAMETERS][p])
    #         for r in self.outcome_variables:
    #             # Output values may vary so take average
    #             self.run_outcome_variable_values[r].append(numpy.average([repetition[epyc.Experiment.RESULTS][r]
    #                                                                       for repetition in data[n]]))
    #
    # def get_all_pearson_correlation_coefficients(self):
    #     coefficients = {}
    #     for p,v in self.lab.get_uncertain_parameters().iteritems():
    #         for r in self.outcome_variables:
    #             coefficients[(p, r)] = self.get_pearson_correlation_coefficient(p, r)
    #     return coefficients
    #
    # def get_pearson_correlation_coefficient(self, parameter, result):
    #     """
    #     For linear trends
    #     :param parameter:
    #     :param result:
    #     :return:
    #     """
    #     assert parameter in self.lab.get_uncertain_parameters()
    #     return scipy.stats.pearsonr(self.run_parameter_values[parameter], self.run_outcome_variable_values[result])
    #
    # def get_all_spearman_rank_correlation_coefficients(self):
    #     coefficients = {}
    #     for p,v in self.lab.get_uncertain_parameters().iteritems():
    #         for r in self.outcome_variables:
    #             coefficients[(p, r)] = self.get_spearman_rank_correlation_coefficient(p, r)
    #     return coefficients
    #
    # def get_spearman_rank_correlation_coefficient(self, parameter, result):
    #     """
    #     For non-linear monotonic trends
    #     :param parameter:
    #     :param result:
    #     :return:
    #     """
    #     assert parameter in self.lab.get_uncertain_parameters()
    #     return scipy.stats.spearmanr(self.run_parameter_values[parameter], self.run_outcome_variable_values[result])
