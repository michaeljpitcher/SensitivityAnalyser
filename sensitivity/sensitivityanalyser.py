#!/usr/bin/python

import os
import epyc
import json
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model

from lhslab import LatinHypercubeLab


class SensitivityAnalyser(object):
    """

    """
    MONOTONICITY = 'monotonicity'
    MONOTONICITY_FOLDER = MONOTONICITY + '/'
    MONOTONICITY_FILENAME_SUFFIX = '_' + MONOTONICITY + '.json'
    LHS_FILENAME = "lhs_output.json"
    PRCC = 'PRCC'
    PRCC_FOLDER = PRCC + '/'

    def __init__(self, model):
        assert isinstance(model, epyc.Experiment), "Model must be an epyc experiment"
        self.model = model

        self._stratifications = 0

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

        try:
            if not os.path.exists(SensitivityAnalyser.PRCC):
                os.makedirs(SensitivityAnalyser.PRCC)
        except OSError:
            print ('Error: Creating directory. ' + SensitivityAnalyser.PRCC)

        self.lhs_params = None
        self.lhs_results = None
        self.ranked_lhs_params = None
        self.ranked_lhs_results = None

    def __setitem__( self, k, r ):
        """Add a parameter using array notation.

        :param k: the parameter name
        :param r: the parameter range"""
        if isinstance(r, int):
            self.add_certain_parameter(k, r)
        elif isinstance(r, list):
            assert len(r) == 2
            # Assumes a uniform distribution
            self.add_uncertain_parameter_uniform_distribution(k, r[0], r[1])

    def set_stratifications(self, stratifications):
        self._stratifications = stratifications

    def add_certain_parameter(self, parameter, value):
        """
        Add a parameter whose value is known a priori
        :param parameter:
        :param value:
        :return:
        """
        self.baseline_values[parameter] = value

    def add_uncertain_parameter_uniform_distribution(self, parameter, start, stop):
        """
        Add a parameter whose value is not known a priori but is assumed to be uniform between two specified values
        :param parameter:
        :param start:
        :param stop:
        :param num:
        :return:
        """
        # Calculate range
        param_range = numpy.linspace(start, stop, self._stratifications)
        # Baseline value is in the middle of the range
        self.baseline_values[parameter] = param_range[len(param_range)/2]
        self.uncertain_parameters[parameter] = param_range

    def _read_data(self, filename):
        with open(filename) as data_file:
            data = json.load(data_file)[epyc.Experiment.RESULTS].values()
        param_data = dict([(p,[]) for p in self.uncertain_parameters])
        result_keys = data[0][0][epyc.Experiment.RESULTS].keys()
        result_data = dict([(p, []) for p in result_keys])
        for param_variation in data:
            for p in self.uncertain_parameters:
                param_data[p].append(param_variation[0][epyc.Experiment.PARAMETERS][p])
            for k in result_keys:
                if len(param_variation) > 1:
                    # Multiple repetitions so we need to average
                    repetition_results = [p[epyc.Experiment.RESULTS] for p in param_variation]
                    result_data[k].append(numpy.average([rep[k] for rep in repetition_results]))
                else:
                    # Only 1 repetition (possibly a deterministic model) so just pass it through
                    result_data[k].append(param_variation[0][epyc.Experiment.RESULTS][k])
        return (pandas.DataFrame(param_data, index=range(1, self._stratifications+1)),
                pandas.DataFrame(result_data, index=range(1, self._stratifications+1)))

    def _create_scatter_plot(self, x_data, y_data, title, filename, x_label, y_label, show=False):
        plt.scatter(x_data, y_data)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(filename + ".png")
        if show:
            plt.show()
        # Refresh figure window
        plt.close()

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
        for param in self.uncertain_parameters:
            # Read data from json file and take average if repetitions
            param_data, result_data = self._read_data(SensitivityAnalyser.MONOTONICITY_FOLDER + param +
                                                      SensitivityAnalyser.MONOTONICITY_FILENAME_SUFFIX)
            # Only interested in the current parameters as all others set to baseline values
            param_data = param_data[param]
            for res in result_data:
                result = result_data[res]
                self._create_scatter_plot(param_data, result,
                                          SensitivityAnalyser.MONOTONICITY + " - " + param + " on " + res,
                                          SensitivityAnalyser.MONOTONICITY_FOLDER + SensitivityAnalyser.MONOTONICITY
                                          + "_" + param + "_" + res, param, res, show)

    def generate_lhs_data(self, repetitions):
        """
        Create Latin Hypercube data. For each parameter sample, select a value from the range of each uncertain
        parameter (without replacement)
        :param repetitions:
        :return:
        """
        notebook = epyc.JSONLabNotebook(SensitivityAnalyser.LHS_FILENAME, create=True, description="LHS")
        lab = LatinHypercubeLab(notebook)
        params = self.baseline_values.copy()
        for q, v in self.uncertain_parameters.iteritems():
            params[q] = v
        for q, v in params.iteritems():
            lab[q] = v
        lab.runExperiment(epyc.RepeatedExperiment(self.model, repetitions))

    def obtain_lhs_results(self):
        self.lhs_params, self.lhs_results = self._read_data(SensitivityAnalyser.LHS_FILENAME)
        self.ranked_lhs_params = pandas.DataFrame.rank(self.lhs_params)
        self.ranked_lhs_results = pandas.DataFrame.rank(self.lhs_results)

    def get_all_pearson_correlation_coefficients(self):
        coefficients = {}
        for p in self.uncertain_parameters:
            for r in self.lhs_results:
                coefficients[(p, r)] = self.get_pearson_correlation_coefficient(p, r)
        return coefficients

    def get_pearson_correlation_coefficient(self, parameter, result):
        """
        For linear trends
        :param parameter:
        :param result:
        :return:
        """
        assert parameter in self.uncertain_parameters
        return stats.pearsonr(self.lhs_params[parameter], self.lhs_results[result])

    def get_all_spearman_rank_correlation_coefficients(self):
        coefficients = {}
        for p in self.uncertain_parameters:
            for r in self.lhs_results:
                coefficients[(p, r)] = self.get_spearman_rank_correlation_coefficient(p, r)
        return coefficients

    def get_spearman_rank_correlation_coefficient(self, parameter, result):
        """
        For non-linear monotonic trends
        :param parameter:
        :param result:
        :return:
        """
        assert parameter in self.uncertain_parameters
        return stats.spearmanr(self.lhs_params[parameter], self.lhs_results[result])

    def get_all_prcc(self, plots=False):
        prccs = dict([(r, self.calculate_prcc(r, plots)) for r in self.ranked_lhs_results])
        return prccs

    def calculate_prcc(self, result, plots=False):
        regr = linear_model.LinearRegression()

        cols = self.ranked_lhs_params.columns

        param_data = numpy.asarray(self.ranked_lhs_params)
        result_data = numpy.asarray(self.ranked_lhs_results[result]).reshape((self._stratifications,1))

        k = param_data.shape[1]

        prccs = {}
        for i in range(k):
            label = cols[i]
            col = [j == i for j in range(k)]
            remaining_col = numpy.logical_not(col)

            param = param_data[:, col]
            remaining_param = param_data[:, remaining_col]

            regr.fit(remaining_param, param)
            linreg_param = regr.predict(remaining_param)
            param_resid = param - linreg_param

            regr.fit(remaining_param, result_data)
            linreg_result = regr.predict(remaining_param)
            result_resid = result_data - linreg_result

            corr, p = stats.pearsonr(param_resid, result_resid)
            prccs[label] = (corr, p)

            if plots:
                title = "PRCC(" + label + ',' + result + "): " + str(corr) + '\n p =' + str(p)
                filename = SensitivityAnalyser.PRCC_FOLDER + label + "_" + result + ".png"
                self._create_scatter_plot(param_resid, result_resid, title, filename, label, result, False)

        return prccs
