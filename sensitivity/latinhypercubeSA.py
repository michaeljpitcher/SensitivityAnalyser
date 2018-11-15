import os
import epyc
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model

from jsondatareader import read_epyc_json_data
from lhslab import LatinHypercubeLab


class LatinHypercubeSensitivityAnalyser(object):
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
            if not os.path.exists(LatinHypercubeSensitivityAnalyser.MONOTONICITY):
                os.makedirs(LatinHypercubeSensitivityAnalyser.MONOTONICITY)
        except OSError:
            print ('Error: Creating directory. ' + LatinHypercubeSensitivityAnalyser.MONOTONICITY)

        try:
            if not os.path.exists(LatinHypercubeSensitivityAnalyser.PRCC):
                os.makedirs(LatinHypercubeSensitivityAnalyser.PRCC)
        except OSError:
            print ('Error: Creating directory. ' + LatinHypercubeSensitivityAnalyser.PRCC)

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
        """
        Specify the number of stratifications for latin hypercube sampling
        :param stratifications:
        :return:
        """
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

    def _create_scatter_plot(self, x_data, y_data, title, filename, x_label, y_label, show=False):
        """
        Create a scatter plot
        :param x_data:
        :param y_data:
        :param title:
        :param filename:
        :param x_label:
        :param y_label:
        :param show:
        :return:
        """
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
            notebook = epyc.JSONLabNotebook(LatinHypercubeSensitivityAnalyser.MONOTONICITY_FOLDER + p +
                                            LatinHypercubeSensitivityAnalyser.MONOTONICITY_FILENAME_SUFFIX,
                                            create=True, description=LatinHypercubeSensitivityAnalyser.MONOTONICITY +
                                                                     " " + p)
            lab = epyc.Lab(notebook)
            params = self.baseline_values.copy()
            params[p] = self.uncertain_parameters[p]
            for q, v in params.iteritems():
                lab[q] = v
            lab.runExperiment(epyc.RepeatedExperiment(self.model, repetitions))

    def create_monotonicity_plots(self, show=False):
        """
        Create plots comparing uncertain parameters against outcomes to check for nonlinearities and non-monotonicities
        :param show:
        :return:
        """
        for param in self.uncertain_parameters:
            # Read data from json file and take average if repetitions
            param_data, result_data = read_epyc_json_data(
                LatinHypercubeSensitivityAnalyser.MONOTONICITY_FOLDER + param +
                LatinHypercubeSensitivityAnalyser.MONOTONICITY_FILENAME_SUFFIX,
                self.uncertain_parameters)
            # Only interested in the current parameters as all others set to baseline values
            param_data = param_data[param]
            for res in result_data:
                result = result_data[res]
                self._create_scatter_plot(param_data, result,
                                          LatinHypercubeSensitivityAnalyser.MONOTONICITY + " - " + param + " on " + res,
                                          LatinHypercubeSensitivityAnalyser.MONOTONICITY_FOLDER +
                                          LatinHypercubeSensitivityAnalyser.MONOTONICITY
                                          + "_" + param + "_" + res, param, res, show)

    def generate_lhs_data(self, repetitions):
        """
        Create Latin Hypercube data. For each parameter sample, select a value from the range of each uncertain
        parameter (without replacement)
        :param repetitions:
        :return:
        """
        notebook = epyc.JSONLabNotebook(LatinHypercubeSensitivityAnalyser.LHS_FILENAME, create=True, description="LHS")
        lab = LatinHypercubeLab(notebook)
        params = self.baseline_values.copy()
        for q, v in self.uncertain_parameters.iteritems():
            params[q] = v
        for q, v in params.iteritems():
            lab[q] = v
        lab.runExperiment(epyc.RepeatedExperiment(self.model, repetitions))

    def obtain_lhs_results(self):
        """
        Load the Latin hypercube sample results
        :return:
        """
        self.lhs_params, self.lhs_results = read_epyc_json_data(LatinHypercubeSensitivityAnalyser.LHS_FILENAME,
                                                                self.uncertain_parameters)
        self.ranked_lhs_params = pandas.DataFrame.rank(self.lhs_params)
        self.ranked_lhs_results = pandas.DataFrame.rank(self.lhs_results)

    def get_all_pearson_correlation_coefficients(self):
        """
        Calculate all Pearson correlation coefficients
        :return:
        """
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
        """
        Calculate all Spearman Rank Correlation Coefficients
        :return:
        """
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
        """
        Calculate a partial rank correlation coefficient (PRCC) value for all uncertain parameters against the output.

        Partial correlation characterizes the linear relationship between an input and an output after the linear
        effects of the remaining inputs on the output are discounted. This is calculated by constructing a linear
        regression model between the input and the remaining inputs, and calculating the residuals between the input
        values and the model. Similarly, a linear regression model between the output and the remaining input is
        created and residuals are calculated. The PRCC is then the correlation coefficient between the input residuals
        and the output residuals (Note all data is rank transformed).
        :param result:
        :param plots:
        :return:
        """
        # Linear regression model
        regr = linear_model.LinearRegression()
        # Column headers
        cols = self.ranked_lhs_params.columns
        # Turn data into numpy arrays
        param_data = numpy.asarray(self.ranked_lhs_params)
        result_data = numpy.asarray(self.ranked_lhs_results[result]).reshape((self._stratifications,1))
        # Number of parameters
        k = param_data.shape[1]

        prccs = {}
        for i in range(k):
            label = cols[i]
            # Create a truth dictionaries to split the parameter_data into the parameter of interest and all other
            # parameters
            col = [j == i for j in range(k)]
            # Numpy array of just the parameter
            param = param_data[:, col]
            remaining_col = numpy.logical_not(col)
            # Numpy array of all other parameters
            remaining_param = param_data[:, remaining_col]

            # Fit a linear regression model between the remaining parameters and the parameter
            regr.fit(remaining_param, param)
            # Use to construct a line
            linreg_param = regr.predict(remaining_param)
            # Calculate residuals
            param_resid = param - linreg_param

            # Fit a linear regression model between the remaining parameters and the parameter
            regr.fit(remaining_param, result_data)
            # Use to construct a line
            linreg_result = regr.predict(remaining_param)
            # Calculate residuals
            result_resid = result_data - linreg_result

            # Determine correlation between residuals
            corr, p = stats.pearsonr(param_resid, result_resid)
            prccs[label] = (corr, p)

            # Draw PRCC plots
            if plots:
                title = "PRCC(" + label + ',' + result + "): " + str(corr) + '\n p =' + str(p)
                filename = LatinHypercubeSensitivityAnalyser.PRCC_FOLDER + label + "_" + result + ".png"
                self._create_scatter_plot(param_resid, result_resid, title, filename, label, result, False)

        return prccs
