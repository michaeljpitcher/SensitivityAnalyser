import pandas
from ..visual.scatterplot import plot_scatter_graph
from scipy import stats
from sklearn import linear_model
import numpy


class LHSAnalyser(object):
    def __init__(self, parameter_set, result_set, output_folder=None):
        if output_folder:
            self.output_folder = output_folder + '/'
        else:
            self.output_folder = ''
        self.uncertain_parameters = parameter_set.uncertain_parameters().keys()
        self.result_set = result_set
        self._stratifications = len(self.result_set)

    def get_all_pearson_correlation_coefficients(self):
        """
        Calculate all Pearson correlation coefficients
        :return:
        """
        coefficients = {}
        for p in self.uncertain_parameters:
            for r in self.result_set.result_keys():
                    coefficients[(p, r)] = self.get_pearson_correlation_coefficient(p, r)
        return coefficients

    def get_pearson_correlation_coefficient(self, parameter, result):
        """
        For linear trends
        :param parameter:
        :param result:
        :param time
        :return:
        """
        assert parameter in self.uncertain_parameters
        return stats.pearsonr(self.result_set.parameter_samples(parameter), self.result_set.result_data(result))

    def get_all_spearman_rank_correlation_coefficients(self):
        """
        Calculate all Spearman Rank Correlation Coefficients
        :return:
        """
        coefficients = {}
        for p in self.uncertain_parameters:
            for r in self.result_set.result_keys():
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
        return stats.spearmanr(self.result_set.parameter_samples(parameter), self.result_set.result_data(result))

    def get_all_prcc(self, plots=False):
        prccs = []
        for result in self.result_set.result_keys():
            for param in self.uncertain_parameters:
                prccs.append(((param, result), self.calculate_prcc(result, param, plot=plots)))
        return prccs

    def calculate_prcc(self, result, parameter, plot=False):
        """
        Calculate a partial rank correlation coefficient (PRCC) value for an uncertain parameters against the output.

        Partial correlation characterizes the linear relationship between an input and an output after the linear
        effects of the remaining inputs on the output are discounted. This is calculated by constructing a linear
        regression model between the input and the remaining inputs, and calculating the residuals between the input
        values and the model. Similarly, a linear regression model between the output and the remaining input is
        created and residuals are calculated. The PRCC is then the correlation coefficient between the input residuals
        and the output residuals (Note all data is rank transformed).
        :param result:
        :param parameter:
        :param plot:
        :return:
        """
        # Linear regression model
        regr = linear_model.LinearRegression()

        ranked_params = pandas.DataFrame.rank(self.result_set.parameter_samples())
        ranked_results = pandas.DataFrame.rank(self.result_set.result_data())

        # Turn data into numpy arrays
        all_params_data = numpy.asarray(ranked_params)
        result_data = numpy.asarray(ranked_results[result]).reshape((self._stratifications,1))
        # Number of parameters
        k = all_params_data.shape[1]

        # Create truth dictionaries to split the parameter_data into the parameter of interest and all other
        # parameters
        # Dictionary indicating where parameter lies (true for param, false for all others)
        param_col_truth_table = [ranked_params.columns[j] == parameter for j in range(k)]

        # Numpy array of just the parameter
        param_data = all_params_data[:, param_col_truth_table]
        # Numpy array of all other parameters
        remaining_param_data = all_params_data[:, numpy.logical_not(param_col_truth_table)]

        # Fit a linear regression model between the remaining parameters and the parameter
        regr.fit(remaining_param_data, param_data)
        # Use to construct a line
        linreg_param = regr.predict(remaining_param_data)
        # Calculate residuals
        param_resid = param_data - linreg_param

        # Fit a linear regression model between the remaining parameters and the result
        regr.fit(remaining_param_data, result_data)
        # Use to construct a line
        linreg_result = regr.predict(remaining_param_data)
        # Calculate residuals
        result_resid = result_data - linreg_result

        # Determine correlation between residuals
        corr, p = stats.pearsonr(param_resid, result_resid)

        # Draw PRCC plots
        if plot:
            title = "PRCC(" + parameter + ',' + result + "): " + str(corr) + '\n p =' + str(p)
            filename = self.output_folder + parameter + "_" + result
            plot_scatter_graph(param_resid, result_resid, title, parameter, result, True, filename, False)

        return (corr, p)