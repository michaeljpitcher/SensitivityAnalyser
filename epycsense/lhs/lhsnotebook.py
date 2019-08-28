from epyc import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy
import scipy.stats as stats
from pandas import DataFrame
from ..aggregated.aggregationnotebook import AggregationJSONNotebook

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt


class LatinHypercubeJSONNotebook(AggregationJSONNotebook):
    def __init__(self, name, create=True, description=None):
        AggregationJSONNotebook.__init__(self, name, create, description)

    def get_all_pearson_correlation_coefficients(self):
        pccs = {}
        for p in self.uncertain_parameters():
            for r in self.result_keys():
                pccs[(p,r)] = self.get_pearson_correlation_coefficient(p,r)
        return pccs

    def get_pearson_correlation_coefficient(self, parameter, result):
        """
        For linear trends
        :param parameter:
        :param result:
        :return:
        """
        df = self.dataframe_aggregated()
        assert parameter in df.columns and result in df.columns, "{0} {1}".format(parameter, df.columns)
        return stats.pearsonr(df[parameter], df[result])

    def get_all_spearman_rank_correlation_coefficients(self):
        """
        Calculate all Spearman Rank Correlation Coefficients
        :return:
        """

        srccs = {}
        for p in self.uncertain_parameters():
            for r in self.result_keys():
                srccs[(p, r)] = self.get_spearman_rank_correlation_coefficient(p, r)
        return srccs

    def get_spearman_rank_correlation_coefficient(self, parameter, result):
        """
        For non-linear monotonic trends
        :param parameter:
        :param result:
        :return:
        """
        df = self.dataframe_aggregated()
        assert parameter in df.columns and result in df.columns
        return stats.spearmanr(df[parameter], df[result])

    def get_all_prcc(self):
        prccs = {}
        for p in self.uncertain_parameters():
            for r in self.result_keys():
                prccs[(p, r)] = self.calculate_prcc(p, r)
        return prccs

    def calculate_prcc(self, parameter, result, plot=False):
        """
        Calculate a partial rank correlation coefficient (PRCC) value for uncertain parameters against the output.

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

        df = self.dataframe_aggregated()

        ranked_params = DataFrame.rank(df[self.uncertain_parameters()])
        ranked_results = DataFrame.rank(df[self._result_keys])

        # Turn data into numpy arrays

        all_params_data = numpy.asarray(ranked_params)
        result_data = numpy.asarray(ranked_results[result]).reshape((len(ranked_results),1))

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

        return (corr, p)
