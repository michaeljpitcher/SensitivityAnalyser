from ..data import *
from ..evaluate import *
from scipy import stats
from sklearn import linear_model


class LHSAnalyser(object):
    def __init__(self, parameter_set, output_folder=None):
        if output_folder:
            self.output_folder = output_folder + '/'
        else:
            self.output_folder = ''
        self.param_data, self.result_data = \
            read_epyc_json_data(self.output_folder + LHS_FILENAME, parameter_set.uncertain_parameters.keys())
        self.uncertain_parameters = self.param_data.columns
        self.result_variables = self.result_data.columns
        self._stratifications = len(self.result_data[self.result_variables[0]])

    def get_all_pearson_correlation_coefficients(self):
        """
        Calculate all Pearson correlation coefficients
        :return:
        """
        coefficients = {}
        for p in self.uncertain_parameters:
            for r in self.result_variables:
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
        return stats.pearsonr(self.param_data[parameter], self.result_data[result])

    def get_all_spearman_rank_correlation_coefficients(self):
        """
        Calculate all Spearman Rank Correlation Coefficients
        :return:
        """
        coefficients = {}
        for p in self.uncertain_parameters:
            for r in self.result_variables:
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
        return stats.spearmanr(self.param_data[parameter], self.result_data[result])

    def get_all_prcc(self, plots=False):
        prccs = dict([(r, self.calculate_prcc(r, plots)) for r in self.result_variables])
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
        ranked_params = pandas.DataFrame.rank(self.param_data)
        ranked_results = pandas.DataFrame.rank(self.result_data)

        # Turn data into numpy arrays
        param_data = numpy.asarray(ranked_params)
        result_data = numpy.asarray(ranked_results[result]).reshape((self._stratifications,1))
        # Number of parameters
        k = param_data.shape[1]

        prccs = {}
        for i in range(k):
            label = self.uncertain_parameters[i]
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
                filename = self.output_folder + label + "_" + result
                create_scatter_plot(param_resid, result_resid, title, filename, label, result, False)

        return prccs