from ..data import *
from ..evaluate import *
from scipy import stats
from sklearn import linear_model


class LHSAnalyser(object):
    def __init__(self, parameter_set, output_folder=None, time_series=False):
        if output_folder:
            self.output_folder = output_folder + '/'
        else:
            self.output_folder = ''
        self.uncertain_parameters = parameter_set.uncertain_parameters.keys()
        if time_series:
            self.result_set = TimeSeriesResultSet(parameter_set)
        else:
            self.result_set = ResultSet(parameter_set)
        self.result_set.load_data_from_json(self.output_folder + LHS_FILENAME)
        self._stratifications = len(self.result_set)

    def get_all_pearson_correlation_coefficients(self):
        """
        Calculate all Pearson correlation coefficients
        :return:
        """
        coefficients = {}
        for p in self.uncertain_parameters:
            if isinstance(self.result_set, TimeSeriesResultSet):
                for t in self.result_set.timesteps():
                    for r in self.result_set.result_keys():
                        coefficients[(t,p,r)] = self.get_pearson_correlation_coefficient(p, r, t)
            else:
                for r in self.result_set.result_keys():
                    coefficients[(p, r)] = self.get_pearson_correlation_coefficient(p, r)
        return coefficients

    def get_pearson_correlation_coefficient(self, parameter, result, time=None):
        """
        For linear trends
        :param parameter:
        :param result:
        :param time
        :return:
        """
        assert parameter in self.uncertain_parameters
        if time:
            return stats.pearsonr(self.result_set.parameter_data(parameter), self.result_set.result_data(time, result))
        else:
            return stats.pearsonr(self.result_set.parameter_data(parameter), self.result_set.result_data(result))

    def get_all_spearman_rank_correlation_coefficients(self):
        """
        Calculate all Spearman Rank Correlation Coefficients
        :return:
        """
        coefficients = {}
        for p in self.uncertain_parameters:
            if isinstance(self.result_set, TimeSeriesResultSet):
                for t in self.result_set.timesteps():
                    for r in self.result_set.result_keys():
                        coefficients[(t,p,r)] = self.get_spearman_rank_correlation_coefficient(p, r, t)
            else:
                for r in self.result_set.result_keys():
                    coefficients[(p, r)] = self.get_spearman_rank_correlation_coefficient(p, r)
        return coefficients

    def get_spearman_rank_correlation_coefficient(self, parameter, result, time=None):
        """
        For non-linear monotonic trends
        :param parameter:
        :param result:
        :return:
        """
        assert parameter in self.uncertain_parameters
        if time:
            return stats.spearmanr(self.result_set.parameter_data(parameter), self.result_set.result_data(time, result))
        else:
            return stats.spearmanr(self.result_set.parameter_data(parameter), self.result_set.result_data(result))

    def get_all_prcc(self, plots=False):
        prccs = []
        for result in self.result_set.result_keys():
            for param in self.uncertain_parameters:
                if isinstance(self.result_set, TimeSeriesResultSet):
                    for time in self.result_set.timesteps():
                        prccs.append(((time, param, result), self.calculate_prcc(result, param, time=time, plot=plots)))
                else:
                    prccs.append(((param, result), self.calculate_prcc(result, param, plot=plots)))
        return prccs

    def calculate_prcc(self, result, parameter, time=None, plot=False):
        """
        Calculate a partial rank correlation coefficient (PRCC) value for an uncertain parameters against the output.

        Partial correlation characterizes the linear relationship between an input and an output after the linear
        effects of the remaining inputs on the output are discounted. This is calculated by constructing a linear
        regression model between the input and the remaining inputs, and calculating the residuals between the input
        values and the model. Similarly, a linear regression model between the output and the remaining input is
        created and residuals are calculated. The PRCC is then the correlation coefficient between the input residuals
        and the output residuals (Note all data is rank transformed).
        :param result:
        :param param:
        :param time:
        :param plots:
        :return:
        """
        # Linear regression model
        regr = linear_model.LinearRegression()

        ranked_params = pandas.DataFrame.rank(self.result_set.parameter_data())
        if time:
            ranked_results = pandas.DataFrame.rank(self.result_set.result_data(time=time))
        else:
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
            if time:
                filename = filename + "_" + time
                title = title + 't =' + time
            create_scatter_plot(param_resid, result_resid, title, filename, parameter, result, False)

        return (corr, p)