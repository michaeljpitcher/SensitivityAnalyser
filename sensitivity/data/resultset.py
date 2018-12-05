import json
import epyc
import numpy
import pandas


class ResultSet(object):
    def __init__(self, parameter_set):
        self._parameter_set = parameter_set
        self._param_data = None
        self._result_keys = None
        self._result_data = None
        self._length = 0

    def __len__(self):
        return self._length

    def parameter_data(self, param_key=None):
        if param_key:
            return self._param_data[param_key]
        else:
            return self._param_data

    def result_data(self, result_key=None):
        if result_key:
            return self._result_data[result_key]
        else:
            return self._result_data

    def result_keys(self):
        return self._result_keys

    def load_data_from_json(self, filename):
        with open(filename) as data_file:
            data = json.load(data_file)[epyc.Experiment.RESULTS].values()
        param_data = {p: [] for p in self._parameter_set.uncertain_parameters}
        self._result_keys = data[0][0][epyc.Experiment.RESULTS].keys()
        result_data = {r: [] for r in self._result_keys}
        # TODO - this assumes result data is just numeric values
        for param_variation in data:
            self._length += 1
            for p in self._parameter_set.uncertain_parameters:
                param_data[p].append(param_variation[0][epyc.Experiment.PARAMETERS][p])
            for k in self._result_keys:
                if len(param_variation) > 1:
                    # Multiple repetitions so we need to average
                    repetition_results = [p[epyc.Experiment.RESULTS] for p in param_variation]
                    result_data[k].append(numpy.average([rep[k] for rep in repetition_results]))
                else:
                    # Only 1 repetition (possibly from a deterministic model) so just pass it through
                    result_data[k].append(param_variation[0][epyc.Experiment.RESULTS][k])

        self._param_data = pandas.DataFrame(param_data)
        self._result_data = pandas.DataFrame(result_data)


class TimeSeriesResultSet(ResultSet):
    def __init__(self, parameter_set):
        ResultSet.__init__(self, parameter_set)
        self._timesteps = []

    def timesteps(self):
        return self._timesteps

    def load_data_from_json(self, filename):
        with open(filename) as data_file:
            data = json.load(data_file)[epyc.Experiment.RESULTS].values()
        param_data = {p: [] for p in self._parameter_set.uncertain_parameters}
        self._timesteps = data[0][0][epyc.Experiment.RESULTS].keys()
        self._result_keys = data[0][0][epyc.Experiment.RESULTS][self._timesteps[0]].keys()
        result_data = {t: {r: [] for r in self._result_keys} for t in self._timesteps}

        for param_variation in data:
            self._length += 1
            for p in self._parameter_set.uncertain_parameters:
                param_data[p].append(param_variation[0][epyc.Experiment.PARAMETERS][p])
            for k in self._result_keys:
                if len(param_variation) > 1:
                    # Multiple repetitions so we need to average
                    repetition_results = [p[epyc.Experiment.RESULTS] for p in param_variation]
                    for t in self._timesteps:
                        result_data[t][k].append(numpy.average([rep[t][k] for rep in repetition_results]))
                else:
                    for t in self._timesteps:
                        # Only 1 repetition (possibly from a deterministic model) so just pass it through
                        result_data[t][k].append(param_variation[0][epyc.Experiment.RESULTS][t][k])

        for t in self._timesteps:
            result_data[t] = pandas.DataFrame(result_data[t])
        self._param_data = pandas.DataFrame(param_data)
        self._result_data = result_data

    def result_data(self, time=None, result_key=None):
        # Specific result and time
        if result_key and time:
            return self._result_data[time][result_key]
        # Just a time
        elif time:
            return self._result_data[time]
        # Just a result
        elif result_key:
            return {t: self._result_data[t][result_key] for t in self._timesteps}
        # Everything
        else:
            return self._result_data
