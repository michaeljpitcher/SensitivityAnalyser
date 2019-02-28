import epyc
import json
import numpy
import pandas


class ResultSet(object):

    def __init__(self, parameter_set=None):
        self._parameter_set = parameter_set
        self._description = self._param_samples = self._result_data = self._result_keys = None

    def __len__(self):
        return len(self._result_data)

    def parameter_samples(self, parameter=None):
        if parameter:
            return self._param_samples[parameter]
        else:
            return self._param_samples

    def result_keys(self):
        return self._result_keys

    def result_data(self, result_key=None):
        if result_key:
            return self._result_data[result_key]
        else:
            return self._result_data

    def load_epyc_results_from_json(self, filename):
        """
        Given filename for an epyc JSON output file, loads the data from that file. Data can be aggregated for cases
        of multiple repetitions
        :param filename:
        :return:
        """
        with open(filename) as data_file:
            json_data = json.load(data_file)
        # Set string value of this class to equal the epyc description
        self._description = json_data['description']
        # Load the results
        results = json_data[epyc.Experiment.RESULTS]
        parameter_samples = []
        result_data = []
        # Loop through all rows of JSON file
        for v in results.values():
            # Take the parameter sample from the first repetition (should be the same for all repetitions)
            parameter_samples.append(v[0][epyc.Experiment.PARAMETERS])
            # Obtain the relevant data from each repetition
            data_to_aggregate = []
            for rep_row in v:
                data_to_aggregate.append(self._get_data(rep_row[epyc.Experiment.RESULTS]))
            # Calculate the average
            aggregated_data = {k: numpy.average([r[k] for r in data_to_aggregate]) for k in data_to_aggregate[0].keys()}

            result_data.append(aggregated_data)
        # Store data as DataFrame objects
        self._param_samples = pandas.DataFrame(parameter_samples)
        self._result_data = pandas.DataFrame(result_data)
        self._result_keys = [c for c in self._result_data.columns]

    def _get_data(self, row):
        """
        Given a row (for a particular repetition of a parameter sample), return the necessary results. Default assumes
        that all recorded data is a single value for each result, and thus returns the row. Can be overridden in order
        to manage and/or aggregate more complicated data structures.
        :param row: Row of an epyc JSON output file
        :return: aggregated data
        """
        return row


class TimeSeriesResultSet(ResultSet):
    def __init__(self, parameter_set=None):
        self._timesteps = None
        ResultSet.__init__(self, parameter_set)

    def load_epyc_results_from_json(self, filename):
        """
        Given filename for an epyc JSON output file, loads the data from that file. Data can be aggregated for cases
        of multiple repetitions
        :param filename:
        :return:
        """
        with open(filename) as data_file:
            json_data = json.load(data_file)
        # Set string value of this class to equal the epyc description
        self._description = json_data['description']
        # Load the results
        results = json_data[epyc.Experiment.RESULTS]
        parameter_samples = []
        result_data = {}
        result_keys = []
        # Loop through all rows (param_variations) of JSON file
        for v in results.values():
            # Take the parameter sample from the first repetition (should be the same for all repetitions)
            parameter_samples.append(v[0][epyc.Experiment.PARAMETERS])
            # Obtain the relevant data from each repetition
            data_to_aggregate = []
            for rep_row in v:
                data_to_aggregate.append(self._get_data(rep_row[epyc.Experiment.RESULTS]))

            if not self._timesteps:
                self._timesteps = data_to_aggregate[0].keys()
                result_data = {t: [] for t in self._timesteps}
                result_keys = data_to_aggregate[0][self._timesteps[0]].keys()

            # Calculate the average
            aggregated_data = {t: {k: numpy.average([d[t][k] for d in data_to_aggregate]) for k in result_keys}
                               for t in self._timesteps}
            for t in self._timesteps:
                result_data[t].append(aggregated_data[t])

        # Store data as DataFrame objects
        self._param_samples = pandas.DataFrame(parameter_samples)
        self._result_data = {t: pandas.DataFrame(d) for t,d in result_data.iteritems()}
        self._result_keys = result_keys

    def result_data(self, timestep=None, result_key=None):

        if timestep is None and result_key is None:
            return self._result_data
        elif timestep is not None and result_key is None:
            return self._result_data[timestep]
        elif timestep is None and result_key is not None:
            return {t: d[result_key] for t, d in self._result_data.iteritems()}
        else:
            return self._result_data[timestep][result_key]

    def __len__(self):
        return len(self._result_data.values()[0])

    def timesteps(self):
        return self._timesteps