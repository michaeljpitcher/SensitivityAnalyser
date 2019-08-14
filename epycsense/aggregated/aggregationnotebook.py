from epyc import *

import numpy
from pandas import DataFrame


class AggregationJSONNotebook(JSONLabNotebook):
    # TODO - using a JSON notebook, could be just a notebook
    def __init__(self, name, create=True, description=None):
        JSONLabNotebook.__init__(self, name, create, description)
        # TODO - aggregated results work a little differently to results: it's a list not a dict as we don't use
        #  parameters as dict index, and we don't have metadata (since that only exists for individual repetitions)
        self._aggregated_results = []

    def uncertain_parameters(self):
        df = self.dataframe_aggregated_parameters()
        return [q for q in df if len(set(df[q])) > 1]

    def certain_parameters(self):
        df = self.dataframe_aggregated_parameters()
        return [q for q in df if len(set(df[q])) == 1]

    def result_keys(self):
        return self.dataframe_aggregated_results().columns

    def _get_results_for_row(self, repetition_data):
        """
        Given the repetition data for a parameter sample, aggregate the data. Default is to average the results if
        > 1 repetitions, or just leave data if 1 repetition. Can be overridden to generate other outputs.
        :param repetition_data:
        :return:
        """
        if len(repetition_data) == 1:
            return repetition_data
        else:
            return {r: numpy.mean([rep[Experiment.RESULTS][r] for rep in repetition_data])
                    for r in repetition_data[0][Experiment.RESULTS]}

    def addResult( self, result, jobids = None ):
        if isinstance(result[Experiment.RESULTS], list):
            aggregation = self._get_results_for_row(result[Experiment.RESULTS])
            self._aggregated_results.append({Experiment.PARAMETERS: result[Experiment.PARAMETERS],
                                             Experiment.RESULTS: aggregation})
        JSONLabNotebook.addResult(self, result, jobids)

    def results_aggregated(self):
        return self._aggregated_results

    def dataframe_aggregated(self):
        def extract(r):
            rd = r[Experiment.PARAMETERS].copy()
            rd.update(r[Experiment.RESULTS])
            return rd

        records = [r for r in map(extract, self._aggregated_results) if r is not None]
        return DataFrame.from_records(records)

    def dataframe_aggregated_parameters(self):
        def extract(r):
            rd = r[Experiment.PARAMETERS].copy()
            return rd

        records = [r for r in map(extract, self._aggregated_results) if r is not None]
        return DataFrame.from_records(records)

    def dataframe_aggregated_results(self):
        def extract(r):
            rd = r[Experiment.RESULTS].copy()
            return rd

        records = [r for r in map(extract, self._aggregated_results) if r is not None]
        return DataFrame.from_records(records)
