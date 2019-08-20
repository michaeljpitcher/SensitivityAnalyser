from epyc import *

import numpy
from pandas import DataFrame


class AggregationJSONNotebook(JSONLabNotebook):
    # TODO - using a JSON notebook, could be just a notebook
    def __init__(self, name, create=True, description=None):
        # TODO - aggregated results work a little differently to results: it's a list not a dict as we don't use
        #  parameters as dict index, and we don't have metadata (since that only exists for individual repetitions)
        self._aggregated_results = []
        self._parameters = []
        self._result_keys = []
        JSONLabNotebook.__init__(self, name, create, description)

    def uncertain_parameters(self):
        df = self.dataframe_aggregated()
        return [q for q in self._parameters if len(set(df[q])) > 1]

    def certain_parameters(self):
        df = self.dataframe_aggregated()
        return [q for q in self._parameters if len(set(df[q])) > 1]

    def result_keys(self):
        return self._result_keys

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

    def _record_aggregated_row(self, parameters, agg_results):
        self._parameters += [p for p in parameters if p not in self._parameters]
        self._result_keys += [r for r in agg_results if r not in self._result_keys]
        self._aggregated_results.append({Experiment.PARAMETERS: parameters,
                                         Experiment.RESULTS: agg_results})

    def addResult( self, result, jobids = None ):
        # TODO - doesn't work for non-repetitions
        # Make sure all repetitions have succeeded, can't aggregate if a failure has occurred
        if isinstance(result[Experiment.RESULTS], list) \
          and all([r[Experiment.METADATA]['status'] for r in result[Experiment.RESULTS]]):
            aggregation = self._get_results_for_row(result[Experiment.RESULTS])
            self._record_aggregated_row(result[Experiment.PARAMETERS], aggregation)
        JSONLabNotebook.addResult(self, result, jobids)

    def _load( self, fn ):
        JSONLabNotebook._load(self, fn)
        # TODO - probably doesn't work for non-repetitions
        for r in self._results.values():
            aggregation = self._get_results_for_row(r)
            self._record_aggregated_row(r[0][Experiment.PARAMETERS], aggregation)

    def results_aggregated(self):
        return self._aggregated_results

    def dataframe_aggregated(self):
        def extract(r):
            rd = r[Experiment.PARAMETERS].copy()
            rd.update(r[Experiment.RESULTS])
            return rd

        records = [r for r in map(extract, self._aggregated_results) if r is not None]
        return DataFrame.from_records(records)

    # def dataframe_aggregated_parameters(self):
    #     def extract(r):
    #         rd = r[Experiment.PARAMETERS].copy()
    #         return rd
    #
    #     records = [r for r in map(extract, self._aggregated_results) if r is not None]
    #     return DataFrame.from_records(records)
    #
    # def dataframe_aggregated_results(self):
    #     def extract(r):
    #         rd = r[Experiment.RESULTS].copy()
    #         return rd
    #
    #     records = [r for r in map(extract, self._aggregated_results) if r is not None]
    #     return DataFrame.from_records(records)
