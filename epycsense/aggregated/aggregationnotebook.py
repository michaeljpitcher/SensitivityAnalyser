from epyc import *

import numpy
from pandas import DataFrame


class AggregationJSONNotebook(JSONLabNotebook):
    def __init__(self, name, create=False, description=None):
        self._aggregated_results = []
        self._parameters = []
        self._result_keys = []
        JSONLabNotebook.__init__(self, name, create, description)

    def aggregate(self):
        # Warn if there's still results pending
        if self._pending:
            print "Warning: Some results are pending"

        for k,r in self._results.iteritems():
            params = r[0][Experiment.PARAMETERS]
            if not self._parameters:
                self._parameters = params.keys()
            results = [k[Experiment.RESULTS] for k in r]
            agg_result = self._aggregate_row(results)
            if not self._result_keys:
                self._result_keys = agg_result.keys()
            self._aggregated_results.append({Experiment.PARAMETERS: params,
                                             Experiment.RESULTS: agg_result})

    def aggregated_results(self):
        return self._aggregated_results

    def _aggregate_row(self, repetition_data):
        if len(repetition_data) == 1:
            return repetition_data
        else:
            return {rk: numpy.mean([rep[rk] for rep in repetition_data]) for rk in repetition_data[0].keys()}

    def dataframe_aggregated(self):
        if not self._aggregated_results:
            self.aggregate()

        def extract(r):
            rd = r[Experiment.PARAMETERS].copy()
            rd.update(r[Experiment.RESULTS])
            return rd

        records = [r for r in map(extract, self._aggregated_results) if r is not None]
        return DataFrame.from_records(records)

    def uncertain_parameters(self):
        df = self.dataframe_aggregated()
        return [q for q in self._parameters if len(set(df[q])) > 1]

    def certain_parameters(self):
        df = self.dataframe_aggregated()
        return [q for q in self._parameters if len(set(df[q])) == 1]

    def result_keys(self):
        return self._result_keys
