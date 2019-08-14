import epyc
from ..aggregated.aggregationnotebook import AggregationJSONNotebook


class EFASTJSONNotebook(AggregationJSONNotebook):
    def __init__(self, name, create, description):
        AggregationJSONNotebook.__init__(self, name, create, description)
