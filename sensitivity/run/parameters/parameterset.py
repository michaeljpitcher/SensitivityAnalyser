import numpy
import pandas


class ParameterSet(object):
    DISTRIBUTION_UNIFORM = 'uniform'

    def __init__(self):
        self._certain_parameters = {}
        self._uncertain_parameters = {}

    def certain_parameters(self):
        return self._certain_parameters

    def uncertain_parameters(self):
        return self._uncertain_parameters

    def add_parameter(self, parameter, values):
        if isinstance(values, list):
            self._uncertain_parameters[parameter] = values
        else:
            self._certain_parameters[parameter] = values

    def create_latin_hypercube_stratifications(self, stratifications):
        values = {}
        assert self.uncertain_parameters, "No uncertain parameters"
        for p, (min_val, max_val, dist) in self._uncertain_parameters.iteritems():
            if dist == ParameterSet.DISTRIBUTION_UNIFORM:
                values[p] = numpy.linspace(min_val, max_val, stratifications)
            else:
                # TODO - normal distribution
                raise NotImplementedError
        return pandas.DataFrame(values)

