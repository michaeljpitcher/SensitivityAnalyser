import numpy
import pandas


class ParameterSet(object):

    DISTRIBUTION_UNIFORM = 'uniform'
    DISTRIBUTION_NORMAL = 'normal'

    def __init__(self):
        self.uncertain_parameters = {}
        self.certain_parameters = {}

    def add_certain_parameter(self, parameter, value):
        self.certain_parameters[parameter] = value

    def add_uncertain_parameter_uniform_dist(self, parameter, min_val, max_val):
        self.uncertain_parameters[parameter] = (min_val, max_val, ParameterSet.DISTRIBUTION_UNIFORM)

    def create_latin_hypercube_stratifications(self, stratifications):
        values = {}
        for p, (min_val, max_val, dist) in self.uncertain_parameters.iteritems():
            if dist == ParameterSet.DISTRIBUTION_UNIFORM:
                values[p] = numpy.linspace(min_val, max_val, stratifications)
            else:
                # TODO - normal distribution
                pass
        return pandas.DataFrame(values)
