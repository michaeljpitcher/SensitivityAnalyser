import epyc
import numpy
from progressepyc import ProgressLab


class LatinHypercubeLab(ProgressLab):
    """
    An epyc lab, which creates a latin hypercube of uncertain parameter values (as opposed to cross-product)
    """
    def __init__(self, notebook):
        epyc.Lab.__init__(self, notebook)

    def parameterSpace( self ):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.

        :returns: the parameter space as a list of dicts"""
        ps = self.parameters()
        if len(ps) == 0:
            return []
        else:
            return self._latin_hypercube_sample_matrix()

    def _latin_hypercube_sample_matrix(self):
        """Internal method to generate the latin Hypercube sample of all parameter
        values, creating the parameter space for the experiment. Each value within
        the parameter range of an uncertain parameter is sampled only once.

        :param ls: an array of parameter names
        :returns: list of dicts"""
        uncertain_params = {}
        certain_params = {}
        stratifications = 0

        # Determine if each parameter is certain or uncertain
        for param, param_range in self._parameters.iteritems():
            if len(param_range) > 1:
                # Check that the stratifications of this uncertain parameter range match previous uncertain parameters
                if not stratifications:
                    stratifications = len(param_range)
                else:
                    assert len(param_range) == stratifications, "All uncertain parameters must have equal number of " \
                                                                "stratifications for Latin Hypercube sampling"
                uncertain_params[param] = param_range
            else:
                # Only one value so parameter is certain
                certain_params[param] = param_range[0]

        # Create the samples
        param_samples = []
        uncertain_values = {}
        # Shuffle the range for each uncertain parameter
        for p,param_range in uncertain_params.iteritems():
            uncertain_values[p] = numpy.random.choice(param_range,len(param_range),replace=False)
        # Assign a parameter set based on the shuffled values
        for i in range(stratifications):
            sample = {}
            for p in uncertain_params:
                sample[p] = uncertain_values[p][i]
            # Set the certain params
            for p, value in certain_params.iteritems():
                sample[p] = value
            param_samples.append(sample)

        # return the complete parameter space
        return param_samples
