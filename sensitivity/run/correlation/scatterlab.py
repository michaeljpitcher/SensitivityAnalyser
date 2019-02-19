import epyc
import numpy
from progressepyc import ProgressLab


class ScatterLab(ProgressLab):
    """
    An epyc lab, which creates a sample space whereby each possible value of an uncertain parameter is run against the
    baseline values of all other parameters (baseline value for uncertain parameters is the middle value of all given
    values)

    e.g.
    Param ALPHA has values [1,2,3,4,5]
    Param BETA has values [7,9,11,13,15]
    Param GAMMA has value [22]

    Samples will be
    [ALPHA: 1, BETA: 11, GAMMA: 22], [ALPHA: 2, BETA: 11, GAMMA: 22],
    [ALPHA: 4, BETA: 11, GAMMA: 22], [ALPHA: 5, BETA: 11, GAMMA: 22], <- ALPHA varies, others fixed
    [ALPHA: 3, BETA:  7, GAMMA: 22], [ALPHA: 3, BETA:  9, GAMMA: 22],
    [ALPHA: 3, BETA: 13, GAMMA: 22], [ALPHA: 3, BETA: 15, GAMMA: 22], <- BETA varies, others fixed
    [ALPHA: 3, BETA: 11, GAMMA: 22]

    Allows the creation of scatter diagrams that demonstrate how an output varies given change of an input. Only really
    for illustrative purposes, to determine if relationship between parameter and output is monotonic/linear/etc. after
    which more thorough sensitivity analyses can be run.

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
            return self._scatter_samples()

    def _scatter_samples(self):
        """Internal method to generate the sample of all parameter values, creating the parameter space for the
        experiment. Each value within the parameter range of an uncertain parameter is sampled and joined with the
        middle value of all other samples.

        :returns: list of dicts"""
        baseline_values = {}

        param_samples = []
        # Determine if each parameter is certain or uncertain
        for param, param_range in self._parameters.iteritems():
            if len(param_range) > 1:
                baseline_values[param] = param_range[len(param_range)/2]
            else:
                # Only one value so parameter is certain
                baseline_values[param] = param_range[0]

        # For every parameter, look all through all possible values
        for param, param_range in self._parameters.iteritems():
            for v in param_range:
                # Don't use the baseline value (all params at baseline is included below)
                if v != baseline_values[param]:
                    # Copy the baseline values
                    sample = baseline_values.copy()
                    # Overwrite the current param with the value
                    sample[param] = v
                    param_samples.append(sample)

        # Add sample with all parameters at baseline
        param_samples.append(baseline_values.copy())

        # Return the parameter space
        return param_samples
