import epyc
import numpy


class LatinHypercubeLab(epyc.Lab):
    """
    An epyc lab, which creates a latin hypercube of parameter values (as opposed to cross-product)
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
            return self._latin_hypercube_sample_matrix(ps)

    def _latin_hypercube_sample_matrix(self, ls):
        """Internal method to generate the latin Hypercube sample of all parameter
        values, creating the parameter space for the experiment. Each value within
        the parameter range of a parameter is sampled only once.

        :param ls: an array of parameter names
        :returns: list of dicts"""
        # TODO: we assume that parameters have been stratified and values within each stratification chosen

        # Ensure all parameters have the same number of options
        stratifications = len(self._parameters[ls[0]])
        assert all(len(q) == stratifications for q in self._parameters.values()), "All parameters must have equal " \
                                                                                  "number of stratifications for " \
                                                                                  "Latin Hypercube sampling"
        ds = []
        chosen = {}
        # Shuffle the range for each parameter
        for p,v in self._parameters.iteritems():
            chosen[p] = numpy.random.choice(v,len(v),replace=False)
        # Assign a parameter set based on the shuffled values
        for i in range(stratifications):
            values = {}
            for p in self._parameters:
                values[p] = chosen[p][i]
            ds.append(values)

        # return the complete parameter space
        return ds
