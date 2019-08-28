import epyc
import numpy
import scipy.stats as stats

UNIFORM_DISTRIBUTION = 'uniform_distribution'
NORMAL_DISTRIBUTION = 'normal_distribution'
LOGNORMAL_DISTRIBUTION = 'lognormal_distribution'


def lhs_samples(parameters, stratifications):
    uncertain_params = {}
    certain_params = {}

    parameters['dummy'] = [0,10,UNIFORM_DISTRIBUTION]

    # Linspace 0-1, split into the number of stratfications sections
    d = numpy.linspace(0,1,stratifications+2)[1:-1]

    # Determine if each parameter is certain or uncertain
    for param, param_range in parameters.iteritems():
        if len(param_range) > 1:
            v1, v2, dist = param_range
            if dist == UNIFORM_DISTRIBUTION:
                values = [v1 + (v2 - v1) * y for y in d]
            elif dist == NORMAL_DISTRIBUTION:
                values = stats.norm.ppf(d, v1, v2)
            elif dist == LOGNORMAL_DISTRIBUTION:
                values = stats.lognorm.ppf(d, v1, v2)
            else:
                raise Exception("Invalid distribtion")
            uncertain_params[param] = values
        else:
            # Only one value so parameter is certain
            certain_params[param] = param_range[0]

    # Create the samples
    param_samples = []

    # Shuffle the range for each uncertain parameter
    for p, param_range in uncertain_params.iteritems():
        numpy.random.shuffle(uncertain_params[p])

    # Assign a parameter set based on the shuffled values
    for i in range(stratifications):
        sample = {p: v[i] for (p,v) in uncertain_params.iteritems()}
        # Set the certain params
        sample.update(certain_params)
        param_samples.append(sample)

    # return the complete parameter space
    return param_samples


class LatinHypercubeLab(epyc.Lab):

    def __init__(self, notebook):
        self._stratifications = 0
        epyc.Lab.__init__(self, notebook)

    def set_stratifications(self, value):
        self._stratifications = value

    def parameterSpace( self ):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.

        :returns: the parameter space as a list of dicts"""
        ps = self.parameters()
        if len(ps) == 0:
            return []
        else:
            assert self._stratifications > 0, "Must set stratification number"
            return lhs_samples(self._parameters, self._stratifications)


class LatinHypercubeClusterLab(epyc.ClusterLab):
    def __init__(self, notebook, profile, debug=False):
        epyc.ClusterLab.__init__(self, notebook, profile=profile, debug=debug)

    def set_stratifications(self, value):
        self._stratifications = value

    def parameterSpace(self):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.

        :returns: the parameter space as a list of dicts"""
        ps = self.parameters()
        if len(ps) == 0:
            return []
        else:
            assert self._stratifications > 0, "Must set stratification number"
            return lhs_samples(self._parameters, self._stratifications)
