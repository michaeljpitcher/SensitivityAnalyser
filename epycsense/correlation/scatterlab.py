import epyc


def scatter_samples(parameters):
    """Internal method to generate the sample of all parameter values, creating the parameter space for the
    experiment. Each value within the parameter range of an uncertain parameter is sampled and joined with the
    middle value of all other samples.

    :returns: list of dicts"""
    baseline_values = {}

    param_samples = []
    # Determine if each parameter is certain or uncertain
    for param, param_range in parameters.iteritems():
        if len(param_range) > 1:
            baseline_values[param] = param_range[len(param_range) / 2]
        else:
            # Only one value so parameter is certain
            baseline_values[param] = param_range[0]

    # For every parameter, look all through all possible values
    for param, param_range in parameters.iteritems():
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


class ScatterLab(epyc.Lab):
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
            return scatter_samples(self._parameters)


class ScatterClusterLab(epyc.ClusterLab):
    def __init__(self, notebook):
        epyc.ClusterLab.__init__(self, notebook)

    def parameterSpace( self ):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.

        :returns: the parameter space as a list of dicts"""
        ps = self.parameters()
        if len(ps) == 0:
            return []
        else:
            return scatter_samples(self._parameters)
