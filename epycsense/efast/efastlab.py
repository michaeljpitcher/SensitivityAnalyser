import epyc
import math
import numpy as np
import scipy.stats as stats

UNIFORM_DISTRIBUTION = 'uniform_distribution'
TRIANGE_DISTRIBUTION = 'triangle_distribution'
NORMAL_DISTRIBUTION = 'normal_distribution'
LOGNORMAL_DISTRIBUTION = 'lognormal_distribution'


def efast_sample_matrix(sample_number, interference, parameters):
    """
    Generate model inputs for the extended Fourier Amplitude Sensitivity Test (FAST).

    Returns a NumPy matrix containing the model inputs required by the Fourier
    Amplitude sensitivity test.  The resulting matrix contains N rows and K
    columns, where K is the number of parameters.

    Code taken from SALib (due to an inability to install) https://salib.readthedocs.io/en/latest/

    :param sample_number: Number of samples
    :param interference: The interference parameter, i.e., the number of harmonics to sum in the
        Fourier series decomposition
    :param parameters:
    :return:
    """
    if sample_number <= 4 * interference ** 2:
        raise ValueError("""
            Sample size N > 4M^2 is required. M=4 by default.""")

    uncertain_params = {p: v for (p, v) in parameters.iteritems() if len(v) > 1}
    certain_params = {p: v[0] for (p, v) in parameters.iteritems() if len(v) == 1}

    k = len(uncertain_params)

    omega = np.zeros([k])
    omega[0] = math.floor((sample_number - 1.0) / (2.0 * interference))

    # Maximum value of complimentary frequencies
    max_omega_comp = math.floor(omega[0] / (2.0 * interference))

    # Determine complimentary frequencies
    if max_omega_comp >= (k - 1):
        # Max is greater than the remainder number of frequencies, so list is (1, max)
        omega[1:] = np.floor(np.linspace(1, max_omega_comp, k - 1))
    else:
        # Max is less than the remainder number of frequencies, so will need to repeat values
        omega[1:] = np.arange(k - 1) % max_omega_comp + 1

    # Discretisation of the frequency space, s (vector of length samples with values 0-2pi)
    s = (2 * math.pi / sample_number) * np.arange(sample_number)

    # Transformation to get points in the X space
    x = np.zeros([sample_number * k, k])
    omega2 = np.zeros([k])

    # Taking each parameter as the parameter of interest
    for parameter_of_interest in range(k):
        # Assign the parameter of interest frequency to this parameter
        omega2[parameter_of_interest] = omega[0]
        # Assign the remaining frequencies to all other parameters
        # Get list [0:k] excluding the parameter of interest
        idx = list(range(parameter_of_interest)) + list(range(parameter_of_interest + 1, k))
        omega2[idx] = omega[1:]

        # Run numbers for this parameters
        run_numbers = range(parameter_of_interest * sample_number, (parameter_of_interest + 1) * sample_number)

        # TODO - resamples?

        # random phase shift on [0, 2pi) following Saltelli et al.
        # Technometrics 1999
        phi = 2 * math.pi * np.random.rand()

        # Assign a value (in range [0,1]) to each parameter for each run based on their frequency and phase shift
        for param in range(k):
            g = 0.5 + (1 / math.pi) * np.arcsin(np.sin(omega2[param] * s + phi))
            x[run_numbers, param] = g

    # Convert 0-1 values into values within the parameter range, based on minima, maxima and distribution. Then
    # add to the sample list
    samples = []
    # Process a sample at a time
    for j in range(x.shape[0]):
        # Initialise the sample as the certain parameter values
        sample = certain_params.copy()
        # For each uncertain parameter
        for p in range(x.shape[1]):
            param = uncertain_params.keys()[p]
            minimum_val = uncertain_params[param][0]
            maximum_val = uncertain_params[param][1]
            dist = uncertain_params[param][2]
            val = x[j, p]
            if dist == TRIANGE_DISTRIBUTION:
                val = stats.triang.ppf(val, c=maximum_val, scale=minimum_val, loc=0)
            elif dist == UNIFORM_DISTRIBUTION:
                val = val * (maximum_val - minimum_val) + minimum_val
            elif dist == NORMAL_DISTRIBUTION:
                val = stats.norm.ppf(val, loc=minimum_val, scale=maximum_val)
            # lognormal distribution (ln-space, not base-10)
            # paramters are ln-space mean and standard deviation
            elif dist == LOGNORMAL_DISTRIBUTION:
                # checking for valid parameters
                val = np.exp(stats.norm.ppf(val, loc=minimum_val, scale=maximum_val))
            else:
                valid_dists = ['unif', 'triang', 'norm', 'lognorm']
                raise ValueError('Distributions: choose one of %s' % ", ".join(valid_dists))
            sample[param] = val
        samples.append(sample)

    return samples


class EFASTLab(epyc.Lab):
    def __init__(self, notebook):
        epyc.Lab.__init__(self, notebook)
        self._sample_number = 0
        self._interference = 0

    def set_sample_number(self, samples):
        self._sample_number = samples

    def set_interference_factor(self, factor):
        self._interference = factor

    def parameterSpace(self):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.
        :returns: the parameter space as a list of dicts"""
        if len(self._parameters) == 0:
            return []
        else:
            assert (self._sample_number > 0), "Sample number invalid: {0}. Set using {1}()"\
                .format(self._sample_number, self.set_sample_number.__name__)
            assert (self._interference > 0), "Interference value invalid: {0}. Set using {1}()"\
                .format(self._interference, self.set_interference_factor.__name__)
            return efast_sample_matrix(self._sample_number, self._interference, self._parameters)


class EFASTClusterLab(epyc.ClusterLab):
    def __init__(self, notebook):
        epyc.ClusterLab.__init__(self, notebook)
        self._sample_number = 0
        self._interference = 0

    def set_sample_number(self, samples):
        self._sample_number = samples

    def set_interference_factor(self, factor):
        self._interference = factor

    def parameterSpace(self):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.
        :returns: the parameter space as a list of dicts"""
        if len(self._parameters) == 0:
            return []
        else:
            assert self._sample_number > 0, self._interference > 0
            return efast_sample_matrix(self._sample_number, self._interference, self._parameters)
