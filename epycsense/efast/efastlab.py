import epyc
import math
import numpy as np
import scipy.stats as stats

UNIFORM_DISTRIBUTION = 'uniform_distribution'
NORMAL_DISTRIBUTION = 'normal_distribution'
LOGNORMAL_DISTRIBUTION = 'lognormal_distribution'

RUN_NUMBER = 'run_number'
PARAMETER_OF_INTEREST = 'parameter_of_interest'

# Reference material for EFAST:
#
# Cukier RI, Fortuin CM, Shuler KE, Petschek AG, Schaibly JH.
# "Study of the sensitivity of coupled reaction systems to uncertainties in rate coefficients. I Theory."
# J Chem Phys 1973; 59: 3873-8. doi:10.1063/1.1680571
#
# Saltelli A, Tarantola S, Chan KP-S.
# "A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output."
# Technometrics 1999; 41: 39. doi:10.2307/1270993
#
# Marino S, Hogue IB, Ray CJ, Kirschner DE.
# "A methodology for performing global uncertainty and sensitivity analysis in systems biology."
# J Theor Biol 2008; 254: 178-96. doi:10.1016/j.jtbi.2008.04.011


def efast_sample_matrix(sample_number, interference, parameters):
    """
    Generate model inputs for the extended Fourier Amplitude Sensitivity Test (FAST).

    Returns a NumPy matrix containing the model inputs required by the Fourier
    Amplitude sensitivity test.  The resulting matrix contains N rows and K
    columns, where K is the number of parameters.

    Code modified from SALib (due to an inability to install) https://salib.readthedocs.io/en/latest/

    :param sample_number: Number of samples
    :param interference: The interference parameter, i.e., the number of harmonics to sum in the
        Fourier series decomposition
    :param parameters: parameters and values (from epyc)
    :return:
    """
    if sample_number <= 4 * interference ** 2:
        raise ValueError("""
            Sample size N > 4M^2 is required. M=4 by default.""")

    # Determine certainty of parameters
    uncertain_params = [(p, v[0], v[1], v[2]) for (p, v) in parameters.iteritems() if len(v) > 1]
    certain_params = {p: v[0] for (p, v) in parameters.iteritems() if len(v) == 1}
    # Number of uncertain parameters
    k = len(uncertain_params)

    # Master list of frequencies. Pos 0 will be used for the parameter of interest, other frequencies will be applied
    # to the other parameters
    omega = np.zeros([k])

    # Frequency of parameter of interest [Saltelli et al. 1999 - eqn 21]
    omega[0] = math.floor((sample_number - 1.0) / (2.0 * interference))

    # Maximum value of complimentary frequencies [Saltelli et al. 1999 - Sect 4.2, pg 47]
    max_omega_comp = math.floor(omega[0] / (2.0 * interference))

    # Determine complimentary frequencies
    # From [Saltelli et al 1999], "The other frequencies for the complementary set are chosen to exhaust the
    # whole range between 1 and max{omega_i}, and according to the two following conflicting requirements:
    # (1) the step between frequencies must be as large as possible and
    # (2) the number of factors to which the same frequency is assigned must be as low as possible"
    if max_omega_comp >= (k - 1):
        # Max is greater than the remainder number of frequencies, so list is (1, max), with number of steps == number
        # of remaining params
        omega[1:] = np.floor(np.linspace(1, max_omega_comp, k - 1))
    else:
        # Max is less than the remainder number of frequencies, so will need to repeat values
        omega[1:] = np.arange(k - 1) % max_omega_comp + 1

    # Discretisation of the frequency space, s (vector of length samples with values 0-2pi)
    s = (2 * math.pi / sample_number) * np.arange(sample_number)

    # Transformation to get points in the X space
    x = np.zeros([sample_number * k, k])

    # Taking each parameter as the parameter of interest
    for parameter_of_interest_pos in range(k):
        # Complimentary frequencies
        omega2 = np.zeros([k])

        # Assign the omega_max frequency to the parameter of interest
        omega2[parameter_of_interest_pos] = omega[0]

        # Assign the remaining frequencies to all other parameters
        # Get list [0:k] excluding the parameter of interest to use as index
        idx = list(range(parameter_of_interest_pos)) + list(range(parameter_of_interest_pos + 1, k))
        # Assign values
        omega2[idx] = omega[1:]

        # Calculate the run ID numbers for this parameters
        run_numbers = range(parameter_of_interest_pos * sample_number, (parameter_of_interest_pos + 1) * sample_number)

        # TODO - resamples?

        # random phase shift on [0, 2pi) following [Saltelli et al. 1999 - Sect 2.2]
        phi = 2 * math.pi * np.random.rand()

        # Assign a value (in range [0,1]) to each parameter for each run based on their frequency and phase shift
        for param in range(k):
            g = 0.5 + (1 / math.pi) * np.arcsin(np.sin(omega2[param] * s + phi))
            x[run_numbers, param] = g

    # Convert 0-1 values into values within the parameter range, based on distribution values and distribution type.
    # Then add to the sample list
    samples = []
    # Process a sample at a time
    for j in range(x.shape[0]):
        # Initialise the sample as the certain parameter values
        sample = certain_params.copy()
        # For each uncertain parameter
        for p in range(len(uncertain_params)):
            dist = uncertain_params[p][3]
            val = x[j, p]
            if dist == UNIFORM_DISTRIBUTION:
                minimum_val = uncertain_params[p][1]
                maximum_val = uncertain_params[p][2]
                assert minimum_val < maximum_val, "Second value must exceed first for uniform distribution"
                val = val * (maximum_val - minimum_val) + minimum_val
            elif dist == NORMAL_DISTRIBUTION:
                loc = uncertain_params[p][1]
                scale = uncertain_params[p][2]
                assert scale > 0, "Standard deviation for normal must exceed 0"
                val = stats.norm.ppf(val, loc=loc, scale=scale)
            # lognormal distribution (ln-space, not base-10)
            # parameters are ln-space mean and standard deviation
            elif dist == LOGNORMAL_DISTRIBUTION:
                loc = uncertain_params[p][1]
                scale = uncertain_params[p][2]
                # checking for valid parameters
                val = np.exp(stats.norm.ppf(val, loc=loc, scale=scale))
            else:
                # Invalid distribution type
                valid_dists = [UNIFORM_DISTRIBUTION, NORMAL_DISTRIBUTION, LOGNORMAL_DISTRIBUTION]
                raise ValueError('Distributions: choose one of %s' % ", ".join(valid_dists))
            # Assign value to parameter
            sample[uncertain_params[p][0]] = val
        # Add in experiment parameter values (needed for analysis)
        sample[RUN_NUMBER] = j
        sample[PARAMETER_OF_INTEREST] = uncertain_params[j / sample_number][0]
        samples.append(sample)
    return samples


class EFASTLab(epyc.Lab):
    def __init__(self, notebook):
        epyc.Lab.__init__(self, notebook)
        self._sample_number = 0
        self._interference = 0

    def set_sample_number(self, samples):
        # self._sample_number = samples
        self._notebook.set_sample_number(samples)

    def set_interference_factor(self, factor):
        # self._interference = factor
        self._notebook.set_interference_factor(factor)

    def parameterSpace(self):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.
        :returns: the parameter space as a list of dicts"""
        if len(self._parameters) == 0:
            return []
        else:
            NS = self._notebook.sample_number()
            Mi = self._notebook.interference_factor()
            assert (NS > 0), "Sample number invalid: {0}. Set using {1}()"\
                .format(self._sample_number, self.set_sample_number.__name__)
            assert (Mi > 0), "Interference value invalid: {0}. Set using {1}()"\
                .format(self._interference, self.set_interference_factor.__name__)
            return efast_sample_matrix(NS, Mi, self._parameters)


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
            NS = self._notebook.sample_number()
            Mi = self._notebook.interference_factor()
            assert (NS > 0), "Sample number invalid: {0}. Set using {1}()" \
                .format(self._sample_number, self.set_sample_number.__name__)
            assert (Mi > 0), "Interference value invalid: {0}. Set using {1}()" \
                .format(self._interference, self.set_interference_factor.__name__)
            return efast_sample_matrix(NS, Mi, self._parameters)
