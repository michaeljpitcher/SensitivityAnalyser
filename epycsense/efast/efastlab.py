import epyc
import math
import numpy as np
import scipy.stats as stats
from .efastnotebook import EFASTJSONNotebook

UNIFORM_DISTRIBUTION = 'uniform_distribution'
NORMAL_DISTRIBUTION = 'normal_distribution'
LOGNORMAL_DISTRIBUTION = 'lognormal_distribution'

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


def efast_sample_matrix(sample_number, interference, parameters, resample_number):
    """
    Generate model inputs for the extended Fourier Amplitude Sensitivity Test (FAST).

    Returns a NumPy matrix containing the model inputs required by the Fourier
    Amplitude sensitivity test.  The resulting matrix contains N rows and K
    columns, where K is the number of parameters.

    Code modified from SALib (due to an inability to install) https://salib.readthedocs.io/en/latest/

    :param sample_number: Number of samples per parameter
    :param interference: The interference parameter, i.e., the number of harmonics to sum in the
        Fourier series decomposition
    :param parameters: parameters and values (from epyc)
    :return:
    """
    assert sample_number > 4 * interference ** 2, "Sample size N > 4M^2 is required. M=4 by default."
    assert resample_number >= 1, "Resample number must be >= 1"

    # Determine certainty of parameters
    uncertain_params = [(p, v[0], v[1], v[2]) for (p, v) in parameters.iteritems() if len(v) > 1]
    # Add a dummy parameter (Marino et al., 2008)
    uncertain_params.append((EFASTJSONNotebook.DUMMY, 0, 10, UNIFORM_DISTRIBUTION))
    # Get parameters that only have one value
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
    x = np.zeros([k * resample_number * sample_number, k])

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

        # Assign a value (in range [0,1]) to each parameter for each run based on their frequency and phase shift
        for rs in range(0, resample_number):
            # Calculate the run IDs for this resample of this parameter
            run_numbers = range(sample_number*(parameter_of_interest_pos*resample_number + rs),
                                sample_number*(parameter_of_interest_pos*resample_number + rs + 1))

            # random phase shift on [0, 2pi) following [Saltelli et al. 1999 - Sect 2.2]

            for param in range(k):
                phi = 2 * math.pi * np.random.rand()
                g = 0.5 + (1 / math.pi) * np.arcsin(np.sin(omega2[param] * s + phi))
                x[run_numbers, param] = g

    # Convert 0-1 values into values within the parameter range, based on distribution values and distribution type.
    for q in range(len(uncertain_params)):
        _, d1, d2, dist = uncertain_params[q]
        if dist == UNIFORM_DISTRIBUTION:
            assert d1 < d2, "Second value must exceed first for uniform distribution"
            x[:, q] = x[:, q] * (d2 - d1) + d1
        elif dist == NORMAL_DISTRIBUTION:
            assert d2 > 0, "Standard deviation for normal must exceed 0"
            x[:, q] = stats.norm.ppf(x[:, q], loc=d1, scale=d2)
        # lognormal distribution (ln-space, not base-10)
        # parameters are ln-space mean and standard deviation
        elif dist == LOGNORMAL_DISTRIBUTION:
            # checking for valid parameters
            x[:, q] = np.exp(stats.norm.ppf(x[:, q], loc=d1, scale=d2))

    # Then add to the sample list
    samples = []
    rows_per_poi = resample_number*sample_number

    for row in range(x.shape[0]):
        sample = certain_params.copy()
        # Calculate the EFAST parameters (needed for analysis)
        sample[EFASTJSONNotebook.PARAMETER_OF_INTEREST] = uncertain_params[row / rows_per_poi][0]
        sample[EFASTJSONNotebook.RESAMPLE_NUMBER] = (row % rows_per_poi) / sample_number
        sample[EFASTJSONNotebook.RUN_NUMBER] = row % sample_number
        for q in range(len(uncertain_params)):
            sample[uncertain_params[q][0]] = x[row,q]
        samples.append(sample)
    return samples


class EFASTLab(epyc.Lab):
    def __init__(self, notebook):
        assert isinstance(notebook, EFASTJSONNotebook), "Notebook must be Efast JSON notebook"
        epyc.Lab.__init__(self, notebook)
        self._sample_number = 0
        self._interference = 0
        self._resample_number = 0

    def set_sample_number(self, samples):
        self._notebook.set_sample_number(samples)

    def set_interference_factor(self, factor):
        self._notebook.set_interference_factor(factor)

    def set_resample_number(self, resamples):
        self._notebook.set_resample_number(resamples)

    def parameterSpace(self):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.
        :returns: the parameter space as a list of dicts"""
        if len(self._parameters) == 0:
            return []
        else:
            NS = self._notebook.sample_number()
            Mi = self._notebook.interference_factor()
            NR = self._notebook.resample_number()
            assert (NS > 0), "Sample number invalid: {0}. Set using {1}()"\
                .format(self._sample_number, self.set_sample_number.__name__)
            assert (Mi > 0), "Interference value invalid: {0}. Set using {1}()"\
                .format(self._interference, self.set_interference_factor.__name__)
            assert (NR >= 1), "Resample value invalid: {0}. Set using {1}()" \
                .format(self._interference, self.set_resample_number.__name__)
            return efast_sample_matrix(NS, Mi, self._parameters, NR)


class EFASTClusterLab(epyc.ClusterLab):
    def __init__(self, notebook):
        assert isinstance(notebook, EFASTJSONNotebook), "Notebook must be Efast JSON notebook"
        epyc.ClusterLab.__init__(self, notebook)
        self._sample_number = 0
        self._interference = 0
        self._resample_number = 0

    def set_sample_number(self, samples):
        self._notebook.set_sample_number(samples)

    def set_interference_factor(self, factor):
        self._notebook.set_interference_factor(factor)

    def set_resample_number(self, resamples):
        self._notebook.set_resample_number(resamples)

    def parameterSpace(self):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.
        :returns: the parameter space as a list of dicts"""
        if len(self._parameters) == 0:
            return []
        else:
            NS = self._notebook.sample_number()
            Mi = self._notebook.interference_factor()
            NR = self._notebook.resample_number()
            assert (NS > 0), "Sample number invalid: {0}. Set using {1}()"\
                .format(self._sample_number, self.set_sample_number.__name__)
            assert (Mi > 0), "Interference value invalid: {0}. Set using {1}()"\
                .format(self._interference, self.set_interference_factor.__name__)
            assert (NR >= 1), "Resample value invalid: {0}. Set using {1}()" \
                .format(self._interference, self.set_resample_number.__name__)
            return efast_sample_matrix(NS, Mi, self._parameters, NR)
