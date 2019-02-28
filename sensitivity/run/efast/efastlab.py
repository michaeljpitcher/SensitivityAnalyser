import epyc
import numpy
from progressepyc import ProgressLab
import collections
import six
import math


class EFastLab(ProgressLab):

    # TODO - more distributions
    UNIFORM_DISTRIBUTION = 'uniform_distribution'
    DISTRIBUTIONS = [UNIFORM_DISTRIBUTION]

    """
    An epyc lab, which creates a sample scheme following eFast algorithm
    """
    def __init__(self, notebook, runs=65, resamples=1, max_fourier_coeff=4):
        self._certain_parameters = []
        self._uncertain_parameters = []

        self._runs = runs
        self._resamples = resamples
        self._max_fourier_coeffs = max_fourier_coeff

        epyc.Lab.__init__(self, notebook)

    def set_runs(self, runs):
        self._runs = runs

    def set_resamples(self, resamples):
        self._resamples = resamples

    def set_max_fourier_coeffs(self, max_fourier_coeffs):
        self._max_fourier_coeffs = max_fourier_coeffs

    def addParameter( self, k, r ):
        """Add a parameter to the experiment's parameter space. k is the
        parameter name, and r is its range values - should be either a single value or
        tuple of (min, max, distribution).

        :param k: parameter name
        :param r: parameter range"""

        if isinstance(r, six.string_types) or not isinstance(r, collections.Iterable):
            # range is a single value (where a string constitutes a single value), make it a list
            r = [ r ]
            self._certain_parameters.append(k)
        else:
            if isinstance(r, collections.Iterable):
                # range is an iterable, make into a list
                r = list(r)
                # Ensure list has 3 elements
                assert len(r) == 3, "Parameter must be defined as (min,max,distribution)"
                assert r[0] < r[1], "Invalid minimum/maximum values"
                assert r[2] in EFastLab.DISTRIBUTIONS, "Invalid distribution"
                self._uncertain_parameters.append(k)
        self._parameters[k] = r

    def parameterSpace( self ):
        """Return the parameter space of the experiment as a list of dicts,
        with each dict mapping each parameter name to a value.

        :returns: the parameter space as a list of dicts"""
        if len(self._parameters) == 0:
            return []
        else:
            return self._efast_sample_matrix()

    def _efast_sample_matrix(self):
        samples = []
        number_parameters = len(self._uncertain_parameters)

        # wanted no. of sample points
        wantedN = self._runs * number_parameters * self._resamples

        # Computation of the frequency for the group of interest
        # TODO - where does this come from?
        parameter_frequency = math.floor(((wantedN/self._resamples)-1)/(2*self._max_fourier_coeffs)/number_parameters)

        # Number of sample points : [Saltelli et al., 1999] - eqn 15
        number_samples = int(2 * self._max_fourier_coeffs * parameter_frequency + 1)

        assert number_samples * self._resamples >= 65, 'Error: sample size must be >= 65 per factor'

        # [Saltelli et al., 1999] - pg 47, col2, section 1
        max_freq_other_params = (1.0 / self._max_fourier_coeffs) * (parameter_frequency / 2.0)

        # Algorithm for selection of a frequency set for the complementary group. Done recursively as described in:
        # Appendix of "Sensitivity Analysis" [Saltelli et al., 2000]
        if number_parameters == 1:
            complementary_frequencies = 1
        elif max_freq_other_params == 1:
            # Set all frequencies to 1
            complementary_frequencies = [1, ] * number_parameters
        else:
            complementary_frequencies = [None, ] * number_parameters
            INFD = min(max_freq_other_params, number_parameters)

            ISTEP = round((max_freq_other_params - 1) / (INFD - 1))
            if (max_freq_other_params == 1):
                ISTEP = 0

            OTMP = numpy.linspace(1, INFD * ISTEP, INFD)
            fl_INFD = int(math.floor(INFD))
            for i in range(number_parameters):
                j = i % fl_INFD
                complementary_frequencies[i] = OTMP[j]

        # Vector of s ranging from -2*pi to 2pi
        S_VEC = numpy.transpose(numpy.pi * (2 * numpy.linspace(1, number_samples, number_samples) - number_samples - 1)
                                / number_samples).reshape(number_samples, 1)

        # Loop over k parameters (input factors)
        for i in range(number_parameters):
            # Algorithm for selecting the set of frequencies.
            # OM is frequency list. See all values to complementary frequencies, overwrite the parameter of interest
            OM = list(complementary_frequencies)
            OM[i] = parameter_frequency
            OM_VEC = numpy.array(OM).reshape(1, number_parameters)

            # Loop over NR resamples
            for L in range(self._resamples):

                # Random phase shift [0, 2*pi).
                phase_shift = numpy.random.rand(1, number_parameters) * 2 * numpy.pi
                # Extend to a k * NS matrix
                phase_shift = numpy.repeat(phase_shift, [number_samples], axis=0)

                # Create parameter values
                ANGLE = OM_VEC * S_VEC + phase_shift

                # Obtain a transformation [Saltelli et al 1999, eqn 20]
                X = 0.5 + (1.0 / numpy.pi) * (numpy.arcsin(numpy.sin(ANGLE)))

                # Transform distributions from 0-1 to min-max for each parameter based on distribution
                for k in range(X.shape[1]):
                    parameter = self._uncertain_parameters[k]
                    min_val, max_val, dist = self._parameters[parameter]
                    # TODO - more distributions
                    if dist == EFastLab.UNIFORM_DISTRIBUTION:
                        X[:, k] = (X[:, k] * (max_val - min_val)) + min_val

                # Add each row to the sample list
                for r in range(X.shape[0]):
                    sample = {c: self._parameters[c][0] for c in self._certain_parameters}
                    for c in range(X.shape[1]):
                        p = self._uncertain_parameters[c]
                        # min_val, max_val, _ = self._parameters[p]
                        # assert min_val <= X[r][c] <= max_val, "Value {0} for {1} out of bounds".format(X[r][c], c)
                        sample[p] = X[r][c]
                    samples.append(sample)
        # Total number of samples is NS * k * NR [Marino et al 2008, pg 183]
        return samples
