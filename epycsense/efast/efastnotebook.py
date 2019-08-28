import numpy as np
import math
import json
import itertools
from ..aggregated.aggregationnotebook import *
from epyc.jsonlabnotebook import MetadataEncoder


class EFASTJSONNotebook(AggregationJSONNotebook):
    RUN_NUMBER = 'run_number'
    PARAMETER_OF_INTEREST = 'parameter_of_interest'
    DUMMY = 'dummy'
    INTERFERENCE_FACTOR = 'interference_factor'
    RESAMPLE_NUMBER = 'resample_number'
    SAMPLE_NUMBER = 'sample_number'

    """
    epyc Notebook for analysing results out of an epyc.EFastLab or epyc.EFastClusterLab
    """

    def __init__(self, name, create=True, description=None):
        AggregationJSONNotebook.__init__(self, name, create, description)

    def uncertain_parameters(self):
        """
        Uncertain parameters (excludes the EFAST parameter values)
        :return:
        """
        params = AggregationJSONNotebook.uncertain_parameters(self)
        params.remove(EFASTJSONNotebook.RUN_NUMBER)
        params.remove(EFASTJSONNotebook.PARAMETER_OF_INTEREST)
        params.remove(EFASTJSONNotebook.RESAMPLE_NUMBER)
        return params

    def generate_sensitivity_indices(self):
        """
        Performs the Fourier Amplitude Sensitivity Test (FAST) on model outputs.

        Returns a dictionary with keys 'S1' and 'ST', where each entry is a list of
        size D (the number of parameters) containing the indices in the same order
        as the parameter file.

        Reference material for EFAST:

        Cukier RI, Fortuin CM, Shuler KE, Petschek AG, Schaibly JH.
        "Study of the sensitivity of coupled reaction systems to uncertainties in rate coefficients. I Theory."
        J Chem Phys 1973; 59: 3873-8. doi:10.1063/1.1680571

        Saltelli A, Tarantola S, Chan KP-S.
        "A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output."
        Technometrics 1999; 41: 39. doi:10.2307/1270993

        Marino S, Hogue IB, Ray CJ, Kirschner DE.
        "A methodology for performing global uncertainty and sensitivity analysis in systems biology."
        J Theor Biol 2008; 254: 178-96. doi:10.1016/j.jtbi.2008.04.011
        :return:
        """
        num_uncertain_params = len(self.uncertain_parameters())

        # Reduce to only the actual results, sort by parameter of interest, then resample, then run number
        data = self.dataframe_aggregated().sort_values(by=[EFASTJSONNotebook.PARAMETER_OF_INTEREST,
                                                           EFASTJSONNotebook.RESAMPLE_NUMBER,
                                                           EFASTJSONNotebook.RUN_NUMBER])
        params_of_interest = data[EFASTJSONNotebook.PARAMETER_OF_INTEREST]
        required_parameters = list(set(params_of_interest))

        sample_number = max(data[EFASTJSONNotebook.RUN_NUMBER]) + 1
        resample_number =  max(data[EFASTJSONNotebook.RESAMPLE_NUMBER]) + 1
        # TODO - hard-coded
        interference_factor = 4

        # Check we have all expected results (NS * k)
        assert len(data) == len(required_parameters) * sample_number * resample_number, "Invalid data length"

        # Recreate the frequency vector used in the sampling
        omega = np.zeros([num_uncertain_params])
        omega[0] = math.floor((sample_number - 1.0) / (2.0 * interference_factor))

        # Recreate the complimentary parameter frequencies
        m = math.floor(omega[0] / (2.0 * interference_factor))
        if m >= (num_uncertain_params - 1):
            omega[1:] = np.floor(np.linspace(1, m, num_uncertain_params - 1))
        else:
            omega[1:] = np.arange(num_uncertain_params - 1) % m + 1

        # First-order sensitivity indices
        S1 = {(rk, p): [] for (rk,p) in itertools.product(self._result_keys, self.uncertain_parameters())}
        # Total-order sensitivity indices
        ST = {(rk, p): [] for (rk,p) in itertools.product(self._result_keys, self.uncertain_parameters())}

        # Loop through each uncertain parameter
        for i in range(len(required_parameters)):
            poi = required_parameters[i]
            for rs in range(resample_number):
                relevant_rows = data[(data.parameter_of_interest==poi) & (data.resample_number==rs)]
                for result_key in self._result_keys:
                    result_data = relevant_rows[result_key]
                    f = np.fft.fft(result_data)
                    Sp = np.power(np.absolute(f[np.arange(1, int((sample_number + 1) / 2))]) /
                                          sample_number, 2)

                    q = np.arange(1, int(interference_factor) + 1) * int(omega[0]) - 1
                    V = 2 * np.sum(Sp)
                    D1 = 2 * np.sum(Sp[q])
                    S1[(result_key, poi)].append(D1 / V)

                    q2 = np.arange(int(omega[0] / 2))
                    Dt = 2 * sum(Sp[q2])
                    ST[(result_key, poi)].append(1 - Dt/V)

        return S1, ST