import numpy as np
import math
import json
from ..aggregated.aggregationnotebook import *
from .efastlab import RUN_NUMBER, PARAMETER_OF_INTEREST
from epyc.jsonlabnotebook import MetadataEncoder


class EFASTJSONNotebook(AggregationJSONNotebook):
    """
    epyc Notebook for analysing results out of an epyc.EFastLab or epyc.EFastClusterLab
    """

    INTERFERENCE_FACTOR = 'interference_factor'
    SAMPLE_NUMBER = 'sample_number'

    def __init__(self, name, create=True, description=None):
        self._interference_factor = 0
        self._sample_number = 0
        AggregationJSONNotebook.__init__(self, name, create, description)

    def set_interference_factor(self, factor):
        self._interference_factor = factor

    def interference_factor(self):
        return self._interference_factor

    def set_sample_number(self, samples):
        self._sample_number = samples

    def sample_number(self):
        return self._sample_number

    def _save( self, fn ):
        """Persist the notebook to the given file. Saves the interference factor and sample number within the JSON file
        so it can be loaded in future.

        :param fn: the file name"""

        # create JSON object (with additional experiment EFAST values in header)
        j = json.dumps({'description': self.description(),
                        'pending': self._pending,
                        'results': self._results,
                        EFASTJSONNotebook.INTERFERENCE_FACTOR: self._interference_factor,
                        EFASTJSONNotebook.SAMPLE_NUMBER: self._sample_number},
                       indent=4,
                       cls=MetadataEncoder)

        # write to file
        with open(fn, 'w') as f:
            f.write(j)

    def _load( self, fn ):
        """
        Load JSON data. Reads the EFAST parameter values (interference number and sample number) from the file.
        :param fn:
        :return:
        """
        AggregationJSONNotebook._load(self, fn)
        # load the JSON object to get EFAST parameters
        with open(fn, "r") as f:
            s = f.read()
            # parse back into appropriate variables
            j = json.loads(s)
            self._interference_factor = j[EFASTJSONNotebook.INTERFERENCE_FACTOR]
            self._sample_number = j[EFASTJSONNotebook.SAMPLE_NUMBER]

    def uncertain_parameters(self):
        """
        Uncertain parameters (excludes the EFAST parameter values)
        :return:
        """
        params = AggregationJSONNotebook.uncertain_parameters(self)
        params.remove(RUN_NUMBER)
        params.remove(PARAMETER_OF_INTEREST)
        return params

    def analyse(self):
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
        # TODO - resamples?

        num_uncertain_params = len(self.uncertain_parameters())

        # Reduce to only the actual results
        data = self.dataframe_aggregated().sort_values(by=RUN_NUMBER)
        params_of_interest = data[PARAMETER_OF_INTEREST]
        results = data[self._result_keys]

        # Check we have all expected results (NS * k)
        assert len(results) == num_uncertain_params * self._sample_number, "Invalid data length"

        # Recreate the frequency vector used in the sampling
        omega = np.zeros([num_uncertain_params])
        omega[0] = math.floor((self._sample_number - 1.0) / (2.0 * self._interference_factor))

        # Recreate the complimentary parameter frequencies
        # TODO - possibly redundant recreating this when we could save it in the data (like NS and Mi)
        m = math.floor(omega[0] / (2.0 * self._interference_factor))
        if m >= (num_uncertain_params - 1):
            omega[1:] = np.floor(np.linspace(1, m, num_uncertain_params - 1))
        else:
            omega[1:] = np.arange(num_uncertain_params - 1) % m + 1

        S1 = {}
        ST = {}

        # Loop through each uncertain parameter
        for i in range(num_uncertain_params):
            # Find the results where the parameter was the parameter of interest (i.e. had high frequency)
            relevant_results = results[i*self._sample_number:(i+1)*self._sample_number]
            # Check it was the parameter of interest for all rows
            poi = set(params_of_interest[i*self._sample_number:(i+1)*self._sample_number])
            assert len(poi) == 1
            poi = poi.pop()
            for result_key in relevant_results.columns:
                output_data = relevant_results[result_key]
                f = np.fft.fft(output_data)
                Sp = np.power(np.absolute(f[np.arange(1, int((self._sample_number + 1) / 2))]) / self._sample_number, 2)

                q = np.arange(1, int(self._interference_factor) + 1) * int(omega[0]) - 1
                V = 2 * np.sum(Sp)
                D1 = 2 * np.sum(Sp[q])
                S1[(result_key, poi)] = D1 / V

                q2 = np.arange(int(omega[0] / 2))
                Dt = 2 * sum(Sp[q2])
                ST[(result_key, poi)] = 1 - Dt/V

        return S1, ST