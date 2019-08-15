import epyc
from ..aggregated.aggregationnotebook import AggregationJSONNotebook


class EFASTJSONNotebook(AggregationJSONNotebook):
    def __init__(self, name, create=True, description=None):
        AggregationJSONNotebook.__init__(self, name, create, description)

    def analyse(self):
        """
        Performs the Fourier Amplitude Sensitivity Test (FAST) on model outputs.

        Returns a dictionary with keys 'S1' and 'ST', where each entry is a list of
        size D (the number of parameters) containing the indices in the same order
        as the parameter file.


        References
        ----------
        .. [1] Cukier, R. I., C. M. Fortuin, K. E. Shuler, A. G. Petschek, and J. H.
           Schaibly (1973).  "Study of the sensitivity of coupled reaction
           systems to uncertainties in rate coefficients."  J. Chem. Phys.,
           59(8):3873-3878, doi:10.1063/1.1680571.

        .. [2] Saltelli, A., S. Tarantola, and K. P.-S. Chan (1999).  "A
          Quantitative Model-Independent Method for Global Sensitivity
          Analysis of Model Output."  Technometrics, 41(1):39-56,
          doi:10.1080/00401706.1999.10485594.
        :return:
        """
        print self._aggregated_results
