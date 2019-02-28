from ..parameters.parameterset import ParameterSet
from ...folder import *
from efastlab import EFastLab
from progressepyc import *
import epyc
import math
import numpy

EFAST_FILENAME = 'efast.json'


def run_efast(model, parameter_set, repetitions, runs=65, resamples=1, max_fourier_coeffs=4, output_folder=None):
    assert isinstance(parameter_set, ParameterSet), "Invalid parameter set"

    create_folder(output_folder)
    output_folder += '/'

    notebook = epyc.JSONLabNotebook(output_folder + EFAST_FILENAME, create=True, description="eFAST")
    lab = EFastLab(notebook)
    for p, v in parameter_set.certain_parameters().iteritems():
        lab[p] = v
    for x, y in parameter_set.uncertain_parameters().iteritems():
        lab[x] = y

    lab.set_runs(runs)
    lab.set_resamples(resamples)
    lab.set_max_fourier_coeffs(max_fourier_coeffs)

    lab.runExperiment(ProgressRepeatedExperiment(model, repetitions))

    # lab.runExperiment(ProgressRepeatedExperiment(model, repetitions))

    # return output_folder + LHS_FILENAME


    # k = len(parameter_set.uncertain_parameters())
    # wantedN = runs * k * resamples  # wanted no. of sample points
    # max_freq = math.floor(((wantedN / resamples) - 1) / (2 * max_fourier_coeffs) / k)
    #
    # # TODO - uncertain why we recalculate runs (NS) here
    # runs = int(2 * max_fourier_coeffs * max_freq + 1)
    # if runs * resamples < 65:
    #     raise Exception('Error: sample size must be >= 65 per factor.\n')
    #
    # OMciMAX = max_freq / 2 / max_fourier_coeffs
    #
    # if k == 1:
    #     OMci = 1
    # elif OMciMAX == 1:
    #     # Set all frequencies to 1
    #     OMci = [1, ] * k
    # else:
    #     OMci = [None, ] * k
    #     INFD = min(OMciMAX, k)
    #
    #     ISTEP = round((OMciMAX - 1) / (INFD - 1))
    #     if OMciMAX == 1:
    #         ISTEP = 0
    #
    #     OTMP = numpy.linspace(1, INFD * ISTEP, INFD)
    #     fl_INFD = int(math.floor(INFD))
    #     for i in range(k):
    #         j = i % fl_INFD
    #         OMci[i] = OTMP[j]
    #
    # S_VEC = numpy.pi * (2 * numpy.linspace(1, runs, runs) - runs - 1) / runs
    # print S_VEC

