from ..parameters.parameterset import ParameterSet
from ...folder import *
from lhslab import LatinHypercubeLab
from progressepyc import *
import epyc

LHS_FILENAME = 'latinhypercube.json'


def run_lhs(model, parameter_set, stratifications, repetitions, output_folder=None):

    assert isinstance(parameter_set, ParameterSet), "Invalid parameter set"
    lhs_vals = parameter_set.create_latin_hypercube_stratifications(stratifications)

    create_folder(output_folder)
    output_folder += '/'

    notebook = epyc.JSONLabNotebook(output_folder + LHS_FILENAME, create=True, description="LHS")
    lab = LatinHypercubeLab(notebook)
    for p, v in parameter_set.certain_parameters().iteritems():
        lab[p] = v
    for x,y in lhs_vals.iteritems():
        lab[x] = y
    lab.runExperiment(ProgressRepeatedExperiment(model, repetitions))

    return output_folder + LHS_FILENAME
