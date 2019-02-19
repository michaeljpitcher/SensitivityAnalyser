from sensitivity.folder import *
from scatterlab import *
from progressepyc import *

SCATTER_FILENAME = 'scatter.json'


def create_correlation_scatter_data(model, parameter_set, stratifications, repetitions, output_folder):
    create_folder(output_folder)
    output_file = output_folder + '/' + SCATTER_FILENAME
    uncertain_ranges = parameter_set.create_latin_hypercube_stratifications(stratifications)

    notebook = epyc.JSONLabNotebook(output_file, True)
    lab = ScatterLab(notebook)
    for p, r in uncertain_ranges.iteritems():
        lab[p] = r
    for p, v in parameter_set.certain_parameters().iteritems():
        lab[p] = v
    lab.runExperiment(ProgressRepeatedExperiment(model, repetitions))
    return output_file
