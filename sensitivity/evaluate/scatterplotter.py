from ..data import *
from ..visual import *
import epyc
import os

SCATTER = 'scatter'
JSON_SUFFIX = '.json'


def create_lhs_scatter_plots(model, parameter_set, stratifications, repetitions, output_folder=None):

    assert isinstance(parameter_set, ParameterSet), "Invalid parameter set"
    lhs_vals = parameter_set.create_latin_hypercube_stratifications(stratifications)

    baseline_values = parameter_set.certain_parameters.copy()
    for p, v in lhs_vals.iteritems():
        baseline_values[p] = v[len(v)/2]

    create_output_folder(output_folder)
    output_folder += '/'

    for uncertain_param, v in lhs_vals.iteritems():
        # TODO - using a JSONnotebook and loading from json file as it preserves repetition structure
        filename = output_folder + uncertain_param + JSON_SUFFIX
        notebook = epyc.JSONLabNotebook(filename, True)
        lab = epyc.Lab(notebook)
        for param, baseline in baseline_values.iteritems():
            lab[param] = baseline
        lab[uncertain_param] = v
        lab.runExperiment(epyc.RepeatedExperiment(model, repetitions))

        param_data, result_data = read_epyc_json_data(filename, [uncertain_param])

        for r in result_data:
            scatter_filename = output_folder + uncertain_param + '_' + r
            create_scatter_plot(param_data[uncertain_param], result_data[r], uncertain_param + ' vs ' + r,
                                scatter_filename, uncertain_param, r)
