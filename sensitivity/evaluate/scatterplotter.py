from ..data import *
from ..visual import *
from outputfolder import create_output_folder
import epyc

SCATTER = 'scatter'
JSON_SUFFIX = '.json'


def create_lhs_scatter_plots(model, parameter_set, stratifications, repetitions, output_folder=None, time_series=False):

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

        if time_series:
            result_set = TimeSeriesResultSet(parameter_set)
            result_set.load_data_from_json(filename)
            for t, d in result_set.result_data().iteritems():
                for r in d:
                    scatter_filename = output_folder + uncertain_param + '_' + r + '_' + t
                    create_scatter_plot(result_set.parameter_data()[uncertain_param], result_set.result_data()[t][r],
                                        uncertain_param + ' vs ' + r + ' \n t = ' + t,
                                        scatter_filename, uncertain_param, r)
        else:
            result_set = ResultSet(parameter_set)
            result_set.load_data_from_json(filename)
            for r in result_set.result_data():
                scatter_filename = output_folder + uncertain_param + '_' + r
                create_scatter_plot(result_set.parameter_data()[uncertain_param], result_set.result_data()[r],
                                    uncertain_param + ' vs ' + r,
                                    scatter_filename, uncertain_param, r)
