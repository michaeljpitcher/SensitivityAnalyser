import epyc
import json
import numpy
import pandas


def read_epyc_json_data(filename, uncertain_parameters):
    """
    Given an epyc JSON output file, reads in the data (and averages the result for multiple repetitions)
    :param filename:
    :param uncertain_parameters:
    :return:
    """
    with open(filename) as data_file:
        data = json.load(data_file)[epyc.Experiment.RESULTS].values()
    param_data = dict([(p, []) for p in uncertain_parameters])
    result_keys = data[0][0][epyc.Experiment.RESULTS].keys()
    result_data = dict([(p, []) for p in result_keys])
    # TODO - this assumes result data is just numeric values
    for param_variation in data:
        for p in uncertain_parameters:
            param_data[p].append(param_variation[0][epyc.Experiment.PARAMETERS][p])
        for k in result_keys:
            if len(param_variation) > 1:
                # Multiple repetitions so we need to average
                repetition_results = [p[epyc.Experiment.RESULTS] for p in param_variation]
                result_data[k].append(numpy.average([rep[k] for rep in repetition_results]))
            else:
                # Only 1 repetition (possibly from a deterministic model) so just pass it through
                result_data[k].append(param_variation[0][epyc.Experiment.RESULTS][k])

    return pandas.DataFrame(param_data), pandas.DataFrame(result_data)
