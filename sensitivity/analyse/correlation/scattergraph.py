from collections import Counter
from sensitivity.analyse.visual.scatterplot import *


def create_correlation_scatter_graphs(result_set, output_folder, result_keys=None, show=False):
    if not result_keys:
        result_keys = result_set.result_keys()
    print
    uncertain_params = []
    baseline_index = range(len(result_set))
    relevant_indices = {}
    for parameter in result_set.parameter_samples():
        samples = result_set.parameter_samples()[parameter]
        counts = Counter([f for f in samples])
        # Get the most common value. This is the baseline
        baseline = counts.most_common(1)[0][0]

        relevant_indices[parameter] = range(len(samples))
        if len(counts) > 1:
            uncertain_params.append(parameter)
            # Remove any samples which aren't the baseline for this param from the baseline index search
            for i in range(len(samples)):
                if samples[i] != baseline:
                    baseline_index.remove(i)
                else:
                    relevant_indices[parameter].remove(i)
    for parameter in uncertain_params:
        indices = relevant_indices[parameter] + [baseline_index[0]]
        samples = [result_set.parameter_samples()[parameter][n] for n in indices]
        all_results = {r: [v[n] for n in indices] for r, v in result_set.result_data().iteritems()}
        for result_key in result_keys:
            result_values = all_results[result_key]
            plot_scatter_graph(samples, result_values, '', parameter, result_key, True,
                               output_folder + '/' + parameter + '_' + result_key, show)

