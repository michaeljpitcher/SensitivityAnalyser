import epyc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import itertools
import numpy


class ScatterJSONNotebook(epyc.JSONLabNotebook):
    # TODO - using a JSON notebook, could be just a notebook
    def __init__(self, name, create=True, description=None):
        epyc.JSONLabNotebook.__init__(self, name, create, description)

    def _aggregate(self, data_row):
        """
        Obtain results. Moved to a separate function to allow aggregation, average over repetitions or
        :return:
        """
        # Multiple repetitions, so process
        if len(data_row) > 1:
            # Get parameters from first rep (will be same for all reps) and average the values over other rows
            agg_row = {epyc.Experiment.PARAMETERS: data_row[0][epyc.Experiment.PARAMETERS],
                       epyc.Experiment.RESULTS: {r: numpy.mean([rep[epyc.Experiment.RESULTS][r] for rep in data_row])
                                                for r in data_row[0][epyc.Experiment.RESULTS]}}
            return agg_row
        # Only 1 run so just return it
        else:
            return data_row[0]

    def plot_scatter_graphs(self, save=True, folder='', show=False):
        if self.pendingResults():
            raise Exception("Not all results complete")

        parameters = self.results()[0][epyc.Experiment.PARAMETERS].keys()
        uncertain_parameters = {p: list(set(self.dataframe()[p].tolist())) for p in parameters if
                                       len(set(self.dataframe()[p].tolist())) > 1}
        baselines = {k: v[len(v)/2] for (k,v) in uncertain_parameters.iteritems()}

        # Get the result keys
        result_keys = self.results()[0][epyc.Experiment.RESULTS].keys()
        # Create a dictionary of (uncertain_param, result_key) tuples
        plots = {k: ([], []) for k in itertools.product(uncertain_parameters, result_keys)}

        # Loop through all rows of the output
        for row in self._results.values():
            row = self._aggregate(row)
            # For every parameter row combination
            for p,r in plots:
                # Check if param value = baseline value, only include this row if all values are baseline
                if row[epyc.Experiment.PARAMETERS][p] == baselines[p] and \
                        any([row[epyc.Experiment.PARAMETERS][q] !=  baselines[q] for q in uncertain_parameters]):
                    pass
                else:
                    plots[(p, r)][0].append(row[epyc.Experiment.PARAMETERS][p])

        for ((param, result), (x_data, y_data)) in plots.iteritems():
            plt.scatter(x_data, y_data)
            plt.title('Correlation: {0} vs {1}'.format(param, result))
            plt.xlabel(param)
            plt.ylabel(result)
            if save:
                if folder:
                    try:
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                    except OSError as o:
                        print ('Error creating directory ' + folder + ":" + o.message)
                    folder = folder + '/'
                plt.savefig(folder + param + '_' + result + ".png")
            if show:
                plt.show()
            # Refresh figure window
            plt.close()
