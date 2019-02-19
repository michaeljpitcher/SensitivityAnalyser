from epyc import *
from progressrepeatedexperiment import *


class ProgressLab(Lab):
    """
    Lab with a progress bar which updates with every parameter sample

    """

    def __init__( self, notebook = None ):
        Lab.__init__(self, notebook)

    def runExperiment(self, e):
        """Run an experiment over all the points in the parameter space.
        The results will be stored in the notebook.

        :param e: the experiment"""

        # create the parameter space
        ps = self.parameterSpace()

        # run the experiment at each point
        nb = self.notebook()
        if isinstance(e, ProgressRepeatedExperiment):
            e.set_total(len(ps))

        for p in ps:
            # print "Running {p}".format(p = p)
            res = e.set(p).run()
            nb.addResult(res)

        # commit the results
        nb.commit()
