__author__ = "Cody Boon"
__version__ = "0.1"

import numpy as np
from numpy.random import multivariate_normal, normal

from SMPyBandits.ContextualBandits.Contexts.BaseContext import BaseContext


class NullContext(BaseContext):
    """ A context generator drawing normalized Gaussian vectors """

    def __init__(self, dimension):
        # print("\nInitiating Null Context\n")
        BaseContext.__init__(self, 1, 1, 0)
        self.dimension = dimension

    def __str__(self):
        return "NullContext"

    def __repr__(self):
        return "NullContext"

    def draw_context(self):
        return np.full(1, self.dimension)

    def get_means(self):
        return np.full(1, self.dimension)
