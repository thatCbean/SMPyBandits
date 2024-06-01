__author__ = "Cody Boon"
__version__ = "0.1"

import numpy as np
from numpy.random import multivariate_normal, normal

from SMPyBandits.ContextualBandits.Contexts.BaseContext import BaseContext


class NullContext(BaseContext):
    """ A context generator drawing normalized Gaussian vectors """

    def __init__(self):
        print("\nInitiating Null Context\n")
        BaseContext.__init__(self, 1, 1, 0)

    def draw_context(self):
        return [1]

    def get_means(self):
        return [1]
