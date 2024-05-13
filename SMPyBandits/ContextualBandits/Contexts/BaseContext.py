__author__ = "Cody Boon"
__version__ = "0.1"

import numpy as np


class BaseContext(object):
    """ Base class for any context generator """

    def __init__(self, dimension, lower=0., amplitude=1.):
        assert dimension > 0, "Dimension needs to be greater than zero"
        self.dimension = dimension
        self.lower = lower
        self.amplitude = amplitude

    def __str__(self):
        return self.__class__.__name__

    def draw_context(self):
        raise NotImplementedError(
            "This method, draw_context(), must be implemented in the child class inheriting from BaseContext"
        )

    def draw_nparray(self, shape=(1,)):
        np.array([self.draw_context() for _ in range(np.multiply(shape))]).reshape(shape)

    def get_means(self):
        raise NotImplementedError(
            "This method, get_means(), must be implemented in the child class inheriting from BaseContext"
        )
