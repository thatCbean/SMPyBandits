__author__ = "Cody Boon"
__version__ = "0.1"

import numpy as np
from numpy.random import multivariate_normal

from SMPyBandits.ContextualBandits.Contexts.BaseContext import BaseContext


class NormalContext(BaseContext):
    """ A context generator drawing normalized Gaussian vectors """

    def __init__(self, means=None, covariance_matrix=None, dimension=1, lower=0., amplitude=1.):
        print(
            "\nInitiating Normal Context with params:\nMeans: {}\nCovariance matrix:\n{}\nDimension: {}\nLower: {}\nAmplitude: {}\n"
            .format(means, covariance_matrix, dimension, lower, amplitude)
        )
        if means is None:
            means = np.full(shape=dimension, fill_value=(1./dimension))
        if covariance_matrix is None:
            covariance_matrix = np.identity(dimension) * 0.5
        if not isinstance(means, np.ndarray):
            means = np.array(means)
        assert len(means) == dimension, "means needs to have <dimension> entries"
        assert covariance_matrix.shape == (dimension, dimension), "covariance_matrix needs to be a <dimension>x<dimension> matrix"

        BaseContext.__init__(self, dimension, lower, amplitude)
        self.means = means
        self.covariance_matrix = covariance_matrix

    def draw_context(self):
        res = multivariate_normal(self.means, self.covariance_matrix)
        res = np.abs(res)
        return res / np.linalg.norm(res) if np.linalg.norm(res) > 1 else res

    def get_means(self):
        return self.means / self.means.sum()
