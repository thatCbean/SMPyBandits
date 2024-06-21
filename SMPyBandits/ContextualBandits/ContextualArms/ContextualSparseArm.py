
__author__ = "Cody Boon"
__version__ = "0.1"

from SMPyBandits.Policies.kullback import klBern
from numpy.random import binomial

import numpy as np

from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm


class ContextualSparseArm(ContextualArm):

    """ An arm that generates a sparse reward based on context"""

    def set_theta_param(self, theta):
        pass

    def __init__(self, beta_star, mean_function='identity'):
        if not isinstance(beta_star, np.ndarray):
            beta_star = np.array(beta_star)
        super(__class__, self).__init__()
        if np.linalg.norm(beta_star) > 1:
            beta_star = beta_star / np.linalg.norm(beta_star)
        self.lower = 0
        self.amplitude = 1
        self.min = 0
        self.max = 1
        self.beta_star = beta_star
        print("Beta star: ", beta_star.shape)
        self.mean_function = mean_function

    def __str__(self):
        return "ContextualSparse"

    def __repr__(self):
        return "ContextualSparse(\\beta^*: {})".format(self.beta_star)

    def draw(self, context, t=None):
        assert isinstance(context, np.ndarray), "context must be an np.ndarray"
        assert self.beta_star.shape == context.shape, "beta star shape must be equal to context"
        return np.abs(np.dot(context, self.beta_star)) + 0.5 * np.random.randn()

    def set(self, beta_star):
        assert isinstance(beta_star, np.ndarray), "beta_star must be an np.ndarray"
        self.beta_star = beta_star

    def is_nonzero(self):
        return np.linalg.norm(self.beta_star) != 0
