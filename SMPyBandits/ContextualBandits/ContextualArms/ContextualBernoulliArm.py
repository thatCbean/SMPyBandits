
__author__ = "Cody Boon"
__version__ = "0.1"

from SMPyBandits.Policies.kullback import klBern
from numpy.random import binomial

import numpy as np

from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm


class ContextualBernoulliArm(ContextualArm):

    """ An arm that generates a reward using a Bernoulli distribution with a probability based on context """
    def __init__(self, theta):
        if not isinstance(theta, np.ndarray):
            theta = np.array(theta)
        super(__class__, self).__init__()
        if np.linalg.norm(theta) > 1:
            theta = theta / np.linalg.norm(theta)
        self.lower = 0
        self.amplitude = 1
        self.min = 0
        self.max = 1
        self.theta = theta

    def __str__(self):
        return "ContextualBernoulli"

    def __repr__(self):
        return "ContextualBernoulli(theta: {})".format(self.theta)

    def draw(self, context, t=None):
        assert isinstance(context, np.ndarray), "context must be an np.ndarray"
        assert self.theta.shape == context.shape, "theta shape must be equal to context"
        p = np.abs(np.inner(context, self.theta))
        return binomial(1, p)

    def set(self, theta):
        assert isinstance(theta, np.ndarray), "theta must be an np.ndarray"
        self.theta = theta

    def is_nonzero(self):
        return np.linalg.norm(self.theta) != 0
