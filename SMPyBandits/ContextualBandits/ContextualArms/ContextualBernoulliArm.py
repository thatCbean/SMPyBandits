
__author__ = "Cody Boon"
__version__ = "0.1"

from SMPyBandits.Policies.kullback import klBern
from numpy.random import binomial

import numpy as np

from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm


class ContextualBernoulliArm(ContextualArm):

    """ An arm that generates a reward using a Bernoulli distribution with a probability based on context """
    def __init__(self):
        super(__class__, self).__init__()
        self.lower = 0
        self.amplitude = 1
        self.min = 0
        self.max = 1

    def __str__(self):
        return "ContextualBernoulli"

    def __repr__(self):
        return "ContextualBernoulli"

    def draw(self, theta_star, context, t=None):
        assert isinstance(context, np.ndarray), "context must be an np.ndarray"
        assert theta_star.shape == context.shape, "theta shape must be equal to context"
        p = np.abs(np.inner(context, theta_star))
        return binomial(1, p)

    def is_nonzero(self):
        return True
