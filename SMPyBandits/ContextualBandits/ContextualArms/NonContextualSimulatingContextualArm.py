
__author__ = "Cody Boon"
__version__ = "0.1"

import math

from SMPyBandits.Policies.kullback import klBern
from numpy.random import binomial, normal

import numpy as np

from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm

REWARD_MEAN = 0.5
REWARD_VAR = 0.2


class NonContextualSimulatingContextualArm(ContextualArm):

    """ An arm that generates a reward using a Normal distribution with a probability based on its mean """
    def __init__(self, reward_mean=REWARD_MEAN, reward_variance=REWARD_VAR, dimension=1):
        super(__class__, self).__init__()
        self.reward_mean = reward_mean
        self.reward_var = reward_variance
        self.lower = 0
        self.amplitude = 1
        self.min = 0
        self.max = 1
        self.dimension = dimension

    def __str__(self):
        return "NonContextualGaussian"

    def __repr__(self):
        return "NonContextualGaussian(mu: {}, sigma^2: {}, dimension: {})".format(self.reward_mean, self.reward_var, self.dimension)

    def draw(self, theta_star, context, t=None):
        return min(
            1.0,
            abs(
                normal(
                    self.reward_mean, self.reward_var
                )
            )
        )

    def is_nonzero(self):
        return True