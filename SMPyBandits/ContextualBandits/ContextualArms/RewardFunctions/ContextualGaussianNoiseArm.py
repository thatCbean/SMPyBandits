
__author__ = "Cody Boon"
__version__ = "0.1"

from SMPyBandits.Policies.kullback import klBern
from numpy.random import binomial, normal

import numpy as np

from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm

NOISE_MEAN = 0
NOISE_VAR = 0.01


class ContextualGaussianNoiseArm(ContextualArm):

    """ An arm that generates a reward using a Bernoulli distribution with a probability based on context """
    def __init__(self, theta, noise_mean=NOISE_MEAN, noise_var=NOISE_VAR):
        if not isinstance(theta, np.ndarray):
            theta = np.array(theta)
        super(__class__, self).__init__()
        if np.linalg.norm(theta) > 1:
            theta = theta / np.linalg.norm(theta)
        self.noise_mean = noise_mean
        self.noise_var = noise_var
        self.lower = 0
        self.amplitude = 1
        self.min = 0
        self.max = 1
        self.theta = theta

    def __str__(self):
        return "ContextualGaussian"

    def __repr__(self):
        return "ContextualGaussian(theta: {}, mu: {}, sigma^2: {})".format(self.theta, self.noise_mean, self.noise_var)

    def draw(self, context, t=None):
        assert isinstance(context, np.ndarray), "context must be an np.ndarray"
        assert self.theta.shape == context.shape, "theta shape must be equal to context"
        return min(1, max(0, np.inner(context, self.theta) + normal(self.noise_mean, self.noise_var)))

    def set(self, theta):
        assert isinstance(theta, np.ndarray), "theta must be an np.ndarray"
        self.theta = theta

    def is_nonzero(self):
        return np.linalg.norm(self.theta) != 0
