
__author__ = "Cody Boon"
__version__ = "0.1"

from SMPyBandits.Policies.kullback import klBern
from numpy.random import binomial, normal

import numpy as np

from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm

NOISE_MEAN = 0
NOISE_VAR = 0.01


class ContextualKernelizedNoiseArm(ContextualArm):

    def __init__(self, noise_mean=NOISE_MEAN, noise_var=NOISE_VAR):
        super(__class__, self).__init__()
        self.noise_mean = noise_mean
        self.noise_var = noise_var
        self.lower = 0
        self.amplitude = 1
        self.min = 0
        self.max = 1

    def __str__(self):
        return "ContextualKernelized"

    def __repr__(self):
        return "ContextualKernelizedWithNoise(mu: {}, sigma^2: {})".format(self.noise_mean, self.noise_var)

    def draw(self, theta_star, context, t=None):
        assert isinstance(context, np.ndarray), "context must be an np.ndarray"
        assert theta_star.shape == context.shape, "theta shape must be equal to context"
        # return min(1, max(0, np.inner(context, self.theta) + normal(self.noise_mean, self.noise_var)))
        x = abs(np.inner(context, theta_star))
        #reward = np.sin(x) + np.cos(x)
        #reward = np.sqrt(x)
        #reward = np.power(x, 3) + np.power(x, 2) + x
        #reward = np.sin(x) / np.power(x, 2)
        #reward = np.sqrt(np.sin(x) * np.cos(x))
        #reward = np.sin(x) * np.cos(x)
        #reward = np.tan(x)
        #reward = x
        reward = np.log(x)
        return min(1, reward),  min(1, reward + normal(self.noise_mean, self.noise_var))

    def is_nonzero(self):
        return True