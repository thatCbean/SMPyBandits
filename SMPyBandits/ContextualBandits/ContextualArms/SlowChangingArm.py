__author__ = "Cody Boon"
__version__ = "0.1"

import math

from SMPyBandits.Policies.kullback import klBern
from numpy.random import binomial

import numpy as np

from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm


class SlowChangingArm(ContextualArm):
    """ An arm that generates a reward using some contextual reward function, with theta based on time """

    def __init__(self, thetas, reward_function, horizon, interpolate=False):
        assert callable(reward_function), \
            "Error: reward_function must be a function that takes some array theta and returns a reward"
        assert isinstance(thetas, np.ndarray), \
            "Error: thetas must be an nd-array"
        super(__class__, self).__init__()
        self.reward_function = reward_function
        self.interpolate = interpolate
        self.thetas = thetas
        self.length = thetas.shape[0]
        self.theta_shape = thetas[0].shape
        if interpolate:
            self.interval = math.floor(horizon / (self.length - 1))
        else:
            self.interval = math.floor(horizon / self.length)

    def __str__(self):
        return "SlowChangingArm"

    def __repr__(self):
        return "SlowChangingArm(reward: {}, thetas: {})".format(self.reward_function, self.thetas)

    def draw(self, context, t=None):
        assert isinstance(context, np.ndarray), "context must be an np.ndarray"
        assert self.theta_shape == context.shape, "theta shape must be equal to context"
        assert t is not None, "changing arm must have t set when drawing a reward"

        if self.interpolate:
            formula = t / self.interval
            theta1 = self.thetas[math.floor(formula)]
            theta2 = self.thetas[math.ceil(formula)]
            interp = (theta1 * (1 - (formula % 1))) + (theta2 * (formula % 1))
            return self.reward_function(interp, context, t)
        else:
            return self.reward_function(self.thetas[math.floor(t / self.interval)], context, t)

    def set(self, thetas):
        assert isinstance(thetas, np.ndarray), "thetas must be an np.ndarray"
        self.thetas = thetas

    # def draw_nparray(self, contexts, shape=(1,)):
    #     assert isinstance(contexts, np.ndarray), "contexts must be an np.ndarray"

    def calculate_mean(self, context):
        su = np.zeros(self.theta_shape)
        for theta in self.thetas:
            su = su + theta
        su = su / self.length
        return np.inner(su, context)

    # TODO: Just copy pasted below for the code to run, need to validate use and function
    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        return klBern(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Bernoulli arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klBern(mu, mumax)
