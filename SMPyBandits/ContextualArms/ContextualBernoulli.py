
__author__ = "Cody Boon"
__version__ = "0.1"

from SMPyBandits.Policies.kullback import klBern
from numpy.random import binomial, normal

from SMPyBandits.Arms import Arm
import numpy as np

from SMPyBandits.ContextualArms.ContextualArm import ContextualArm


class ContextualBernoulli(ContextualArm):

    """ An arm that generates a reward using a Bernoulli distribution with a probability based on context """
    def __init__(self, theta):
        if not isinstance(theta, np.ndarray):
            theta = np.array(theta)
        super(__class__, self).__init__()
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

    # def draw_nparray(self, contexts, shape=(1,)):
    #     assert isinstance(contexts, np.ndarray), "contexts must be an np.ndarray"

    def calculate_mean(self, context):
        assert isinstance(context, np.ndarray), "context must be an np.ndarray"
        assert context.shape == self.theta.shape, "context must be of the same dimension as arm"
        self.mean = np.inner(context, self.theta)
        return self.mean

    # TODO: Just copy pasted these for the code to run, need to validate use and function
    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        return klBern(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Bernoulli arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klBern(mu, mumax)
