# -*- coding: utf-8 -*-

from __future__ import division, print_function  # Python 2 compatibility

import numpy as np

# Local imports
try:
    from .Arm import Arm
except ImportError:
    from Arm import Arm


class Adversarial(Arm):
    """ Arm with a reward that changes as a function of time.

    - `reward_function` is the reward function,
    - `lower`, `amplitude` default to `floor(constant_reward)`,
    """

    def __init__(self, reward_function, mean, time_weight, error, lower=0., amplitude=1.):
        """ New arm."""
        self.reward_function = reward_function  #: Constant value of rewards
        self.lower = lower  #: Known lower value of rewards
        self.time_weight = time_weight
        self.error = error
        self.amplitude = amplitude  #: Known amplitude of rewards
        self.mean = mean  #: Mean for this Constant arm

    # --- Random samples

    def draw(self, t):
        """ Draw one adversarial sample. The time t is used to compute the reward."""
        return self.reward_function(self.time_weight * t + self.error)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of constant samples, of a certain shape."""
        support_array = np.arange(shape) + 1
        return self.reward_function(self.time_weight * support_array + self.error)
        # return np.full(shape, self.constant_reward)

    def set_mean_param(self, mean):
        self.mean = mean

    # --- Printing

    def __str__(self):
        return "Adversarial"

    def __repr__(self):
        return "Adversarial({:.3g})".format(self.mean)

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The `kl(x, y) = abs(x - y)` to use for this arm."""
        return abs(x - y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Constant arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / abs(mumax - mu)


__all__ = ["Adversarial"]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
