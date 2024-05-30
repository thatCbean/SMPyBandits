# -*- coding: utf-8 -*-
""" Base class for any policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""
from __future__ import division, print_function

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy
from SMPyBandits.Policies import BasePolicy  # Python 2 compatibility

__author__ = "Cody Boon"
__version__ = "0.1"

import numpy as np


class ContextualBasePolicyWithDelay(BasePolicy):
    """ Base class for any contextual policy."""
    
    def pull_arm(self, arm):
        self.t += 1
        self.pulls[arm] += 1
        return arm

    def update_reward(self, arm, reward):
        reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] += reward

    def update_estimators(self, arm, reward, contexts):
        raise NotImplementedError("This method update_estimators() has to be implemented in the child class inheriting from BasePolicy.")

    # def estimatedOrder(self, context):
    #     """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means.
    #
    #     - For a base policy, it is completely random.
    #     """
    #     return np.random.permutation(self.nbArms)
