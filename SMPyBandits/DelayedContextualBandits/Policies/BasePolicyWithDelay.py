# -*- coding: utf-8 -*-
""" Base class for any policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""
from __future__ import division, print_function

from SMPyBandits.Policies import BasePolicy  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np

#: If True, every time a reward is received, a warning message is displayed if it lies outsides of ``[lower, lower + amplitude]``.
CHECKBOUNDS = True
CHECKBOUNDS = False


class BasePolicyWithDelay(BasePolicy):
         
    def pull_arm(self, arm):
        self.t += 1
        self.pulls[arm] += 1
        return arm

    def update_reward(self, arm, reward):
        reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] += reward

    def update_estimators(self, arm, reward):
        raise NotImplementedError("This method update_estimators() has to be implemented in the child class inheriting from BasePolicy.")