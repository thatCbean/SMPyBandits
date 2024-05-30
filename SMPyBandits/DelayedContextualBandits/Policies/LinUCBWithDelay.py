# -*- coding: utf-8 -*-
"""
The linUCB policy.

Reference:
    [Contextual Bandits with Linear Payoff Functions, W. Chu, L. Li, L. Reyzin, R.E. Schapire, Algorithm 1 (linUCB)]
"""
from __future__ import division, print_function  # Python 2 compatibility

import math

import numpy as np

from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.DelayedContextualBandits.Policies.ContextualBasePolicyWithDelay import ContextualBasePolicyWithDelay


class LinUCBWithDelay(LinUCB, ContextualBasePolicyWithDelay):

    def update_estimators(self, arm, reward, contexts):
        self.A = self.A + (np.outer(contexts[arm], contexts[arm]))
        self.b = self.b + (contexts[arm] * np.full(self.dimension, reward))