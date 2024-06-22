# -*- coding: utf-8 -*-
"""
The linUCB policy.

Reference:
    [Contextual Bandits with Linear Payoff Functions, W. Chu, L. Li, L. Reyzin, R.E. Schapire, Algorithm 1 (linUCB)]
"""
from __future__ import division, print_function  # Python 2 compatibility

from collections import deque
import math

import numpy as np

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy
from SMPyBandits.DelayedContextualBandits.Policies.ContextualBasePolicyWithDelay import ContextualBasePolicyWithDelay

#: Default :math:`\alpha` parameter.
ALPHA = 0.01


class OTFLinUCB(ContextualBasePolicy):
    """
    The linUCB contextual bandit policy.
    """

    def __init__(self, nbArms, dimension, horizon, alpha=ALPHA, lambda_reg=1, m=500,
                 lower=0., amplitude=1.):
        super(OTFLinUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha > 0, "Error: the 'alpha' parameter for the LinUCB class must be greater than 0"
        assert dimension > 0, "Error: the 'dimension' parameter for the LinUCB class must be greater than 0"
        print("Initiating policy LinUCB with {} arms, dimension: {}, alpha: {}".format(nbArms, dimension, alpha))
        self.alpha = alpha
        self.k = nbArms
        self.dimension = dimension
        self.m = m
        self.lambda_reg = lambda_reg
        self.time_window = deque(maxlen=m)
        self.f = self.compute_f_values(lambda_reg, alpha, dimension, horizon)
        self.V = self.lambda_reg * np.identity(self.dimension)
        self.Vinv = np.linalg.inv(self.V)
        self.B = np.zeros(self.dimension)

    def __str__(self):
        return r"OTFLinUCB($\alpha: {:.3g}$, $m={}$)".format(self.alpha, self.m)

    def update_covariance_matrix(self, context):
        self.V += np.outer(context, context) 
        self.Vinv = np.linalg.inv(self.V)
        self.time_window.append(context)
    def update_beta(self, delay, reward):
        if delay < len(self.time_window):
            self.B += reward * self.time_window[-1 - delay]

    # def update_estimators(self, delay, reward, context):
    #     if delay < len(self.time_window):
    #         self.B += reward * self.time_window[-1 - delay]
    #     self.V += np.outer(context, context) 
    #     self.Vinv = np.linalg.inv(self.V)

    def helper(self):
        time_window_sum = 0
        for context in self.time_window:
            time_window_sum += self.get_L2_norm(context, self.Vinv)

        return time_window_sum


    def choice(self, contexts):
        theta_t = self.Vinv @ self.B
        #confidence_interval_width = 2 * self.f[self.t] + self.helper()
       
        # print(self.helper())
        max_val = -np.inf
        index = -1
        for arm in range(self.k):
            # print(np.inner(theta_t, contexts[arm]) + 0.01 * self.get_L2_norm(contexts[arm], self.Vinv),
            #        np.inner(theta_t, contexts[arm]) + confidence_interval_width * self.get_L2_norm(contexts[arm], self.Vinv))
            # print(confidence_interval_width)
            #print(confidence_interval_width * self.get_L2_norm(contexts[arm], self.Vinv))
            p_ta = np.inner(theta_t, contexts[arm]) + self.alpha * self.get_L2_norm(contexts[arm], self.Vinv)
            # print(confidence_interval_width)
            # print(self.f[self.t])
            # print(confidence_interval_width * self.get_L2_norm(contexts[arm], self.Vinv))
            # print("Reward: ", np.inner(theta_t, contexts[arm]), "exploitation: ", confidence_interval_width * self.get_L2_norm(contexts[arm], self.Vinv))
            if p_ta > max_val:
                max_val = p_ta
                index = arm

        return index
    
    def get_L2_norm(self, x, M):
        # return np.sqrt(x.T @ ( M @ x))
        return np.sqrt(np.dot(x.T, np.dot(M, x)))
    
    def compute_f_values(self, lambda_, delta, d, T):
        f_values = []

        for t in range(1, T + 2):
            f_t = np.sqrt(lambda_) + np.sqrt(2 * np.log10(1 / delta) + d * np.log10((d * lambda_ + t) / (d * lambda_)))
            f_values.append(f_t)
        return f_values

# Init
# Inputs: α ∈ R+, K, d ∈ N
# 1: A ← Id {The d-by-d identity matirx}
# 2: b ← 0d


# 3: for t = 1, 2, 3, . . . , T do
# 4: θt ← A−1b
# 5: Observe K features, xt,1, xt,2, · · · , xt,K ∈ Rd
# 6: for a = 1, 2, . . . , K do
# 7: pt,a ← θ>
# t  xt,a + α√( x^T_(t,a) * A^(−1)x_(t,a) ) {Compute upper confidence bound}
# 8: end for

# 9: Choose action a_t = arg max_a (p_(t,a)) with ties broken arbitrarily
# 10: Observe payoff rt ∈ {0, 1}
# 11: A ← A + x(t,a_t) * x^T_(t,a_t)
# 12: b ← b + x_(t,a_t) * r_t
# 13: end for
