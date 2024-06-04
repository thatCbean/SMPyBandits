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


class DeLinUCB(ContextualBasePolicyWithDelay):
    """
    The linUCB contextual bandit policy.
    """

    def __init__(self, nbArms, dimension, horizon, alpha=ALPHA, lambda_reg=0.001, m=1500,
                 lower=0., amplitude=1.):
        super(DeLinUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha > 0, "Error: the 'alpha' parameter for the LinUCB class must be greater than 0"
        assert dimension > 0, "Error: the 'dimension' parameter for the LinUCB class must be greater than 0"
        print("Initiating policy LinUCB with {} arms, dimension: {}, alpha: {}".format(nbArms, dimension, alpha))
        self.alpha = alpha
        self.k = nbArms
        self.dimension = dimension
        self.lambda_reg = lambda_reg
        self.time_window = deque(maxlen=m)
        self.time_window.append(np.zeros(dimension))
        self.f = self.compute_f_values(lambda_reg, alpha, dimension, horizon)
        self.Vinv = np.identity(dimension)


    def startGame(self):
        """Start with uniform weights."""
        super(DeLinUCB, self).startGame()
        self.V = np.identity(self.dimension)
        self.B = np.zeros(self.dimension)
        self.Vinv = np.identity(self.dimension)
        self.a = 2 * self.f[0]

    def __str__(self):
        return r"DelinUCB($\alpha: {:.3g}$)".format(self.alpha)

    def update_estimators(self, arm, reward, contexts):
        self.V = self.V + np.outer(contexts[arm], contexts[arm]) + self.lambda_reg * np.identity(self.dimension)
        self.Vinv = np.linalg.inv(self.V)
        ##TODO
        ##update_estimators is called for rewards that have already been observed, so they can be added to B
        self.B = self.B + reward * contexts[arm]
        self.time_window.popleft()
        self.a = 2 * self.f[self.t] + self.helper()
        self.time_window.append(contexts[arm])

    def helper(self):
        time_window_sum = 0
        for context in self.time_window:
            time_window_sum = time_window_sum + self.get_L2_norm(context, self.Vinv)

        return time_window_sum


    def choice(self, contexts):
        theta_t = self.Vinv @ self.B
        max_val = -np.inf
        index = -1
        for arm in range(self.k):
            p_ta = np.inner(theta_t, contexts[arm]) + np.inner(self.a, self.get_L2_norm(contexts[arm], self.Vinv))
            if p_ta > max_val:
                max_val = p_ta
                index = arm

        return index

    def get_L2_norm(self, a, M):
        return np.sqrt(np.transpose(a) @ (M @ a))
    
    def compute_f_values(self, lambda_, delta, d, T):
        f_values = []
        for t in range(1, 2*T):
            term1 = np.sqrt(lambda_)
            term2 = np.sqrt(2 * np.log(1 / delta) + d * np.log((d * lambda_ + t) / (d * lambda_)))
            f_t = term1 + term2
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
