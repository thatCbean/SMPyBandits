# -*- coding: utf-8 -*-
""" The linUCB policy.

Reference: [Contextual Bandits with Linear Payoff Functions, W. Chu, L. Li, L. Reyzin, R.E. Schapire, Algorithm 1 (linUC)]
"""
from __future__ import division, print_function  # Python 2 compatibility

import math

import numpy as np
import numpy.random as rn

from SMPyBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

#: Default :math:`\alpha` parameter.
ALPHA = 0.01


class LinUCB(ContextualBasePolicy):
    """
    The linUCB contextual bandit policy.
    """

    def __init__(self, nbArms, dimension, alpha=ALPHA,
                 lower=0., amplitude=1.):
        super(LinUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha > 0, "Error: the 'alpha' parameter for the LinUCB class must be greater than 0"
        assert dimension > 0, "Error: the 'dimension' parameter for the LinUCB class must be greater than 0"
        self.alpha = alpha
        self.k = nbArms
        self.dimension = dimension
        self.A = np.identity(dimension)
        self.b = np.zeros(dimension)

    def startGame(self):
        """Start with uniform weights."""
        super(LinUCB, self).startGame()
        self.A = np.identity(self.dimension)
        self.b = np.zeros(self.dimension)

    def __str__(self):
        return r"linUCB3($\alpha: {:.3g}$)".format(self.alpha)

    def getReward(self, arm, reward, context):
        r"""Give a reward: accumulate rewards on that arm k, then update the weight :math:`w_k(t)` and renormalize the weights.

        - With unbiased estimators, divide by the trust on that arm k, i.e., the probability of observing arm k: :math:`\tilde{r}_k(t) = \frac{r_k(t)}{\mathrm{trusts}_k(t)}`.
        - But with a biased estimators, :math:`\tilde{r}_k(t) = r_k(t)`.

        .. math::

           w'_k(t+1) &= w_k(t) \times \exp\left( \frac{\tilde{r}_k(t)}{\gamma_t N_k(t)} \right) \\
           w(t+1) &= w'(t+1) / \sum_{k=1}^{K} w'_k(t+1).
        """
        super(LinUCB, self).getReward(arm, reward, context)  # XXX Call to BasePolicy
        self.A = self.A + np.matmul(context, np.transpose(context))
        self.b = self.b + np.multiply(context, reward)

    def choice(self, context):
        theta_t = np.matmul(np.linalg.inv(self.A), self.b)
        max_val = -1
        index = -1
        for i in range(self.k):
            p_ta = (np.matmul(np.transpose(theta_t), context) +
                    self.alpha * math.sqrt(
                        np.matmul(np.transpose(context), np.matmul(np.linalg.inv(self.A), context))))
            if p_ta > max_val:
                max_val = p_ta
                index = i

        return index

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
