# -*- coding: utf-8 -*-
"""
The CW_OFUL policy.

Reference:
    [J. He, D. Zhou, T. Zhang, and Q. Gu, “Nearly optimal algorithms for linear contextual bandits with adversarial
    corruptions,” Advances in neural information processing systems, vol. 35, pp. 34 614–34 625, 2022]
"""
from __future__ import division, print_function  # Python 2 compatibility

import math

import numpy as np

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

#: Default :math:`\alpha, \beta, \lambda` parameters.
ALPHA = 0.01
BETA = 0.01
LAMBDA = 0.01


class CW_OFUL(ContextualBasePolicy):
    """
    The CW-OFUL contextual adversarial bandit policy.
    """

    def __init__(self, nbArms, dimension, alpha=ALPHA,
                 beta=BETA, labda=LAMBDA, lower=0.,
                 amplitude=1.):
        super(CW_OFUL, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha > 0, "Error: the 'alpha' parameter for the CW_OFUL class must be greater than 0"
        assert beta > 0, "Error: the 'beta' parameter for the CW_OFUL class must be greater than 0"
        assert labda > 0, "Error: the 'lambda' parameter for the CW_OFUL class must be greater than 0"
        assert dimension > 0, "Error: the 'dimension' parameter for the LinUCB class must be greater than 0"
        self.alpha = alpha
        self.labda = labda
        self.beta = beta
        self.k = nbArms
        self.dimension = dimension
        self.sigma = np.identity(dimension) * labda
        self.b = np.zeros(dimension)
        self.theta_k = np.zeros(dimension)
        self.omega_k = 1

    def startGame(self):
        """Start with uniform weights."""
        super(CW_OFUL, self).startGame()
        self.sigma = np.identity(self.dimension) * self.labda
        self.b = np.zeros(self.dimension)
        self.theta_k = np.zeros(self.dimension)

    def __str__(self):
        return r"CW_OFUL($\alpha: {:.3g}, \beta: {:.3g}, \lambda: {:.3g}$)".format(self.alpha, self.beta, self.labda)

    def getReward(self, arm, reward, contexts, t=0):
        r"""Process the received reward
        """
        super(CW_OFUL, self).getReward(arm, reward, contexts)  # XXX Call to BasePolicy
        self.sigma = self.sigma + (self.omega_k * (contexts[arm] @ np.transpose(contexts[arm])))
        self.b = self.b + (self.omega_k * contexts[arm] * reward)
        self.theta_k = np.linalg.inv(self.sigma) @ self.b

    def choice(self, contexts, t=0):
        maxx = -np.inf
        index = -1
        for k in range(self.k):
            x_k = (np.transpose(self.theta_k) @ contexts[k] + (self.beta * np.sqrt(np.transpose(contexts[k]) @ np.linalg.inv(self.sigma) @ contexts[k])))
            if x_k > maxx:
                maxx = x_k
                index = k
        self.omega_k = min(1, self.alpha / np.sqrt(np.transpose(contexts[index]) @ np.linalg.inv(self.sigma)))
