# -*- coding: utf-8 -*-
"""
The SW-UCB policy.

Reference:
    [W. C. Cheung, D. Simchi-Levi, and R. Zhu, “Learning to optimize under non-stationarity,” in The 22nd
    International Conference on Artificial Intelligence and Statistics, PMLR, 2019, pp. 1079–1087.]
"""
from __future__ import division, print_function  # Python 2 compatibility

import math

import numpy as np

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

#: Default :math:`\alpha, \beta, \lambda` parameters.
LAMBDA = 0.01
DELTA = 0.5
WINDOW_SIZE = 10
_R = 0.1
_L = 1
_S = 1


class SW_UCB(ContextualBasePolicy):
    """
    The SW-UCB contextual changing-reward bandit policy.
    """

    def __init__(self, nbArms, dimension, window_size=WINDOW_SIZE, R=_R, L=_L, S=_S, labda=LAMBDA, delta=DELTA, lower=0.,
                 amplitude=1.):
        super(SW_UCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)

        self.labda = labda
        self.delta = delta
        self.dimension = dimension
        self.window_size = window_size
        self.R = R
        self.L = L
        self.S = S

        self.window_contexts = np.zeros((window_size, dimension))
        self.window_rewards = np.zeros(window_size)
        self.window_index = 0

        self.V_t = np.identity(dimension)

    def startGame(self):
        """Start with uniform weights."""
        super(SW_UCB, self).startGame()
        self.V_t = np.identity(self.dimension) * self.labda
        self.window_contexts = np.zeros((self.window_size, self.dimension))
        self.window_rewards = np.zeros(self.window_size)
        self.window_index = 0

    def __str__(self):
        return r"SW_UCB()".format()

    def getReward(self, arm, reward, contexts, t=0):
        r"""Process the received reward
        """
        super(SW_UCB, self).getReward(arm, reward, contexts)  # XXX Call to BasePolicy
        self.window_contexts[self.window_index] = contexts[arm]
        self.window_rewards[self.window_index] = reward
        self.window_index = (self.window_index + 1) % self.window_size
        sum_matrix = np.zeros((self.dimension, self.dimension))
        for context in self.window_contexts:
            sum_matrix += np.outer(context, context)
        self.V_t = np.identity(self.dimension) * self.labda + sum_matrix

    def choice(self, contexts, t=0):
        sum_vector = np.zeros(self.dimension)
        for i in range(self.window_size):
            sum_vector += self.window_contexts[i] * self.window_rewards[i]
        thetaHat_t = np.linalg.inv(self.V_t) @ sum_vector
        X_t = -1
        highest = -np.inf
        for i in range(len(contexts)):
            res = (
                    np.inner(contexts[i], thetaHat_t) +
                    (
                            np.sqrt(np.inner(contexts[i], self.V_t @ contexts[i])) *
                            (
                                    self.R *
                                    math.sqrt(
                                        self.dimension * math.log(
                                            (1 + ((self.window_size * (self.L ** 2)) / self.labda)) / self.delta)
                                    )
                            ) +
                            (np.sqrt(self.labda) * self.S)
                    )
            )
            if res > highest:
                highest = res
                X_t = i
        return X_t
