# -*- coding: utf-8 -*-
"""
The BOB policy.

Reference:
    [W. C. Cheung, D. Simchi-Levi, and R. Zhu, “Learning to optimize under non-stationarity,” in The 22nd
    International Conference on Artificial Intelligence and Statistics, PMLR, 2019, pp. 1079–1087.]
"""
from __future__ import division, print_function  # Python 2 compatibility

import math

import numpy as np
from numpy.random import binomial
from scipy.special import bernoulli

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

#: Default :math:`\alpha, \beta, \lambda` parameters.
_R = 0.1
_L = 1
_S = 1
LAMBDA = 0.01


class BOB(ContextualBasePolicy):
    """
    The BOB contextual changing-reward bandit policy.
    """

    def __init__(self, nbArms, dimension, horizon, R=_R,
                 L=_L, S=_S, labda=LAMBDA, lower=0.,
                 amplitude=1.):
        super(BOB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.R = R
        self.L = L
        self.S = S
        self.labda = labda
        self.dimension = dimension
        self.horizon = horizon

        self.H = (dimension ** (2 / 3)) * (horizon ** (1 / 2))
        self.DELTA = np.log(self.H, np.e)
        self.J = [1.0]
        J_step = (1 - np.floor(self.H ** (1 / self.DELTA)))
        while self.J[-1] < self.H:
            if self.J[-1] + J_step < self.H:
                self.J.append(self.J[-1] + J_step)
        self.J.append(self.H)

        self.gamma = min(1, np.sqrt(
            (
                    ((self.DELTA + 1) * np.log(self.DELTA + 1, np.e)) /
                    ((np.e - 1) * np.ceil(self.horizon / self.H))
            )
        ))
        self.s_j1 = np.full(np.ceil(self.DELTA + 1), 1)
        self.j = np.zeros(self.DELTA)
        self.w_i = 0
        self.V_i = np.identity(self.dimension) * self.labda
        self.contexts = np.zeros((self.horizon, self.dimension))
        self.rewards = np.zeros(self.horizon)
        self.p_i = list()

    def startGame(self):
        """Start with uniform weights."""
        super(BOB, self).startGame()
        self.s_j1 = np.full(np.ceil(self.DELTA + 1), 1)
        self.j = np.zeros(self.DELTA)
        self.w_i = 0
        self.V_i = np.identity(self.dimension) * self.labda
        self.contexts = np.zeros((self.horizon, self.dimension))
        self.rewards = np.zeros(self.horizon)
        self.p_i = list()

    def __str__(self):
        return r"BOB($$)".format()

    def getReward(self, arm, reward, contexts, t=0):
        r"""Process the received reward
        """
        super(BOB, self).getReward(arm, reward, contexts)  # XXX Call to BasePolicy
        self.rewards[t] = reward
        self.contexts[t] = contexts[arm]

        sigma = np.zeros((self.dimension, self.dimension))
        for s in range(max(0, t - self.w_i + 1), max(0, t)):
            sigma += np.outer(self.contexts[s], self.contexts[s])

        self.V_i = (np.identity(self.dimension) * self.labda) + sigma

        if t % self.H == self.H - 1:
            j = np.floor(t / self.H)
            self.s_j1[j] *= (
                np.e ** (
                    (self.gamma / ((self.DELTA + 1) * self.p_i[j])) * (
                        1/2 + (
                            np.sum(self.rewards[t-self.H+1:]) / (
                                (2 * self.H) + (4 * self.R * np.sqrt(self.H * np.log(self.horizon / np.sqrt(self.H))))
                            )
                        )
                    )
                )
            )

    def choice(self, contexts, t=0):
        if t % self.H == 0:
            self.p_i = list()
            for j in range(self.DELTA + 1):
                self.p_i.append(
                    (1 - self.gamma) *
                    ((self.s_j1[j] / np.sum(self.s_j1)) + (self.gamma / (self.DELTA + 1)))
                )
                self.j[j] = binomial(1, self.p_i[j]) * self.j[j]
            self.w_i = np.floor(self.H ** (self.j[t] / self.DELTA))
            self.V_i = np.identity(self.dimension) * self.labda

        theta_t = np.zeros(self.dimension)
        for s in range(max(0, t - 1), max(0, t - self.w_i)):
            theta_t += (np.linalg.inv(self.V_i) * (self.contexts[s] * self.rewards[s]))

        highest = -np.inf
        index = -1
        for i, context in enumerate(contexts):
            res = np.inner(context, theta_t) + (
                np.sqrt(
                    np.inner(context, (np.linalg.inv(self.V_i) @ context))
                ) * (
                    (np.sqrt(self.labda) * self.S) + self.R * (
                        np.sqrt(
                            self.dimension * np.log((
                                self.horizon * (1 + (self.w_i * (self.L ** 2) / self.labda))
                            ), np.e)
                        )
                    )
                )
            )
            if res > highest:
                highest = res
                index = i

        return index


