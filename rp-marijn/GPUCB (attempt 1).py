# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 compatibility

import math

import numpy as np
import GPy

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

#: Default :math:`\delta` parameter.
DELTA = 0.80


class GPUCB2(ContextualBasePolicy):
    """
    The GP-UCB contextual bandit policy.
    """

    def __init__(self, nbArms, dimension, kernel, sampler, delta=DELTA,
                 lower=0., amplitude=1.):
        super(GPUCB2, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert delta > 0, "Error: the 'delta' parameter for the GP-UCB class must be greater than 0"
        assert dimension > 0, "Error: the 'dimension' parameter for the GP-UCB class must be greater than 0"
        print("Initiating policy GP-UCB with {} arms, dimension: {}, delta: {}".format(nbArms, dimension, delta))
        self.actions = np.arange(0, nbArms, 1.)
        self.sampler = sampler
        self.beta = None
        self.delta = delta
        self.k = nbArms
        self.dimension = dimension
        self.kernel = kernel
        self.gp = None
        self.X = []
        self.Y = []

    def startGame(self, contexts):
        """Start with uniform weights."""
        super(GPUCB2, self).startGame()
        self.contexts = contexts
        print("ACTIONS: " + str(self.actions))
        print("CONTEXTS: " + str(self.contexts))
        self.input_mesh = np.array(np.meshgrid, self.actions, self.contexts)
        self.number_of_actions_in_each_context = np.size(self.input_mesh, 2)
        self.input_space = self.input_mesh.reshape(self.input_mesh.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.input_space.shape[0])])
        self.sigma = np.array([.5 for _ in range(self.input_space.shape[0])])

    def __str__(self):
        return r"GP-UCB($\delta: {:.3g}$)".format(self.delta)

    def getReward(self, arm, reward, contexts):
        super(GPUCB2, self).getReward(arm, reward, contexts)  # XXX Call to BasePolicy

    def choice(self):
        self.beta = self.optimal_beta()
        grid_idx = self.cgp_ucb_rule
        self.sample(self.input_space[grid_idx])
        self.gp = GPy.models.GPRegression(np.array(self.X), np.array(self.Y), self.kernel)
        self.gp.optimize(messages = False)
        self.mu, variances = self.gp.predict(self.input_space)
        self.sigma = np.sqrt(variances)
        #theta_t = np.linalg.inv(self.A) @ self.b
        #max_val = -np.inf
        #index = -1
        #for a in range(self.k):
        #    p_ta = (np.inner(theta_t, contexts[a])) + (
        #            self.alpha * math.sqrt(
        #                # TODO: Currently taking absolute value to prevent domain errors, but may need to change this
        #                np.abs(np.inner(
        #                    contexts[a],
        #                    (np.linalg.inv(self.A) @ contexts[a])
        #                ))
        #            )
        #    )
        #                # max(0, np.transpose(contexts[a]) @ (np.linalg.inv(self.A) @ contexts[a]))))
        #    if p_ta > max_val:
        #        max_val = p_ta
        #        index = a
        #return index

    def cgp_ucb_rule(self, context_index):
        """
        this point selection strategy combines a greedy of choice of choosing a point with high mu, together with
        an exploratory choice of choosing a point with high variance; achieving a balance of exploration & exploitation.
        :param context_index: index of the context you are referring to.
        :return:  next point to be sampled.
        """

        # deduce the indices of the actions for the given context.
        context = int(context_index)
        lower_bound_on_actions = context*self.number_of_actions_in_each_context
        upper_bound_on_actions = (context+1)*self.number_of_actions_in_each_context

        # point selection rule
        return lower_bound_on_actions + np.argmax(
        self.mu[lower_bound_on_actions:upper_bound_on_actions] +
        self.sigma[lower_bound_on_actions:upper_bound_on_actions] * np.sqrt(self.beta))

    def optimal_beta(self):
        """
        :param t: the current round t.
        :param input_space_size: |D| of input space D.
        :param delta: hyperparameter delta where 0 < delta < 1, exclusively.
        :return: optimal beta for exploration_exploitation trade-off at round t.
        """
        return 2 * np.log(self.input_space.size * (self.t ** 2) * (np.pi ** 2) / (6 * self.delta))
    
    def sample(self, x):
        """
        :param x: the point to be sampled from environment.
        :return:
        """
        y = self.sampler(x)
        self.X.append(x)
        self.Y.append(y)
        return