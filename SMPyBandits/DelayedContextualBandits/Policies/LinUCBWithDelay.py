# -*- coding: utf-8 -*-
"""
The linUCB policy.

Reference:
    [Contextual Bandits with Linear Payoff Functions, W. Chu, L. Li, L. Reyzin, R.E. Schapire, Algorithm 1 (linUCB)]
"""
from __future__ import division, print_function  # Python 2 compatibility

import math

import numpy as np

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.DelayedContextualBandits.Policies.ContextualBasePolicyWithDelay import ContextualBasePolicyWithDelay


ALPHA = 0.01

class LinUCBWithDelay(ContextualBasePolicy): # XXX Call to BasePolicy

    def __init__(self, nbArms, dimension, alpha=ALPHA,
                 lower=0., amplitude=1.):
        super(LinUCBWithDelay, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha > 0, "Error: the 'alpha' parameter for the LinUCB class must be greater than 0"
        assert dimension > 0, "Error: the 'dimension' parameter for the LinUCB class must be greater than 0"
        print("Initiating policy LinUCB with {} arms, dimension: {}, alpha: {}".format(nbArms, dimension, alpha))
        self.alpha = alpha
        self.k = nbArms
        self.dimension = dimension
        self.A = np.identity(dimension)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros(dimension)

    def __str__(self):
        return r"LinUCB($\alpha: {:.3g}$)".format(self.alpha)    
        
    def update_covariance_matrix(self, context):
        self.A +=  np.outer(context, context)
        self.Ainv = np.linalg.inv(self.A)

    def update_beta(self, reward, context):
        self.b += reward * context

    # def update_estimators(self, reward, context):
    #     self.A = self.A + np.outer(context, context)
    #     self.b = self.b + reward * context


    def choice(self, contexts):
        theta_t = self.Ainv @ self.b
        max_val = -np.inf
        index = -1
        for a in range(self.k):
            p_ta = (np.inner(theta_t, contexts[a])) + (
                    self.alpha * math.sqrt(
                        # TODO: Currently taking absolute value to prevent domain errors, but may need to change this
                        np.abs(np.inner(
                            contexts[a],
                            (self.Ainv @ contexts[a])
                        ))
                    )
            )
            # print("Reward: ", np.inner(theta_t, contexts[a]), "exploitation: ", self.alpha * math.sqrt(np.abs(np.inner(contexts[a],(np.linalg.inv(self.A) @ contexts[a])))))
                        # max(0, np.transpose(contexts[a]) @ (np.linalg.inv(self.A) @ contexts[a]))))
            if p_ta > max_val:
                max_val = p_ta
                index = a

        return index