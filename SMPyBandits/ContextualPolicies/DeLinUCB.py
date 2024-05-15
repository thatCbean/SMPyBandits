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

    def __init__(self, nbArms, dimension, alpha=ALPHA, lambda_reg=1, delta=0.1,
                 lower=0., amplitude=1.):
        super(LinUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha > 0, "Error: the 'alpha' parameter for the LinUCB class must be greater than 0"
        assert dimension > 0, "Error: the 'dimension' parameter for the LinUCB class must be greater than 0"
        self.alpha = alpha
        self.k = nbArms
        self.dimension = dimension
        self.A = np.identity(dimension)
        self.b = np.zeros(dimension)
        #CHnage this 
        T = 1000000
        self.beta = [2 * np.log(1/self.delta) + self.dim * (np.log(1+t  / (self.lambda_reg * self.dim)))  for t in range(1,T)] 

    def startGame(self):
        """Start with uniform weights."""
        super(LinUCB, self).startGame()
        self.A = np.identity(self.dimension)
        self.b = np.zeros(self.dimension)

    def __str__(self):
        return r"linUCB($\alpha: {:.3g}$)".format(self.alpha)

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
        #theta_t = np.matmul(np.linalg.inv(self.A), self.b)
        covxa = np.inner(self.A,context.T)
        max_val = -1
        index = -1
        for i in range(self.k):
            # p_ta = (np.matmul(np.transpose(theta_t), context) +
            #         self.alpha * math.sqrt(
            #             np.matmul(np.transpose(context), np.matmul(np.linalg.inv(self.A), context))))
            p_ta = np.dot(self.hat_theta, context) + \
                            self.alpha * (np.sqrt(self.beta[self.t-1] + self.lambda_reg)
                            + np.sum(self.bias)) * (np.dot(context,covxa)) 
            if p_ta > max_val:
                max_val = p_ta
                index = i

        return index
    
    # is this the main difference: + np.sum(self.bias)) * (np.dot(context,covxa)) 
    # self.beta = [2 * log(1/self.delta) + self.dim * (log(1+t  / (self.lambda_reg * self.dim)))  for t in range(1,T)] 
    #elements are not dependent on each other, so we can calculate them one at a time
    
    # invcov -> A
    # def selectArm(self,arms):
    #     """
    #     This function implements the randomized LinUCB algorithm in delayed environment.
    #     It discards all observations received within the last m time steps.
    #     Input:
    #     -------
    #     arms : (K x d) array containing K arms in dimension d

    #     Output:
    #     -------
    #     chosen_arm : index of the pulled arm
    #     """
    #     if not self.initialized:
    #         return None     # Better raise error

    #     K = len(arms)
    #     self.UCBs = np.zeros(K)
        
    #     for i in range(K):
    #         context = arms[i,:]
    #         covxa = np.inner(self.A,context.T)     
    #         self.UCBs[i] = np.dot(self.hat_theta, context) 
    #                        + self.alpha * (np.sqrt(self.beta[self.t-1] + self.lambda_reg)
    #                        + np.sum(self.bias)) * (np.dot(context,covxa)) 
            
        
    #     #print(self.bias)
    #     #print(np.sum(self.bias))
    #     #print(self.UCBs)
        
    #     #bias update (exact elliptical norm)
    #     A_norm = np.dot(arms[chosen_arm,:],np.inner(self.A,arms[chosen_arm,:].T))
    #     self.bias.append(A_norm) #automatically remove overflow
        
    #     xxt = np.outer(arms[chosen_arm,:],arms[chosen_arm,:].T)
    #     self.cov += xxt
    #     self.A = pinv(self.cov)
    #     #self.Delta = min(self.m, self.Delta +1)

    #     return chosen_arm


        
    # def updateState(self, disclosure):
    #     "disclosure is a list of T lists containing all data to be displayed at time t"

    #     for feedback in disclosure: 
    #         delay, reward, features = feedback
    #         self.xy += reward * features

            
    #     self.hat_theta = np.inner(self.A,self.xy)
    #     self.t += 1

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
