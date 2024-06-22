# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 compatibility

import math

import numpy as np

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

class KernelUCB(ContextualBasePolicy):
    """
    The Kernel-UCB contextual bandit policy.
    """

    def __init__(self, nbArms, dimension, kname, kern, eta, gamma,
                 lower=0., amplitude=1.):
        super(KernelUCB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert dimension > 0, "Error: the 'dimension' parameter for the KernelUCB class must be greater than 0"
        print("Initiating policy KernelUCB with {} arms, dimension: {}, eta: {}, gamma: {}".format(nbArms, dimension, eta, gamma))
        self.k = nbArms
        self.dimension = dimension

        self.narms = nbArms
        # Number of context features
        self.ndims = dimension
        # regularization parameter
        self.eta = eta
        # exploration parameter
        self.gamma = gamma
        # kernel function name
        self.kname = kname
        # kernel function
        self.kern = kern
        # u_n_t values
        self.u = np.zeros(self.narms)
        # sigma_n_t values
        self.sigma = np.zeros(self.narms)
        # list of contexts of chosen actions to the moment
        self.pulled = []
        # list of rewards corresponding to chosen actions to the moment
        self.a_rewards = []
        self.Kinv = []
        self.KinvLast = []
        

    def startGame(self):
        """Start with uniform weights."""
        super(KernelUCB, self).startGame()

    def __str__(self):
        return r"KernelUCB(k: {}, $\eta: {:.3g}, \gamma: {:.3g}$)".format(self.kname, self.eta, self.gamma)

    def getReward(self, arm, reward, context):
        # get the flattened context and reshape it to an array of shape (narms,ndims)
        context = np.reshape(context, (self.narms, self.ndims))
        # append the context of choesn arm (index = [arm]) with the previous list of contexts (self.pulled)
        # the obserbved context is being reshaped into a column vector simultanesously for future kernel calculations
        self.pulled.append(context[arm].reshape(1,-1))
        # set currently observed context of chosen arm as x_t
        x_t = context[arm].reshape(1,-1)
        
        #========================================
        #    Calculating all possible k_x ...
        #========================================
        
        # To perform kernel UCB in the least and efficient time as possible I propose to
        # calculate k_x for all of the contexts and not just for chosen context (x_t)
        # this will be hugely beneficiary to calculating sigma_n_t step in for loop
        
        # calculate the kernel between each of the contexts of narms and the pulled 
        # contexts of chosen arms to the moment
        
        # self.pulled is just a list of arrays, and hence reshaping it to a valid
        # numpy array of shape (t+1,ndims). Since t is starting from zero
        # it is being added by 1 to give valid shape in each round especially for
        # the first round
        k_x = self.kern(context,np.reshape(self.pulled,(self.t+1,self.ndims)))
        
        # append the observed reward value of chosen action to the previous list of rewards
        self.a_rewards.append(reward)
        # generate array of y. Since t is starting from zero
        # it is being added by 1 to give valid shape in each round especially for
        # the first round
        self.y = np.reshape(self.a_rewards,(self.t+1,1))
        
        # building inverse of kernel matrix for first round is different from consequent rounds.
        if self.t==0:
            self.Kinv = 1.0/(self.kern(x_t,x_t) + self.gamma)
        else:
            # set inverse of kernel matrix as the kernel matrix inverse of the previous round
            Kinv = self.KinvLast
            # set b as k_(x_t) excluding the kernel value of the current round
            b = k_x[arm][:-1]
            # reshape b into the valid numpy column vector
            b = b.reshape(self.t,1)
            # compute b.T.dot(kernel matrix inverse)
            bKinv = np.dot(b.T,Kinv)
            # compute (kernel matrix inverse).dot(b)
            Kinvb = np.dot(Kinv,b)
            
            #==========================================================================
            #    Calculating components of current Kernel matrix inverse (Kinv_t)
            #==========================================================================
            
            K22 = 1.0/(k_x[arm][-1] + self.gamma - np.dot(bKinv,b))            
            K11 = Kinv + K22*np.dot(Kinvb,bKinv)
            K12 = -K22*Kinvb
            K21 = -K22*bKinv
            K11 = np.reshape(K11,(self.t,self.t))
            K12 = np.reshape(K12,(self.t,1))
            K21 = np.reshape(K21,(1,self.t))
            K22 = np.reshape(K22,(1,1))
            # stack components into an array of shape(self.t, self.t)
            self.Kinv = np.vstack((np.hstack((K11,K12)),np.hstack((K21,K22)))) 

        super(KernelUCB, self).getReward(arm, reward, context)  # XXX Call to BasePolicy

    def choice(self, context):
        # get the flattened context and reshape it to an array of shape (narms,ndims)
        context = np.reshape(context, (self.narms,self.ndims))
        self.KinvLast = self.Kinv
        if self.t == 0:
            # Always start with action 1
            self.u[0] = 1.0
        else:
            
            #========================================
            #    Calculating all possible k_x ...
            #========================================
        
            # To perform kernel UCB in the least and efficient time as possible I propose to
            # calculate k_x for all of the contexts and not just for chosen context (x_t)
            # this will be hugely beneficiary to calculating sigma_n_t step in for loop
        
            # calculate the kernel between each of the contexts of narms and the pulled 
            # contexts of chosen arms to the moment
        
            # self.pulled is just a list of arrays, and hence reshaping it to a valid
            # numpy array of shape (t+1,ndims). Since t is starting from zero
            # it is being added by 1 to give valid shape in each round especially for
            # the first round
            
            k_x = self.kern(context, np.reshape(self.pulled, (self.t, self.ndims)))
            
            #===============================
            #    MAIN LOOP ...
            #===============================
            
            for i in range(self.narms):
                self.sigma[i] = np.sqrt(
                    self.kern(context[i].reshape(1, -1), context[i].reshape(1,-1)) -
                        k_x[i].T.dot(self.KinvLast).dot(k_x[i]))  
                self.u[i] = k_x[i].T.dot(self.KinvLast).dot(self.y) + (self.eta/np.sqrt(self.gamma))*self.sigma[i]
            
        # Breaking ties arbitrarily
        action = np.random.choice(np.where(self.u==max(self.u))[0])
        return action