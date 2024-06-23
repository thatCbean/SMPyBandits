# -*- coding: utf-8 -*-
""" Base class for any policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Cody Boon"
__version__ = "0.1"

import numpy as np


class ContextualBasePolicy(object):
    """ Base class for any contextual policy."""

    def __init__(self, nbArms, lower=0., amplitude=1., group=-1):
        """ New policy."""
        # Parameters
        assert nbArms > 0, "Error: the 'nbArms' parameter of a {} object cannot be <= 0.".format(self)  # DEBUG
        self.nbArms = nbArms  #: Number of arms
        self.lower = lower  #: Lower values for rewards
        assert amplitude > 0, "Error: the 'amplitude' parameter of a {} object cannot be <= 0.".format(self)  # DEBUG
        self.amplitude = amplitude  #: Larger values for rewards
        self.group = group
        # Internal memory
        self.t = 0  #: Internal time
        self.pulls = np.zeros(nbArms, dtype=int)  #: Number of pulls of each arms
        self.rewards = np.zeros(nbArms)  #: Cumulated rewards of each arms

    def __str__(self):
        """ -> str"""
        return self.__class__.__name__

    def __repr__(self):
        """ -> repr str"""
        return self.__class__.__name__

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        self.t = 0
        self.pulls.fill(0)
        self.rewards.fill(0)

    def getReward(self, arm, reward, contexts, t=0):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
        self.t += 1
        self.pulls[arm] += 1
        # reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] += reward

    def choice(self, context, t=0):
        """ Not defined."""
        raise NotImplementedError("This method choice() has to be implemented in the child class inheriting from BasePolicy.")

    # def estimatedOrder(self, context):
    #     """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means.
    #
    #     - For a base policy, it is completely random.
    #     """
    #     return np.random.permutation(self.nbArms)
