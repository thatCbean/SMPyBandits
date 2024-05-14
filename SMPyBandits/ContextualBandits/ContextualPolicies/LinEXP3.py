from __future__ import division, print_function  # Python 2 compatibility

import numpy as np
import numpy.random as rn

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

ETA = 0.1
GAMMA = 0.1

class LinEXP3(ContextualBasePolicy):
    """
    The linEXP3 contextual bandit policy.
    """

    def __init__(self, nbArms, dimension, eta=ETA, gamma=GAMMA, lower=0., amplitude=1.):
        super(LinEXP3, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert eta > 0, "Error: the 'eta' parameter for the LinEXP3 class must be greater than 0"
        assert 0 < gamma < 1, "Error: the 'gamma' parameter must be in the range (0, 1)"
        assert dimension > 0, "Error: the 'dimension' parameter for the LinEXP3 class must be greater than 0"

        self.eta = eta
        self.gamma = gamma
        self.dimension = dimension
        self.k = nbArms
        self.weights = np.ones(nbArms)
        self.theta_hats = np.zeros((nbArms, dimension))

    def startGame(self):
        """Initialize weights and parameter estimates."""
        super(LinEXP3, self).startGame()
        self.weights = np.ones(self.k)
        self.theta_hats = np.zeros((self.k, self.dimension))

    def __str__(self):
        return r"linEXP3($\eta: {:.3g}$, $\gamma: {:.3g}$)".format(self.eta, self.gamma)

    def getReward(self, arm, reward, context):
        """Update the parameter estimates for the chosen arm."""
        super(LinEXP3, self).getReward(arm, reward, context)  # Call to BasePolicy
        self.theta_hats[arm] += self.eta * reward * context[arm]

    def choice(self, context):
        """Choose an arm based on the LINEXP3 policy."""
        # Update weights
        for a in range(self.k):
            exponent = -self.eta * np.dot(context[a], self.theta_hats[a])
            self.weights[a] = np.exp(exponent)

        # Compute probabilities
        sum_weights = np.sum(self.weights)
        probabilities = (1 - self.gamma) * (self.weights / sum_weights) + self.gamma / self.k

        # Choose arm
        return rn.choice(self.k, p=probabilities)
