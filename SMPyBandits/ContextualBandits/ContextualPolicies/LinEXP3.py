import numpy as np
import numpy.random as rn
from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

ETA = 0.1
GAMMA = 0.1
BETA = 0.5
M = 1000

class LinEXP3(ContextualBasePolicy):
    """
    The linEXP3 contextual bandit policy with a sophisticated estimator.
    """

    def __init__(self, nbArms, dimension, eta=ETA, gamma=GAMMA, beta=BETA, m=M, lower=0., amplitude=1.):
        super(LinEXP3, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert eta > 0, "Error: the 'eta' parameter for the LinEXP3 class must be greater than 0"
        assert 0 < gamma < 1, "Error: the 'gamma' parameter must be in the range (0, 1)"
        assert dimension > 0, "Error: the 'dimension' parameter for the LinEXP3 class must be greater than 0"
        assert beta > 0, "Error: the 'beta' parameter for the LinEXP3 class must be greater than 0"
        assert m > 0, "Error: the 'm' parameter for the LinEXP3 class must be greater than 0"

        self.eta = eta
        self.gamma = gamma
        self.beta = beta
        self.m = m
        self.dimension = dimension
        self.k = nbArms
        self.theta_hats = np.zeros((nbArms, dimension))
        self.cumulative_theta_hats = np.zeros((nbArms, dimension))
        self.Sigma = np.zeros((nbArms, dimension, dimension))
        for a in range(nbArms):
            self.Sigma[a] = np.eye(dimension)  # Initialize covariance matrix with identity matrix

    def startGame(self):
        """Initialize weights and parameter estimates."""
        super(LinEXP3, self).startGame()
        self.theta_hats = np.zeros((self.k, self.dimension))
        self.cumulative_theta_hats = np.zeros((self.k, self.dimension))
        self.Sigma = np.zeros((self.k, self.dimension, self.dimension))
        for a in range(self.k):
            self.Sigma[a] = np.eye(self.dimension)  # Reinitialize covariance matrix with identity matrix

    def __str__(self):
        return r"linEXP3($\eta: {:.3g}$, $\gamma: {:.3g}$)".format(self.eta, self.gamma)

    # def getReward(self, arm, reward, context): # basic covariance matrix computation
    #     """Update the parameter estimates for the chosen arm."""
    #     super(LinEXP3, self).getReward(arm, reward, context)

    #     self.cumulative_loss[arm] -= reward

    #     self.Sigma[arm] += np.outer(context[arm], context[arm])

    #     inv_Sigma = np.linalg.inv(self.Sigma[arm])
    #     self.theta_hats[arm] = np.dot(inv_Sigma, context[arm]) * reward

    def getReward(self, arm, reward, context):
        """Update the parameter estimates for the chosen arm."""
        super(LinEXP3, self).getReward(arm, reward, context)

        self.Sigma[arm] += np.outer(context[arm], context[arm])

        inv_Sigma = np.linalg.inv(self.Sigma[arm])
        inner_product = np.dot(context[arm], self.theta_hats[arm])
        self.theta_hats[arm] = inv_Sigma @ (inner_product * context[arm]) * (-reward)
        for a in range(self.k):
            if a == arm:
                self.theta_hats[arm] += inv_Sigma @ (inner_product * context[arm]) * (-reward)
            else:
                self.cumulative_theta_hats[a] += self.theta_hats[a]

    def choice(self, context):
        """Choose an arm based on the LINEXP3 policy."""
        weights = np.zeros(self.k)
        
        # Update weights based on cumulative sum of past contexts and losses
        for a in range(self.k):
            weights[a] = np.exp(-self.eta * np.dot(context[a], self.cumulative_theta_hats[a]))

        # Compute probabilities
        sum_weights = np.sum(weights)
        probabilities = (1 - self.gamma) * (weights / sum_weights) + self.gamma / self.k

        # Choose arm
        return rn.choice(self.k, p=probabilities)