import numpy as np
import numpy.random as rn
from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

ETA = 0.1
GAMMA = 0.1

class LinEXP3(ContextualBasePolicy):
    """
    The linEXP3 contextual bandit policy with a sophisticated estimator.
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
        self.theta_hats = np.zeros((nbArms, dimension))
        self.cumulative_loss = np.zeros((nbArms, dimension))
        self.Sigma = np.zeros((nbArms, dimension, dimension))
        for a in range(nbArms):
            self.Sigma[a] = np.eye(dimension)  # Initialize covariance matrix with identity matrix

    def startGame(self):
        """Initialize weights and parameter estimates."""
        super(LinEXP3, self).startGame()
        self.theta_hats = np.zeros((self.k, self.dimension))
        self.cumulative_loss = np.zeros((self.k, self.dimension))
        self.Sigma = np.zeros((self.k, self.dimension, self.dimension))
        for a in range(self.k):
            self.Sigma[a] = np.eye(self.dimension)  # Reinitialize covariance matrix with identity matrix

    def __str__(self):
        return r"linEXP3($\eta: {:.3g}$, $\gamma: {:.3g}$)".format(self.eta, self.gamma)

    def getReward(self, arm, reward, context):
        """Update the parameter estimates for the chosen arm."""
        super(LinEXP3, self).getReward(arm, reward, context)  # Call to BasePolicy

        loss = -reward  # Assuming reward is given, and we need to convert it to loss
        self.cumulative_loss[arm] += loss
        
        # Update the covariance matrix
        self.Sigma[arm] += np.outer(context[arm], context[arm])
        
        # Update the theta_hat estimator using the sophisticated formula
        inv_Sigma = np.linalg.inv(self.Sigma[arm])
        self.theta_hats[arm] = np.dot(inv_Sigma, context[arm]) * loss

    def choice(self, context):
        """Choose an arm based on the LINEXP3 policy."""
        weights = np.zeros(self.k)
        
        # Update weights based on cumulative sum of past contexts and losses
        for a in range(self.k):
            weights[a] = np.exp(-self.eta * np.dot(context[a], self.theta_hats[a]))

        # Compute probabilities
        sum_weights = np.sum(weights)
        probabilities = (1 - self.gamma) * (weights / sum_weights) + self.gamma / self.k

        # Choose arm
        return rn.choice(self.k, p=probabilities)