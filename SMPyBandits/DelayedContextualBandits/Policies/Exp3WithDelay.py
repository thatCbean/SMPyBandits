
import numpy as np
from SMPyBandits.DelayedContextualBandits.Policies.BasePolicyWithDelay import BasePolicyWithDelay
from SMPyBandits.Policies.Exp3 import Exp3


class Exp3WithDelay(Exp3, BasePolicyWithDelay):

    def update_estimators(self, arm, reward):
        if self.unbiased:
            reward = reward / self.trusts[arm]
        # Multiplicative weights
        self.weights[arm] *= np.exp(reward * (self.gamma / self.nbArms))
        # Renormalize weights at each step
        self.weights /= np.sum(self.weights)