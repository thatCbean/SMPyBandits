import math
import numpy as np
from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm

class ACArm(ContextualArm):
    """ An arm that generates a reward using the given reward function arbitrarily"""

    def __init__(self, w, phi):
        assert isinstance(w, np.ndarray), \
            "Error: w must be an nd-array"
        assert isinstance(phi, np.ndarray), \
            "Error: phi must be an nd-array"
        assert w.shape[0] == phi.shape[0], \
            "Error: w and phi must have compatible shapes"

        super(__class__, self).__init__()
        self.w = w
        self.phi = phi

    def __str__(self):
        return "ArbitraryChangingArm"

    def __repr__(self):
        return "ArbitraryChangingArm( w: {}, phi: {})".format(self.w, self.phi)

    def reward_function(self, context, t):
        return np.dot(np.sin(self.w * t + self.phi), context)

    def draw(self, theta_star, context, t=None):
        return self.reward_function(context, t)
    
    def optimalLoss(self):
        return None

    def is_nonzero(self):
        return np.linalg.norm(self.w) != 0 and np.linalg.norm(self.phi) != 0