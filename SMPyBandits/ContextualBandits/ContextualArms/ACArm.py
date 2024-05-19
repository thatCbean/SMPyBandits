import math
import numpy as np
from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm

class ACArm(ContextualArm):
    """ An arm that generates a reward using the given reward function arbitrarily"""

    def __init__(self, theta, w, phi):
        if not isinstance(theta, np.ndarray):
            theta = np.array(theta)
        assert isinstance(w, np.ndarray), \
            "Error: w must be an nd-array"
        assert isinstance(phi, np.ndarray), \
            "Error: phi must be an nd-array"
        assert theta.shape[1] == w.shape[0] == phi.shape[0], \
            "Error: thetas, w, and phi must have compatible shapes"

        super(__class__, self).__init__()
        self.w = w
        self.phi = phi
        self.theta = theta
        self.length = theta.shape[0]
        self.theta_shape = theta[0].shape
        self.t = 0

    def __str__(self):
        return "SlowChangingArm"

    def __repr__(self):
        return "SlowChangingArm(theta: {}, w: {}, phi: {})".format(self.theta, self.w, self.phi)

    def reward_function(self, context, t):
        # Compute reward based on the given formula: theta_{i, a} = sin(w_i t + phi_i)
        return np.sin(self.w * t + self.phi).sum()

    def draw(self, context, t=None):
        assert isinstance(context, np.ndarray), "context must be an np.ndarray"
        assert self.theta_shape == context.shape, "theta shape must be equal to context"

        self.t = self.t + 1
        return self.reward_function(context, self.t)

    def set(self, thetas):
        assert isinstance(thetas, np.ndarray), "thetas must be an np.ndarray"
        self.thetas = thetas

    def is_nonzero(self):
        return np.linalg.norm(self.theta) != 0