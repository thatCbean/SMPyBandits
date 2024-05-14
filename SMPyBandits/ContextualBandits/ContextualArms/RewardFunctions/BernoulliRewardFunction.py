import numpy as np
from numpy.random import binomial


def bernoulliReward(theta, context, t):
    p = np.abs(np.inner(context, theta))
    return binomial(1, p)


def debugBernoulliReward(theta, context, t):
    print("BernoulliReward called with theta: {} and context {} at time {}".format(theta, context, t))
    p = np.abs(np.inner(context, theta))
    return binomial(1, p)
