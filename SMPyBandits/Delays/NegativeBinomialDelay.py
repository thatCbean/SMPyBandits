import numpy as np

from SMPyBandits.Delays.BaseDelay import BaseDelay


class NegativeBinomialDelay(BaseDelay):
    """ Geometric delay, with a fixed parameter. """

    def __init__(self, dimension, min_delay=0.0, max_delay=0.0, mean = 50, n = 100):
        super(NegativeBinomialDelay, self).__init__(dimension, min_delay=min_delay, max_delay=max_delay, mean=mean)
        self.p = n / (n + mean)
        self.n = n

    def draw_delay(self):
        return np.random.negative_binomial(self.n, self.p)
    
    def __repr__(self):
        return f"NegativeBinomialDelay(mean={self.mean}, n={self.n})"