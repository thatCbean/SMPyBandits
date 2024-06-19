
import numpy as np

from SMPyBandits.Delays.BaseDelay import BaseDelay


class GeometricDelay(BaseDelay):
    """ Geometric delay, with a fixed parameter. """

    def __init__(self, dimension, min_delay=0.0, max_delay=0.0, mean = 50):
        super(GeometricDelay, self).__init__(dimension, min_delay=min_delay, max_delay=max_delay, mean=mean)
        self.p = 1 / mean

    def draw_delay(self):
        return np.random.geometric(self.p)
    
    def __repr__(self):
        return f"GeometricDelay(mean={self.mean})"