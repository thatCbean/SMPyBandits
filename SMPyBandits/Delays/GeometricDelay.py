
import numpy as np


class GeometricDelay(BaseDelay):
    """ Geometric delay, with a fixed parameter. """

    def __init__(self, dimension, min_delay=0.0, max_delay=0.0, p=0.5):
        super(GeometricDelay, self).__init__(dimension, min_delay=min_delay, max_delay=max_delay)
        self.p = p

    def draw_delay(self):
        return np.random.geometric(self.p)