
import numpy as np
from SMPyBandits.Delays.BaseDelay import BaseDelay


class PoissonDelay(BaseDelay):
    
    def __init__(self, dimension, min_delay=0.0, max_delay=0.0, lam=1.0):
        super(PoissonDelay, self).__init__(dimension, min_delay=min_delay, max_delay=max_delay)
        self.lam = lam

    def draw_delay(self):
        return np.random.poisson(self.lam)