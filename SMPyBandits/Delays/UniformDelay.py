
import numpy as np
from SMPyBandits.Delays.BaseDelay import BaseDelay


class UniformDelay(BaseDelay):
    

    def draw_delay(self):
        return np.random.randint(self.min_delay, self.max_delay + 1)
    
    def __repr__(self):
        return f"UniformDelay(min={self.min_delay}, max={self.max_delay})"