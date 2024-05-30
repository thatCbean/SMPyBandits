
import numpy as np


class BaseDelay(object):

    def __init__(self, dimension, min_delay=0.0, max_delay=0.0):
        assert dimension > 0, "Dimension needs to be greater than zero"
        self.dimension = dimension

    def __str__(self):
        return self.__class__.__name__

    def draw(self):
        raise NotImplementedError(
            "This method, draw_delay(), must be implemented in the child class inheriting from BaseDelay"
        )
    
    def draw_nparray(self, shape=(1,)):
        np.array([self.draw_delay() for _ in range(np.multiply(shape))]).reshape(shape)

    



    