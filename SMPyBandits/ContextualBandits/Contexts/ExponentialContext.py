import numpy as np

from SMPyBandits.ContextualBandits.Contexts.BaseContext import BaseContext


class ExponentialContext(BaseContext):
    """
    A context generator that draws samples from an exponential distribution.
    """

    def get_means(self):
        return 1.0 / self.rate

    def __init__(self, dimension, rate, lower=0., amplitude=1.):
        """
        Initialize the ExponentialContext with parameters for the exponential distribution.

        Args:
        dimension (int): Dimensionality of the output sample vectors.
        rate (float or np.array): Rate parameter of the exponential distribution (lambda), can be a scalar or an array for multivariate.
        lower (float): Lower bound for context values, used in base class.
        amplitude (float): Amplitude for context scaling, used in base class.
        """
        super().__init__(dimension, lower, amplitude)
        self.rate = rate
        self.dimension = dimension
        self.means = [1.0 / self.rate] * self.dimension

    def draw_context(self):
        """
        Draw a sample vector from the exponential distribution.

        Returns:
        np.array: A vector sampled from the exponential distribution.
        """
        if np.isscalar(self.rate):
            sample = np.random.exponential(scale=1.0 / self.rate, size=self.dimension)
        else:
            sample = np.random.exponential(scale=1.0 / np.array(self.rate), size=self.dimension)
        print(sample)
        return sample / np.linalg.norm(sample)
