import numpy as np
import scipy.stats as stats
from SMPyBandits.ContextualBandits.Contexts.BaseContext import BaseContext


class SkewedNormalContext(BaseContext):
    """
    A context generator that draws vectors from a skewed normal distribution.
    """

    def __init__(self, alpha, means=None, covariance_matrix=None, dimension=1, lower=0., amplitude=1.):
        """
        Initialize the SkewedNormalContext with parameters for skewness, means, covariance matrix, and dimensions.

        Args:
        alpha (float): Skewness parameter. Positive values imply right skew, negative values imply left skew.
        means (np.array): Mean values for the Gaussian distributions, defaults to uniform distribution across dimensions.
        covariance_matrix (np.array): Covariance matrix for the Gaussian distributions, defaults to scaled identity matrix.
        dimension (int): Dimensionality of the Gaussian vectors.
        lower (float): Lower bound for context values, used in base class.
        amplitude (float): Amplitude for context scaling, used in base class.
        """
        self.alpha = alpha

        if means is None:
            means = np.full(shape=dimension, fill_value=(1. / dimension))
        else:
            assert dimension == len(means), f"Dimension mismatch: {dimension} != {len(means)}"

        if covariance_matrix is None:
            covariance_matrix = np.identity(dimension) * 0.5

        if not isinstance(means, np.ndarray):
            means = np.array(means)

        assert len(means) == dimension, "Means array must have <dimension> entries"
        assert covariance_matrix.shape == (dimension, dimension), "Covariance matrix must be <dimension>x<dimension>"

        super().__init__(dimension, lower, amplitude)

        self.means = means
        self.covariance_matrix = covariance_matrix

        print(f"\nInitiating Skewed Normal Context with params:\n"
              f"Alpha (Skewness): {self.alpha}\n"
              f"Means: {self.means}\n"
              f"Covariance matrix:\n{self.covariance_matrix}\n"
              f"Dimension: {self.dimension}\n"
              f"Lower: {self.lower}\n"
              f"Amplitude: {self.amplitude}\n")

    def draw_context(self):
        """
        Draw a skewed Gaussian vector using the defined mean, covariance matrix, and skewness parameter.

        Returns:
        np.array: A vector sampled from the skewed normal distribution.
        """
        # We assume independence between dimensions
        res = stats.skewnorm.rvs(a=self.alpha, loc=self.means, scale=np.sqrt(np.diag(self.covariance_matrix)))
        return res / np.linalg.norm(res) if np.linalg.norm(res) > 1 else res

    def get_means(self):
        """
        Normalize and retrieve the means used in this context.

        Returns:
        np.array: Normalized means.
        """
        return self.means / np.linalg.norm(self.means)
