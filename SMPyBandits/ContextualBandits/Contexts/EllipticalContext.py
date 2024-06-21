import numpy as np
import scipy.stats as stats
from SMPyBandits.ContextualBandits.Contexts.BaseContext import BaseContext

class EllipticalContext(BaseContext):
    """
    A context generator that draws vectors from an elliptical distribution.
    """
    def __init__(self, radial_dist, means=None, dispersion_matrix=None, dimension=1, lower=0., amplitude=1.):
        """
        Initialize the EllipticalContext with parameters for the radial distribution, means, and dispersion matrix.

        Args:
        radial_dist (callable): A function that generates random samples from the radial distribution.
        means (np.array): Mean values for the distribution, defaults to a uniform distribution across dimensions.
        dispersion_matrix (np.array): Dispersion matrix for the distribution, defaults to scaled identity matrix.
        dimension (int): Dimensionality of the vectors.
        lower (float): Lower bound for context values, used in base class.
        amplitude (float): Amplitude for context scaling, used in base class.
        """
        self.radial_dist = radial_dist

        if means is None:
            means = np.full(shape=dimension, fill_value=(1. / dimension))
        else:
            assert dimension == len(means), f"Dimension mismatch: {dimension} != {len(means)}"

        if dispersion_matrix is None:
            dispersion_matrix = np.identity(dimension) * 0.5

        if not isinstance(means, np.ndarray):
            means = np.array(means)

        assert len(means) == dimension, "Means array must have <dimension> entries"
        assert dispersion_matrix.shape == (dimension, dimension), "Dispersion matrix must be <dimension>x<dimension>"

        super().__init__(dimension, lower, amplitude)

        self.means = means
        self.dispersion_matrix = dispersion_matrix

        print(f"\nInitiating Elliptical Context with params:\n"
              f"Means: {self.means}\n"
              f"Dispersion matrix:\n{self.dispersion_matrix}\n"
              f"Dimension: {self.dimension}\n"
              f"Lower: {self.lower}\n"
              f"Amplitude: {self.amplitude}\n")

    def draw_context(self):
        """
        Draw a vector from the elliptical distribution using the defined mean, dispersion matrix, and radial distribution.

        Returns:
        np.array: A vector sampled from the elliptical distribution.
        """
        normal_sample = np.random.multivariate_normal(self.means, self.dispersion_matrix)
        radial_component = self.radial_dist()
        return (normal_sample * radial_component) / np.linalg.norm(normal_sample * radial_component) if np.linalg.norm(normal_sample * radial_component) > 1 else normal_sample * radial_component

    def get_means(self):
        """
        Normalize and retrieve the means used in this context.

        Returns:
        np.array: Normalized means.
        """
        return self.means / np.linalg.norm(self.means)
