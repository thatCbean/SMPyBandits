import numpy as np
import scipy.stats as stats
from SMPyBandits.ContextualBandits.Contexts.EllipticalContext import EllipticalContext
from SMPyBandits.ContextualBandits.Contexts.ExponentialContext import ExponentialContext

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.Contexts.SkewedNormalContext import SkewedNormalContext

from SMPyBandits.ContextualBandits.ContextualArms.ContextualGaussianNoiseArm import ContextualGaussianNoiseArm

DIMENSION = 100
N = 100


def create_sparse_coefficients(dimension, sparsity):
    """Create a sparse coefficient vector of given dimension and sparsity"""
    beta = np.zeros(dimension)
    indices = np.random.choice(range(dimension), sparsity, replace=False)
    beta[indices] = np.random.uniform(0., 1., sparsity)
    return beta


def generate_environment(num_arms, num_contexts, dimension, sparsity):
    theta_star = create_sparse_coefficients(dimension, sparsity)
    arms = [ContextualGaussianNoiseArm(0, 0.01) for _ in range(num_arms)]
    contexts = [NormalContext(np.zeros(dimension), np.identity(dimension) * 0.5, dimension)
                for _ in range(num_contexts)]
    return {
        "theta_star": theta_star,
        "arms": arms,
        "contexts": contexts
    }


def generate_environment_skewed_normal(num_arms, num_contexts, dimension, sparsity, skewness=1.0):
    theta_star = create_sparse_coefficients(dimension, sparsity)
    arms = [ContextualGaussianNoiseArm(0, 0.01) for _ in range(num_arms)]
    contexts = [
        SkewedNormalContext(skewness, np.zeros(dimension), np.identity(dimension) * 0.5,
                            dimension) for _ in range(num_contexts)]
    return {
        "theta_star": theta_star,
        "arms": arms,
        "contexts": contexts
    }


def radial_dist():
    return stats.expon(scale=1).rvs()  # Exponential distribution with scale 1


def generate_environment_elliptical(num_arms, num_contexts, dimension, sparsity):
    theta_star = create_sparse_coefficients(dimension, sparsity)
    arms = [ContextualGaussianNoiseArm(0, 0.01) for _ in range(num_arms)]

    means = np.random.rand(dimension)
    dispersion_matrix = np.random.rand(dimension, dimension)
    dispersion_matrix = np.dot(dispersion_matrix, dispersion_matrix.T)
    contexts = [EllipticalContext(radial_dist, means, dispersion_matrix, dimension) for _ in range(num_contexts)]
    return {
        "theta_star": theta_star,
        "arms": arms,
        "contexts": contexts
    }


def generate_environment_exponential(num_arms, num_contexts, dimension, sparsity, rate):
    theta_star = create_sparse_coefficients(dimension, sparsity)
    arms = [ContextualGaussianNoiseArm(0, 0.01) for _ in range(num_arms)]
    contexts = [ExponentialContext(dimension, rate) for _ in range(num_contexts)]
    return {
        "theta_star": theta_star,
        "arms": arms,
        "contexts": contexts
    }


# Generate the environment with n arms, n contexts, dimension d, sparsity s_0
environments = [generate_environment(N, N, DIMENSION, 5)]
# environments = [generate_environment_skewed_normal(N, N, DIMENSION, 5, 3)]
# environments = [generate_environment_elliptical(N, N, DIMENSION, 5)]
# environments = [generate_environment_exponential(N, N, DIMENSION, 5, 3)]
