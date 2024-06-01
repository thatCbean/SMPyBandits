import numpy as np


from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext


from SMPyBandits.ContextualBandits.ContextualArms.ContextualGaussianNoiseArm import ContextualGaussianNoiseArm


def create_sparse_coefficients(dimension, sparsity):
    """Create a sparse coefficient vector of given dimension and sparsity"""
    beta = np.zeros(dimension)
    indices = np.random.choice(range(dimension), sparsity, replace=False)
    beta[indices] = np.random.uniform(0., 1., sparsity)
    return beta


def generate_environment(num_arms, num_contexts, dimension, sparsity):
    theta_star = create_sparse_coefficients(dimension, sparsity)
    arms = [ContextualGaussianNoiseArm(0, 0.01) for _ in range(num_arms)]
    contexts = [NormalContext(create_sparse_coefficients(dimension, dimension), np.identity(dimension) * 0.5, dimension) for _ in range(num_contexts)]
    return {
        "theta_star": theta_star,
        "arms": arms,
        "contexts": contexts
    }


# Generate the environment with n arms, n contexts, dimension d, sparsity s_0
environments = [generate_environment(10, 10, 10, 5)]