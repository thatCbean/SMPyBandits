import numpy as np
from SMPyBandits.Arms import Constant
from SMPyBandits.ContextualBandits.Contexts.GaussianContext import GaussianContext

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualBernoulliArm import ContextualBernoulliArm
from SMPyBandits.ContextualBandits.ContextualArms.ContextualConstantArm import ContextualConstantArm
from SMPyBandits.ContextualBandits.ContextualArms.ContextualSparseArm import ContextualSparseArm
from SMPyBandits.ContextualBandits.ContextualArms.RewardFunctions.ContextualGaussianNoiseArm import \
    ContextualGaussianNoiseArm
from rowczarskiResearchProject.context.ContextNDimensionalGaussian import ContextNDimensionalGaussian

SPARSITY = 1


def create_sparse_coefficients(dimension, sparsity):
    """Create a sparse coefficient vector of given dimension and sparsity"""
    beta = np.zeros(dimension)
    indices = np.random.choice(range(dimension), sparsity, replace=False)
    beta[indices] = np.random.uniform(0., 1., sparsity)
    return beta


def generate_environment(num_arms, num_contexts, dimension, sparsity):
    inx = create_sparse_coefficients(dimension, sparsity)
    arms = [ContextualSparseArm(inx) for _ in range(num_arms)]

    return {"arms": arms, "contextNDimensional": ContextNDimensionalGaussian(armNumber=num_arms, dimension=dimension)}


# Generate the environment with n arms, n contexts, dimension d, sparsity s_0
environments = [generate_environment(20, 20, 100, 5)]

# environments = [
#     {
#         "arms": [
#             ContextualSparseArm([0.1, 0.0, 0.15, 0, 0, 0, 0, 0.12, 0, 0.1, 0, 0]),
#             ContextualSparseArm([0.1, 0.0, 0, 0, 0, 0, 0.12, 0, 0.15, 0.1, 0, 0]),
#         ],
#         "contexts": [
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#         ],
#     }
# ]

# environments = [
#     {
#         "arms": [
#             ContextualSparseArm([0.1, 0, 0.15, 0, 0, 0.12, 0]),
#             ContextualSparseArm([0, 0.1, 0, 0.05, 0.1, 0, 0]),
#         ],
#         "contexts": [
#             NormalContext([0.1]*7, np.identity(7) * 0.1, 7),
#             NormalContext([0.2]*7, np.identity(7) * 0.2, 7),
#         ],
#     }
# ]

# environments = [
#     {
#         "arms": [
#             ContextualSparseArm([0.1, 0, 0, 0, 0.05]),
#             ContextualSparseArm([0, 0.1, 0.05, 0, 0]),
#             ContextualSparseArm([0.05, 0, 0.1, 0, 0]),
#         ],
#         "contexts": [
#             NormalContext([0.2]*5, np.identity(5) * 0.05, 5),
#             NormalContext([0.1]*5, np.identity(5) * 0.1, 5),
#             NormalContext([0.1]*5, np.identity(5) * 0.1, 5),
#         ],
#     }
# ]


# environments = [
#     {
#         "arms": [
#             ContextualSparseArm([0.1, 0.0, 0.15, 0, 0, 0, 0, 0.12, 0, 0.1, 0, 0]),
#             ContextualSparseArm([0.1, 0.0, 0, 0, 0, 0, 0.12, 0, 0.15, 0.1, 0, 0]),
#             ContextualSparseArm([0.1, 0.0, 0.15, 0.1, 0, 0, 0, 0.12, 0, 0, 0, 0]),
#             ContextualSparseArm([0.1, 0.0, 0.15, 0, 0.1, 0, 0, 0.12, 0, 0, 0, 0]),
#             ContextualSparseArm([0.1, 0.0, 0.15, 0, 0, 0, 0, 0.12, 0, 0.1, 0, 0]),
#             ContextualSparseArm([0.1, 0.0, 0.15, 0, 0.15, 0, 0, 0.12, 0, 0, 0, 0]),
#             ContextualSparseArm([0.0, 0.1, 0.15, 0, 0, 0, 0, 0.12, 0, 0.15, 0, 0]),
#             ContextualSparseArm([0.1, 0.12, 0.0, 0, 0, 0, 0, 0.12, 0, 0.1, 0, 0]),
#             ContextualSparseArm([0.12, 0.0, 0, 0, 0, 0, 0, 0.12, 0.12, 0.1, 0, 0]),
#         ],
#         "contexts": [
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#             NormalContext([0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], np.identity(12) * [0.2, 0.1, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, 0.1, 0.3, 0.2, 0.1], 12),
#         ],
#     }
# ]


# environments = [
#     {
#         "arms": [
#             ContextualBernoulliArm([0.1, 0.0, 0.15]), ContextualBernoulliArm([0.1, 0.12, 0.11]), ContextualBernoulliArm([0.2, 0.04, 0.1]), ContextualConstantArm(0), ContextualConstantArm(0), ContextualConstantArm(0), ContextualConstantArm(0), ContextualConstantArm(0), ContextualConstantArm(0)
#         ],
#         "contexts": [
#             NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
#             NormalContext([0.2, 0.2, 0.1], np.identity(3) * [0.1, 0.4, 0.1], 3),
#             NormalContext([0.1, 0.1, 0.7], np.identity(3) * [0.1, 0.2, 0.3], 3),
#             NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
#             NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
#             NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
#             NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
#             NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
#             NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
#         ],
#     }
# ]
