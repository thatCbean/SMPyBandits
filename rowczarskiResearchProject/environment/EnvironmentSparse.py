import numpy as np

from SMPyBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualArms.ContextualBernoulli import ContextualBernoulli


SPARSITY = 5

environments = [
    {
        "arm_type": ContextualBernoulli,
        "arm_params": [np.array([0.1, 0.2, 0.15]), np.array([0.1, 0.12, 0.11]), np.array([0.3, 0.04, 0.1]), np.array([0.2, 0.12, 0.13]), np.array([0.1, 0.2, 0.15]), np.array([0.1, 0.12, 0.11]), np.array([0.3, 0.04, 0.1])],
        "context_type": NormalContext,
        "context_params": [np.array([0.2, 0.1, 0.3]), np.identity(3) * [0.1, 0.2, 0.3], 3],
        "sparsity": SPARSITY
    }
]
