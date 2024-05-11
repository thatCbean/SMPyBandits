import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualBernoulliArm import ContextualBernoulliArm

SPARSITY = 1

environments = [
    {
        "arm_type": ContextualBernoulliArm,
        "arm_params": [[0.1, 0.2, 0.15], [0.1, 0.12, 0.11], [0.3, 0.04, 0.1]],
        "context_type": NormalContext,
        "context_params":  [
                    [0.2, 0.1, 0.3],
                    np.identity(3) * [0.1, 0.2, 0.3],
                    3
                ],
        "sparsity": SPARSITY
    }
]
