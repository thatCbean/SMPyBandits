import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualBernoulliArm import ContextualBernoulliArm

environments = [
    {
        "arm_type": ContextualBernoulliArm,
        "arm_params": [0.1, 0.5, 0.9],
        "context_type": NormalContext,
        "context_params":
                    np.identity(3) * [0.1, 0.2, 0.3]

    }
]
