import numpy as np
from SMPyBandits.Arms import Constant

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualBernoulliArm import ContextualBernoulliArm
from SMPyBandits.ContextualBandits.ContextualArms.ContextualConstantArm import ContextualConstantArm

SPARSITY = 1





environments = [
    {
        "arms": [
            ContextualBernoulliArm([0.1, 0.0, 0.15]), ContextualBernoulliArm([0.1, 0.12, 0.11]), ContextualBernoulliArm([0.2, 0.04, 0.1]), ContextualConstantArm(0), ContextualConstantArm(0), ContextualConstantArm(0), ContextualConstantArm(0), ContextualConstantArm(0), ContextualConstantArm(0)
        ],
        "contexts": [
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
            NormalContext([0.2, 0.2, 0.1], np.identity(3) * [0.1, 0.4, 0.1], 3),
            NormalContext([0.1, 0.1, 0.7], np.identity(3) * [0.1, 0.2, 0.3], 3),
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
        ],
    }
]
