import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualBernoulliArm import ContextualBernoulliArm
from SMPyBandits.ContextualBandits.ContextualArms.RewardFunctions.ContextualGaussianNoiseArm import \
    ContextualGaussianNoiseArm
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.Policies import UCB, Exp3

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb

# horizon = 20000
# horizon = 5000
# horizon = 200
horizon = 1000000
repetitions = 8

# Has nice looking graphs
# horizon = 100
# repetitions = 30

n_jobs = 1
verbosity = 2

environments = [
    {
        "arms": [
            ContextualGaussianNoiseArm([0.15, 0.15, 0.15], 0, 0.01),
            ContextualGaussianNoiseArm([0.6, 0.2, 0.2], 0, 0.01),
            ContextualGaussianNoiseArm([0.2, 0.3, 0.7], 0, 0.01),
        ],
        "contexts": [
            NormalContext([0.4, 0.3, 0.2], np.identity(3) * 0.1, 3),
            NormalContext([0.4, 0.3, 0.2], np.identity(3) * 0.5, 3),
            NormalContext([0.4, 0.3, 0.2], np.identity(3) * 0.2, 3)
        ]
    },
    {
        "arms": [
            ContextualGaussianNoiseArm([0.1, 0.1, 0.0], 0, 0.01),
            ContextualGaussianNoiseArm([0.0, 0.1, 0.1], 0, 0.01),
            ContextualGaussianNoiseArm([0.1, 0.0, 0.1], 0, 0.01),
        ],
        "contexts": [
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.7, 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.3, 3)
        ]
    },
    {
        "arms": [
            ContextualGaussianNoiseArm([0.1, 0.1, 0.0], 0, 0.01),
            ContextualGaussianNoiseArm([0.0, 0.1, 0.1], 0, 0.01),
            ContextualGaussianNoiseArm([0.1, 0.0, 0.1], 0, 0.01),
        ],
        "contexts": [
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3)
        ]
    },
    {
        "arms": [
            ContextualGaussianNoiseArm([0.1, 0.0, 0.0], 0, 0.01),
            ContextualGaussianNoiseArm([0.0, 0.1, 0.0], 0, 0.01),
            ContextualGaussianNoiseArm([0.0, 0.0, 0.1], 0, 0.01),
        ],
        "contexts": [
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3)
        ]
    },
    {
        "arms": [
            ContextualGaussianNoiseArm([0.7, 0.7, 0.0], 0, 0.01),
            ContextualGaussianNoiseArm([0.0, 0.7, 0.7], 0, 0.01),
            ContextualGaussianNoiseArm([0.7, 0.0, 0.7], 0, 0.01),
        ],
        "contexts": [
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3)
        ]
    }
]

environments_5d = [
    {
        "arms": [
            ContextualGaussianNoiseArm([0.5, 0.5, 0.0, 0.0, 0.5], 0, 0.01),
            ContextualGaussianNoiseArm([0.0, 0.5, 0.5, 0.5, 0.0], 0, 0.01),
            ContextualGaussianNoiseArm([0.5, 0.0, 0.5, 0.5, 0.0], 0, 0.01),
        ],
        "contexts": [
            NormalContext([0.4, 0.4, 0.4, 0.4, 0.4], np.identity(5) * 0.5, 5),
            NormalContext([0.4, 0.4, 0.4, 0.4, 0.4], np.identity(5) * 0.5, 5),
            NormalContext([0.4, 0.4, 0.4, 0.4, 0.4], np.identity(5) * 0.5, 5)
        ]
    },
    {
        "arms": [
            ContextualGaussianNoiseArm([1.0, 0.0, 0.0, 0.0, 0.0], 0, 0.01),
            ContextualGaussianNoiseArm([0.0, 0.0, 1.0, 0.0, 0.0], 0, 0.01),
            ContextualGaussianNoiseArm([0.0, 0.0, 0.0, 0.0, 1.0], 0, 0.01),
        ],
        "contexts": [
            NormalContext([0.5, 0.5, 0.5, 0.5, 0.5], np.identity(5) * 0.5, 5),
            NormalContext([0.5, 0.5, 0.5, 0.5, 0.5], np.identity(5) * 0.5, 5),
            NormalContext([0.5, 0.5, 0.5, 0.5, 0.5], np.identity(5) * 0.5, 5)
        ]
    }
]

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.05}},
    {"archtype": Exp3, "params": {"gamma": 0.1}},
    {"archtype": Exp3, "params": {"gamma": 0.25}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 100.0}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 50.0}},
    # {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 20.0}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 10.0}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 5.0}},
    # {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 2.0}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 1.0}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.5}},
    # {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.2}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.1}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.05}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.01}}
]

policies_5d = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.05}},
    {"archtype": Exp3, "params": {"gamma": 0.1}},
    {"archtype": Exp3, "params": {"gamma": 0.25}},
    {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 100.0}},
    {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 50.0}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 20.0}},
    {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 10.0}},
    {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 5.0}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 2.0}},
    {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 1.0}},
    {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 0.5}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 0.2}},
    {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 0.1}},
    {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 0.05}},
    {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 0.01}}
]

configuration = {
    "horizon": horizon,
    "repetitions": repetitions,
    "n_jobs": n_jobs,
    "verbosity": verbosity,
    "environment": environments,
    "policies": policies
}

configuration_5d = {
    "horizon": horizon,
    "repetitions": repetitions,
    "n_jobs": n_jobs,
    "verbosity": verbosity,
    "environment": environments_5d,
    "policies": policies_5d
}

evaluator = EvaluatorContextual(configuration)
evaluator_5d = EvaluatorContextual(configuration_5d)

evaluator.startAllEnv()
evaluator_5d.startAllEnv()


def plot_env(evaluation, environment_id):
    evaluation.printFinalRanking(environment_id)
    # evaluation.plotRegrets(environment_id)
    evaluation.plotRegrets(environment_id, relativeRegret=True)
    # evaluation.plotRegrets(environment_id, altRegret=True, relativeRegret=True)
    evaluation.plotRegrets(environment_id, regretOverMaxReturn=True)
    # evaluation.plotRegrets(environment_id, meanReward=True)
    # evaluation.plotRegrets(environment_id, meanReward=True, relativeRegret=True)
    # evaluation.plotRegrets(environment_id, semilogy=True)


for env_id in range(len(environments)):
    plot_env(evaluator, env_id)

for env_id in range(len(environments_5d)):
    plot_env(evaluator_5d, env_id)
