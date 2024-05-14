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

horizon = 1000
repetitions = 1
n_jobs = 8
verbosity = 2

environments = [
    {
        "arms": [
            ContextualGaussianNoiseArm([0.4, 0.5, 0.6], 0, 0.01),
            ContextualGaussianNoiseArm([0.6, 0.5, 0.4], 0, 0.01),
            ContextualGaussianNoiseArm([0.5, 0.5, 0.5], 0, 0.01),
        ],
        "contexts": [
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * [0.02, 0.02, 0.02], 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * [0.02, 0.02, 0.02], 3),
            NormalContext([0.4, 0.4, 0.4], np.identity(3) * [0.02, 0.02, 0.02], 3)
        ]
    }
]

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.05}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.04}}
]

configuration = {
    "horizon": horizon,
    "repetitions": repetitions,
    "n_jobs": n_jobs,
    "verbosity": verbosity,
    "environment": environments,
    "policies": policies
}

evaluator = EvaluatorContextual(configuration)

evaluator.startAllEnv()


def plot_env(evaluation, environment_id):
    evaluation.printFinalRanking(environment_id)
    evaluation.plotRegrets(environment_id)
    evaluation.plotAvgRewards(environment_id)
    # evaluation.plotRegrets(environment_id, semilogx=True)


for env_id in range(len(environments)):
    plot_env(evaluator, env_id)
