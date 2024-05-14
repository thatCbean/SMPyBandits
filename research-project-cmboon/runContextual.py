import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualBernoulliArm import ContextualBernoulliArm
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.Policies import UCB, Exp3

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb

horizon = 5000
repetitions = 1
n_jobs = 8
verbosity = 2

environments = [
    {
        "arms": [
            ContextualBernoulliArm([0.1, 0.0, 0.15]), ContextualBernoulliArm([0.1, 0.12, 0.11]), ContextualBernoulliArm([0.2, 0.04, 0.1])
        ],
        "contexts": [
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], 3),
            NormalContext([0.2, 0.2, 0.1], np.identity(3) * [0.1, 0.4, 0.1], 3),
            NormalContext([0.1, 0.1, 0.7], np.identity(3) * [0.1, 0.2, 0.3], 3)
        ]
    }
]

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.02}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.01}}
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
    evaluation.plotRegrets(environment_id, semilogx=True)


for env_id in range(len(environments)):
    plot_env(evaluator, env_id)
