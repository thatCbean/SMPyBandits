import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ChangingArm import ChangingArm
from SMPyBandits.ContextualBandits.ContextualArms.RewardFunctions.BernoulliRewardFunction import bernoulliReward, debugBernoulliReward
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.Policies import UCB, Exp3

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb

horizon = 10000
repetitions = 25
n_jobs = 8
verbosity = 4

environments = [
    {
        "arms": [
                       ChangingArm(np.array([[0.1, 0.2, 0.15], [0.3, 0.1, 0.05]]), bernoulliReward, horizon, True),
                       ChangingArm(np.array([[0.3, 0.3, 0.3], [0.1, 0.5, 0.3]]), bernoulliReward, horizon, True),
                       ChangingArm(np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.1], [0.2, 0.2, 0.2]]), bernoulliReward, horizon, True)
                       ],
        "contexts": [
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.1, 0.2, 0.3], 3),
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.1, 0.2, 0.3], 3),
            NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.1, 0.2, 0.3], 3),
        ]
    }
]

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.01}},
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
    evaluation.plotRegrets(environment_id, meanReward=True)
    # evaluation.plotBestArmPulls(environment_id)


for env_id in range(len(environments)):
    plot_env(evaluator, env_id)
