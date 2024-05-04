import numpy as np

from SMPyBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualArms.ContextualBernoulli import ContextualBernoulli
from SMPyBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.Environment.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.Policies import UCB, Exp3

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb

environments = [
    {
        "arm_type": ContextualBernoulli,
        "arm_params": [[0.1, 0.2, 0.15], [0.1, 0.12, 0.11], [0.3, 0.04, 0.1]],
        "context_type": NormalContext,
        "context_params": [[0.2, 0.1, 0.3], np.identity(3) * [0.1, 0.2, 0.3], 3]
    }
]

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.01}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.01}}
]

configuration = {
    "horizon": 1000,
    "repetitions": 10,
    "n_jobs": 4,
    "verbosity": 6,
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
