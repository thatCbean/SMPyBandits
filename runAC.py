import numpy as np

from SMPyBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualArms.ContextualBernoulli import ContextualBernoulli
from SMPyBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualPolicies.LinEXP3 import LinEXP3
from SMPyBandits.Environment.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.Policies import UCB, Exp3

# Sinusoidal reward function
def custom_reward(t, w, phi):
    return np.sin(w * t + phi)

horizon = 1000
d = 3 # number of arms

change_points = np.arange(0, horizon, 200).tolist()
w = np.random.uniform(0.1, 1.0, d)  # Random frequencies
phi = np.random.uniform(0, 2 * np.pi, d)  # Random phase shifts

list_of_means = [
    [custom_reward(t, w[i], phi[i]) for i in range(d)]
    for t in change_points
]

environments = [
    {
        "arm_type": ContextualBernoulli,
        "arm_params": [[0.1, 0.2, 0.15], [0.1, 0.12, 0.11], [0.3, 0.04, 0.1]],
        "params": {
                "listOfMeans": list_of_means,
                "changePoints": change_points,
            },
        "context_type": NormalContext,
        "context_params": [[0.2, 0.1, 0.3], np.identity(3) * [0.1, 0.2, 0.3], 3]
    }
]

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.01}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.05}},
    {"archtype": LinEXP3, "params": {"dimension": 3, "eta" : 0.2, "gamma": 0.05}}
]

configuration = {
    "horizon": horizon,    # Finite horizon of the simulation
    "repetitions": 10,  # number of repetitions
    "n_jobs": 1,        # Maximum number of cores for parallelization: use ALL your CPU
    "verbosity": 5,      # Verbosity for the joblib calls
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
