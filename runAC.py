import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualBandits.ContextualPolicies.LinEXP3 import LinEXP3
from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.ContextualBandits.ContextualArms.ACArm import ACArm
from SMPyBandits.Policies import UCB, Exp3



horizon = 1000
d = 3 

w_1 = np.random.rand(d)
phi_1 = np.random.rand(d)

w_2 = np.random.rand(d)
phi_2 = np.random.rand(d)

w_3 = np.random.rand(d)
phi_3 = np.random.rand(d)


arms = [
    ACArm(np.array([[0.1, 0.2, 0.15], [0.3, 0.1, 0.05], [0.05, 0.1, 0.5]]), w_1, phi_1),
    ACArm(np.array([[0.3, 0.3, 0.3], [0.1, 0.5, 0.3], [0.1, 0.2, 0.6]]), w_2, phi_2),
    ACArm(np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.1], [0.15, 0.05, 0.2]]), w_3, phi_3)
]

contexts = [
    NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], d),
    NormalContext([0.2, 0.2, 0.1], np.identity(3) * [0.1, 0.4, 0.1], d),
    NormalContext([0.1, 0.1, 0.7], np.identity(3) * [0.1, 0.2, 0.3], d)
]

environments = [
    {
        "arms": arms,
        "contexts": contexts
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
    # evaluation.plotRegrets(environment_id)
    # evaluation.plotRegrets(environment_id, semilogx=True)
    evaluation.plotRegrets(environment_id, meanReward=True)
    # evaluation.plotBestArmPulls(environment_id)


for env_id in range(len(environments)):
    plot_env(evaluator, env_id)
    
