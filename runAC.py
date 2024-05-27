import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualBandits.ContextualPolicies.LinEXP3 import LinEXP3
from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.ContextualBandits.ContextualArms.ACArm import ACArm
from SMPyBandits.Policies import UCB, Exp3


HORIZON = 10000
REPETITIONS = 20
d = 3 

HORIZON = 1000

# Environment 1: Draw nbArms of w_i and phi_i of d dimensions as the reward function setup
w_list = [
    np.array([0.2, 0.1, 0.3]),
    np.array([0.2, 0.2, 0.1]),
    np.array([0.05, 0.1, 0.82]),
    np.array([0.44, 0.6, 0.2])
]

phi_list = [
    np.array([0.2, 0.3, 0.1]),
    np.array([0.1, 0.4, 0.1]),
    np.array([0.2, 0.1, 0.45]),
    np.array([0.4, 0.9, 0.6])
]

# Create a list of ACArm instances
arms = [ACArm(w, phi) for w, phi in zip(w_list, phi_list)]

multivariate_contexts = [NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0.2, 0.3, 0.1], d),
                     NormalContext([0.2, 0.2, 0.1], np.identity(3) * [0.1, 0.4, 0.1], d),
                     NormalContext([0.1, 0.1, 0.7], np.identity(3) * [0.1, 0.2, 0.3], d),
                     NormalContext([0.5, 0.5, 0.35], np.identity(3) * [0.6, 0.6, 0.4], d)]

N = NormalContext([0.6, 0.4, 0.65], np.identity(3) * [0.4, 0.4, 0.7], d)
single_context = [N for _ in range(len(w_list))]

environments = [{
        "theta_star": [0.5, 0.5, 0.5], # could be ignored in this environment
        "arms": arms,
        # NormalContext(means, covariance_matrix, dimension)
        "contexts": single_context
}]


policies = [
    {"archtype": LinEXP3, "params": {"dimension": 3, "eta" : 0.2, "gamma": 0.8}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.2}},
    {"archtype": Exp3, "params": {"gamma": 0.8}},
    {"archtype": UCB, "params": {}},
    

]

configuration = {
    "horizon": HORIZON,    # Finite horizon of the simulation
    "repetitions": REPETITIONS,  # number of repetitions
    "n_jobs": 4,        # Maximum number of cores for parallelization: use ALL your CPU
    "verbosity": 5,      # Verbosity for the joblib calls
    "environment": environments,
    "policies": policies
}

evaluator = EvaluatorContextual(configuration)

evaluator.startAllEnv()

def find_highest_reward(w_list, phi_list, context, t):
    highest_reward = 0

    for t in range(HORIZON):
        best = float('-inf')
        for a in range(len(arms)):
            reward = calculate_reward(w_list[a], phi_list[a], context, t)
            if reward > best:
                best = reward
            
        highest_reward += best


    return highest_reward

def calculate_reward(w, phi, context, t):
    return np.dot(np.sin(w * t + phi), context.means)


def plot_env(evaluation, environment_id):
    highest_reward = find_highest_reward(w_list, phi_list, N, HORIZON)
    print("The best possible reward is: {:.6f}".format(highest_reward))
    evaluation.printFinalRanking(environment_id)
    # evaluation.plotRegrets(environment_id)
    # evaluation.plotRegrets(environment_id, semilogx=True)
    # evaluation.plotRegrets(environment_id, meanReward=True)
    # evaluation.plotBestArmPulls(environment_id)


for env_id in range(len(environments)):
    plot_env(evaluator, env_id)
    
