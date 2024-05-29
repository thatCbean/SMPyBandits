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
nbArms = 4 

# Environment 1: Draw nbArms of w_i and phi_i of d dimensions as the reward function setup
w = np.array([0.44, 0.6, 0.2])

phi = np.array([0.2, 0.1, 0.45])

# Create a list of ACArm instances
arms = [ACArm(w, phi) for _ in range(nbArms)]

multivariate_contexts = [NormalContext([0.2, 0.1, 0.3], np.identity(3) * [0., 0., 0.], d),
                     NormalContext([0.5, 0.3, 0.4], np.identity(3) * [0., 0., 0.], d),
                     NormalContext([0.4, 0.3, 0.7], np.identity(3) * [0., 0., 0.], d),
                     NormalContext([0.5, 0.5, 0.35], np.identity(3) * [0., 0., 0.], d)]

environments = [{
        "theta_star": [0.5, 0.5, 0.5], # could be ignored in this environment
        "arms": arms,
        # NormalContext(means, covariance_matrix, dimension)
        "contexts": multivariate_contexts
}]


policies = [
    {"archtype": LinEXP3, "params": {"dimension": d, 
                                     "m": int(np.sqrt(HORIZON)*nbArms*d/10), 
                                     "eta" : 0.2, "gamma": 0.8}},
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

def find_highest_reward():
    highest_reward = 0

    for t in range(HORIZON):
        best = float('-inf')
        for a in range(len(arms)):
            context = multivariate_contexts[a]
            reward = calculate_reward(w, phi, context, t)
            if reward > best:
                best = reward
            
        highest_reward += best


    return highest_reward

def random_policy():
    res = 0
    for t in range(HORIZON):
        r = np.random.randint(0, nbArms)
        context = multivariate_contexts[r]
        res += calculate_reward(w, phi, context, t)
        
    return res

def calculate_reward(w, phi, context, t):
    return np.dot(np.sin(w * t + phi), context.means)


def plot_env(evaluation, environment_id):
    highest_reward = find_highest_reward()
    # The best possible reward is: 422.303516
    # The random dumb reward is: -8.059394
    print("The best possible reward is: {:.6f}".format(highest_reward))

    random_reward = random_policy()
    print("The random dumb reward is: {:.6f}".format(random_reward))
    evaluation.printFinalRanking(environment_id)
    evaluation.plotRegrets(environment_id)
    evaluation.plotRegrets(environment_id, semilogx=True)
    evaluation.plotRegrets(environment_id, meanReward=True)
    # evaluation.plotBestArmPulls(environment_id)


for env_id in range(len(environments)):
    plot_env(evaluator, env_id)
    
