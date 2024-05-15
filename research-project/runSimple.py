import numpy as np

#remove following two lines if you are running this script from the root directory of the project
#TODO remove these linesbefore commiting
import sys


sys.path.insert(0, 'C:\\Users\\Dragos\\Desktop\\SMPyBandits')

from SMPyBandits.Environment import Evaluator
from SMPyBandits.Arms import Gaussian
from SMPyBandits.Arms import Adversarial
from SMPyBandits.Policies import UCB, Exp3

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb


reward_function_sin = lambda t: abs(np.sin(t))
reward_function_sin_squared = lambda t: abs(np.sin(t) ** 2)
reward_function_sin_cubed = lambda t: abs(np.sin(t) ** 3)
mean_of_reward_sin = 2 / np.pi
mean_of_reward_sin_squared = 1 / 2
mean_of_reward_sin_cubed = 0.42


environments = [
    # {
    #     "arm_type": Gaussian,
    #     "params": [(0.2, 0.25), (0.25, 0.25), (0.3, 0.25), (0.3, 0.25), (0.4, 0.25), (0.5, 0.25), (0.55, 0.25)]
    # },
    {
        "arm_type": Adversarial,
        "params": [(reward_function_sin, mean_of_reward_sin, 0.2, 0.01),
                    (reward_function_sin, mean_of_reward_sin, 0.4, 0.05),
                    (reward_function_sin, mean_of_reward_sin, 0.6, 0.1),
                    (reward_function_sin_squared, mean_of_reward_sin_squared, 0.2, 0.01),
                    (reward_function_sin_squared, mean_of_reward_sin_squared, 0.4, 0.05),
                    (reward_function_sin_cubed, mean_of_reward_sin_cubed, 0.2, 0.01), 
                    (reward_function_sin_cubed, mean_of_reward_sin_cubed, 0.4, 0.05)]
    }
]

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.01}},
]

configuration = {
    "horizon": 100000,
    "repetitions": 10,
    "n_jobs": 4,
    "verbosity": 6,
    "environment": environments,
    "policies": policies,
}

evaluator = Evaluator(configuration)

evaluator.startAllEnv()


def plot_env(evaluation, environment_id):
    evaluation.printFinalRanking(environment_id)
    evaluation.plotRegrets(environment_id)
    evaluation.plotRegrets(environment_id, semilogx=True)
    evaluation.plotRegrets(environment_id, meanReward=True)
    # evaluation.plotBestArmPulls(environment_id)


for env_id in range(len(environments)):
    plot_env(evaluator, env_id)
