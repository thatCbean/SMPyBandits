from SMPyBandits.Environment import Evaluator
from SMPyBandits.Arms import Gaussian
from SMPyBandits.Policies import UCB, Exp3

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb

environments = [
    {
        "arm_type": Gaussian,
        "params": [(0.2, 0.25), (0.25, 0.25), (0.3, 0.25), (0.3, 0.25), (0.4, 0.25), (0.5, 0.25), (0.55, 0.25)]
    }
]

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.01}},
]

configuration = {
    "horizon": 1000,
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
