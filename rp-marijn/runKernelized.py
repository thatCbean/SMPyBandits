import datetime
from errno import EEXIST
from os import makedirs, path

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import ExpSineSquared
from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.Contexts.BaseContext import BaseContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualGaussianNoiseArm import ContextualGaussianNoiseArm
from SMPyBandits.ContextualBandits.ContextualArms.ContextualKernelizedNoiseArm import ContextualKernelizedNoiseArm
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from KernelUCB import KernelUCB
from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.Policies import UCB, Exp3, EpsilonGreedy
from plotter import Plotting

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb
# https://github.com/akhadangi/Multi-armed-Bandits/blob/master/Multi-armed%20Bandits.ipynb
# https://papers.nips.cc/paper_files/paper/2011/file/f3f1b7fc5a8779a9e618e1f23a7b7860-Paper.pdf

matern_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
rq_kernel = RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15))
expsine_kernel = 1.0 * ExpSineSquared(
    length_scale=1.0,
    periodicity=3.0,
    length_scale_bounds=(0.1, 10.0),
    periodicity_bounds=(1.0, 10.0),
)

start_time = datetime.datetime.now()
print("Starting run at {}")

environments = [
    {
        "theta_star": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "arms": [
            ContextualKernelizedNoiseArm(0, 0.01),
            ContextualKernelizedNoiseArm(0, 0.01),
            ContextualKernelizedNoiseArm(0, 0.01)
        ],
        "contexts": [
            NormalContext([0.3, 0.5, 0.2, 0.4, 0.6, 0.3, 0.5, 0.2, 0.4, 0.6, 0.3, 0.5, 0.2, 0.4, 0.6, 0.3, 0.5, 0.2, 0.4, 0.6], np.identity(20) * 0.5, 20),
            NormalContext([0.7, 0.3, 0.3, 0.2, 0.6, 0.7, 0.3, 0.3, 0.2, 0.6, 0.7, 0.3, 0.3, 0.2, 0.6, 0.7, 0.3, 0.3, 0.2, 0.6], np.identity(20) * 0.5, 20),
            NormalContext([0.4, 0.6, 0.1, 0.2, 0.7, 0.4, 0.6, 0.1, 0.2, 0.7, 0.4, 0.6, 0.1, 0.2, 0.7, 0.4, 0.6, 0.1, 0.2, 0.7], np.identity(20) * 0.5, 20)
        ]
    }
]

policies = [
    {"archtype": EpsilonGreedy, "params": {"epsilon": 1}},
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.01}},
    {"archtype": LinUCB, "params": {"dimension": 20, "alpha": 0.01}},
    {"archtype": KernelUCB, "params": {"dimension": 20, "kname": "Mátern", "kern": matern_kernel, "eta": 0.5, "gamma": 0.1}},
    {"archtype": KernelUCB, "params": {"dimension": 20, "kname": "RBF", "kern": rbf_kernel, "eta": 0.5, "gamma": 0.1}},
    {"archtype": KernelUCB, "params": {"dimension": 20, "kname": "RQ", "kern": rq_kernel, "eta": 0.5, "gamma": 0.1}},
    {"archtype": KernelUCB, "params": {"dimension": 20, "kname": "ExpSine", "kern": expsine_kernel, "eta": 0.5, "gamma": 0.1}}
]

# Paper:
# T = 1000
# reps = 10
# n_jobs = 3
# K = 3
# d = 20
# eta = 0.5
# gamma = 0.1

horizon = 1000
repetitions = 10
n_jobs = 3
verbosity = 2
reward_function = "ln(x)"
plot_title = "Cumulative regrets averaged over {} repetitions\nK={}, d={}, r={}".format(repetitions, 3, 20, reward_function)

configuration = {
    "horizon": horizon,
    "repetitions": repetitions,
    "n_jobs": n_jobs,
    "verbosity": verbosity,
    "environment": environments,
    "policies": policies
}

evaluator = EvaluatorContextual(configuration)
N = len(evaluator.envs)

# Create the folder for the plots
plotter = Plotting(evaluator, configuration=configuration, saveAllFigures=True)

for environmentId, environment in enumerate(evaluator.envs):
    # hash for this specific run
    hashValue = abs(hash((tuple(configuration.keys()), tuple(
        [(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))

    evaluator.startOneEnv(environmentId, environment)
    evaluator.printFinalRanking(environmentId)

    # Display the final regrets and rankings for that env
    # evaluator.printLastRegrets(environmentId)
    # evaluator.printMemoryConsumption(environmentId)
    # evaluator.printNumberOfCPDetections(environmentId)

    # Plotting
    plotter.create_subfolder(N, environmentId, hashValue)
    plotter.plot_all(environmentId, plot_title)

end_time = datetime.datetime.now()
equals_string = "".join(["=" for i in range(100)])
print("\n\n{}\n\nStarted run at {}\nFinished at {}\nTotal time taken: {}".format(equals_string, str(start_time), str(end_time), str(end_time - start_time)))
