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
from SMPyBandits.Policies import UCB, Exp3

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb
# https://github.com/akhadangi/Multi-armed-Bandits/blob/master/Multi-armed%20Bandits.ipynb
# https://papers.nips.cc/paper_files/paper/2011/file/f3f1b7fc5a8779a9e618e1f23a7b7860-Paper.pdf

# horizon = 30000
# horizon = 5000
# horizon = 200
horizon = 1000
repetitions = 1
# repetitions = 5
# repetitions = 2

# Has nice looking graphs
# horizon = 100
# repetitions = 30

# For quick testing
# horizon = 100
# repetitions = 4

n_jobs = 1
verbosity = 2

plot_rewards = False
plot_regret_normalized = True
plot_regret_absolute = True
plot_expectation_based_regret_normalized = False
plot_expectation_based_regret_absolute = False
plot_regret_over_max_return = True
plot_regret_logy = True
plot_regret_log = True
plot_min_max = True

start_time = datetime.datetime.now()
print("Starting run at {}")

environments = [
    {
        #"theta_star": [0.5, 0.5],
        "theta_star": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "arms": [
            ContextualKernelizedNoiseArm(0, 0.01),
            ContextualKernelizedNoiseArm(0, 0.01),
            ContextualKernelizedNoiseArm(0, 0.01)
        ],
        "contexts": [
            #NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
            #NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
            #NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3)
            NormalContext([0.3, 0.5, 0.2, 0.4, 0.6, 0.3, 0.5, 0.2, 0.4, 0.6, 0.3, 0.5, 0.2, 0.4, 0.6, 0.3, 0.5, 0.2, 0.4, 0.6], np.identity(20) * 0.5, 20),
            NormalContext([0.7, 0.3, 0.3, 0.2, 0.6, 0.7, 0.3, 0.3, 0.2, 0.6, 0.7, 0.3, 0.3, 0.2, 0.6, 0.7, 0.3, 0.3, 0.2, 0.6], np.identity(20) * 0.5, 20),
            NormalContext([0.4, 0.6, 0.1, 0.2, 0.7, 0.4, 0.6, 0.1, 0.2, 0.7, 0.4, 0.6, 0.1, 0.2, 0.7, 0.4, 0.6, 0.1, 0.2, 0.7], np.identity(20) * 0.5, 20)
        ]
    }
]

matern_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
rq_kernel = RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15))
expsine_kernel = 1.0 * ExpSineSquared(
    length_scale=1.0,
    periodicity=3.0,
    length_scale_bounds=(0.1, 10.0),
    periodicity_bounds=(1.0, 10.0),
)

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.05}},
    # {"archtype": Exp3, "params": {"gamma": 0.1}},
    # {"archtype": Exp3, "params": {"gamma": 0.2}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 100.0}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 50.0}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 20.0}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 10.0}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 5.0}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 2.0}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 1.0}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 0.5}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 0.2}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 0.1}},
    # {"archtype": LinUCB, "params": {"dimension": 5, "alpha": 0.05}},
    {"archtype": KernelUCB, "params": {"dimension": 20, "kname": "Mátern", "kern": matern_kernel, "eta": 1.0, "gamma": 1.5}},
    {"archtype": KernelUCB, "params": {"dimension": 20, "kname": "RBF", "kern": rbf_kernel, "eta": 1.0, "gamma": 1.5}},
    {"archtype": KernelUCB, "params": {"dimension": 20, "kname": "RQ", "kern": rq_kernel, "eta": 1.0, "gamma": 1.5}},
    {"archtype": KernelUCB, "params": {"dimension": 20, "kname": "ExpSine", "kern": expsine_kernel, "eta": 1.0, "gamma": 1.5}},
    {"archtype": LinUCB, "params": {"dimension": 20, "alpha": 0.01}}
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
# evaluator.startOneEnv(4, evaluator.envs[4])

figures_list = []
text_list = []


def plot_env(evaluation, environment_id, start_plot_title_index=1):
    figures = list()

    _, _, text = evaluation.printFinalRanking(environment_id)
    text_list.append(text)

    if plot_regret_normalized:
        figures.append(evaluation.plotRegrets(environment_id, show=False, normalizedRegret=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_absolute:
        figures.append(evaluation.plotRegrets(environment_id, show=False, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_expectation_based_regret_normalized:
        figures.append(evaluation.plotRegrets(environment_id, show=False, altRegret=True, normalizedRegret=True, subtitle="Alt Regret Calculation\nEnvironment #" + str(environment_id + start_plot_title_index)))
    if plot_expectation_based_regret_absolute:
        figures.append(evaluation.plotRegrets(environment_id, show=False, altRegret=True, subtitle="Alt Regret Calculation\nEnvironment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_over_max_return:
        figures.append(evaluation.plotRegrets(environment_id, show=False, regretOverMaxReturn=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_logy:
        figures.append(evaluation.plotRegrets(environment_id, show=False, semilogy=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_log:
        figures.append(evaluation.plotRegrets(environment_id, show=False, loglog=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_min_max:
        figures.append(evaluation.plotRegrets(environment_id, show=False, plotMaxMin=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))

    figures_list.append(figures)


for env_id in range(len(environments)):
    evaluator.startOneEnv(env_id, evaluator.envs[env_id])
    plot_env(evaluator, env_id)

end_time = datetime.datetime.now()
equals_string = "".join(["=" for i in range(100)])
gen_plots = True
if gen_plots:
    file_root = "./plots/{}/".format(end_time.strftime("%Y-%m-%d %H;%M;%S"))

    for env, figure_list in enumerate(figures_list):
        file_path = file_root + "environment_{}".format(env)
        try:
            makedirs(file_path)
        except OSError as exc:
            if exc.errno == EEXIST and path.isdir(file_path):
                pass
            else:
                raise
        if env < len(environments):
            with open(file_root + "regrets_environment_{}.txt".format(env), "w") as f:
                evaluator.getEnvCumulatedRegrets(env).astype(str).tofile(f, ",")
        else:
            with open(file_root + "regrets_environment_{}.txt".format(env), "w") as f:
                evaluator.getEnvCumulatedRegrets(env - len(environments)).astype(str).tofile(f, ",")

        for i, figure in enumerate(figure_list):
            figure.savefig("{}/SMPyBandits plot {}.png".format(file_path, i))

    rankings_text = equals_string.join(text_list)

    with open(file_root + "rankings.txt", "w") as f:
        f.write(rankings_text)

    with open(file_root + "rewards.txt", "w") as f:
        for value in evaluator.all_rewards.values():
            value.astype(str).tofile(f, ",")

    with open(file_root + "contexts.txt", "w") as f:
        for value in evaluator.all_contexts.values():
            value.astype(str).tofile(f, ",")

    with open(file_root + "chosen_rewards.txt", "w") as f:
        evaluator.rewards.astype(str).tofile(f, ",")

print("\n\n{}\n\nStarted run at {}\nFinished at {}\nTotal time taken: {}".format(equals_string, str(start_time), str(end_time), str(end_time - start_time)))
