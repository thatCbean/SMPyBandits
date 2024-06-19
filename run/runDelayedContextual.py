import datetime
from errno import EEXIST
from os import makedirs, path
import sys
import numpy as np

sys.path.append("C:\\Users\\Dragos\\Desktop\\SMPyBandits")

from SMPyBandits.Delays.NegativeBinomialDelay import NegativeBinomialDelay
from SMPyBandits.Delays.UniformDelay import UniformDelay
from SMPyBandits.Delays.GeometricDelay import GeometricDelay
from SMPyBandits.DelayedContextualBandits.Policies.DeLinUCB import DeLinUCB
from SMPyBandits.ContextualBandits.ContextualPolicies import LinUCB
from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

from SMPyBandits.Delays.PoissonDelay import PoissonDelay

from SMPyBandits.DelayedContextualBandits.Policies.Exp3WithDelay import Exp3WithDelay
from SMPyBandits.DelayedContextualBandits.Policies.LinUCBWithDelay import LinUCBWithDelay
from SMPyBandits.DelayedContextualBandits.Policies.UCBWithDelay import UCBWithDelay
from SMPyBandits.DelayedContextualBandits.DelayedContextualEnvironments.EvaluatorDelayedContextual import EvaluatorDelayedContextual




from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualGaussianNoiseArm import \
    ContextualGaussianNoiseArm

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb
horizon = 5000
repetitions = 10
m = 100
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

include_delay_info = False
include_std_dev = True

start_time = datetime.datetime.now()
print("Starting run at {}")

environments = [
    {
        "theta_star": [0.5, 0.5, 0.5],
        "arms": [
            ContextualGaussianNoiseArm(0, 0.01),
            ContextualGaussianNoiseArm(0, 0.01),
            ContextualGaussianNoiseArm(0, 0.01),
            ContextualGaussianNoiseArm(0, 0.01),
            ContextualGaussianNoiseArm(0, 0.01),
            ContextualGaussianNoiseArm(0, 0.01)
        ],
        "contexts": [
            NormalContext([0.2, 0.4, 0.6], np.identity(3) * 0.3, 3),
            NormalContext([0.1, 0.3, 0.5], np.identity(3) * 0.5, 3),
            NormalContext([0.6, 0.05, 0.01], np.identity(3) * 0.7, 3),
            NormalContext([0.2, 0.7, 0.001], np.identity(3) * 0.3, 3),
            NormalContext([0.4, 0.1, 0.2], np.identity(3) * 0.5, 3),
            NormalContext([0.3, 0.5, 0.15], np.identity(3) * 0.7, 3),
        ],
        "delays": [
            # PoissonDelay(3, 0, 0, 500),
            # PoissonDelay(3, 0, 0, 400),
            # PoissonDelay(3, 0, 0, 300),
            # PoissonDelay(3, 0, 0, 600),
            # PoissonDelay(3, 0, 0, 800),
            # PoissonDelay(3, 0, 0, 900),
            UniformDelay(3, 0, 100),
            UniformDelay(3, 0, 100),
            UniformDelay(3, 0, 100),
            UniformDelay(3, 0, 100),
            UniformDelay(3, 0, 100),
            UniformDelay(3, 0, 100),
            # GeometricDelay(3, 0, 0, 400),
            # GeometricDelay(3, 0, 0, 500),
            # GeometricDelay(3, 0, 0, 600),
            # GeometricDelay(3, 0, 0, 700),
            # GeometricDelay(3, 0, 0, 800),
            # GeometricDelay(3, 0, 0, 900),
        ]
    }
]

policies = [
    # {"archtype": UCBWithDelay, "params": {}},
    # {"archtype": Exp3WithDelay, "params": {"gamma": 0.01}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 700}},

    {"archtype": LinUCBWithDelay, "params": {"dimension": 3, "alpha": 0.01}},
    {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 100}},
    {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 90}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 80}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 70}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 60}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 50}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 40}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 30}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 20}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 10}},
    # {"archtype": DeLinUCB, "params": {"dimension": 3, "alpha": 0.01, "horizon": horizon, "lambda_reg" : 1, "m" : 0}},
]

configuration = {
    "horizon": horizon,
    "repetitions": repetitions,
    "n_jobs": n_jobs,
    "verbosity": verbosity,
    "environment": environments,
    "policies": policies,
}

evaluator = EvaluatorDelayedContextual(configuration)

figures_list = []
text_list = []


def plot_env(evaluation, environment_id, start_plot_title_index=1):
    figures = list()

    _, _, text = evaluation.printFinalRanking(environment_id)
    text_list.append(text)
    if plot_regret_normalized:
        figures.append(evaluation.plotRegrets(environment_id, show=False, normalizedRegret=True, include_delay_info = include_delay_info,
            plotSTD = include_std_dev, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_absolute:
        figures.append(evaluation.plotRegrets(environment_id, show=False, include_delay_info = include_delay_info,
            plotSTD = include_std_dev, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_expectation_based_regret_normalized:
        figures.append(evaluation.plotRegrets(environment_id, show=False, altRegret=True, normalizedRegret=True,
            plotSTD = include_std_dev, subtitle="Alt Regret Calculation\nEnvironment #" + str(environment_id + start_plot_title_index)))
    if plot_expectation_based_regret_absolute:
        figures.append(evaluation.plotRegrets(environment_id, show=False, altRegret=True, include_delay_info = include_delay_info,
            plotSTD = include_std_dev, subtitle="Alt Regret Calculation\nEnvironment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_over_max_return:
        figures.append(evaluation.plotRegrets(environment_id, show=False, regretOverMaxReturn=True, include_delay_info = include_delay_info,
            plotSTD = include_std_dev, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_logy:
        figures.append(evaluation.plotRegrets(environment_id, show=False, semilogy=True, include_delay_info = include_delay_info,\
            plotSTD = include_std_dev, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_log:
        figures.append(evaluation.plotRegrets(environment_id, show=False, loglog=True, include_delay_info = include_delay_info,
            plotSTD = include_std_dev, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_min_max:
        figures.append(evaluation.plotRegrets(environment_id, show=False, plotMaxMin=True, include_delay_info = include_delay_info,
            plotSTD = include_std_dev, subtitle="Environment #" + str(environment_id + start_plot_title_index)))

    figures_list.append(figures)


for env_id in range(len(environments)):
    evaluator.startOneEnv(env_id, evaluator.envs[env_id])
    plot_env(evaluator, env_id)

end_time = datetime.datetime.now()
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

equals_string = "".join(["=" for i in range(100)])

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
