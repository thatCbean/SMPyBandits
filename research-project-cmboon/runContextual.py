import datetime
from errno import EEXIST
from os import makedirs, path

import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualGaussianNoiseArm import \
    ContextualGaussianNoiseArm
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.Policies import UCB, Exp3
from EnvironmentConfigurations import EnvironmentConfigurations
from PolicyConfigurations import PolicyConfigurations

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb

environments_gen = EnvironmentConfigurations()

policies_gen = PolicyConfigurations()


horizon = 100
repetitions = 2

secondEnv = False

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


start_time = datetime.datetime.now()
print("Starting run at {}")

dimension = 20

# environments = environments_gen.getEnv1(horizon)
environments = environments_gen.getEnv2(horizon) + environments_gen.getEnv3(horizon) + environments_gen.getEnv4(horizon)

policies = policies_gen.generatePolicySetContextualOneEach(dimension, horizon)

configuration = {
    "horizon": horizon,
    "repetitions": repetitions,
    "n_jobs": n_jobs,
    "verbosity": verbosity,
    "environment": environments,
    "policies": policies
}

evaluator = EvaluatorContextual(configuration)

for env_id in range(len(environments)):
    evaluator.startOneEnv(env_id, evaluator.envs[env_id])
    plot_env(evaluator, env_id)

if secondEnv:
    dimension_2 = 20

    # environments_2 = environments_gen.getEnv2(horizon) + environments_gen.getEnv3(horizon) + environments_gen.getEnv4(horizon)
    environments_2 = []

    policies_2 = policies_gen.generatePolicySetContextualOneEach(dimension_2, horizon)

    configuration_2 = {
        "horizon": horizon,
        "repetitions": repetitions,
        "n_jobs": n_jobs,
        "verbosity": verbosity,
        "environment": environments_2,
        "policies": policies_2
    }

    evaluator_2 = EvaluatorContextual(configuration_2)
    for env_id in range(len(environments_2)):
        evaluator_2.startOneEnv(env_id, evaluator_2.envs[env_id])
        plot_env(evaluator_2, env_id, start_plot_title_index=len(environments) + 1)

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

if secondEnv:
    with open(file_root + "rewards_env2.txt", "w") as f:
        for value in evaluator_2.all_rewards.values():
            value.astype(str).tofile(f, ",")

    with open(file_root + "contexts_env2.txt", "w") as f:
        for value in evaluator_2.all_contexts.values():
            value.astype(str).tofile(f, ",")

    with open(file_root + "chosen_rewards_env2.txt", "w") as f:
        evaluator_2.rewards.astype(str).tofile(f, ",")

print("\n\n{}\n\nStarted run at {}\nFinished at {}\nTotal time taken: {}".format(equals_string, str(start_time), str(end_time), str(end_time - start_time)))
