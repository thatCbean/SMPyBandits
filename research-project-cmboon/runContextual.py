import datetime
import sys
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

equals_string = "".join(["=" for i in range(100)])

plot_std = True

plot_rewards = False
plot_regret_normalized = False
plot_regret_absolute = False
plot_best_policy_regret = False
plot_relative_to_best_policy_regret = False
plot_expectation_based_regret_normalized = False # Does not work in changing environments yet!!!
plot_expectation_based_regret_absolute = False # Does not work in changing environments yet!!!
plot_regret_over_max_return = False
plot_regret_logy = False
plot_regret_log = False
plot_min_max = False

plot_rewards_best_in_group = True

plot_regret_normalized_best_in_group = False
plot_regret_absolute_best_in_group = True
plot_best_policy_regret_best_in_group = False
plot_relative_to_best_policy_regret_best_in_group = False
plot_expectation_based_regret_normalized_best_in_group = False # Does not work in changing environments yet!!!
plot_expectation_based_regret_absolute_best_in_group = False # Does not work in changing environments yet!!!
plot_regret_over_max_return_best_in_group = False
plot_regret_logy_best_in_group = True
plot_regret_log_best_in_group = False
plot_min_max_best_in_group = False

figures_list = []
text_list = []

start_time = datetime.datetime.now()
file_root = "./plots/{}/".format(start_time.strftime("%Y-%m-%d %H;%M;%S"))
plots_path = "./plots/{}/plots/".format(start_time.strftime("%Y-%m-%d %H;%M;%S"))
print("Starting run at {}".format(start_time))

try:
    makedirs(plots_path)
except OSError as exc:
    if exc.errno == EEXIST and path.isdir(plots_path):
        pass
    else:
        raise


horizon = 5000
repetitions = 24

n_jobs = 12
verbosity = 2

dimension = 20

print("Constructing environments and policies...")

environments = []
# environments += environments_gen.getEnvStochasticOld(horizon, dimension)
# environments += environments_gen.getEnvContextualOld(horizon, dimension)
# environments += environments_gen.getEnvPerturbedOld(horizon, dimension)
# environments += environments_gen.getEnvSlowChangingOld(horizon, dimension)
#
# environments += environments_gen.getEnvStochastic(horizon, dimension)
# environments += environments_gen.getEnvContextual(horizon, dimension)
# environments += environments_gen.getEnvPerturbed(horizon, dimension)[0:51]
environments += environments_gen.getEnvSlowChanging(horizon, dimension)[18:]

# environments = environments[43:]

# policies = policies_gen.generatePolicySetContextualOneEach(dimension, horizon)
policies = policies_gen.generatePolicySetContextualMany(dimension, horizon)
# policies = policies_gen.generatePolicySetStochastic(dimension, horizon)

print("Finished constructing {} environments and {} policies".format(len(environments), len(policies)))


def plot_env(evaluation, environment_id, start_plot_title_index=1):
    figures = list()

    _, _, text = evaluation.printFinalRanking(environment_id)
    # text_list.append(text)

    if plot_regret_normalized:
        figures.append(evaluation.plotRegrets(environment_id, show=False, normalizedRegret=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_absolute:
        figures.append(evaluation.plotRegrets(environment_id, show=False, plotSTD=plot_std, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_best_policy_regret:
        figures.append(evaluation.plotRegrets(environment_id, show=False, bestPolicyRegret=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_relative_to_best_policy_regret:
        figures.append(evaluation.plotRegrets(environment_id, show=False, relativeToBestPolicy=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_expectation_based_regret_normalized:
        figures.append(evaluation.plotRegrets(environment_id, show=False, altRegret=True, normalizedRegret=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_expectation_based_regret_absolute:
        figures.append(evaluation.plotRegrets(environment_id, show=False, altRegret=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_over_max_return:
        figures.append(evaluation.plotRegrets(environment_id, show=False, regretOverMaxReturn=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_logy:
        figures.append(evaluation.plotRegrets(environment_id, show=False, semilogy=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_log:
        figures.append(evaluation.plotRegrets(environment_id, show=False, loglog=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_min_max:
        figures.append(evaluation.plotRegrets(environment_id, show=False, plotMaxMin=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_rewards:
        figures.append(evaluation.plotRegrets(environment_id, show=False, meanReward=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))

    if plot_regret_normalized_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, normalizedRegret=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_absolute_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, plotSTD=plot_std, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_best_policy_regret_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, bestPolicyRegret=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_relative_to_best_policy_regret_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, relativeToBestPolicy=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_expectation_based_regret_normalized_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, altRegret=True, normalizedRegret=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_expectation_based_regret_absolute_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, altRegret=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_over_max_return_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, regretOverMaxReturn=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_logy_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, semilogy=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_regret_log_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, loglog=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_min_max_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, plotMaxMin=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))
    if plot_rewards_best_in_group:
        figures.append(evaluation.plotRegrets(environment_id, show=False, meanReward=True, showOnlyBestInGroup=True, subtitle="Environment #" + str(environment_id + start_plot_title_index)))

    # figures_list.append(figures)
    return figures, text


keep_characters = (' ', '.', '_')


def savePlot(env, figure_list, rankings_text):
    env_name = evaluator.envs[env].name

    # file_path = file_root + "environment_{}_{}/".format(env + 1, "".join(c for c in env_name if c.isalnum() or c in keep_characters).rstrip())
    #
    # try:
    #     makedirs(file_path)
    # except OSError as exc:
    #     if exc.errno == EEXIST and path.isdir(file_path):
    #         pass
    #     else:
    #         raise
    # if env < len(environments):
    #     with open(file_path + "regrets_environment_{}.txt".format(env + 1), "w") as f:
    #         evaluator.getEnvCumulatedRegrets(env).astype(str).tofile(f, ",")
    # else:
    #     with open(file_path + "regrets_environment_{}.txt".format(env + 1), "w") as f:
    #         evaluator.getEnvCumulatedRegrets(env - len(environments)).astype(str).tofile(f, ",")

    for i, figure in enumerate(figure_list):
        figure.savefig("{}Plot_{}_Environment_{} {}.png".format(plots_path, i, env + 1, env_name))

    # rankings_text = equals_string.join(text_list)

    with open(file_root + "rankings_environment_{}.txt".format(env + 1), "w") as f:
        f.write(rankings_text)

    #
    # with open(file_path + "rewards.txt", "w") as f:
    #     for value in evaluator.all_rewards.values():
    #         value.astype(str).tofile(f, ",")
    #
    # with open(file_path + "contexts.txt", "w") as f:
    #     for value in evaluator.all_contexts.values():
    #         value.astype(str).tofile(f, ",")
    #
    # with open(file_path + "chosen_rewards.txt", "w") as f:
    #     evaluator.rewards.astype(str).tofile(f, ",")


configuration = {
    "horizon": horizon,
    "repetitions": repetitions,
    "n_jobs": n_jobs,
    "verbosity": verbosity,
    "environment": environments,
    "policies": policies
}

with open(file_root + "configuration.txt", "w") as f:
    f.write(str(configuration))

evaluator = EvaluatorContextual(configuration)

for env_id in range(len(environments)):
    env_start_time = datetime.datetime.now()
    evaluator.startOneEnv(env_id, evaluator.envs[env_id])
    fig, txt = plot_env(evaluator, env_id)
    savePlot(env_id, fig, txt)
    env_end_time = datetime.datetime.now()
    print("\n\n{}\n\nFinished environment {} / {} at {} in {}\nTotal time taken: {}\nProjected time remaining: {}\n\n{}\n\n".format(equals_string, env_id + 1, len(environments), str(env_end_time), str(env_end_time - env_start_time), str(env_end_time - start_time), str(((env_end_time - start_time) / (env_id + 1)) * (len(environments) - env_id - 1)), equals_string))


end_time = datetime.datetime.now()

print("\n\n{}\n\nStarted run at {}\nFinished at {}\nTotal time taken: {}\n\n{}\n\n".format(equals_string, str(start_time), str(end_time), str(end_time - start_time), equals_string))
