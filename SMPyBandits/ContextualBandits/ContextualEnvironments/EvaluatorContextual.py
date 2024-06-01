# -*- coding: utf-8 -*-
""" Evaluator class to wrap and run the simulations.
Lots of plotting methods, to have various visualizations.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import math
# Generic imports
import sys
import pickle

from SMPyBandits.Policies import BasePolicy

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy
from SMPyBandits.ContextualBandits.ContextualEnvironments.ContextualMAB import ContextualMAB
from rowczarskiResearchProject.SparsityAgnosticLassoBandit.SparsityAgnosticLassoBandit import \
    SparsityAgnosticLassoBandit

USE_PICKLE = False  #: Should we save the figure objects to a .pickle file at the end of the simulation?
import random
import time
from copy import deepcopy
# Scientific imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import inspect

# Local imports, libraries
from SMPyBandits.Environment.usejoblib import USE_JOBLIB, Parallel, delayed
from SMPyBandits.Environment.usetqdm import USE_TQDM, tqdm
# Local imports, tools and config
from SMPyBandits.Environment.plotsettings import BBOX_INCHES, signature, maximizeWindow, palette, makemarkers, \
    add_percent_formatter, \
    legend, show_and_save, nrows_ncols, violin_or_box_plot, adjust_xticks_subplots, table_to_latex
from SMPyBandits.Environment.sortedDistance import weightedDistance, manhattan, kendalltau, spearmanr, gestalt, \
    meanDistance, \
    sortedDistance
# Local imports, objects and functions
from SMPyBandits.Environment.Result import Result
from SMPyBandits.Environment.memory_consumption import getCurrentMemory, sizeof_fmt

REPETITIONS = 1  #: Default nb of repetitions
DELTA_T_PLOT = 50  #: Default sampling rate for plotting

plot_lowerbound = True  #: Default is to plot the lower-bound

USE_BOX_PLOT = True  #: True to use boxplot, False to use violinplot.

# Parameters for the random events
random_shuffle = False  #: Use basic random events of shuffling the arms?
random_invert = False  #: Use basic random events of inverting the arms?
nb_break_points = 0  #: Default nb of random events

# Flag for experimental aspects
STORE_ALL_REWARDS = False  #: Store all rewards?
STORE_REWARDS_SQUARED = False  #: Store rewards squared?
MORE_ACCURATE = False  #: Use the count of selections instead of rewards for a more accurate mean/var reward measure.
FINAL_RANKS_ON_AVERAGE = True  #: Final ranks are printed based on average on last 1% rewards and not only the last rewards
USE_JOBLIB_FOR_POLICIES = True  #: Don't use joblib to parallelize the simulations on various policies (we parallelize the random Monte Carlo repetitions)


def _nbOfArgs(function):
    try:
        return len(inspect.signature(function).parameters)
    except NameError:
        return len(inspect.getargspec(function).args)


class EvaluatorContextual(object):
    """ Evaluator class to run contextual simulations."""

    def __init__(self, configuration,
                 finalRanksOnAverage=FINAL_RANKS_ON_AVERAGE, averageOn=5e-3,
                 useJoblibForPolicies=USE_JOBLIB_FOR_POLICIES,
                 moreAccurate=MORE_ACCURATE):
        self.use_box_plot = True
        self.cfg = configuration  #: Configuration dictionary
        # Attributes
        self.nbPolicies = len(self.cfg['policies'])  #: Number of policies
        print("Number of policies in this comparison:", self.nbPolicies)
        self.horizon = self.cfg['horizon']  #: Horizon (number of time steps)
        print("Time horizon:", self.horizon)
        self.repetitions = self.cfg.get('repetitions', REPETITIONS)  #: Number of repetitions
        print("Number of repetitions:", self.repetitions)
        self.delta_t_plot = 1 if self.horizon <= 10000 else self.cfg.get('delta_t_plot',
                                                                         DELTA_T_PLOT)  #: Sampling rate for plotting
        print("Sampling rate for plotting, delta_t_plot:", self.delta_t_plot)
        print("Number of jobs for parallelization:", self.cfg['n_jobs'])

        # Flags
        self.finalRanksOnAverage = finalRanksOnAverage  #: Final display of ranks are done on average rewards?
        self.averageOn = averageOn  #: How many last steps for final rank average rewards
        self.useJoblibForPolicies = useJoblibForPolicies  #: Use joblib to parallelize for loop on policies (useless)
        self.useJoblib = USE_JOBLIB and self.cfg[
            'n_jobs'] != 1  #: Use joblib to parallelize for loop on repetitions (useful)
        self.use_box_plot = USE_BOX_PLOT
        self.showplot = True

        self.verbosity = self.cfg['verbosity'] if "verbosity" in self.cfg else 3

        # Internal object memory

        self.envs = []  #: List of environments
        self.policies = []  #: List of policies
        self.dimension = -1
        self.__initEnvironments__()

        if "seeds" in self.cfg:
            assert len(self.cfg['seeds']) == len(self.envs) or len(self.cfg['seeds']) == 1, \
                "Error: Number of seeds must be equal to one or the number of environments"
            self.seeds = self.cfg['seeds']
        else:
            self.seeds = None

        # Internal vectorial memory
        self.rewards = np.zeros((self.repetitions, self.nbPolicies, len(self.envs),
                                 self.horizon))  #: For each env, history of rewards, ie accumulated rewards
        self.sumRewards = np.zeros((self.nbPolicies, len(self.envs),
                                    self.repetitions))  #: For each env, last accumulated rewards, to compute variance and histogram of whole regret R_T
        self.minCumRewards = np.full((self.nbPolicies, len(self.envs), self.horizon),
                                     +np.inf)  #: For each env, history of minimum of rewards, to compute amplitude (+- STD)
        self.maxCumRewards = np.full((self.nbPolicies, len(self.envs), self.horizon),
                                     -np.inf)  #: For each env, history of maximum of rewards, to compute amplitude (+- STD)

        self.runningTimes = dict()  #: For each env, keep the history of running times
        self.memoryConsumption = dict()  #: For each env, keep the history of running times
        self.numberOfCPDetections = dict()  #: For each env, store the number of change-point detections by each algorithms, to print it's average at the end (to check if a certain Change-Point detector algorithm detects too few or too many changes).
        self.all_contexts = dict()
        self.all_rewards = dict()
        # XXX: WARNING no memorized vectors should have dimension duration * repetitions, that explodes the RAM consumption!
        for envId in range(len(self.envs)):
            self.runningTimes[envId] = np.zeros((self.nbPolicies, self.repetitions))
            self.memoryConsumption[envId] = np.zeros((self.nbPolicies, self.repetitions))
            self.numberOfCPDetections[envId] = np.zeros((self.nbPolicies, self.repetitions), dtype=np.int32)
            self.all_contexts[envId] = np.zeros(
                (self.repetitions, self.horizon, self.envs[envId].nbArms, self.dimension))
            self.all_rewards[envId] = np.zeros((self.repetitions, self.horizon, self.envs[envId].nbArms))
        print("Number of environments to try:", len(self.envs))
        # To speed up plotting
        self._times = np.arange(1, 1 + self.horizon)

    # --- Init methods

    def __initEnvironments__(self):
        """ Create environments."""
        for configuration_envs in self.cfg['environment']:
            print("Using this dictionary to create a new environment:\n", configuration_envs)  # DEBUG
            if isinstance(configuration_envs, dict) \
                    and "theta_star" in configuration_envs \
                    and "arms" in configuration_envs \
                    and "contexts" in configuration_envs:
                dim = len(configuration_envs["theta_star"])
                assert self.dimension == -1 or self.dimension == dim, "Error: All contexts must have the same dimension"
                self.dimension = dim
                self.envs.append(ContextualMAB(configuration_envs))

    def __initPolicies__(self, env):
        """ Create or initialize policies."""
        for policyId, policy in enumerate(self.cfg['policies']):
            print("- Adding policy #{} = {} ...".format(policyId + 1, policy))  # DEBUG
            if isinstance(policy, dict):
                print("  Creating this policy from a dictionnary 'self.cfg['policies'][{}]' = {} ...".format(policyId,
                                                                                                             policy))  # DEBUG
                self.policies.append(policy['archtype'](env.nbArms, **policy['params']))
            else:
                print("  Using this already created policy 'self.cfg['policies'][{}]' = {} ...".format(policyId,
                                                                                                       policy))  # DEBUG
                self.policies.append(policy)
        for policyId in range(self.nbPolicies):
            self.policies[policyId].__cachedstr__ = str(self.policies[policyId])

    # --- Start computation

    def compute_cache_rewards(self, env, env_id):
        """ Compute only once the rewards, then launch the experiments with the same matrix (r_{k,t})."""
        if self.seeds is not None and len(self.seeds) > 1:
            random.seed(self.seeds[env_id])
            np.random.seed(self.seeds[env_id])

        rewards = np.zeros((self.repetitions, self.horizon, env.nbArms))
        contexts = np.zeros((self.repetitions, self.horizon, env.nbArms, self.dimension))
        print(
            "\n===> Pre-computing the rewards ... Of shape {} ...\n    In order for all simulated algorithms to face the same random rewards (robust comparison of A1,..,An vs Aggr(A1,..,An)) ...\n".format(
                np.shape(rewards)))  # DEBUG
        if self.verbosity > 2:
            for repetitionId in tqdm(range(self.repetitions), desc="Repetitions"):
                for t in tqdm(range(self.horizon), desc="Time steps"):
                    for arm_id in tqdm(range(len(env.arms)), desc="Arms"):
                        contexts[repetitionId, t, arm_id], rewards[repetitionId, t, arm_id] = env.draw(arm_id, t)
        else:
            for repetitionId in tqdm(range(self.repetitions), desc="Repetitions"):
                for t in range(self.horizon):
                    for arm_id in range(len(env.arms)):
                        contexts[repetitionId, t, arm_id], rewards[repetitionId, t, arm_id] = env.draw(arm_id, t)
        return contexts, rewards

    def startAllEnv(self):
        """Simulate all envs."""
        for envId, env in enumerate(self.envs):
            self.startOneEnv(envId, env)

    def startOneEnv(self, envId, env):
        """Simulate that env."""
        plt.close('all')
        print("\n\nEvaluating environment:", repr(env))
        self.policies = []
        self.__initPolicies__(env)

        if self.seeds is not None and len(self.seeds) == 1:
            random.seed(self.seeds[0])
            np.random.seed(self.seeds[0])

        # Precompute rewards
        all_contexts, all_rewards = self.compute_cache_rewards(env, envId)
        self.all_contexts[envId], self.all_rewards[envId] = all_contexts, all_rewards

        def store(r, policyId, repeatId):
            """ Store the result of the #repeatId experiment, for the #policyId policy."""
            self.rewards[repeatId][policyId][envId] = r.rewards
            self.sumRewards[policyId, envId, repeatId] = np.sum(r.rewards)
            if hasattr(self, 'minCumRewards'):
                self.minCumRewards[policyId, envId, :] = np.minimum(self.minCumRewards[policyId, envId, :], np.cumsum(
                    r.rewards)) if repeatId > 1 else np.cumsum(r.rewards)
            if hasattr(self, 'maxCumRewards'):
                self.maxCumRewards[policyId, envId, :] = np.maximum(self.maxCumRewards[policyId, envId, :], np.cumsum(
                    r.rewards)) if repeatId > 1 else np.cumsum(r.rewards)
            self.memoryConsumption[envId][policyId, repeatId] = r.memory_consumption
            self.runningTimes[envId][policyId, repeatId] = r.running_time
            self.numberOfCPDetections[envId][policyId, repeatId] = r.number_of_cp_detections

        # Start for all policies
        for policyId, policy in enumerate(self.policies):
            total = 100 * (((envId * self.nbPolicies) + policyId) / (len(self.envs) * self.nbPolicies))
            print("\n\n\n- Evaluating environment {}/{}\n    policy {}/{}\n    total {}%\n    {}".format(envId + 1, len(self.envs), policyId + 1, self.nbPolicies, str(total)[:5], policy))
            if self.useJoblib:
                repeatIdout = 0
                for r in Parallel(n_jobs=self.cfg['n_jobs'], pre_dispatch='3*n_jobs', verbose=self.cfg['verbosity'])(
                        delayed(delayed_play)(env, policy, self.horizon, self.all_contexts, self.all_rewards, envId,
                                              repeatId=repeatId, verbose=(self.verbosity > 4))
                        for repeatId in range(self.repetitions)
                ):
                    store(r, policyId, repeatIdout)
                    repeatIdout += 1

            else:
                for repeatId in (
                        tqdm(range(self.repetitions), desc="Repeat") if self.verbosity > 3 else range(
                            self.repetitions)):
                    r = delayed_play(env, policy, self.horizon, self.all_contexts, self.all_rewards, envId,
                                     repeatId=repeatId, verbose=(self.verbosity > 4))
                    store(r, policyId, repeatId)

    def getRunningTimes(self, envId=0):
        """Get the means and stds and list of running time of the different policies."""
        all_times = [self.runningTimes[envId][policyId, :] for policyId in range(self.nbPolicies)]
        means = [np.mean(times) for times in all_times]
        stds = [np.std(times) for times in all_times]
        return means, stds, all_times

    def getMemoryConsumption(self, envId=0):
        """Get the means and stds and list of memory consumptions of the different policies."""
        all_memories = [self.memoryConsumption[envId][policyId, :] for policyId in range(self.nbPolicies)]
        for policyId in range(self.nbPolicies):
            all_memories[policyId] = [m for m in all_memories[policyId] if m > 0]
        means = [np.mean(memories) if len(memories) > 0 else 0 for memories in all_memories]
        stds = [np.std(memories) if len(memories) > 0 else 0 for memories in all_memories]
        return means, stds, all_memories

    def getNumberOfCPDetections(self, envId=0):
        """Get the means and stds and list of numberOfCPDetections of the different policies."""
        all_number_of_cp_detections = [self.numberOfCPDetections[envId][policyId, :] for policyId in
                                       range(self.nbPolicies)]
        means = [np.mean(number_of_cp_detections) for number_of_cp_detections in all_number_of_cp_detections]
        stds = [np.std(number_of_cp_detections) for number_of_cp_detections in all_number_of_cp_detections]
        return means, stds, all_number_of_cp_detections

    # self.all_rewards[envId] = np.zeros((self.repetitions, self.horizon, self.envs[envId].nbArms))
    # all_rewards[envId][repetitions][horizon][nbArms]
    def getHighestRewards(self, envId):
        return \
            np.array([[np.max(self.all_rewards[envId][repetition][t]) for t in range(self.horizon)] for repetition in
                      range(self.repetitions)])

    def getExpectedHighestReward(self, envId):
        return np.mean(self.getHighestRewards(envId))

    def getLowestRewards(self, envId):
        return \
            np.array([[np.min(self.all_rewards[envId][repetition][t]) for t in range(self.horizon)] for repetition in
                      range(self.repetitions)])

    # self.rewards = np.zeros((self.repetitions, self.nbPolicies, len(self.envs), self.horizon))
    def getHighestRewardsPolicy(self, policyId, envId):
        return np.array([np.max(self.rewards[:, policyId, envId, t]) for t in range(self.horizon)])

    def getLowestRewardsPolicy(self, policyId, envId):
        return np.array([np.min(self.rewards[:, policyId, envId, t]) for t in range(self.horizon)])

    # self.rewards = np.zeros((self.repetitions, self.nbPolicies, len(self.envs), self.horizon))
    def getRegretAmplitude(self, policyId, envId):
        highest_rewards = self.getHighestRewards(envId)
        return np.array([highest_rewards[repetition] - self.rewards[repetition, policyId, envId]
                         for repetition in range(self.repetitions)])

    def getCumulatedRegrets(self, policyId, envId):
        highest_rewards = self.getHighestRewards(envId)
        return np.array([np.cumsum(highest_rewards[repetition] - self.rewards[repetition, policyId, envId])
                         for repetition in range(self.repetitions)])

    def getCumulatedRegretAverage(self, policyId, envId):
        cumulated_regrets = self.getCumulatedRegrets(policyId, envId)
        return np.array([np.mean(cumulated_regrets[:, t]) for t in range(self.horizon)])

    def getEnvCumulatedRegrets(self, envId):
        return np.array([
            self.getCumulatedRegrets(policyId, envId)
            for policyId in range(self.nbPolicies)
        ])

    def getCumulatedRegretSum(self, policyId, envId):
        cumulated_regrets = self.getCumulatedRegrets(policyId, envId)
        return np.array([np.mean(cumulated_regrets[:, t]) for t in range(self.horizon)])

    def getCumulatedRewardAverage(self, policyId, envId):
        cumulative_rewards = np.array(
            [np.cumsum(self.rewards[repetition, policyId, envId, :]) for repetition in range(self.repetitions)])
        return np.array([np.mean(cumulative_rewards[:, t]) for t in range(self.horizon)])

    def getHighestRewardsAverage(self, envId):
        highest_rewards = self.getHighestRewards(envId)
        return np.array([np.mean(highest_rewards[:, t]) for t in range(self.horizon)])

    def getCumulativeRegretAvgOverCumulativeMaxReward(self, policyId, envId):
        cumulative_regrets = self.getCumulatedRegretAverage(policyId, envId)
        cumulative_highest_rewards = np.cumsum(self.getHighestRewardsAverage(envId))
        return cumulative_regrets / cumulative_highest_rewards

    def getCumulatedRewardAmplitude(self, policyId, envId):
        highest = np.cumsum(self.getHighestRewardsPolicy(policyId, envId))
        lowest = np.cumsum(self.getLowestRewardsPolicy(policyId, envId))
        return highest - lowest

    def getLastRegrets(self, policyId, envId):
        return (self.getCumulatedRegrets(policyId, envId))[:, -1]

    def getBestMeanReward(self, envId):
        arms = self.envs[envId].arms
        contexts = self.envs[envId].contexts
        theta_star = self.envs[envId].theta_star
        context_means = np.array([context.means for context in contexts])
        mean_rewards = np.array(np.abs(
            [
                np.inner(context_means, theta_star)
                for arm in range(len(arms))
            ]
        ))
        return np.max(mean_rewards)

    def getAltRegret(self, policyId, envId):
        highest_reward = self.getBestMeanReward(envId)
        full_regret = np.array([
            np.cumsum(np.full(self.horizon, highest_reward) - self.rewards[repetition, policyId, envId])
            for repetition in range(self.repetitions)
        ])
        return np.array([
            np.mean(full_regret[:, t])
            for t in range(self.horizon)
        ])

    # def getRegretRelativeToHighest(self, envId):
    #    TODO: WIP
    # --- Plotting methods

    def printAndReturn(self, strii):
        print(strii)
        return strii

    def printFinalRanking(self, envId=0):
        """Print the final ranking of the different policies."""
        stri = self.printAndReturn("\nGiving the final ranks ...")
        assert 0 < self.averageOn < 1, "Error, the parameter averageOn of a EvaluatorMultiPlayers classs has to be in (0, 1) strictly, but is = {} here ...".format(
            self.averageOn)  # DEBUG

        stri += self.printAndReturn("\nFinal ranking for this environment #{}".format(envId))

        nbPolicies = self.nbPolicies
        totalRegret = np.zeros(nbPolicies)
        altRegret = np.zeros(nbPolicies)
        totalRewards = np.zeros(nbPolicies)
        lastRegret = np.zeros(nbPolicies)
        lastAltRegret = np.zeros(nbPolicies)
        for i, policy in enumerate(self.policies):
            Y = self.getCumulatedRegretAverage(i, envId)
            Z = self.getAltRegret(i, envId)
            totalRegret[i] = Y[-1]
            altRegret[i] = Z[-1]
            lastRegret[i] = Y[-1] - Y[-2]
            lastAltRegret[i] = Z[-1] - Z[-2]
            totalRewards[i] = np.sum(self.rewards[:, i, envId, :]) / self.repetitions
        # Sort lastRegret and give ranking
        index_of_sorting = np.argsort(totalRegret)
        for i, k in enumerate(index_of_sorting):
            policy = self.policies[k]
            stri += self.printAndReturn(
                "- Policy '{}'\twas ranked\t{} / {} for this simulation\n\t(last regret = {:.5g},\ttotal regret = {:.5g},\ttotal reward = {:.5g}.".format(
                    policy.__cachedstr__, i + 1, nbPolicies, lastRegret[k], totalRegret[k], totalRewards[k]))
            stri += self.printAndReturn("    Alternative regret calculation results in regret of [{}] and last regret of [{}]".format(altRegret[k], lastAltRegret[k]))
        return totalRegret, index_of_sorting, stri

    def _xlabel(self, envId, *args, **kwargs):
        """Add xlabel to the plot, and if the environment has change-point, draw vertical lines to clearly identify the locations of the change points."""
        env = self.envs[envId]
        if hasattr(env, 'changePoints'):
            ymin, ymax = plt.ylim()
            taus = self.envs[envId].changePoints
            if len(taus) > 25:
                print(
                    "WARNING: Adding vlines for the change points with more than 25 change points will be ugly on the plots...")  # DEBUG
            if len(taus) > 50:  # Force to NOT add the vlines
                return plt.xlabel(*args, **kwargs)
            for tau in taus:
                if tau > 0 and tau < self.horizon:
                    plt.vlines(tau, ymin, ymax, linestyles='dotted', alpha=0.5)
        return plt.xlabel(*args, **kwargs)

    def plotRegrets(
            self, envId=0, show=True,
            savefig=None, meanReward=False,
            plotSTD=False, plotMaxMin=False,
            semilogx=False, semilogy=False, loglog=False,
            normalizedRegret=False, relativeRegret=False,
            regretOverMaxReturn=False, altRegret=False,
            relativeToBestPolicy=False, subtitle=""
            ):
        """
        Plot the centralized cumulated regret, support more than one environment
        (use evaluators to give a list of other environments).
        """
        fig = plt.figure()
        ymin = None
        colors = palette(self.nbPolicies)
        markers = makemarkers(self.nbPolicies)
        # range_start = min(50, math.floor(self.horizon / 20))
        # print("Starting plots at index {}", range_start)
        X = self._times - 1
        # X = X[range_start:]
        plot_method = plt.loglog if loglog else plt.plot
        plot_method = plt.semilogy if semilogy else plot_method
        plot_method = plt.semilogx if semilogx else plot_method
        if subtitle != "":
            subtitle = "\n" + subtitle

        if relativeRegret:
            if meanReward:
                highest = np.max(
                    np.array([self.getCumulatedRewardAverage(policyId, envId) for policyId in range(self.nbPolicies)])[
                    :, -1])
            else:
                highest = np.max(
                    np.array([self.getCumulatedRegretAverage(policyId, envId) for policyId in range(self.nbPolicies)])[
                    :, -1])
        for policyId, policy in enumerate(self.policies):
            if meanReward:
                Y = np.array(self.getCumulatedRewardAverage(policyId, envId))
            elif regretOverMaxReturn:
                Y = np.array(self.getCumulativeRegretAvgOverCumulativeMaxReward(policyId, envId))
            elif altRegret:
                Y = np.array(self.getAltRegret(policyId, envId))
            elif relativeToBestPolicy:
                Y = np.array()
            else:
                Y = np.array(self.getCumulatedRegretAverage(policyId, envId))
            if normalizedRegret and not regretOverMaxReturn:
                Y /= self._times
            elif relativeRegret and not regretOverMaxReturn:
                Y /= highest

            # Y = Y[range_start:]

            if ymin is None:
                ymin = np.min(Y)
            else:
                ymin = min(ymin, np.min(Y))
            lw = 8
            if len(self.policies) > 8: lw -= 1
            if semilogx or loglog:
                # FIXED for semilogx plots, truncate to only show t >= 100
                X_to_plot_here = X[X >= 100]
                Y_to_plot_here = Y[X >= 100]
                plot_method(X_to_plot_here[::self.delta_t_plot], Y_to_plot_here[::self.delta_t_plot],
                            label=policy.__cachedstr__, color=colors[policyId], marker=markers[policyId],
                            markevery=(policyId / 50., 0.1),
                            lw=lw, ms=int(1.5 * lw))
            else:
                plot_method(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=policy.__cachedstr__,
                            color=colors[policyId],
                            marker=markers[policyId], lw=1, ms=2)
            if semilogx or loglog:  # Manual fix for issue https://github.com/SMPyBandits/SMPyBandits/issues/38
                plt.xscale('log')
            if semilogy or loglog:  # Manual fix for issue https://github.com/SMPyBandits/SMPyBandits/issues/38
                plt.yscale('log')
            # Print standard deviation of regret
            # TODO
            # if plotSTD and self.repetitions > 1:
            #     stdY = self.getSTDRegret(policyId, envId, meanReward=meanReward)
            #     if normalizedRegret:
            #         stdY /= np.log(2 + X)
            #     plt.fill_between(X[::self.delta_t_plot], Y[::self.delta_t_plot] - stdY[::self.delta_t_plot],
            #                      Y[::self.delta_t_plot] + stdY[::self.delta_t_plot], facecolor=colors[policyId], alpha=0.2)

            # Print amplitude of regret
            if plotMaxMin and self.repetitions > 1:
                MaxMinY = self.getCumulatedRewardAmplitude(policyId, envId)
                if normalizedRegret:
                    MaxMinY /= self.horizon
                plt.fill_between(X[::self.delta_t_plot], Y[::self.delta_t_plot] - MaxMinY[::self.delta_t_plot],
                                 Y[::self.delta_t_plot] + MaxMinY[::self.delta_t_plot], facecolor=colors[policyId],
                                 alpha=0.2)


        self._xlabel(envId, r"Time steps $t = 1...T$, horizon $T = {}$".format(self.horizon))

        if not meanReward:
            if semilogy or loglog:
                ymin = max(0, ymin)
            plt.ylim(ymin, plt.ylim()[1])
        # Get a small string to add to ylabel
        ylabel2 = r"%s%s" % (r", $\pm 1$ standard deviation" if (plotSTD and not plotMaxMin) else "",
                             r", $\pm 1$ amplitude" if (plotMaxMin and not plotSTD) else "")
        if meanReward:
            legend()
            plt.ylabel("Mean reward" + ylabel2)
            plt.title(
                "Mean rewards for different bandit algorithms, averaged over ${}$ repetitions{}".format(
                    self.repetitions, subtitle
                ))

        elif normalizedRegret:
            legend()
            plt.ylabel("Normalized regret" + ylabel2)
            plt.title(
                "Normalized cumulated regrets, averaged over ${}$ repetitions{}".format(
                    self.repetitions, subtitle
                ))
        elif regretOverMaxReturn:
            legend()
            plt.ylabel(r"$\frac{R_t}{r_{max}}$")
            plt.title("Regrets $R_t$ relative to maximum rewards $r_{{max}}$, averaged over ${}$ repetitions{}".format(
                self.repetitions, subtitle)
            )
        else:

            # FIXED for semilogx plots, truncate to only show t >= 100
            if semilogx or loglog:
                X = X[X >= 100]
            else:
                X = X[X >= 1]
            legend()
            plt.ylabel(r"Average regret" + ylabel2)
            plt.title("Cumulative regrets, averaged over ${}$ repetitions{}".format(
                self.repetitions, subtitle)
            )
            # mpl.rcParams['lines.linewidth'] = 1
            # mpl.rcParams['lines.markersize'] = 1
            for policyId, policy in enumerate(self.policies):
                x = [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]
                y = self.getMeanYForErrors(policyId, envId)
                print(x)
                print(y)
                y_errors = self.getErrors(policyId, envId)
                print(y_errors)
                print(y.shape)
                plt.errorbar(x, y, y_errors, fmt='o', color=colors[policyId], label='Error bars', linewidth=2)

        if show:
            show_and_save(self.showplot, savefig, fig=fig, pickleit=USE_PICKLE)

        return fig

    def getErrors(self, policyId, envId):
        regrets = self.getCumulatedRegrets(policyId, envId)
        return np.array([np.std(regrets[:, t]) for t in range(0, self.horizon, 111)])

    def getMeanYForErrors(self, policyId, envId):
        meanY = self.getCumulatedRegretAverage(policyId, envId)
        return np.array([meanY[t] for t in range(0, self.horizon, 111)])

    def printRunningTimes(self, envId=0, precision=3):
        """Print the average+-std running time of the different policies."""
        print("\nGiving the mean and std running times ...")
        try:
            from IPython.core.magics.execution import _format_time
        except ImportError:
            _format_time = str
        means, stds, _ = self.getRunningTimes(envId)
        for policyId in np.argsort(means):
            policy = self.policies[policyId]
            print("\nFor policy #{} called '{}' ...".format(policyId, policy.__cachedstr__))
            mean_time, var_time = means[policyId], stds[policyId]
            if self.repetitions <= 1:
                print(u"    {} (mean of 1 run)".format(_format_time(mean_time, precision)))
            else:
                print(u"    {} ± {} per loop (mean ± std. dev. of {} run)".format(_format_time(mean_time, precision),
                                                                                  _format_time(var_time, precision),
                                                                                  self.repetitions))
        for policyId, policy in enumerate(self.policies):
            print("For policy #{} called '{}' ...".format(policyId, policy.__cachedstr__))
            mean_time, var_time = means[policyId], stds[policyId]
            print(r"T^{%i}_{T=%i,K=%i} = " % (policyId + 1, self.horizon, self.envs[envId].nbArms) + r"{} pm {}".format(
                int(round(1000 * mean_time)), int(round(1000 * var_time))))  # XXX in milli seconds
        # table_to_latex(mean_data=means, std_data=stds, labels=[policy.__cachedstr__ for policy in self.policies], fmt_function=_format_time)

    def plotRunningTimes(self, envId=0, savefig=None, base=1, unit="seconds"):
        """Plot the running times of the different policies, as a box plot for each."""
        means, _, all_times = self.getRunningTimes(envId=envId)
        # order by increasing mean time
        index_of_sorting = np.argsort(means)
        labels = [policy.__cachedstr__ for policy in self.policies]
        labels = [labels[i] for i in index_of_sorting]
        all_times = [np.asarray(all_times[i]) / float(base) for i in index_of_sorting]
        fig = plt.figure()
        violin_or_box_plot(data=all_times, labels=labels, boxplot=self.use_box_plot)
        plt.xlabel("Bandit algorithms")
        ylabel = "Running times (in {}), for {} repetitions".format(unit, self.repetitions)
        plt.ylabel(ylabel)
        adjust_xticks_subplots(ylabel=ylabel, labels=labels)
        plt.title(
            "Running times for different bandit algorithms, horizon $T={}$, averaged ${}$ times\n${}$ arms{}: {}".format(
                self.horizon, self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(),
                self.envs[envId].reprarms(1, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=USE_PICKLE)
        return fig

    def printMemoryConsumption(self, envId=0):
        """Print the average+-std memory consumption of the different policies."""
        print("\nGiving the mean and std memory consumption ...")
        means, stds, _ = self.getMemoryConsumption(envId)
        for policyId in np.argsort(means):
            policy = self.policies[policyId]
            print("\nFor policy #{} called '{}' ...".format(policyId, policy.__cachedstr__))
            mean_memory, var_memory = means[policyId], stds[policyId]
            if self.repetitions <= 1:
                print(u"    {} (mean of 1 run)".format(sizeof_fmt(mean_memory)))
            else:
                print(
                    u"    {} ± {} (mean ± std. dev. of {} runs)".format(sizeof_fmt(mean_memory), sizeof_fmt(var_memory),
                                                                        self.repetitions))
        for policyId, policy in enumerate(self.policies):
            print("For policy #{} called '{}' ...".format(policyId, policy.__cachedstr__))
            mean_memory, var_memory = means[policyId], stds[policyId]
            print(r"M^{%i}_{T=%i,K=%i} = " % (policyId + 1, self.horizon, self.envs[envId].nbArms) + r"{} pm {}".format(
                int(round(mean_memory)), int(round(var_memory))))  # XXX in B
        # table_to_latex(mean_data=means, std_data=stds, labels=[policy.__cachedstr__ for policy in self.policies], fmt_function=sizeof_fmt)

    def plotMemoryConsumption(self, envId=0, savefig=None, base=1024, unit="KiB"):
        """Plot the memory consumption of the different policies, as a box plot for each."""
        means, _, all_memories = self.getMemoryConsumption(envId=envId)
        # order by increasing mean memory consumption
        index_of_sorting = np.argsort(means)
        labels = [policy.__cachedstr__ for policy in self.policies]
        labels = [labels[i] for i in index_of_sorting]
        all_memories = [np.asarray(all_memories[i]) / float(base) for i in index_of_sorting]
        fig = plt.figure()
        violin_or_box_plot(data=all_memories, labels=labels, boxplot=self.use_box_plot)
        plt.xlabel("Bandit algorithms")
        ylabel = "Memory consumption (in {}), for {} repetitions".format(unit, self.repetitions)
        plt.ylabel(ylabel)
        adjust_xticks_subplots(ylabel=ylabel, labels=labels)
        plt.title(
            "Memory consumption for different bandit algorithms, horizon $T={}$, averaged ${}$ times\n${}$ arms{}: {}".format(
                self.horizon, self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(),
                self.envs[envId].reprarms(1, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=USE_PICKLE)

    def printNumberOfCPDetections(self, envId=0):
        """Print the average+-std number_of_cp_detections of the different policies."""
        means, stds, _ = self.getNumberOfCPDetections(envId)
        if np.max(means) == 0: return None
        print("\nGiving the mean and std number of CP detections ...")
        for policyId in np.argsort(means):
            policy = self.policies[policyId]
            print("\nFor policy #{} called '{}' ...".format(policyId, policy.__cachedstr__))
            mean_number_of_cp_detections, var_number_of_cp_detections = means[policyId], stds[policyId]
            if self.repetitions <= 1:
                print(u"    {:.3g} (mean of 1 run)".format(mean_number_of_cp_detections))
            else:
                print(u"    {:.3g} ± {:.3g} (mean ± std. dev. of {} runs)".format(mean_number_of_cp_detections,
                                                                                  var_number_of_cp_detections,
                                                                                  self.repetitions))
        # table_to_latex(mean_data=means, std_data=stds, labels=[policy.__cachedstr__ for policy in self.policies])

    def plotNumberOfCPDetections(self, envId=0, savefig=None):
        """Plot the number of change-point detections of the different policies, as a box plot for each."""
        means, _, all_number_of_cp_detections = self.getNumberOfCPDetections(envId=envId)
        if np.max(means) == 0: return None
        # order by increasing mean nb of change-point detection
        index_of_sorting = np.argsort(means)
        labels = [policy.__cachedstr__ for policy in self.policies]
        labels = [labels[i] for i in index_of_sorting]
        all_number_of_cp_detections = [np.asarray(all_number_of_cp_detections[i]) for i in index_of_sorting]
        fig = plt.figure()
        violin_or_box_plot(data=all_number_of_cp_detections, labels=labels, boxplot=self.use_box_plot)
        plt.xlabel("Bandit algorithms")
        ylabel = "Number of detected change-points, for {} repetitions".format(self.repetitions)
        plt.ylabel(ylabel)
        adjust_xticks_subplots(ylabel=ylabel, labels=labels)
        plt.title(
            "Detected change-points for different bandit algorithms, horizon $T={}$, averaged ${}$ times\n${}$ arms{}: {}".format(
                self.horizon, self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(),
                self.envs[envId].reprarms(1, latex=True)))
        show_and_save(self.showplot, savefig, fig=fig, pickleit=USE_PICKLE)
        return fig

    def printLastRegrets(self, envId=0, moreAccurate=False):
        """Print the last regrets of the different policies."""
        print("\nGiving the vector of final regrets ...")
        for policyId, policy in enumerate(self.policies):
            print("\n  For policy #{} called '{}' ...".format(policyId, policy.__cachedstr__))
            last_regrets = self.getLastRegrets(policyId, envId=envId)
            print("  Last regrets (for all repetitions) have:")
            print("Min of    last regrets R_T = {:.3g}".format(np.min(last_regrets)))
            print("Mean of   last regrets R_T = {:.3g}".format(np.mean(last_regrets)))
            print("Median of last regrets R_T = {:.3g}".format(np.median(last_regrets)))
            print("Max of    last regrets R_T = {:.3g}".format(np.max(last_regrets)))
            print("Standard deviation     R_T = {:.3g}".format(np.std(last_regrets)))
        for policyId, policy in enumerate(self.policies):
            print("For policy #{} called '{}' ...".format(policyId, policy.__cachedstr__))
            last_regrets = self.getLastRegrets(policyId, envId=envId)
            print(r"R^{%i}_{T=%i,K=%i} = " % (policyId + 1, self.horizon, self.envs[envId].nbArms) + r"{} pm {}".format(
                int(round(np.mean(last_regrets))), int(round(np.std(last_regrets)))))
        means = [np.mean(self.getLastRegrets(policyId, envId=envId)) for policyId in
                 range(self.nbPolicies)]
        stds = [np.std(self.getLastRegrets(policyId, envId=envId)) for policyId in
                range(self.nbPolicies)]
        # table_to_latex(mean_data=means, std_data=stds, labels=[policy.__cachedstr__ for policy in self.policies])

    def plotLastRegrets(self, envId=0,
                        normed=False, subplots=True, nbbins=10, log=False,
                        all_on_separate_figures=False, sharex=False, sharey=False,
                        boxplot=False, normalized_boxplot=True,
                        savefig=None, moreAccurate=False):
        """Plot histogram of the regrets R_T for all policies."""
        N = self.nbPolicies
        if N == 1:
            subplots = False  # no need for a subplot
        colors = palette(N)
        markers = makemarkers(N)
        if self.repetitions == 1:
            boxplot = True
        if boxplot:
            all_last_regrets = []
            for policyId, policy in enumerate(self.policies):
                last_regret = self.getLastRegrets(policyId, envId=envId)
                if normalized_boxplot:
                    last_regret /= np.log(self.horizon)
                all_last_regrets.append(last_regret)
            means = [np.mean(last_regrets) for last_regrets in all_last_regrets]
            # order by increasing mean regret
            index_of_sorting = np.argsort(means)
            labels = [policy.__cachedstr__ for policy in self.policies]
            labels = [labels[i] for i in index_of_sorting]
            all_last_regrets = [np.asarray(all_last_regrets[i]) for i in index_of_sorting]
            fig = plt.figure()
            plt.xlabel("Bandit algorithms")
            ylabel = "{}egret value $R_T{}$,\nfor $T = {}$, for {} repetitions".format(
                "Normalized r" if normalized_boxplot else "R", r"/\log(T)" if normalized_boxplot else "", self.horizon,
                self.repetitions)
            plt.ylabel(ylabel, fontsize="x-small")
            violin_or_box_plot(data=all_last_regrets, labels=labels, boxplot=self.use_box_plot)
            adjust_xticks_subplots(ylabel=ylabel, labels=labels)
            plt.title(
                "Regret for different bandit algorithms, horizon $T={}$, averaged ${}$ times\n${}$ arms{}: {}".format(
                    self.horizon, self.repetitions, self.envs[envId].nbArms, self.envs[envId].str_sparsity(),
                    self.envs[envId].reprarms(1, latex=True)))
        elif all_on_separate_figures:
            figs = []
            for policyId, policy in enumerate(self.policies):
                fig = plt.figure()
                plt.title(
                    "Histogram of regrets for {}\n${}$ arms{}: {}".format(policy.__cachedstr__, self.envs[envId].nbArms,
                                                                          self.envs[envId].str_sparsity(),
                                                                          self.envs[envId].reprarms(1, latex=True)))
                plt.xlabel("Regret value $R_T$, horizon $T = {}".format(self.horizon))
                plt.ylabel("Density of observations, ${}$ repetitions".format(self.repetitions))
                last_regrets = self.getLastRegrets(policyId, envId=envId)
                try:
                    sns.distplot(last_regrets, hist=True, bins=nbbins, color=colors[policyId],
                                 kde_kws={'cut': 0, 'marker': markers[policyId], 'markevery': (policyId / 50., 0.1)})
                except np.linalg.linalg.LinAlgError:
                    print(
                        "WARNING: a call to sns.distplot() failed because of a stupid numpy.linalg.linalg.LinAlgError exception... See https://api.travis-ci.org/v3/job/528931259/log.txt")  # WARNING
                legend()
                show_and_save(self.showplot,
                              None if savefig is None else "{}__Algo_{}_{}".format(savefig, 1 + policyId, 1 + N),
                              fig=fig, pickleit=USE_PICKLE)
                figs.append(fig)
            return figs
        elif subplots:
            nrows, ncols = nrows_ncols(N)
            fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey)
            fig.suptitle(
                "Histogram of regrets for different bandit algorithms\n${}$ arms{}: {}".format(self.envs[envId].nbArms,
                                                                                               self.envs[
                                                                                                   envId].str_sparsity(),
                                                                                               self.envs[
                                                                                                   envId].reprarms(1,
                                                                                                                   latex=True)))
            # XXX See https://stackoverflow.com/a/36542971/
            ax0 = fig.add_subplot(111, frame_on=False)  # add a big axes, hide frame
            ax0.grid(False)  # hide grid
            ax0.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                            right=False)  # hide tick and tick label of the big axes
            # Add only once the ylabel, xlabel, in the middle
            ax0.set_ylabel(
                "{} of observations, ${}$ repetitions".format("Frequency" if normed else "Histogram and density",
                                                              self.repetitions))
            ax0.set_xlabel("Regret value $R_T$, horizon $T = {}$".format(self.horizon))
            for policyId, policy in enumerate(self.policies):
                i, j = policyId % nrows, policyId // nrows
                ax = axes[i, j] if ncols > 1 else axes[i]
                last_regrets = self.getLastRegrets(policyId, envId=envId)
                try:
                    sns.distplot(last_regrets, ax=ax, hist=True, bins=nbbins, color=colors[policyId],
                                 kde_kws={'cut': 0, 'marker': markers[policyId],
                                          'markevery': (policyId / 50., 0.1)})  # XXX
                except np.linalg.linalg.LinAlgError:
                    print(
                        "WARNING: a call to sns.distplot() failed because of a stupid numpy.linalg.linalg.LinAlgError exception... See https://api.travis-ci.org/v3/job/528931259/log.txt")  # WARNING
                ax.set_title(policy.__cachedstr__, fontdict={
                    'fontsize': 'xx-small'})  # XXX one of x-large, medium, small, None, xx-large, x-small, xx-small, smaller, larger, large
                ax.tick_params(axis='both', labelsize=8)  # XXX https://stackoverflow.com/a/11386056/
        else:
            fig = plt.figure()
            plt.title(
                "Histogram of regrets for different bandit algorithms\n${}$ arms{}: {}".format(self.envs[envId].nbArms,
                                                                                               self.envs[
                                                                                                   envId].str_sparsity(),
                                                                                               self.envs[
                                                                                                   envId].reprarms(1,
                                                                                                                   latex=True)))
            plt.xlabel("Regret value $R_T$, horizon $T = {}$".format(self.horizon))
            plt.ylabel(
                "{} of observations, ${}$ repetitions".format("Frequency" if normed else "Number", self.repetitions))
            all_last_regrets = []
            labels = []
            for policyId, policy in enumerate(self.policies):
                all_last_regrets.append(self.getLastRegrets(policyId, envId=envId))
                labels.append(policy.__cachedstr__)
            if self.nbPolicies > 6: nbbins = int(nbbins * self.nbPolicies / 6)
            for policyId in range(self.nbPolicies):
                try:
                    sns.distplot(all_last_regrets[policyId], label=labels[policyId], hist=False, color=colors[policyId],
                                 kde_kws={'cut': 0, 'marker': markers[policyId],
                                          'markevery': (policyId / 50., 0.1)})  #, bins=nbbins)  # XXX
                except np.linalg.linalg.LinAlgError:
                    print(
                        "WARNING: a call to sns.distplot() failed because of a stupid numpy.linalg.linalg.LinAlgError exception... See https://api.travis-ci.org/v3/job/528931259/log.txt")  # WARNING
            legend()
        # Common part
        show_and_save(self.showplot, savefig, fig=fig, pickleit=USE_PICKLE)
        return fig


# Helper function for the parallelization

def delayed_play(env, policy, horizon,
                 all_contexts, all_rewards,
                 env_id, repeatId=0, verbose=False):
    """Helper function for the parallelization."""
    start_time = time.time()
    start_memory = getCurrentMemory(thread=False)

    # We have to deepcopy because this function is Parallel-ized
    env = deepcopy(env)
    policy = deepcopy(policy)

    # Start game
    policy.startGame()
    result = Result(env.nbArms, horizon)  # One Result object, for every policy

    # self.all_contexts[envId] = np.zeros((self.repetitions, self.horizon, self.envs[envId].nbArms))
    # self.all_rewards[envId] = np.zeros((self.repetitions, self.horizon, self.envs[envId].nbArms))

    pretty_range = tqdm(range(horizon), desc="Time t") if repeatId == 0 and verbose == True else range(horizon)
    for t in pretty_range:
        # 1. A context is drawn
        contexts = all_contexts[env_id][repeatId, t]

        if isinstance(policy, ContextualBasePolicy):
            # 2. The player's policy choose an arm
            choice = policy.choice(contexts)

            # 3. A random reward is drawn, from this arm at this time
            reward = all_rewards[env_id][repeatId, t, choice]

            # 4. The policy sees the reward
            policy.getReward(choice, reward, contexts)
        else:
            # 2. The player's policy choose an arm
            choice = policy.choice()

            # self.all_rewards[envId] = np.zeros((self.repetitions, self.horizon, self.envs[envId].nbArms))
            # 3. A random reward is drawn, from this arm at this time
            reward = all_rewards[env_id][repeatId, t, choice]

            # 4. The policy sees the reward

            policy.getReward(choice, reward)

        # 5. Finally we store the results
        result.store(t, choice, reward)

    # Finally, store running time and consumed memory
    result.running_time = time.time() - start_time
    memory_consumption = getCurrentMemory(thread=False) - start_memory
    if memory_consumption == 0:
        # XXX https://stackoverflow.com/a/565382/
        memory_consumption = sys.getsizeof(pickle.dumps(policy))
        # if repeatId == 0: print("Warning: unable to get the memory consumption for policy {}, so we used a trick to measure {} bytes.".format(policy, memory_consumption))  # DEBUG
    result.memory_consumption = memory_consumption
    return result


# --- Helper for loading a previous Evaluator object

# def EvaluatorFromDisk(filepath='/tmp/saveondiskEvaluator.hdf5'):
#     """ Create a new Evaluator object from the HDF5 file given in argument."""
#     with open(filepath, 'r') as hdf:
#         configuration = hdf.configuration
#         evaluator = EvaluatorContextual(configuration)
#         evaluator.loadfromdisk(hdf)
#     return evaluator


# --- Utility function

from random import shuffle
from copy import copy


def shuffled(mylist):
    """Returns a shuffled version of the input 1D list. sorted() exists instead of list.sort(), but shuffled() does not exist instead of random.shuffle()...

    >>> from random import seed; seed(1234)  # reproducible results
    >>> mylist = [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9]
    >>> shuffled(mylist)
    [0.9, 0.4, 0.3, 0.6, 0.5, 0.7, 0.1, 0.2, 0.8]
    >>> shuffled(mylist)
    [0.4, 0.3, 0.7, 0.5, 0.8, 0.1, 0.9, 0.6, 0.2]
    >>> shuffled(mylist)
    [0.4, 0.6, 0.9, 0.5, 0.7, 0.2, 0.1, 0.3, 0.8]
    >>> shuffled(mylist)
    [0.8, 0.7, 0.3, 0.1, 0.9, 0.5, 0.6, 0.2, 0.4]
    """
    copiedlist = copy(mylist)
    shuffle(copiedlist)
    return copiedlist


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod

    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
