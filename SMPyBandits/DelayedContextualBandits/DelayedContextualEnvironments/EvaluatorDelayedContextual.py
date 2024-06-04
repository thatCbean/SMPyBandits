
import random
import time
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import pickle
from copy import deepcopy
from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.DelayedContextualBandits.DelayedContextualEnvironments.DelayedContextualMAB import DelayedContextualMAB
from SMPyBandits.DelayedContextualBandits.Policies.ContextualBasePolicyWithDelay import ContextualBasePolicyWithDelay
from SMPyBandits.Environment.Result import Result
from SMPyBandits.Environment.memory_consumption import getCurrentMemory, sizeof_fmt
from SMPyBandits.Environment.usetqdm import USE_TQDM, tqdm

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
USE_JOBLIB_FOR_POLICIES = False  #: Don't use joblib to parallelize the simulations on various policies (we parallelize the random Monte Carlo repetitions)

class EvaluatorDelayedContextual(EvaluatorContextual):

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
                self.envs.append(DelayedContextualMAB(configuration_envs))
    
    def __init__(self, configuration,
                 finalRanksOnAverage=FINAL_RANKS_ON_AVERAGE, averageOn=5e-3,
                 useJoblibForPolicies=USE_JOBLIB_FOR_POLICIES,
                 moreAccurate=MORE_ACCURATE):
        super(EvaluatorDelayedContextual, self).__init__(configuration, finalRanksOnAverage, averageOn, useJoblibForPolicies, moreAccurate)
        self.all_delays = {}

    def compute_cache_rewards(self, env, env_id):
        """ Compute only once the rewards, then launch the experiments with the same matrix (r_{k,t})."""
        if self.seeds is not None and len(self.seeds) > 1:
            random.seed(self.seeds[env_id])
            np.random.seed(self.seeds[env_id])

        rewards = np.zeros((self.repetitions, self.horizon, env.nbArms))
        contexts = np.zeros((self.repetitions, self.horizon, env.nbArms, self.dimension))
        delays =  np.zeros((self.repetitions, self.horizon, env.nbArms))
        print(
            "\n===> Pre-computing the rewards ... Of shape {} ...\n    In order for all simulated algorithms to face the same random rewards (robust comparison of A1,..,An vs Aggr(A1,..,An)) ...\n".format(
                np.shape(rewards)))  # DEBUG
        if self.verbosity > 2:
            for repetitionId in tqdm(range(self.repetitions), desc="Repetitions"):
                for t in tqdm(range(self.horizon), desc="Time steps"):
                    for arm_id in tqdm(range(len(env.arms)), desc="Arms"):
                        contexts[repetitionId, t, arm_id], rewards[repetitionId, t, arm_id], delays[repetitionId, t, arm_id] = env.draw(arm_id, t)
        else:
            for repetitionId in tqdm(range(self.repetitions), desc="Repetitions"):
                for t in range(self.horizon):
                    for arm_id in range(len(env.arms)):
                        contexts[repetitionId, t, arm_id], rewards[repetitionId, t, arm_id], delays[repetitionId, t, arm_id] = env.draw(arm_id, t)
        return contexts, rewards, delays
    
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
        all_contexts, all_rewards, all_delays = self.compute_cache_rewards(env, envId)
        self.all_contexts[envId], self.all_rewards[envId], self.all_delays[envId] = all_contexts, all_rewards, all_delays

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
            for repeatId in (tqdm(range(self.repetitions), desc="Repeat") if self.verbosity > 3 else range(self.repetitions)):
                if self.useJoblib:
                    repeatIdout = 0
                    for r in Parallel(n_jobs=self.cfg['n_jobs'], pre_dispatch='3*n_jobs', verbose=self.cfg['verbosity'])(
                            delayed(delayed_play)(env, policy, self.horizon, self.all_contexts, self.all_rewards, self.all_delays, envId,
                                                    repeatId=repeatId, verbose=(self.verbosity > 4))
                            for repeatId in range(self.repetitions)
                    ):
                        store(r, policyId, repeatIdout)
                        repeatIdout += 1
                else:
                    for repeatId in (tqdm(range(self.repetitions), desc="Repeat") if self.verbosity > 3 else range(self.repetitions)):
                        r = delayed_play(env, policy, self.horizon, self.all_contexts, self.all_rewards, envId,
                                        repeatId=repeatId, verbose=(self.verbosity > 4))
                        store(r, policyId, repeatId)


    


    ##current time + delay should be used here
def add_to_window(window, t, arm_id, reward):
    if t not in window:
        window[t] = []
    window[t].append((arm_id, reward))

##current time is used here
def remove_from_window(window, t):
    if t in window:
        return window.pop(t)
    else:
        return []



def delayed_play(env, policy, horizon,
                 all_contexts, all_rewards, all_delays,
                 env_id, repeatId=0, verbose=False):
    """Helper function for the parallelization."""
    start_time = time.time()
    start_memory = getCurrentMemory(thread=False)

    # We have to deepcopy because this function is Parallel-ized
    env = deepcopy(env)
    policy = deepcopy(policy)

    window = {}
    
    # Start game
    policy.startGame()
    result = Result(env.nbArms, horizon)  # One Result object, for every policy

    # self.all_contexts[envId] = np.zeros((self.repetitions, self.horizon, self.envs[envId].nbArms))
    # self.all_rewards[envId] = np.zeros((self.repetitions, self.horizon, self.envs[envId].nbArms))

    pretty_range = tqdm(range(horizon), desc="Time t") if repeatId == 0 and verbose == True else range(horizon)
    ##TODO change the 4th sttep. Instead of observing the reward instantly, we should observe the reward after a delay
    ##keep a data structure with rewards, time steps and so on
    for t in pretty_range:
        # 1. A context is drawn
        contexts = all_contexts[env_id][repeatId, t]

        if isinstance(policy, ContextualBasePolicyWithDelay):
            # 2. The player's policy choose an arm
            choice = policy.choice(contexts)

            policy.pull_arm(choice)

            # 3. A random reward is drawn, from this arm at this time
            reward = all_rewards[env_id][repeatId, t, choice]

            #4. A random delay is drawn, from this arm at this time, the current reward(if its exists) is observed and the drawn reward is delayed for later observation
            delay = all_delays[env_id][repeatId, t, choice]
            add_to_window(window, t + delay, choice, reward)
            for(arm_id, reward) in remove_from_window(window, t):
                policy.update_reward(arm_id, reward)
                policy.update_estimators(arm_id, reward, contexts)

        else:
            # 2. The player's policy choose an arm
            choice = policy.choice()

            policy.pull_arm(choice)

            # self.all_rewards[envId] = np.zeros((self.repetitions, self.horizon, self.envs[envId].nbArms))
            # 3. A random reward is drawn, from this arm at this time
            reward = all_rewards[env_id][repeatId, t, choice]

            #4. A random delay is drawn, from this arm at this time, the current reward(if its exists) is observed and the drawn reward is delayed for later observation
            delay = all_delays[env_id][repeatId, t, choice]
            add_to_window(window, t + delay, choice, reward)
            for(arm_id, reward) in remove_from_window(window, t):
                policy.update_reward(arm_id, reward)
                policy.update_estimators(arm_id, reward)
        # 5. Store the result
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