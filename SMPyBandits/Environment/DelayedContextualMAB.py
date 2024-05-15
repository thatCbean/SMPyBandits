__author__ = "Cody Boon"
__version__ = "0.1"

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm

from SMPyBandits.ContextualArms.ContextualArm import ContextualArm
from SMPyBandits.Environment import ContextualMAB
from SMPyBandits.Environment.MAB import binomialCoefficient

from SMPyBandits.Environment.plotsettings import wraplatex, wraptext, legend, signature, show_and_save, palette


class DelayedContextualMAB(object):
    """ Contextual Multi-Armed Bandit problem, for stochastic and i.i.d. arms.

    - configuration can be a dict with 'arm_type' and 'params' keys. 'arm_type' is a class from the Arms module, and 'params' is a dict, used as a list/tuple/iterable of named parameters given to 'arm_type'. Example::

        configuration = {
                'arm_type': ContextualBernoulli,
                'arm_params':   [0.1, 0.5, 0.9]
                'context_type': NormalContext,
                'context_params':  [
                    [0.2, 0.1, 0.3],
                    np.identity(3) * [0.1, 0.2, 0.3]
                ]
        }

    """
    def __init__(self, configuration):
        """New Contextual MAB."""

        assert isinstance(configuration, dict), "Error: configuration should be a dict"

        print("\n\nCreating a new Contextual MAB problem ...")  # DEBUG
        self.isChangingAtEachRepetition = False  #: Flag to know if the problem is changing at each repetition or not.
        self.isDynamic = False  #: Flag to know if the problem is static or not.
        self.arms = []  #: List of arms
        self.context = None
        self.context_draws = {}  #: Storage for context draws
        self._sparsity = None


        print("  Reading arms of this Contextual MAB problem from a dictionary 'configuration' = {} ...".format(configuration))  # DEBUG
        arm_type = configuration["arm_type"]
        print(" - with 'arm_type' =", arm_type)  # DEBUG
        arm_params = configuration["arm_params"]
        print(" - with 'arm_params' =", arm_params)  # DEBUG
        # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian) or a dictionnary
        for param in arm_params:
            self.arms.append(arm_type(*param) if isinstance(param, (dict, tuple)) else arm_type(param))
        # XXX try to read sparsity
        self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None

        print("  and contexts...")
        context_type = configuration["context_type"]
        print(" - with 'context_type' =", context_type) # DEBUG
        context_params = configuration["context_params"]
        print(" - with 'context_params' =", arm_params)  # DEBUG

        assert isinstance(context_params, (dict, tuple, list)), "Error: Context params must be iterable"
        self.context = context_type(*context_params)

        # Compute the means and stats
        print(" - with 'arms' =", self.arms)  # DEBUG
        self.means = np.array([arm.calculate_mean(self.context.get_means()) if isinstance(arm, ContextualArm) else arm.mean for arm in self.arms])  #: Means of arms
        print(" - with 'means' =", self.means)  # DEBUG
        self.nbArms = len(self.arms)  #: Number of arms
        print(" - with 'nbArms' =", self.nbArms)  # DEBUG
        if self._sparsity is not None:
            print(" - with 'sparsity' =", self._sparsity)  # DEBUG
        self.maxArm = np.max(self.means)  #: Max mean of arms
        print(" - with 'maxArm' =", self.maxArm)  # DEBUG
        self.minArm = np.min(self.means)  #: Min mean of arms
        print(" - with 'minArm' =", self.minArm)  # DEBUG

        #Assign delay conifguration
        self.max_delay = configuration["max_delay"]  #: Max mean of arms
        print(" - with 'maxDelay' =", self.max_delay)  # DEBUG
        self.average_delay = configuration["average_delay"]  #: Min mean of arms
        print(" - with 'averageDelay' =", self.average_delay)  # DEBUG
        self.delay_buffer = self.initialize_delay_buffer(self.arms)
        #print(" - with 'delayBuffer' =", self.delay_buffer)  # DEBUG

        # Print lower bound and HOI factor
        # Print lower bound and HOI factor
        print("\nThis MAB problem has: \n - a [Lai & Robbins] complexity constant C(mu) = {:.3g} ... \n - a Optimal Arm Identification factor H_OI(mu) = {:.2%} ...".format(self.lowerbound(), self.hoifactor()))  # DEBUG
        print(" - with 'arms' represented as:", self.reprarms(1, latex=True))  # DEBUG


    def new_order_of_arm(self, arms):
        """ Feed a new order of the arms to the environment.

        - Updates :attr:`means` correctly.
        - Return the new position(s) of the best arm (to count and plot ``BestArmPulls`` correctly).

        warning:: This only changes the order of the arms, leaving the context the same
        """
        assert sorted([arm.mean for arm in self.arms]) == sorted([arm.mean for arm in arms]), "Error: the new list of arms = {} does not have the same means as the previous ones."  # DEBUG
        assert set(self.arms) == set(arms), "Error: the new list of arms = {} does not have the same arms as the previous one."  # DEBUG
        self.arms = arms
        self.means = np.array([arm.mean for arm in self.arms])
        self.maxArm = np.max(self.means)
        self.minArm = np.min(self.means)

    def __repr__(self):
        return "{}(nbArms: {}, arms: {})".format(self.__class__.__name__, self.nbArms, self.arms)

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*', latex=True):
        """ Return a str representation of the list of the arms (like `repr(self.arms)` but better)

        - If nbPlayers > 0, it surrounds the representation of the best arms by openTag, endTag (for plot titles, in a multi-player setting).

        - Example: openTag = '', endTag = '^*' for LaTeX tags to put a star exponent.
        - Example: openTag = '<red>', endTag = '</red>' for HTML-like tags.
        - Example: openTag = r'\textcolor{red}{', endTag = '}' for LaTeX tags.
        """
        if nbPlayers is None:
            text = repr(self.arms)
        else:
            assert nbPlayers >= 0, "Error, the 'nbPlayers' argument for reprarms method of a MAB object has to be a non-negative integer."  # DEBUG
            append_to_repr = ""

            means = self.means
            best_mean = np.max(means)
            best_arms = np.argsort(means)[-min(nbPlayers, self.nbArms):]
            repr_arms = [repr(arm) for arm in self.arms]

            # WARNING improve display for Gaussian arms that all have same variance
            if all("Gaussian" in str(type(arm)) for arm in self.arms) and len({arm.sigma for arm in self.arms}) == 1:
                sigma = self.arms[0].sigma
                repr_arms = [s.replace(', {:.3g}'.format(sigma), '') for s in repr_arms]
                append_to_repr = r", \sigma^2={:.3g}".format(sigma) if latex else ", sigma2={:.3g}".format(sigma)

            if nbPlayers == 0:
                best_arms = []

            text = '[{}]'.format(', '.join(
                openTag + repr_arms[armId] + endTag
                if (nbPlayers > 0 and (armId in best_arms or np.isclose(arm.mean, best_mean)))
                else repr_arms[armId]
                for armId, arm in enumerate(self.arms))
            )
            text += append_to_repr
        return wraplatex('$' + text + '$') if latex else wraptext(text)
    
    def initialize_delay_buffer(self, arms):
        # delay_buffer = [[0]*self.max_delay for _ in range(self.nbArms)]
        delay_buffer = [[] for _ in range(self.nbArms)]
        
        for (armId, arm) in enumerate(arms):
            print(f"Arm {armId} has mean {arm.mean}")
            for i in range(self.max_delay):
                context_draw = self.draw_context()
                current_reward = self.arms[armId].draw(context_draw, i)
                print(f"Reward for arm {armId} at time {i} is {current_reward}")
                delay_buffer[armId].append(current_reward)
        print(delay_buffer)
        return delay_buffer
    
    def get_delayed_reward(self, armId):
        x = truncnorm.rvs(1, self.max_delay, loc=self.average_delay, scale=0.5)
        rewards_for_arm = self.delay_buffer[armId]
        time = self.max_delay - int(x)
        assert time >= 0, "Error: time should be greater than or equal to 0"
        assert time < len(rewards_for_arm), f"Error: time should be smaller than{len(rewards_for_arm)}, got {time}"
        reward = rewards_for_arm[time]
        #assert 0 > 1, f"Reward: {rewards_for_arm}"
        return reward

    # --- Draw samples

    def draw(self, armId, t=1):
        """ Return a random sample from the armId-th arm, at time t. """
        context_draw = self.draw_context()
        current_reward = self.arms[armId].draw(context_draw, t)
        self.delay_buffer[armId].append(current_reward)
        return context_draw, self.get_delayed_reward(armId)

    def draw_with_context(self, armId, context, t=1):
        """ Return a random sample from the armId-th arm, at time t. """
        current_reward = self.arms[armId].draw(context, t)
        self.delay_buffer[armId].append(current_reward)
        return self.get_delayed_reward(armId)

    def draw_nparray(self, armId, shape=(1,)):
        """
            Return a numpy array of random sample from the armId-th arm, of a certain shape.
            The contexts will not be stored.
        """
        contexts = self.context.draw_nparray(shape)
        return contexts, self.arms[armId].draw_nparray(shape, contexts)

    def draw_each(self, t=1):
        """ Return a random sample from each arm, at time t. """

        return np.array([self.draw(armId, t) for armId in range(self.nbArms)])

    def draw_each_nparray(self, shape=(1,)):
        """
            Return a numpy array of random sample from each arm, of a certain shape.
            The contexts will not be stored
        """
        contexts = []
        rewards = []
        for armId in range(self.nbArms):
            c, r = self.draw_nparray(armId, shape)
            contexts.append(c)
            rewards.append(r)
        return np.array(contexts), np.array(rewards)

    def draw_context(self):
        return self.context.draw_context()

    # def get_or_draw_context(self, t):
    #     if t not in self.context_draws:
    #         self.context_draws[t] = self.draw_context()
    #     return self.context_draws[t]

    #
    # --- Helper to compute sets Mbest and Mworst

    def Mbest(self, M=1):
        """ Set of M best means."""
        sortedMeans = np.sort(self.means)
        return sortedMeans[-M:]

    def Mworst(self, M=1):
        """ Set of M worst means."""
        sortedMeans = np.sort(self.means)
        return sortedMeans[:-M]

    def sumBestMeans(self, M=1):
        """ Sum of the M best means."""
        return np.sum(self.Mbest(M=M))

    #
    # --- Helper to compute vector of min arms, max arms, all arms

    def get_minArm(self, horizon=None):
        """Return the vector of min mean of the arms.

        - It is a vector of length horizon.
        """
        return np.full(horizon, self.minArm)
        # return self.minArm  # XXX Nope, it's not a constant!

    def get_maxArm(self, horizon=None):
        """Return the vector of max mean of the arms.

        - It is a vector of length horizon.
        """
        return np.full(horizon, self.maxArm)
        # return self.maxArm  # XXX Nope, it's not a constant!

    def get_maxArms(self, M=1, horizon=None):
        """Return the vector of sum of the M-best means of the arms.

        - It is a vector of length horizon.
        """
        return np.full(horizon, self.sumBestMeans(M))

    def get_allMeans(self, horizon=None):
        """Return the vector of means of the arms.

        - It is a numpy array of shape (nbArms, horizon).
        """
        # allMeans = np.tile(self.means, (horizon, 1)).T
        allMeans = np.zeros((self.nbArms, horizon))
        for t in range(horizon):
            allMeans[:, t] = self.means
        return allMeans

    #
    # --- Estimate sparsity

    @property
    def sparsity(self):
        """ Estimate the sparsity of the problem, i.e., the number of arms with positive means."""
        if self._sparsity is not None:
            return self._sparsity
        else:
            return np.count_nonzero(self.means > 0)

    def str_sparsity(self):
        """ Empty string if ``sparsity = nbArms``, or a small string ', $s={}$' if the sparsity is strictly less than the number of arm."""
        s, K = self.sparsity, self.nbArms
        assert 0 <= s <= K, "Error: sparsity s = {} has to be 0 <= s <= K = {}...".format(s, K)
        # WARNING
        # disable this feature when not working on sparse simulations
        # return ""
        # or bring back this feature when working on sparse simulations
        return "" if s == K else ", $s={}$".format(s)

    #
    # --- Compute lower bounds

    def lowerbound(self):
        r""" Compute the constant :math:`C(\mu)`, for the [Lai & Robbins] lower-bound for this MAB problem (complexity), using functions from ``kullback.py`` or ``kullback.so`` (see :mod:`Arms.kullback`). """
        return sum(a.oneLR(self.maxArm, a.mean) for a in self.arms if a.mean != self.maxArm)

    def lowerbound_sparse(self, sparsity=None):
        """ Compute the constant :math:`C(\mu)`, for [Kwon et al, 2017] lower-bound for sparse bandits for this MAB problem (complexity)

        - I recomputed suboptimal solution to the optimization problem, and found the same as in [["Sparse Stochastic Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)].
        """
        if hasattr(self, "sparsity") and sparsity is None:
            sparsity = self._sparsity
        if sparsity is None:
            sparsity = self.nbArms

        try:
            try:
                from ..Policies.OSSB import solve_optimization_problem__sparse_bandits
            except ImportError:  # WARNING ModuleNotFoundError is only Python 3.6+
                from SMPyBandits.Policies.OSSB import solve_optimization_problem__sparse_bandits
            ci = solve_optimization_problem__sparse_bandits(self.means, sparsity=sparsity, only_strong_or_weak=False)
            # now we use these ci to compute the lower-bound
            gaps = [self.maxArm - a.mean for a in self.arms]
            lower_bound = sum(delta * c for (delta, c) in zip(gaps, ci))
        except (ImportError, ValueError, AssertionError):
            lower_bound = np.nan
        return lower_bound

    def hoifactor(self):
        """ Compute the HOI factor H_OI(mu), the Optimal Arm Identification (OI) factor, for this MAB problem (complexity). Cf. (3.3) in Navikkumar MODI's thesis, "Machine Learning and Statistical Decision Making for Green Radio" (2017)."""
        return sum(a.oneHOI(self.maxArm, a.mean) for a in self.arms if a.mean != self.maxArm) / float(self.nbArms)

    def plotHistogram(self, horizon=10000, savefig=None, bins=50, alpha=0.9, density=None):
        """Plot a horizon=10000 draws of each arms."""
        arms = self.arms
        rewards = np.zeros((len(arms), horizon))
        colors = palette(len(arms))
        for armId, arm in enumerate(arms):
            # if hasattr(arm, 'draw_nparray'):  # XXX Use this method to speed up computation
            #     rewards[armId] = arm.draw_nparray((horizon,))
            # else:  # Slower
            # draw_nparray not yet implemented correctly for ContextualMAB related classes
            for t in range(horizon):
                rewards[armId, t] = arm.draw(t)
        # Now plot
        fig = plt.figure()
        for armId, arm in enumerate(arms):
            plt.hist(rewards[armId, :], bins=bins, density=density, color=colors[armId], label='$%s$' % repr(arm), alpha=alpha)
        legend()
        plt.xlabel("Rewards")
        if density:
            plt.ylabel("Empirical density of the rewards")
        else:
            plt.ylabel("Empirical count of observations of the rewards")
        plt.title("{} draws of rewards from these arms.\n{} arms: {}{}".format(horizon, self.nbArms, self.reprarms(latex=True), signature))
        show_and_save(showplot=True, savefig=savefig, fig=fig, pickleit=False)
        return fig