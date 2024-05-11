__author__ = "Cody Boon"
__version__ = "0.1"

import numpy as np
from matplotlib import pyplot as plt

from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm

from SMPyBandits.Environment.plotsettings import wraplatex, wraptext, legend, signature, show_and_save, palette


class ContextualMAB(object):
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
        self._sparsity = None
        self.non_zero_means = 0

        print("  Reading arms of this Contextual MAB problem from a dictionary 'configuration' = {} ...".format(
            configuration))  # DEBUG
        arm_type = configuration["arm_type"]
        print(" - with 'arm_type' =", arm_type)  # DEBUG
        arm_params = configuration["arm_params"]
        print(" - with 'arm_params' =", arm_params)  # DEBUG
        # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian) or a dictionnary
        for param in arm_params:
            self.arms.append(arm_type(*param) if isinstance(param, (dict, tuple)) else arm_type(param))
            if self.arms[-1].is_nonzero():
                self.non_zero_means += 1
        # XXX try to read sparsity
        self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None

        print("  and contexts...")
        context_type = configuration["context_type"]
        print(" - with 'context_type' =", context_type)  # DEBUG
        context_params = configuration["context_params"]
        print(" - with 'context_params' =", arm_params)  # DEBUG

        assert isinstance(context_params, (dict, tuple, list)), "Error: Context params must be iterable"
        self.context = context_type(*context_params)

        # Compute the means and stats
        print(" - with 'arms' =", self.arms)  # DEBUG
        self.nbArms = len(self.arms)  #: Number of arms
        print(" - with 'nbArms' =", self.nbArms)  # DEBUG
        if self._sparsity is not None:
            print(" - with 'sparsity' =", self._sparsity)  # DEBUG
        print(" - with 'arms' represented as:", self.reprarms(1, latex=True))  # DEBUG

    def new_order_of_arm(self, arms):
        """ Feed a new order of the arms to the environment.

        - Updates :attr:`means` correctly.
        - Return the new position(s) of the best arm (to count and plot ``BestArmPulls`` correctly).

        warning:: This only changes the order of the arms, leaving the context the same
        """
        assert set(self.arms) == set(
            arms), "Error: the new list of arms = {} does not have the same arms as the previous one."  # DEBUG
        self.arms = arms


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
            repr_arms = [repr(arm) for arm in self.arms]

            text = '[{}]'.format(', '.join(
                openTag + repr_arms[armId] + endTag
                for armId, arm in enumerate(self.arms))
            )
            text += append_to_repr
        return wraplatex('$' + text + '$') if latex else wraptext(text)

    # --- Draw samples

    def draw(self, armId, t=1):
        """ Return a random sample from the armId-th arm, at time t. """
        context_draw = self.draw_context()
        return context_draw, self.arms[armId].draw(context_draw, t)

    def draw_nparray(self, armId, shape=(1,)):
        """
            Return a numpy array of contexts and samples from the armId-th arm, of a certain shape.
        """
        contexts = self.context.draw_nparray(shape)
        return contexts, self.arms[armId].draw_nparray(shape, contexts)

    def draw_context(self):
        return self.context.draw_context()

    @property
    def sparsity(self):
        """ Estimate the sparsity of the problem, i.e., the number of arms with positive means."""
        if self._sparsity is not None:
            return self._sparsity
        else:
            return self.non_zero_means

    def str_sparsity(self):
        """ Empty string if ``sparsity = nbArms``, or a small string ', $s={}$' if the sparsity is strictly less than the number of arm."""
        s, K = self.sparsity, self.nbArms
        assert 0 <= s <= K, "Error: sparsity s = {} has to be 0 <= s <= K = {}...".format(s, K)
        # WARNING
        # disable this feature when not working on sparse simulations
        # return ""
        # or bring back this feature when working on sparse simulations
        return "" if s == K else ", $s={}$".format(s)

    def plotHistogram(self, horizon=10000, savefig=None, bins=50, alpha=0.9, density=None):
        """Plot a horizon=10000 draws of each arms."""
        rewards = np.zeros((self.nbArms, horizon))
        colors = palette(self.nbArms)
        for armId in range(self.nbArms):
            for t in range(horizon):
                rewards[armId, t] = self.draw(armId, t)
        # Now plot
        fig = plt.figure()
        for armId in range(self.nbArms):
            plt.hist(rewards[armId, :], bins=bins, density=density, color=colors[armId],
                     label='$%s$' % repr(self.arms[armId]),
                     alpha=alpha)
        legend()
        plt.xlabel("Rewards")
        if density:
            plt.ylabel("Empirical density of the rewards")
        else:
            plt.ylabel("Empirical count of observations of the rewards")
        plt.title("{} draws of rewards from these arms.\n{} arms: {}{}".format(horizon, self.nbArms,
                                                                               self.reprarms(latex=True), signature))
        show_and_save(showplot=True, savefig=savefig, fig=fig, pickleit=False)
        return fig