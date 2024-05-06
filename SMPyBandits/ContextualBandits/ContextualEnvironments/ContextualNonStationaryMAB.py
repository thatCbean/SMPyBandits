import numpy as np
from SMPyBandits.Environment.plotsettings import legend, show_and_save, signature
from matplotlib import pyplot as plt

from SMPyBandits.Environment import wraptext, palette, makemarkers

from SMPyBandits.ContextualBandits.ContextualEnvironments.ContextualMAB import ContextualMAB

VERBOSE = True


class ContextualPieceWiseStationaryMAB(ContextualMAB):
    r"""Like a stationary Contextual MAB problem, but piece-wise stationary.
    TODO: Change doctext to reflect contextual bandits
    - Give it a list of vector of means, and a list of change-point locations.

    - You can use :meth:`plotHistoryOfMeans` to see a nice plot of the history of means.

    .. note:: This is a generic class to implement one "easy" kind of non-stationary bandits, abruptly changing non-stationary bandits, if changepoints are fixed and decided in advanced.

    .. warning:: It works fine, but it is still experimental, be careful when using this feature.

    .. warning:: The number of arms is fixed, see https://github.com/SMPyBandits/SMPyBandits/issues/123 if you are curious about bandit problems with a varying number of arms (or sleeping bandits where some arms can be enabled or disabled at each time).
    """

    def __init__(self, configuration, verbose=VERBOSE):
        """New PieceWiseStationaryMAB."""
        self.isChangingAtEachRepetition = False  #: The problem is not changing at each repetition.
        self.isDynamic = True  #: The problem is dynamic.
        self.isMarkovian = False  #: The problem is not Markovian.
        self._sparsity = None

        assert isinstance(configuration, dict) \
               and "arm_type" in configuration \
               and "arm_params" in configuration \
               and "listOfMeans" in configuration["arm_params"] \
               and "changePoints" in configuration["arm_params"] \
               and "context_type" in configuration \
               and "context_params" in configuration, \
            ("Error: this ContextualPieceWiseStationaryMAB is not really a contextual non-stationary MAB, "
             "you should use a different MAB or provide the right configuration!")  # DEBUG
        self._verbose = verbose

        print("  Special contextual MAB problem, with arm (possibly) changing at "
              "every time step, read from a dictionnary 'configuration' = {} ...".format(configuration))  # DEBUG

        #: Kind of context (ContextualPieceWiseStationaryMAB are homogeneous)
        self.context_type = context_type = configuration["context_type"]
        print(" - with 'context_type' =", context_type)  # DEBUG
        context_params = configuration["context_params"]
        print(" - with 'context_params' =", context_params)  # DEBUG

        #: Kind of arm (ContextualPieceWiseStationaryMAB are homogeneous)
        self.arm_type = arm_type = configuration["arm_type"]
        print(" - with 'arm_type' =", arm_type)  # DEBUG
        arm_params = configuration["arm_params"]
        print(" - with 'arm_params' =", arm_params)  # DEBUG

        self.listOfMeans = np.array(arm_params["listOfMeans"])  #: The list of parameters
        self.nbArms = len(self.listOfMeans[0])  #: Number of arms
        assert all(len(arms) == self.nbArms for arms in
                   self.listOfMeans), "Error: the number of arms cannot be different between change-points."  # DEBUG
        print(" - with 'listOfMeans' =", self.listOfMeans)  # DEBUG

        self.changePoints = arm_params["changePoints"]  #: List of the change points
        print(" - with 'changePoints' =", self.changePoints)  # DEBUG
        # XXX Maybe we need to add 0 in the list of changePoints
        if 0 not in self.changePoints and len(self.listOfMeans) == len(self.changePoints) - 1:
            self.changePoints = [0] + self.changePoints
        assert len(self.listOfMeans) == len(
            self.changePoints), ("Error: the list of means {} does not has the same length as the list of change "
                                 "points {}...").format(
            self.listOfMeans, self.changePoints)  # DEBUG

        # XXX try to read sparsity
        self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None

        print("\n\n ==> Creating the dynamic arms ...")  # DEBUG

        self.listOfArms = [
            [self.arm_type(theta) for theta in thetas]
            for thetas in self.listOfMeans
        ]

        self.currentInterval = 0  # current number of the interval we are in

        print("   - with 'nbArms' =", self.nbArms)  # DEBUG
        print("   - with 'arms' =", self.arms)  # DEBUG
        print(" - Initial draw of 'means' =", self.means)  # DEBUG

    def __repr__(self):
        if len(self.listOfArms) > 0:
            return "{}(nbArms: {}, arms: {})".format(self.__class__.__name__, self.nbArms, self.arms)
        else:
            return "{}(nbArms: {}, armType: {})".format(self.__class__.__name__, self.nbArms, self.arm_type)

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*', latex=True):
        """Cannot represent the dynamic arms, so print the ContextualPieceWiseStationaryMAB object"""
        text = r"{text}, {context}, {arm} with $\Upsilon={M}$ break-points".format(
            text="Non-Stationary Contextual MAB",
            context=str(self.context),
            arm=str(self.arms[0]),
            M=len([tau for tau in self.changePoints if tau > 0]),
            # we do not count 0 and horizon
        )
        return wraptext(text)

    def newRandomArms(self, t=None, onlyOneArm=None, verbose=VERBOSE):
        """Fake function, there is nothing random here, it is just to tell the piece-wise stationary MAB problem to maybe use the next interval.
        """
        if t > 0 and t in self.changePoints:
            if verbose: print(
                "  - BREAKPOINT For a PieceWiseStationaryMAB object, the function newRandomArms was called, with t = {}, and current interval was {}, so means was = {} and will be = {}...".format(
                    t, self.currentInterval, self.listOfMeans[self.currentInterval],
                    self.listOfMeans[self.currentInterval + 1]))  # DEBUG
            self.currentInterval += 1  # next interval!
        else:
            if verbose: print(
                "  - For a PieceWiseStationaryMAB object, the function newRandomArms was called, with t = {}, and current interval is {}, so means is = {}...".format(
                    t, self.currentInterval, self.listOfMeans[self.currentInterval]))  # DEBUG
        # return the latest generate means
        return self.listOfMeans[self.currentInterval]

    # --- Plot utility

    def plotHistoryOfMeans(self, horizon=None, savefig=None, forceTo01=False, showplot=True, pickleit=False):
        """Plot the history of means, as a plot with x axis being the time, y axis the mean rewards, and K curves one for each arm."""
        if horizon is None:
            horizon = max(self.changePoints)
        allMeans = self.get_allMeans(horizon=horizon)
        colors = palette(self.nbArms)
        markers = makemarkers(self.nbArms)
        # Now plot
        fig = plt.figure()
        for armId in range(self.nbArms):
            meanOfThisArm = allMeans[armId, :]
            plt.plot(meanOfThisArm, color=colors[armId], marker=markers[armId], markevery=(armId / 50., 0.1),
                     label='Arm #{}'.format(armId), lw=4, alpha=0.9)
        legend()
        ymin, ymax = plt.ylim()
        if forceTo01:
            ymin, ymax = min(0, ymin), max(1, ymax)
            plt.ylim(ymin, ymax)
        if len(self.changePoints) > 20:
            print(
                "WARNING: Adding vlines for the change points with more than 20 change points will be ugly on the plots...")  # DEBUG
        if len(self.changePoints) < 30:  # add the vlines only if not too many change points
            for tau in self.changePoints:
                if tau > 0 and tau < horizon:
                    plt.vlines(tau, ymin, ymax, linestyles='dotted', alpha=0.7)
        plt.xlabel(r"Time steps $t = 1...T$, horizon $T = {}${}".format(horizon, signature))
        plt.ylabel(r"Successive means of the $K = {}$ arms".format(self.nbArms))
        plt.title("History of means for {}".format(self.reprarms(latex=True)))
        show_and_save(showplot=showplot, savefig=savefig, fig=fig, pickleit=pickleit)
        return fig

    # All these properties arms, means, minArm, maxArm cannot be attributes, as the means of arms change at every experiments

    @property
    def arms(self):
        """Return the *current* list of arms. at time :math:`t` , the return mean of arm :math:`k` is the mean during the time interval containing :math:`t`."""
        return self.listOfArms[self.currentInterval]

    @property
    def means(self):
        """ Return the list of means of arms for this PieceWiseStationaryMAB: at time :math:`t` , the return mean of arm :math:`k` is the mean during the time interval containing :math:`t`.
        """
        return self.listOfMeans[self.currentInterval]

    #
    # --- Helper to compute values minArm and maxArm

    @property
    def minArm(self):
        """Return the smallest mean of the arms, for the current vector of means."""
        return np.min(self.means)

    @property
    def maxArm(self):
        """Return the largest mean of the arms, for the current vector of means."""
        return np.max(self.means)

    #
    # --- Helper to compute vector of min arms, max arms, all arms

    def get_minArm(self, horizon=None):
        """Return the smallest mean of the arms, for a piece-wise stationary MAB

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self.changePoints)
        mapOfMinArms = [np.min(means) for means in self.listOfMeans]
        meansOfMinArms = np.zeros(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMinArms[t] = mapOfMinArms[nbChangePoint]
        return meansOfMinArms

    def get_minArms(self, M=1, horizon=None):
        """Return the vector of sum of the M-worst means of the arms, for a piece-wise stationary MAB.

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self.changePoints)

        def Mworst(unsorted_list):
            sorted_list = np.sort(unsorted_list)
            return np.sum(sorted_list[:-M])

        mapOfMworstMaxArms = [Mworst(means) for means in self.listOfMeans]
        meansOfMworstMaxArms = np.ones(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMworstMaxArms[t] = mapOfMworstMaxArms[nbChangePoint]
        return meansOfMworstMaxArms

    def get_maxArm(self, horizon=None):
        """Return the vector of max mean of the arms, for a piece-wise stationary MAB.

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self.changePoints)
        mapOfMaxArms = [np.max(means) for means in self.listOfMeans]
        meansOfMaxArms = np.ones(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMaxArms[t] = mapOfMaxArms[nbChangePoint]
        return meansOfMaxArms

    def get_maxArms(self, M=1, horizon=None):
        """Return the vector of sum of the M-best means of the arms, for a piece-wise stationary MAB.

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self.changePoints)

        def Mbest(unsorted_list):
            sorted_list = np.sort(unsorted_list)
            return np.sum(sorted_list[-M:])

        mapOfMBestMaxArms = [Mbest(means) for means in self.listOfMeans]
        meansOfMBestMaxArms = np.ones(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMBestMaxArms[t] = mapOfMBestMaxArms[nbChangePoint]
        return meansOfMBestMaxArms

    def get_allMeans(self, horizon=None):
        """Return the vector of mean of the arms, for a piece-wise stationary MAB.

        - It is a numpy array of shape (nbArms, horizon).
        """
        if horizon is None:
            horizon = np.max(self.changePoints)
        meansOfArms = np.ones((self.nbArms, horizon))
        for armId in range(self.nbArms):
            nbChangePoint = 0
            for t in range(horizon):
                if nbChangePoint < len(self.changePoints) - 1 and t >= self.changePoints[nbChangePoint + 1]:
                    nbChangePoint += 1
                meansOfArms[armId][t] = self.listOfMeans[nbChangePoint][armId]
        return meansOfArms

    #
    # --- Compute lower bounds
    # TODO include knowledge of piece-wise stationarity in the lower-bounds

    # def lowerbound(self):
    #     """ Compute the constant C(mu), for [Lai & Robbins] lower-bound for this MAB problem (complexity), using functions from :mod:`kullback` (averaged on all the draws of new means)."""
    #     raise NotImplementedError

    # def hoifactor(self):
    #     """ Compute the HOI factor H_OI(mu), the Optimal Arm Identification (OI) factor, for this MAB problem (complexity). Cf. (3.3) in Navikkumar MODI's thesis, "Machine Learning and Statistical Decision Making for Green Radio" (2017) (averaged on all the draws of new means)."""
    #     raise NotImplementedError

    # def lowerbound_multiplayers(self, nbPlayers=1):
    #     """ Compute our multi-players lower bound for this MAB problem (complexity), using functions from :mod:`kullback`. """
    #     raise NotImplementedError


class ContextualNonStationaryMAB(ContextualPieceWiseStationaryMAB):
    r"""Like a stationary contextual MAB problem, but the arms *can* be modified *at each time step*, with the :meth:`newRandomArms` method.
    TODO: Change doctext to reflect contextual bandits
    - ``M.arms`` and ``M.means`` is changed after each call to :meth:`newRandomArms`, but not ``nbArm``. All the other methods are carefully written to still make sense (``Mbest``, ``Mworst``, ``minArm``, ``maxArm``).

    .. note:: This is a generic class to implement different kinds of non-stationary bandits:

        - Abruptly changing non-stationary bandits, in different variants: changepoints are randomly drawn (once for all ``n`` repetitions or at different location fo each repetition).
        - Slowly varying non-stationary bandits, where the underlying mean of each arm is slowing randomly modified and a bound on the speed of change (e.g., Lipschitz constant of :math:`t \mapsto \mu_i(t)`) is known.

    .. warning:: It works fine, but it is still experimental, be careful when using this feature.

    .. warning:: The number of arms is fixed, see https://github.com/SMPyBandits/SMPyBandits/issues/123 if you are curious about bandit problems with a varying number of arms (or sleeping bandits where some arms can be enabled or disabled at each time).
    """

    def __init__(self, configuration, verbose=VERBOSE):
        """New NonStationaryMAB."""
        self.isChangingAtEachRepetition = False  #: The problem is not changing at each repetition.
        self.isDynamic = True  #: The problem is dynamic.
        self.isMarkovian = False  #: The problem is not Markovian.
        self._sparsity = None

        assert isinstance(configuration, dict) \
               and "arm_type" in configuration and "params" in configuration \
               and "newMeans" in configuration["params"] \
               and "changePoints" in configuration["params"] \
               and "args" in configuration["params"], \
            "Error: this NonStationaryMAB is not really a non-stationary MAB, you should use a simple MAB instead!"  # DEBUG
        self._verbose = verbose

        print(
            "  NonStationary MAB problem, with arm (possibly) changing at every time step, read from a dictionnary 'configuration' = {} ...".format(
                configuration))  # DEBUG

        self.arm_type = arm_type = configuration["arm_type"]  #: Kind of arm (NonStationaryMAB are homogeneous)
        print(" - with 'arm_type' =", arm_type)  # DEBUG
        params = configuration["params"]
        print(" - with 'params' =", params)  # DEBUG
        self.newMeans = params["newMeans"]  #: Function to generate the means
        print(" - with 'newMeans' =", self.newMeans)  # DEBUG
        self.changePoints = params["changePoints"]  #: List of the change points
        print(" - with 'changePoints' =", self.changePoints)  # DEBUG
        self.onlyOneArm = params.get("onlyOneArm",
                                     None)  #: None by default, but can be "uniform" to only change *one* arm at each change point.
        print(" - with 'onlyOneArm' =", self.onlyOneArm)  # DEBUG
        self.args = params["args"]  #: Args to give to function
        print(" - with 'args' =", self.args)  # DEBUG
        # XXX try to read sparsity
        self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None
        print("\n\n ==> Creating the dynamic arms ...")  # DEBUG
        # Keep track of the successive mean vectors
        self._historyOfMeans = dict()  # Historic of the means vectors, storing time of {changepoint: newMeans}
        self._historyOfChangePoints = []  # Historic of the change points
        self._t = 0  # nb of calls to the function for generating new arms
        # Generate a first mean vector
        self.newRandomArms(0)
        print("   - drawing a random set of arms")
        self.nbArms = len(self.arms)  #: Means of arms
        print("   - with 'nbArms' =", self.nbArms)  # DEBUG
        print("   - with 'arms' =", self.arms)  # DEBUG
        print(" - Example of initial draw of 'means' =", self.means)  # DEBUG

    def reprarms(self, nbPlayers=None, openTag='', endTag='^*', latex=True):
        """Cannot represent the dynamic arms, so print the NonStationaryMAB object"""
        # print("reprarms of a NonStationaryMAB object...")  # DEBUG
        # print("  It has self._historyOfMeans =\n{}".format(self._historyOfMeans))  # DEBUG
        # print("  It has self.means =\n{}".format(self.means))  # DEBUG
        text = "{text}, {arm} with uniform means on [{dollar}{lower:.3g}, {upper:.3g}{dollar}]{mingap}{sparsity}".format(
            text="Non-Stationary MAB",
            arm=str(self._arms[0]),
            lower=self.args["lower"],
            upper=self.args["lower"] + self.args["amplitude"],
            mingap="" if self.args["mingap"] is None or self.args["mingap"] == 0 else r", min gap=$%.3g$" % self.args[
                "mingap"],
            sparsity="" if self._sparsity is None else ", sparsity = {dollar}{s}{dollar}".format(s=self._sparsity,
                                                                                                 dollar="$" if latex else ""),
            dollar="$" if latex else "",
        )
        return wraptext(text)

    #
    # --- Dynamic arms and means

    def newRandomArms(self, t=None, onlyOneArm=None, verbose=VERBOSE):
        """Generate a new list of arms, from ``arm_type(params['newMeans'](t, **params['args']))``.

        - If ``onlyOneArm`` is given and is an integer, the change of mean only occurs for this arm and the others stay the same.
        - If ``onlyOneArm="uniform"``, the change of mean only occurs for one arm and the others stay the same, and the changing arm is chosen uniformly at random.

        .. note:: Only the *means* of the arms change (and so, their order), not their family.

        .. warning:: TODO? So far the only change points we consider is when the means of arms change, but the family of distributions stay the same. I could implement a more generic way, for instance to be able to test algorithms that detect change between different families of distribution (e.g., from a Gaussian of variance=1 to a Gaussian of variance=2, with different or not means).
        """
        if ((t > 0 and t not in self.changePoints) or (t in self._historyOfChangePoints)):
            # return the latest generate means
            return self._historyOfMeans[self._historyOfChangePoints[-1]]
        self._historyOfChangePoints.append(t)
        one_draw_of_means = self.newMeans(**self.args)
        self._t += 1  # new draw!
        if onlyOneArm is not None and len(self._historyOfMeans) > 0:
            if onlyOneArm == "uniform":  # - Handling the option to change only one arm
                onlyOneArm = np.random.randint(self.nbArms)
            elif isinstance(onlyOneArm, int):  # - Or a set of arms
                onlyOneArm = np.random.choice(self.nbArms, min(onlyOneArm, self.nbArms), False)
            if np.ndim(onlyOneArm) == 0:
                onlyOneArm = [onlyOneArm]
            elif np.ndim(onlyOneArm) == 1 and np.size(onlyOneArm) == 1:
                onlyOneArm = [onlyOneArm[0]]  # force to extract the list then wrap it back
            # - If only one arm, and not the first random means, change only one
            # print("onlyOneArm =", onlyOneArm)  # DEBUG
            for arm in range(self.nbArms):
                if arm not in onlyOneArm:
                    one_draw_of_means[arm] = self._historyOfMeans[self._historyOfChangePoints[-2]][arm]
        self._historyOfMeans[t] = one_draw_of_means
        self._arms = [self.arm_type(mean) for mean in one_draw_of_means]
        self.nbArms = len(self._arms)  # useless
        if verbose or self._verbose:
            print("\n  - Creating a new dynamic list of means = {} for arms: NonStationaryMAB = {} ...".format(
                np.array(one_draw_of_means), repr(self)))  # DEBUG
            # print("Currently self._t = {} and self._historyOfMeans = {} ...".format(self._t, self._historyOfMeans))  # DEBUG
        return one_draw_of_means

    def get_minArm(self, horizon=None):
        """Return the smallest mean of the arms, for a non-stationary MAB

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self._historyOfChangePoints)
        mapOfMinArms = [np.min(self._historyOfMeans[tau]) for tau in sorted(self._historyOfChangePoints)]
        meansOfMinArms = np.zeros(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self._historyOfChangePoints) - 1 and t >= self._historyOfChangePoints[
                nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMinArms[t] = mapOfMinArms[nbChangePoint]
        return meansOfMinArms

    def get_maxArm(self, horizon=None):
        """Return the vector of max mean of the arms, for a non-stationary MAB.

        - It is a vector of length horizon.
        """
        if horizon is None:
            horizon = np.max(self._historyOfChangePoints)
        mapOfMaxArms = [np.max(self._historyOfMeans[tau]) for tau in sorted(self._historyOfChangePoints)]
        meansOfMaxArms = np.ones(horizon)
        nbChangePoint = 0
        for t in range(horizon):
            if nbChangePoint < len(self._historyOfChangePoints) - 1 and t >= self._historyOfChangePoints[
                nbChangePoint + 1]:
                nbChangePoint += 1
            meansOfMaxArms[t] = mapOfMaxArms[nbChangePoint]
        return meansOfMaxArms

    def get_allMeans(self, horizon=None):
        """Return the vector of mean of the arms, for a non-stationary MAB.

        - It is a numpy array of shape (nbArms, horizon).
        """
        if horizon is None:
            horizon = np.max(self._historyOfChangePoints)
        mapOfArms = [self._historyOfMeans[tau] for tau in sorted(self._historyOfChangePoints)]
        meansOfArms = np.ones((self.nbArms, horizon))
        for armId in range(self.nbArms):
            nbChangePoint = 0
            for t in range(horizon):
                if nbChangePoint < len(self._historyOfChangePoints) - 1 and t >= self._historyOfChangePoints[
                    nbChangePoint + 1]:
                    nbChangePoint += 1
                meansOfArms[armId][t] = mapOfArms[nbChangePoint][armId]
        return meansOfArms
