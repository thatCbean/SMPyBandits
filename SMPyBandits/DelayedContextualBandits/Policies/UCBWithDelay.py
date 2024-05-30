
from math import log, sqrt

import numpy as np
from SMPyBandits.DelayedContextualBandits.Policies.BasePolicyWithDelay import BasePolicyWithDelay
from SMPyBandits.DelayedContextualBandits.Policies.IndexPolicyWithDelay import IndexPolicyWithDelay
from SMPyBandits.Policies.UCB import UCB


class UCBWithDelay(UCB, BasePolicyWithDelay):

    def update_estimators(self, arm, reward):
        pass