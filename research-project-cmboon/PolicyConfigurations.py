from SMPyBandits.ContextualBandits.ContextualPolicies.BOB import BOB
from SMPyBandits.ContextualBandits.ContextualPolicies.CW_OFUL import CW_OFUL
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualBandits.ContextualPolicies.SW_UCB import SW_UCB
from SMPyBandits.Policies import UCB, Exp3


class PolicyConfigurations(object):

    def generatePolicySetStochastic(self, dimension, horizon):
        return [
            {"archtype": UCB, "params": {"group": 0}},

            {"archtype": Exp3, "params": {"gamma": 0.01, "group": 1}},
            {"archtype": Exp3, "params": {"gamma": 0.05, "group": 1}},
            {"archtype": Exp3, "params": {"gamma": 0.1, "group": 1}},
            {"archtype": Exp3, "params": {"gamma": 0.25, "group": 1}},
            {"archtype": Exp3, "params": {"gamma": 0.5, "group": 1}},
            {"archtype": Exp3, "params": {"gamma": 0.75, "group": 1}},
        ]

    def generatePolicySetContextualMany(self, dimension, horizon):
        return [
            {"archtype": UCB, "params": {"group": 0}},

            {"archtype": Exp3, "params": {"gamma": 0.01, "group": 1}},
            {"archtype": Exp3, "params": {"gamma": 0.1, "group": 1}},
            {"archtype": Exp3, "params": {"gamma": 0.25, "group": 1}},
            {"archtype": Exp3, "params": {"gamma": 0.5, "group": 1}},
            {"archtype": Exp3, "params": {"gamma": 0.75, "group": 1}},

            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 100.0, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 50.0, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 20.0, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 10.0, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 5.0, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 2.0, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 1.0, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.5, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.2, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.1, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.05, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.01, "group": 2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.001, "group": 2}},

            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.5, "labda": 0.1, "group": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.5, "labda": 0.1, "group": 3}},

            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.1, "labda": 0.1, "group": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.1, "labda": 0.1, "group": 3}},

            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.5, "labda": 0.5, "group": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.5, "labda": 0.5, "group": 3}},

            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.1, "labda": 3, "group": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.5, "labda": 3, "group": 3}},

            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.5, "labda": 3, "group": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.5, "labda": 10, "group": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.001, "beta": 0.5, "labda": 3, "group": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.001, "beta": 0.5, "labda": 10, "group": 3}},

            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 50, "R": 0.1, "L": 1, "S": 1, "labda": 1, "delta": 1, "group": 4}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 200, "R": 0.1, "L": 1, "S": 1, "labda": 1, "delta": 1, "group": 4}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 50, "R": 0.1, "L": 1, "S": 1, "labda": 3, "delta": 1, "group": 4}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 200, "R": 0.1, "L": 1, "S": 1, "labda": 3, "delta": 1, "group": 4}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 50, "R": 0.4, "L": 1, "S": 1, "labda": 3, "delta": 1, "group": 4}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 200, "R": 0.4, "L": 1, "S": 1, "labda": 3, "delta": 1, "group": 4}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 50, "R": 0.1, "L": 1, "S": 1, "labda": 3, "delta": 0.5, "group": 4}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 200, "R": 0.1, "L": 1, "S": 1, "labda": 3, "delta": 0.5, "group": 4}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 50, "R": 0.1, "L": 1, "S": 1, "labda": 0.5, "delta": 0.5, "group": 4}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 200, "R": 0.1, "L": 1, "S": 1, "labda": 0.5, "delta": 0.5, "group": 4}}
        ]

    def generatePolicySetContextualManySpecific(self, dimension, horizon):
        return [
            {"archtype": UCB, "params": {"group": 0}},

            {"archtype": Exp3, "params": {"gamma": 0.1, "group": 1}},

            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.5, "group": 2}},

            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.5, "labda": 0.5, "group": 3}},

            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 500, "R": 0.4, "L": 1, "S": 1, "labda": 3, "delta": 1, "group": 4}},
        ]

    def generatePolicySetContextualOneEach(self, dimension, horizon):
        return [
            {"archtype": UCB, "params": {"group": 0}},

            {"archtype": Exp3, "params": {"gamma": 0.1, "group": 1}},

            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.1, "group": 2}},

            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.5, "labda": 0.1, "group": 3}},

            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 1000, "R": 0.01, "L": 1, "S": 1, "labda": 1, "delta": 0.2, "group": 4}},
        ]