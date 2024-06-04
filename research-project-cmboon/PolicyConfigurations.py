from SMPyBandits.ContextualBandits.ContextualPolicies.BOB import BOB
from SMPyBandits.ContextualBandits.ContextualPolicies.CW_OFUL import CW_OFUL
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualBandits.ContextualPolicies.SW_UCB import SW_UCB
from SMPyBandits.Policies import UCB, Exp3


class PolicyConfigurations(object):

    def generatePolicySetStochastic(self, dimension, horizon):
        return [
            {"archtype": UCB, "params": {}},

            # {"archtype": Exp3, "params": {"gamma": 0.05}},
            {"archtype": Exp3, "params": {"gamma": 0.1}},
            # {"archtype": Exp3, "params": {"gamma": 0.25}},
            # {"archtype": Exp3, "params": {"gamma": 0.5}},
            # {"archtype": Exp3, "params": {"gamma": 0.75}},
        ]

    def generatePolicySetContextualMany(self, dimension, horizon):
        return [
            {"archtype": UCB, "params": {}},

            {"archtype": Exp3, "params": {"gamma": 0.01}},
            {"archtype": Exp3, "params": {"gamma": 0.1}},
            # {"archtype": Exp3, "params": {"gamma": 0.25}},
            # {"archtype": Exp3, "params": {"gamma": 0.5}},
            # {"archtype": Exp3, "params": {"gamma": 0.75}},

            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 100.0}},
            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 50.0}},
            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 20.0}},
            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 10.0}},
            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 5.0}},
            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 2.0}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 1.0}},
            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.5}},
            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.2}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.1}},
            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.05}},
            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.01}},
            # {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.001}},

            # {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.5, "labda": 0.1}},
            # {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.5, "labda": 0.1}},
            # {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.1, "labda": 0.1}},
            # {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.1, "labda": 0.1}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.5, "labda": 0.5}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.5, "labda": 0.5}},
            # {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.1, "labda": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.5, "labda": 3}},
            # {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.5, "labda": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.01, "beta": 0.5, "labda": 10}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.001, "beta": 0.5, "labda": 3}},
            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.001, "beta": 0.5, "labda": 10}},

            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 50, "R": 0.1, "L": 1, "S": 1, "labda": 3, "delta": 1}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 100, "R": 0.1, "L": 1, "S": 1, "labda": 3, "delta": 1}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 200, "R": 0.1, "L": 1, "S": 1, "labda": 3, "delta": 1}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 500, "R": 0.1, "L": 1, "S": 1, "labda": 3, "delta": 1}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 50, "R": 0.1, "L": 1, "S": 1, "labda": 10, "delta": 1}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 100, "R": 0.1, "L": 1, "S": 1, "labda": 10, "delta": 1}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 200, "R": 0.1, "L": 1, "S": 1, "labda": 10, "delta": 1}},
            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 500, "R": 0.1, "L": 1, "S": 1, "labda": 10, "delta": 1}},

            # {"archtype": BOB, "params": {"dimension": dimension, "horizon": horizon, "R": 0.1, "L": 1, "S": 1, "labda": 3}}
        ]

    def generatePolicySetContextualOneEach(self, dimension, horizon):
        return [
            {"archtype": UCB, "params": {}},

            {"archtype": Exp3, "params": {"gamma": 0.1}},

            {"archtype": LinUCB, "params": {"dimension": dimension, "alpha": 0.1}},

            {"archtype": CW_OFUL, "params": {"dimension": dimension, "alpha": 0.1, "beta": 0.5, "labda": 0.1}},

            {"archtype": SW_UCB, "params": {"dimension": dimension, "window_size": 1000, "R": 0.01, "L": 1, "S": 1, "labda": 1, "delta": 0.2}},

            # {"archtype": BOB, "params": {"dimension": dimension, "horizon": horizon, "R": 0.01, "L": 1, "S": 1, "labda": 1}}
        ]