from SMPyBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.Policies import UCB, Exp3, SparseUCB

SPARSITY = 5
LOWER = 0
AMPLITUDE = 1

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.01}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.01}},
    {"archtype": SparseUCB, "params": {
                "alpha": 1,
                "sparsity": SPARSITY,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }}
]
