from SMPyBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.Policies import UCB, Exp3, SparseUCB
from rowczarskiResearchProject.SparsityAgnosticLassoBandit.SparsityAgnosticLassoBandit import \
    SparsityAgnosticLassoBandit

SPARSITY = 5
LOWER = 0
AMPLITUDE = 1

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.01}},
    {"archtype": LinUCB, "params": {"dimension": 3, "alpha": 0.01}},
    {"archtype": SparsityAgnosticLassoBandit, "params": {
                "lambda_zero": 0.5, "dimension": 3,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }}
]
