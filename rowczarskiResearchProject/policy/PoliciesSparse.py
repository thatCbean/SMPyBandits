from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.Policies import UCB, Exp3
from rowczarskiResearchProject.SparsityAgnosticLassoBandit.SparsityAgnosticLassoBandit import \
    SparsityAgnosticLassoBandit

SPARSITY = 15
LOWER = 0
AMPLITUDE = 1

policies = [
    # {"archtype": UCB, "params": {}},
    # {"archtype": Exp3, "params": {"gamma": 0.01}},
    {"archtype": LinUCB, "params": {"dimension": 100, "alpha": 0.01}},
    # {"archtype": SparsityAgnosticLassoBandit, "params": {
    #             "lambda_zero": 0.2, "dimension": 100,
    #             "lower": LOWER, "amplitude": AMPLITUDE,
    #         }},
]
