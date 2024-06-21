from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.Policies import UCB, Exp3
from rowczarskiResearchProject.SparsityAgnosticLassoBandit.SparsityAgnosticLassoBandit import \
    SparsityAgnosticLassoBandit

SPARSITY = 15
LOWER = 0
AMPLITUDE = 1
DIMENSION = 5

policies = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.1}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.8}},
    {"archtype": SparsityAgnosticLassoBandit, "params": {
                "lambda_zero": 0.001, "dimension": DIMENSION,
                "lower": LOWER, "amplitude": AMPLITUDE,
            }},
]

policies_test = [
    {"archtype": UCB, "params": {}},
    {"archtype": Exp3, "params": {"gamma": 0.1}},
    {"archtype": Exp3, "params": {"gamma": 0.9}},
    {"archtype": Exp3, "params": {"gamma": 0.4}},
    {"archtype": Exp3, "params": {"gamma": 0.6}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.01}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.1}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.001}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.2}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.4}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.8}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.3}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.9}},
    {"archtype": LinUCB, "params": {"dimension": DIMENSION, "alpha": 0.5}},
    {"archtype": SparsityAgnosticLassoBandit, "params": {
        "lambda_zero": 0.001, "dimension": DIMENSION,
        "lower": LOWER, "amplitude": AMPLITUDE,
    }},
{"archtype": SparsityAgnosticLassoBandit, "params": {
        "lambda_zero": 0.01, "dimension": DIMENSION,
        "lower": LOWER, "amplitude": AMPLITUDE,
    }},
{"archtype": SparsityAgnosticLassoBandit, "params": {
        "lambda_zero": 0.00001, "dimension": DIMENSION,
        "lower": LOWER, "amplitude": AMPLITUDE,
    }},
{"archtype": SparsityAgnosticLassoBandit, "params": {
        "lambda_zero": 0.0001, "dimension": DIMENSION,
        "lower": LOWER, "amplitude": AMPLITUDE,
    }},
]
    
