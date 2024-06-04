import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from SMPyBandits.Policies import UCB, Exp3
from EnvironmentConfigurations import EnvironmentConfigurations

# Code based on:
# https://github.com/SMPyBandits/SMPyBandits/blob/master/notebooks/Example_of_a_small_Single-Player_Simulation.ipynb

horizon = 10000
repetitions = 25
dimension = 20
n_jobs = 8
verbosity = 4

envcfg = EnvironmentConfigurations()
# environments = envcfg.getEnvStochastic(horizon)
# environments = envcfg.getEnvContextual(horizon, dimension)
# environments = envcfg.getEnvPerturbed(horizon, dimension)
environments = envcfg.getEnvSlowChanging(horizon, dimension)

print(len(environments))
