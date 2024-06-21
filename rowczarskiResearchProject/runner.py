from SMPyBandits.ContextualBandits.ContextualEnvironments.EvaluatorContextual import EvaluatorContextual
from rowczarskiResearchProject.configuration.Configuration import Configuration
from rowczarskiResearchProject.environment.EnvironmentSparse import environments as environments_sparse
from rowczarskiResearchProject.policy.PoliciesSparse import policies as policies_sparse
from rowczarskiResearchProject.policy.PoliciesSparse import policies_test as policies_sparse_test

from rowczarskiResearchProject.plotting.Plotting import Plotting


# Configure environments
#environments = environments_bernoulli_contextual
environments = environments_sparse

# Configure policies
#policies = policies_default
policies = policies_sparse
policies_test = policies_sparse_test

configuration = Configuration(environments=environments, policies=policies_test).getConfigurations()

evaluator = EvaluatorContextual(configuration) # For environments with context
# evaluator = Evaluator(configuration) # Only when using non-contextual environments

N = len(evaluator.envs)

# Create the folder for the plots
plotter = Plotting(evaluator, configuration=configuration, saveAllFigures=True)


for environmentId, environment in enumerate(evaluator.envs):
    # hash for this specific run
    hashValue = abs(hash((tuple(configuration.keys()), tuple(
        [(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))

    evaluator.startOneEnv(environmentId, environment)
    evaluator.printFinalRanking(environmentId)

    # Display the final regrets and rankings for that env
    # evaluator.printLastRegrets(environmentId)
    # evaluator.printMemoryConsumption(environmentId)
    # evaluator.printNumberOfCPDetections(environmentId)

    # Plotting
    plotter.create_subfolder(N, environment, environmentId, hashValue)
    plotter.plot_all(environmentId)
