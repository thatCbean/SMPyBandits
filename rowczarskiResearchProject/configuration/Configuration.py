from rowczarskiResearchProject.environment.EnvironmentBernoulli import environments as environments_bernoulli
from rowczarskiResearchProject.environment.EnvironmentBernoulliContextual import environments as environments_bernoulli_contextual
from rowczarskiResearchProject.policy.PoliciesDefault import policies as policies_default

HORIZON = 1000
REPETITIONS = 10
N_JOBS = 10
VERBOSITY = 6
ENVIRONMENTS = environments_bernoulli_contextual
POLICIES = policies_default


class Configuration:
    def __init__(
            self,
            horizon=HORIZON,
            repetitions=REPETITIONS,
            n_jobs=N_JOBS,
            verbosity=VERBOSITY,
            environments=ENVIRONMENTS,
            policies=POLICIES):
        self.horizon = horizon
        self.repetitions = repetitions
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.environments = environments
        self.policies = policies

    def getConfigurations(self):
        return {
            "horizon": self.horizon,
            "repetitions": self.repetitions,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
            "environment": self.environments,
            "policies": self.policies
        }

    def getHorizon(self):
        return self.horizon

    def getRepetitions(self):
        return self.repetitions

    def getNJobs(self):
        return self.n_jobs

    def getVerbosity(self):
        return self.verbosity

    def getEnvironments(self):
        return self.environments

    def getPolicies(self):
        return self.policies
