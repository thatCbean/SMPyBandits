
from SMPyBandits.Environment.EvaluatorContextual import EvaluatorContextual
from rowczarskiResearchProject.configuration.Configuration import Configuration

configuration = Configuration().getConfigurations()

print(configuration)

evaluator = EvaluatorContextual(configuration)

evaluator.startAllEnv()

