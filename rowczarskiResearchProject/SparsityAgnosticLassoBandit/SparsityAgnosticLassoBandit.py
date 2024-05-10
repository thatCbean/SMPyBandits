import numpy as np

from sklearn.linear_model import Lasso
from SMPyBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

"""Reference : Sparsity-Agnostic Lasso Bandit Min-Hwan Oh, Garud Iyengar, Assaf Zeevi Proceedings of the 38th 
International Conference on Machine Learning, PMLR 139:8271-8280, 2021.

"""

__author__ = "Rafal Owczarski"
__version__ = "0.1"

LAMBDA_ZERO = 1.0


class SparsityAgnosticLassoBandit(ContextualBasePolicy):

    def __init__(self, nbArms, dimension, lambda_zero=LAMBDA_ZERO, lower=0., amplitude=1.):
        super(SparsityAgnosticLassoBandit, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.lambda_zero = lambda_zero
        self.dimension = dimension
        self.lambda_t = np.eye(self.dimension) * self.lambda_zero
        self.k = nbArms
        self.betaHat = np.zeros(dimension)
        self.armT = None
        self.lassoModel = Lasso(alpha=self.lambda_zero)

    def __str__(self):
        return r"SparsityAgnosticLassoBandit($\lambda_0: {:.3g}$)".format(self.lambda_zero)

    def choice(self, context):
        all_value = [self.betaHat * np.transpose(context[arm]) for arm in range(0, self.nbArms - 1)]

        chosen_arm = np.argmax(all_value)

        self.update_lamda_t()
        self.lassoModel.fit([context[chosen_arm]], [self.rewards[chosen_arm]])
        self.betaHat = self.lassoModel.coef_

        return chosen_arm

    def update_lamda_t(self):
        self.lambda_t = self.lambda_zero * np.sqrt((4 * np.log(self.t) + 2 * np.log(self.dimension)) / self.t)
