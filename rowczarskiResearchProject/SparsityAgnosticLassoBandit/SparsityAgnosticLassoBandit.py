import numpy as np

from sklearn.linear_model import Lasso

from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy

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
        self.lambda_t = self.lambda_zero
        self.k = nbArms
        self.betaHat = np.zeros(dimension)
        self.armT = None
        self.lassoModel = Lasso(alpha=self.lambda_zero, max_iter=10000)
        self.context_up_to_t = []
        self.observed_rewards = []

    def __str__(self):
        return r"SALasso($\lambda_0: {:.3g}$)".format(self.lambda_zero)

    def choice(self, context):
        estimated_rewards = context.dot(self.betaHat)
        chosen_arm = np.argmax(estimated_rewards)

        return chosen_arm

    def getReward(self, arm, reward, contexts):
        super(SparsityAgnosticLassoBandit, self).getReward(arm, reward, contexts)
        self.update(arm, reward, contexts)

    def update(self, arm, reward, context):
        self.context_up_to_t.append(context[arm])
        self.observed_rewards.append(reward)

        if self.t > 5:
            self.update_lamda_t()
            self.lassoModel.alpha = self.lambda_t
            all_context = np.array(self.context_up_to_t)
            all_reward = np.array(self.observed_rewards)

            self.lassoModel.fit(all_context, all_reward)
            self.betaHat = self.lassoModel.coef_

    def update_lamda_t(self):
        self.lambda_t = self.lambda_zero * np.sqrt((4 * np.log(self.t) + 2 * np.log(self.dimension)) / self.t)
