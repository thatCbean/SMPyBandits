from SMPyBandits.ContextualBandits.ContextualArms.ContextualArm import ContextualArm


class ContextualConstantArm(ContextualArm):

    """ An arm that always returns the same reward, disregarding the context """

    def set_theta_param(self, theta):
        pass

    def __init__(self, reward):
        assert isinstance(reward, (int, float)), "Error: reward should be a number"
        assert reward >= 0, "Error: reward should be positive"
        super(__class__, self).__init__()
        self.reward = reward

    def __str__(self):
        return "ContextualConstant"

    def __repr__(self):
        return "ContextualConstant(reward: {})".format(self.reward)

    def draw(self, context, t=None):
        return self.reward

    def set(self, theta):
        pass

    def is_nonzero(self):
        return self.reward != 0