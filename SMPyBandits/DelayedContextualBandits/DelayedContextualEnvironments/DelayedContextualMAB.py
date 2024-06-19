
from SMPyBandits.ContextualBandits.ContextualEnvironments.ContextualMAB import ContextualMAB


class DelayedContextualMAB(ContextualMAB):

    def __init__(self, configuration):
        super().__init__(configuration)
        self.delays = []

        for delay in configuration["delays"]:
            self.delays.append(delay)

        assert len(self.delays) == len(self.arms), \
            "Error: The number of contexts should be equal to the number of arms"
        
        print(" - with 'contexts' =", self.delays)

    def draw(self, armId, t=1):
        context, reward, reward_with_noise = super().draw(armId, t)
        drawn_delay = self.delays[armId].draw_delay()
        return context, reward, reward_with_noise, drawn_delay
    
    def draw_nparray(self, armId, shape=(1,)):
        contexts, rewards, rewards_with_noise = super().draw_nparray(armId, shape)
        drawn_delays = self.delays.draw_nparray(shape)

        return contexts, rewards, rewards_with_noise, drawn_delays