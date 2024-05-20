import numpy as np
from matplotlib import pyplot as plt

from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualGaussianNoiseArm import \
    ContextualGaussianNoiseArm
from SMPyBandits.ContextualBandits.ContextualPolicies.ContextualBasePolicy import ContextualBasePolicy
from SMPyBandits.ContextualBandits.ContextualPolicies.LinUCB import LinUCB
from SMPyBandits.Policies import UCB, Exp3, BasePolicy
from SMPyBandits.Policies.IndexPolicy import IndexPolicy

horizon = 10000
repetitions = 1

# contexts = [
#             NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
#             NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3),
#             NormalContext([0.4, 0.4, 0.4], np.identity(3) * 0.5, 3)
# ]
#
# arms = [
#             ContextualGaussianNoiseArm([0.5, 0.7, 0.5], 0, 0.01),
#             ContextualGaussianNoiseArm([1, 0.0, 0.0], 0, 0.01),
#             ContextualGaussianNoiseArm([0.3, 0.3, 0.8], 0, 0.01)
# ]


arm = ContextualGaussianNoiseArm([0.0, 0.1, 0.1, 0.2, 0.3, 0.1, 0.7, 0.2, 0.5, 0.1, 0.3, 0.25], 0, 0.01)

contexts = [
    NormalContext([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], np.identity(12) * 0.5, 12),
    NormalContext([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], np.identity(12) * 0.5, 12),
    NormalContext([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], np.identity(12) * 0.5, 12),
    NormalContext([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], np.identity(12) * 0.5, 12),
    NormalContext([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], np.identity(12) * 0.5, 12),
    NormalContext([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], np.identity(12) * 0.5, 12),
    NormalContext([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], np.identity(12) * 0.5, 12),
    NormalContext([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], np.identity(12) * 0.5, 12),
    NormalContext([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], np.identity(12) * 0.5, 12)
]


policies = [
    UCB(9),
    Exp3(9, 0.1),
    LinUCB(9, 12, 0.01)
]

context_vectors = list()
rewards_arms = list()
rewards_highest = list()
rewards_highest_cumulative = list()
rewards_picked = list()
rewards_picked_cumulative = list()


def initiate():
    for policy in policies:
        policy.startGame()
        rewards_picked.append(list())
        rewards_picked_cumulative.append(list())


def drawRewards(time=0):
    context_vectors_t = list()
    rewards_arms_t = list()
    for context_id in range(len(contexts)):
        context_vectors_t.append(contexts[context_id].draw_context())
        rewards_arms_t.append(arm.draw(context_vectors_t[context_id]))
    context_vectors.append(context_vectors_t)
    rewards_arms.append(rewards_arms_t)
    rewards_highest.append(np.max(rewards_arms_t))
    if time != 0:
        rewards_highest_cumulative.append(np.max(rewards_arms_t) + rewards_highest_cumulative[time-1])
    else:
        rewards_highest_cumulative.append(np.max(rewards_arms_t))


def chooseRewards(time=0):
    arms_picked_t = list()

    for i, policy in enumerate(policies):

        if isinstance(policy, ContextualBasePolicy):
            arm_id = policy.choice(context_vectors[time])
            arms_picked_t.append(arm_id)

            reward = rewards_arms[time][arm_id]

            policy.getReward(arm_id, reward, context_vectors[time])

        elif isinstance(policy, BasePolicy) or isinstance(policy, IndexPolicy):
            arm_id = policy.choice()
            arms_picked_t.append(arm_id)

            reward = rewards_arms[time][arm_id]

            policy.getReward(arm_id, reward)

        else:
            reward = -1

        rewards_picked[i].append(reward)
        rewards_picked_cumulative[i].append(((rewards_picked_cumulative[i][time - 1] + reward) if time != 0 else reward))


def calculateRegret():
    return np.array(
        [
            np.array(rewards_highest_cumulative) - np.array(rewards_picked_cumulative[policy_id])
            for policy_id in range(len(policies))
        ]
    )


for t in range(horizon):
    initiate()
    drawRewards(t)
    chooseRewards(t)

regrets = calculateRegret()
# timestamps = np.array([np.arange(1, horizon + 1) for i in range(len(policies))])
timestamps = np.arange(0, horizon)
for policy_id, policy in enumerate(policies):
    plt.plot(timestamps, regrets[policy_id], label=str(policy))
plt.legend()
plt.show()




