import math

import numpy as np

from SMPyBandits.ContextualBandits.Contexts.NullContext import NullContext
from SMPyBandits.ContextualBandits.ContextualArms.ContextualGaussianNoiseArm import ContextualGaussianNoiseArm
from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext
from SMPyBandits.ContextualBandits.ContextualArms.NonContextualSimulatingContextualArm import \
    NonContextualSimulatingContextualArm


class EnvironmentConfigurations(object):

    def generateSimulatedStochasticEnvironment(self, name, arm_count, reward_means, reward_variances, dimension):
        cfg = dict()

        cfg['name'] = name
        cfg['theta_star'] = np.full(dimension, 1)
        cfg['arms'] = list()
        cfg['contexts'] = list()

        for i in range(arm_count):
            cfg['arms'].append(
                NonContextualSimulatingContextualArm(reward_means[i], reward_variances[i], dimension=dimension)
            )
            cfg['contexts'].append(NullContext(dimension=dimension))

        return cfg

    def generateContextualGaussianNoiseNormalContextEnvironment(self, name, arm_count, noise_variance, theta_star, context_means, context_variances, dimension):
        cfg = dict()

        cfg['name'] = name
        cfg['theta_star'] = theta_star
        cfg['arms'] = list()
        cfg['contexts'] = list()

        for i in range(arm_count):
            cfg['arms'].append(ContextualGaussianNoiseArm(0, noise_variance))
            cfg['contexts'].append(NormalContext([context_means[i] for _ in range(dimension)], np.identity(dimension) * context_variances[i], dimension))

        return cfg

    def generatePerturbedContextualGaussianNoiseNormalContextEnvironment(self, name, arm_count, noise_variance, thetas, change_points, change_durations,
                                                                         context_means, context_variances, dimension):
        cfg = dict()

        cfg['name'] = name
        cfg['thetas'] = thetas
        cfg['change_points'] = change_points
        cfg['change_durations'] = change_durations
        cfg['perturbed'] = True
        cfg['arms'] = list()
        cfg['contexts'] = list()

        for i in range(arm_count):
            cfg['arms'].append(ContextualGaussianNoiseArm(0, noise_variance))
            cfg['contexts'].append(NormalContext([context_means[i] for _ in range(dimension)], np.identity(dimension) * context_variances[i], dimension))

        return cfg

    def generateSlowChangingContextualGaussianNoiseNormalContextEnvironment(self, name, arm_count, noise_variance, thetas, context_means, context_variances,
                                                                            dimension):
        cfg = dict()

        cfg['name'] = name
        cfg['thetas'] = thetas
        cfg['slow_changing'] = True
        cfg['arms'] = list()
        cfg['contexts'] = list()

        for i in range(arm_count):
            cfg['arms'].append(ContextualGaussianNoiseArm(0, noise_variance))
            cfg['contexts'].append(NormalContext([context_means[i] for _ in range(dimension)], np.identity(dimension) * context_variances[i], dimension))

        return cfg

    arm_counts = [5]
    noise_variances = [0.1]

    def base_vectors(self, dimension):
        return [
            np.full(dimension, 0.1),
            # np.full(dimension, 0.2),
            np.full(dimension, 0.4),

            np.full(dimension, -0.1),
            # np.full(dimension, -0.2),
            np.full(dimension, -0.4),

            np.linspace(0, 1, dimension),
            np.linspace(1, 0, dimension),

            # np.linspace(0.1, 0.4, dimension),
            # np.linspace(0.4, 0.1, dimension),

            np.linspace(-0.4, 0.4, dimension),
            # np.linspace(-0.4, 0.2, dimension),
            # np.linspace(-0.2, 0.4, dimension),
            # np.linspace(-1.0, 1.0, dimension),

            np.linspace(-1.0, 0.0, dimension),
            # np.linspace(0.0, -1.0, dimension)
        ]

    def direct_vector_subset(self, dimension):
        vectors = self.base_vectors(dimension)
        return [
            vectors[0],
            vectors[1],
            vectors[3],
            vectors[4],
            # vectors[6],
            vectors[7]
        ]

    def non_neg_vector_subset(self, dimension):
        vectors = self.base_vectors(dimension)
        return [
            vectors[0],
            vectors[1],
            vectors[4],
        ]

    def change_schemes(self, count):
        schemes = [np.full(count, vector_id) for vector_id in [0, 1, 3, 4]]

        vector = list()
        j = 0
        for i in range(count):
            vector.append(j + 1)
            j = (j + 1) % 2
        schemes.append(vector)

        # vector = list()
        # j = 0
        # for i in range(count):
        #     vector.append(j + 4)
        #     j = (j + 1) % 2
        # schemes.append(vector)
        #
        # vector = list()
        # j = 0
        # for i in range(count):
        #     vector.append(j + 6)
        #     j = (j + 1) % 2
        # schemes.append(vector)

        # vector = list()
        # j = 0
        # for i in range(count):
        #     vector.append(j + 8)
        #     j = (j + 1) % 2
        # schemes.append(vector)
        #
        # vector = list()
        # j = 0
        # for i in range(count):
        #     vector.append(j + 10)
        #     j = (j + 1) % 4
        # schemes.append(vector)
        #
        # vector = list()
        # j = 0
        # for i in range(count):
        #     vector.append(j + 14)
        #     j = (j + 1) % 2
        # schemes.append(vector)

        return schemes

    def thetas_for_perturbed(self, dimension, changes):
        thetas_vectors = self.direct_vector_subset(dimension)
        change_schemes = self.change_schemes(changes)
        res = list()
        for i in range(len(thetas_vectors)):
            for j in range(len(change_schemes)):
                if i == j:
                    continue

                thetas_vectors_i = [thetas_vectors[i]]
                for k in range(changes):
                    thetas_vectors_i.append(thetas_vectors[change_schemes[j][k]])
                res.append(thetas_vectors_i)
        return res

    def thetas_for_slow_changing(self, dimension):
        thetas_vectors = self.base_vectors(dimension)
        res = list()
        res.append([thetas_vectors[0]])
        res.append([thetas_vectors[1]])
        res.append([thetas_vectors[0], thetas_vectors[1]])
        res.append([thetas_vectors[1], thetas_vectors[3]])
        res.append([thetas_vectors[4], thetas_vectors[5]])
        return res


    def base_change_counts_perturbed(self, horizon):
        return [
            # 0,
            # 1,
            # 2,
            # 4,
            math.floor(horizon / 5),
            math.floor(horizon / 10),
            # math.floor(horizon / 20),
        ]

    def base_interval(self, horizon, change_count):
        if change_count == 0:
            return horizon
        else:
            return math.floor(horizon / (change_count + 2))

    def base_change_points(self, horizon, change_count):
        interval = self.base_interval(horizon, change_count)

        return [(j + 1) * interval for j in range(change_count)]

    def base_change_durations(self, horizon, change_count):
        interval = self.base_interval(horizon, change_count)
        durations = [
            # 0.5,
            # 0.25,
            0.3,
            0.15,
            # 0.1
        ]

        res = list()
        for duration in durations:
            res.append(np.full(change_count, math.floor(interval * duration)))
        return res

    def getEnvStochastic(self, horizon, dimension):
        env = list()
        for arm_id, arm_count in enumerate(self.arm_counts):
            for reward_mean_id, reward_means in enumerate(self.non_neg_vector_subset(arm_count)):
                for reward_variance_id, reward_variance in enumerate(self.non_neg_vector_subset(arm_count)):
                    env.append(self.generateSimulatedStochasticEnvironment("Stochastic environment", arm_count, reward_means, reward_variance, dimension))
        return env

    def getEnvContextual(self, horizon, dimension):
        env = list()
        for arm_id, arm_count in enumerate(self.arm_counts):
            for noise_variance_id, noise_variance in enumerate(self.noise_variances):
                for theta_star_id, theta_star in enumerate(self.direct_vector_subset(dimension)):
                    for context_mean_id, context_means in enumerate(self.non_neg_vector_subset(arm_count)):
                        for context_variance_id, context_variance in enumerate(self.non_neg_vector_subset(arm_count)):
                            env.append(self.generateContextualGaussianNoiseNormalContextEnvironment("Contextual environment", arm_count, noise_variance, theta_star, context_means, context_variance, dimension))
        return env

    def getEnvPerturbed(self, horizon, dimension):
        # name, arm_count, noise_variance, thetas, change_points, change_durations,
        # context_means, context_variances, dimension
        env = list()
        for arm_id, arm_count in enumerate(self.arm_counts):
            for noise_variance_id, noise_variance in enumerate(self.noise_variances):
                for change_id, changes in enumerate(self.base_change_counts_perturbed(horizon)):
                    for thetas_id, thetas in enumerate(self.thetas_for_perturbed(dimension, changes)):
                        for change_durations_id, change_durations in enumerate(self.base_change_durations(horizon, changes)):
                            change_points = self.base_change_points(horizon, changes)
                            for context_mean_id, context_means in enumerate(self.non_neg_vector_subset(arm_count)):
                                for context_variance_id, context_variance in enumerate(self.non_neg_vector_subset(arm_count)):
                                    env.append(self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("Perturbed contextual environment", arm_count, noise_variance, thetas, change_points, change_durations, context_means, context_variance, dimension))
        return env

    def getEnvSlowChanging(self, horizon, dimension):
        env = list()
        for arm_id, arm_count in enumerate(self.arm_counts):
            for noise_variance_id, noise_variance in enumerate(self.noise_variances):
                for thetas_id, thetas in enumerate(self.thetas_for_slow_changing(dimension)):
                    for context_mean_id, context_means in enumerate(self.non_neg_vector_subset(arm_count)):
                        for context_variance_id, context_variance in enumerate(self.non_neg_vector_subset(arm_count)):
                            env.append(self.generateSlowChangingContextualGaussianNoiseNormalContextEnvironment("Slow changing contextual environment", arm_count, noise_variance, thetas, context_means, context_variance, dimension))
        return env

    def getEnvStochasticOld(self, horizon, dimension):
        dim0 = 20
        armCount0 = 5
        noiseVariance0 = 0.1
        baseTheta0 = np.full(dim0, 0.4)
        baseMeans0 = baseTheta0
        baseVariances0 = np.full(armCount0, 0.5)
        env1 = [
            # gnrateContextualGaussianNoiseNormalContextEnvironment(armCount1, noiseVariance1, theta*,                                    , context means                             , context variances             ,              dim1),
            self.generateContextualGaussianNoiseNormalContextEnvironment("", armCount0, noiseVariance0, baseTheta0, baseMeans0, baseVariances0, dim0),
            self.generateContextualGaussianNoiseNormalContextEnvironment("", armCount0, noiseVariance0, baseTheta0, np.linspace(0, 1.0, armCount0), baseVariances0, dim0),
        ]

        return env1

    def getEnvContextualOld(self, horizon, dimension):
        dim2 = 20
        armCount2 = 5
        noiseVariance2 = 0.1
        env2 = [
            # gnrateContextualGaussianNoiseNormalContextEnvironment(armCount2, noiseVariance2, theta*,                                         context means,                                  context variances,                             dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment("2.: ", armCount2, noiseVariance2, np.full(dim2, 0.4), np.full(armCount2, 0.4), np.full(armCount2, 0.5), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment("2.: ", armCount2, noiseVariance2, np.full(dim2, 0.4), np.full(armCount2, 0.15), np.full(armCount2, 0.5), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment("2.: ", armCount2, noiseVariance2, np.full(dim2, 0.4), np.linspace(0.05, 0.25, armCount2), np.full(armCount2, 0.5), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment("2.: ", armCount2, noiseVariance2, np.full(dim2, 0.4), np.full(armCount2, 0.4), np.full(armCount2, 0.15), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment("2.: ", armCount2, noiseVariance2, np.full(dim2, 0.4), np.full(armCount2, 0.4), np.linspace(0.1, 0.9, armCount2), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment("2.: ", armCount2, noiseVariance2, np.full(dim2, 0.15), np.full(armCount2, 0.4), np.full(armCount2, 0.5), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment("2.: ", armCount2, noiseVariance2, np.linspace(0.05, 0.25, dim2), np.full(armCount2, 0.4), np.full(armCount2, 0.5), dim2),
        ]

        return env2

    def getEnvPerturbedOld(self, horizon, dimension):

        dim3 = 20
        armCount3 = 5
        noiseVariance3 = 0.1
        baseTheta3 = np.full(dim3, 0.4)
        baseMeans3 = np.linspace(0.3, 0.7, armCount3)
        baseVariances3 = np.full(armCount3, 0.5)
        thetaLow3 = np.full(dim3, 0.1)
        thetaAscending3 = np.linspace(0.3, 0.7, dim3)
        thetaDescending3 = np.linspace(0.7, 0.3, dim3)
        thetaNegative3 = np.zeros(dim3) - baseTheta3
        thetas30 = [baseTheta3]
        changePoints30 = []
        changeDurations30 = []

        interval31 = math.floor(horizon / 17)
        duration31 = math.floor(horizon / 100)
        changes31 = 15
        thetas31 = thetas30 + [thetaLow3 for _ in range(changes31)]
        changePoints31 = [max(0, (interval31 * i) - duration31) for i in range(1, changes31)]
        changeDurations31 = [duration31 for _ in range(changes31)]

        thetas32 = thetas30 + [thetaAscending3 for _ in range(changes31)]
        thetasAlternating = []
        for i in range(changes31):
            if i % 2 == 0:
                thetasAlternating.append(thetaAscending3)
            else:
                thetasAlternating.append(thetaDescending3)
        thetas33 = thetas30 + thetasAlternating
        thetas34 = thetas30 + [thetaNegative3 for _ in range(changes31)]

        baseMeans32 = np.full(armCount3, 0.5)


        env3 = [
            # gnratePerturbedContextualGaussianNoiseNormalContextEnvironment(armCount3, noiseVariance3, thetas, change_points, change_durations, context means, context variances, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.0: Non-perturbed baseline, different contexts", armCount3, noiseVariance3, thetas30, changePoints30, changeDurations30, baseMeans3, baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.1: Low weight perturbations, different contexts", armCount3, noiseVariance3, thetas31, changePoints31, changeDurations31, baseMeans3, baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.2: Repeated weight perturbations, different contexts", armCount3, noiseVariance3, thetas32, changePoints31, changeDurations31, baseMeans3, baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.3: Alternating weight perturbations, different contexts", armCount3, noiseVariance3, thetas33, changePoints31, changeDurations31, baseMeans3, baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.4: Repeated negative weight perturbations, different contexts", armCount3, noiseVariance3, thetas34, changePoints31, changeDurations31, baseMeans3, baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.5: Non-perturbed baseline, identical contexts", armCount3, noiseVariance3, thetas30, changePoints30, changeDurations30, baseMeans32,baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.6: Low weight perturbations, identical contexts", armCount3, noiseVariance3, thetas31, changePoints31, changeDurations31, baseMeans32,baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.7: Repeated weight perturbations, identical contexts", armCount3, noiseVariance3, thetas32, changePoints31, changeDurations31, baseMeans32,baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.8: Alternating weight perturbations, identical contexts", armCount3, noiseVariance3, thetas33, changePoints31, changeDurations31, baseMeans32, baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment("3.9: Repeated negative weight perturbations, identical contexts", armCount3, noiseVariance3, thetas34, changePoints31, changeDurations31, baseMeans32, baseVariances3, dim3),
        ]

        return env3

    def getEnvSlowChangingOld(self, horizon, dimension):

        dim4 = 20
        armCount4 = 5
        noiseVariance4 = 0.1
        baseTheta4 = np.full(dim4, 0.4)
        baseMeans4 = baseTheta4
        baseVariances4 = np.full(armCount4, 0.5)
        thetaLow4 = np.full(dim4, 0.1)

        thetas40 = [baseTheta4, baseTheta4]
        thetas41 = [baseTheta4, thetaLow4]

        env4 = [
            # gnrateSlowChangingContextualGaussianNoiseNormalContextEnvironment(armCount4, noiseVariance4, thetas, context means, context variances, dim4),
            self.generateSlowChangingContextualGaussianNoiseNormalContextEnvironment("4.0: Non-changing baseline", armCount4, noiseVariance4, thetas40, baseMeans4, baseVariances4, dim4),
            self.generateSlowChangingContextualGaussianNoiseNormalContextEnvironment("4.1: Slowly decreasing weights", armCount4, noiseVariance4, thetas41, baseMeans4, baseVariances4, dim4),
        ]

        return env4
