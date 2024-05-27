import math

import numpy as np

from SMPyBandits.ContextualBandits.ContextualArms.ContextualGaussianNoiseArm import ContextualGaussianNoiseArm
from SMPyBandits.ContextualBandits.Contexts.NormalContext import NormalContext


class EnvironmentConfigurations(object):

    def generateContextualGaussianNoiseNormalContextEnvironment(self, arm_count, noise_variance, theta_star, context_means, context_variances, dimension):
        cfg = dict()

        cfg['theta_star'] = theta_star
        cfg['arms'] = list()
        cfg['contexts'] = list()

        for i in range(arm_count):
            cfg['arms'].append(ContextualGaussianNoiseArm(0, noise_variance))
            cfg['contexts'].append(NormalContext([context_means[i] for _ in range(dimension)], np.identity(dimension) * context_variances[i], dimension))

        return cfg

    def generatePerturbedContextualGaussianNoiseNormalContextEnvironment(self, arm_count, noise_variance, thetas, change_points, change_durations,
                                                                         context_means, context_variances, dimension):
        cfg = dict()

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

    def generateSlowChangingContextualGaussianNoiseNormalContextEnvironment(self, arm_count, noise_variance, thetas, context_means, context_variances,
                                                                            dimension):
        cfg = dict()

        cfg['thetas'] = thetas
        cfg['slow_changing'] = True
        cfg['arms'] = list()
        cfg['contexts'] = list()

        for i in range(arm_count):
            cfg['arms'].append(ContextualGaussianNoiseArm(0, noise_variance))
            cfg['contexts'].append(NormalContext([context_means[i] for _ in range(dimension)], np.identity(dimension) * context_variances[i], dimension))

        return cfg

    def getEnv1(self, horizon=1000):
        dim1 = 3
        armCount1 = 3
        noiseVariance1 = 0.1
        baseTheta1 = np.full(dim1, 0.4)
        baseMeans1 = baseTheta1
        baseVariances1 = np.full(armCount1, 0.5)
        env1 = [
            # gnrateContextualGaussianNoiseNormalContextEnvironment(armCount1, noiseVariance1, theta*,                                    , context means                             , context variances             ,              dim1),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount1, noiseVariance1, baseTheta1, baseMeans1, baseVariances1, dim1),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount1, noiseVariance1, baseTheta1, np.full(armCount1, 0.15), baseVariances1, dim1),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount1, noiseVariance1, baseTheta1, np.linspace(0.1, 0.3, armCount1), baseVariances1, dim1),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount1, noiseVariance1, baseTheta1, baseMeans1, np.full(armCount1, 0.15), dim1),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount1, noiseVariance1, baseTheta1, baseMeans1, np.linspace(0.2, 0.6, armCount1), dim1),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount1, noiseVariance1, np.full(dim1, 0.15), baseMeans1, baseVariances1, dim1),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount1, noiseVariance1, np.linspace(0.1, 0.3, dim1), baseMeans1, baseVariances1, dim1),
        ]

        return env1

    def getEnv2(self, horizon=1000):
        dim2 = 20
        armCount2 = 5
        noiseVariance2 = 0.1
        env2 = [
            # gnrateContextualGaussianNoiseNormalContextEnvironment(armCount2, noiseVariance2, theta*,                                         context means,                                  context variances,                             dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount2, noiseVariance2, np.full(dim2, 0.4), np.full(armCount2, 0.4),
                                                                    np.full(armCount2, 0.5), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount2, noiseVariance2, np.full(dim2, 0.4), np.full(armCount2, 0.15),
                                                                    np.full(armCount2, 0.5), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount2, noiseVariance2, np.full(dim2, 0.4), np.linspace(0.05, 0.25, armCount2),
                                                                    np.full(armCount2, 0.5), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount2, noiseVariance2, np.full(dim2, 0.4), np.full(armCount2, 0.4),
                                                                    np.full(armCount2, 0.15), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount2, noiseVariance2, np.full(dim2, 0.4), np.full(armCount2, 0.4),
                                                                    np.linspace(0.1, 0.9, armCount2), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount2, noiseVariance2, np.full(dim2, 0.15), np.full(armCount2, 0.4),
                                                                    np.full(armCount2, 0.5), dim2),
            self.generateContextualGaussianNoiseNormalContextEnvironment(armCount2, noiseVariance2, np.linspace(0.05, 0.25, dim2), np.full(armCount2, 0.4),
                                                                    np.full(armCount2, 0.5), dim2),
        ]

        return env2

    def getEnv3(self, horizon=1000):

        dim3 = 20
        armCount3 = 5
        noiseVariance3 = 0.1
        baseTheta3 = np.full(dim3, 0.4)
        baseMeans3 = baseTheta3
        baseVariances3 = np.full(armCount3, 0.5)
        thetaLow3 = np.full(dim3, 0.1)
        thetas30 = [baseTheta3]
        changePoints30 = []
        changeDurations30 = []

        interval31 = math.floor(horizon / 6)
        duration31 = math.floor(horizon / 100)
        changes31 = 5
        thetas31 = thetas30 + [thetaLow3 for _ in range(changes31)]
        changePoints31 = [max(0, (interval31 * i) - duration31) for i in range(changes31)]
        changeDurations31 = [duration31 for _ in range(changes31)]

        env3 = [
            # gnratePerturbedContextualGaussianNoiseNormalContextEnvironment(armCount3, noiseVariance3, thetas, change_points, change_durations, context means, context variances, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment(armCount3, noiseVariance3, thetas30, changePoints30, changeDurations30, baseMeans3,
                                                                             baseVariances3, dim3),
            self.generatePerturbedContextualGaussianNoiseNormalContextEnvironment(armCount3, noiseVariance3, thetas31, changePoints31, changeDurations31, baseMeans3,
                                                                             baseVariances3, dim3),
        ]

        return env3

    def getEnv4(self, horizon=1000):

        dim4 = 20
        armCount4 = 5
        noiseVariance4 = 0.1
        baseTheta4 = np.full(dim4, 0.4)
        baseMeans4 = baseTheta4
        baseVariances4 = np.full(armCount4, 0.5)
        thetaLow4 = np.full(dim4, 0.1)

        thetas40 = [baseTheta4]
        thetas41 = [baseTheta4, thetaLow4]

        env4 = [
            # gnrateSlowChangingContextualGaussianNoiseNormalContextEnvironment(armCount4, noiseVariance4, thetas, context means, context variances, dim4),
            self.generateSlowChangingContextualGaussianNoiseNormalContextEnvironment(armCount4, noiseVariance4, thetas40, baseMeans4, baseVariances4, dim4),
            self.generateSlowChangingContextualGaussianNoiseNormalContextEnvironment(armCount4, noiseVariance4, thetas41, baseMeans4, baseVariances4, dim4),
        ]

        return env4
