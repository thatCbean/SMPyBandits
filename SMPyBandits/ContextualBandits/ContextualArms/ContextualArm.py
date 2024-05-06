# -*- coding: utf-8 -*-
""" Base class for an arm class."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Cody Boon"
__version__ = "0.1"

import numpy as np


class ContextualArm(object):
    """ Base class for a contextual arm class."""

    def __init__(self, lower=0., amplitude=1.):
        """ Base class for an arm class."""
        self.lower = lower  #: Lower value of rewards
        self.amplitude = amplitude  #: Amplitude of value of rewards
        self.min = lower  #: Lower value of rewards
        self.max = lower + amplitude  #: Higher value of rewards
        self.mean = lower + (amplitude / 2)  #: Default mean, child classes should use a better calculation

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        if hasattr(self, 'lower') and hasattr(self, 'amplitude'):
            return self.lower, self.amplitude
        elif hasattr(self, 'min') and hasattr(self, 'max'):
            return self.min, self.max - self.min
        else:
            raise NotImplementedError("This method lower_amplitude() has to be implemented in the class inheriting from Arm.")

    # --- Printing

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dir__)

    # --- Random samples

    def draw(self, context, t=None):
        """ Draw one random sample."""
        raise NotImplementedError("This method draw(context, t) has to be implemented in the class inheriting from ContextualArm.")

    def set_theta_param(self, theta):
        raise NotImplementedError("This method set_theta_param(theta) has to be implemented in the class inheriting from ContextualArm.")

    def draw_nparray(self, contexts, shape=(1,)):
        assert isinstance(contexts, np.ndarray)
        assert contexts.shape == shape or np.multiply(contexts.shape) == np.multiply(shape)
        np.array([self.draw(context) for context in contexts.flat]).reshape(shape)

    def calculate_mean(self, context):
        raise NotImplementedError("This method calculate_mean(context) has to be implemented in the class inheriting from ContextualArm.")

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        raise NotImplementedError("This method kl(x, y) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - mu) / KL(mu, mumax). """
        raise NotImplementedError("This method oneLR(mumax, mu) has to be implemented in the class inheriting from Arm.")


    @staticmethod
    def oneHOI(mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu)
