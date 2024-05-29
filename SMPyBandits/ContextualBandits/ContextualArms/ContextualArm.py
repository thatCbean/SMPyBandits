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

    def draw(self, theta_star, context, t=None):
        """ Draw one random sample."""
        raise NotImplementedError("This method draw(context, t) has to be implemented in the class inheriting from ContextualArm.")

    def draw_nparray(self, theta_star, contexts, shape=(1,)):
        assert isinstance(contexts, np.ndarray)
        assert contexts.shape == shape or np.multiply(contexts.shape) == np.multiply(shape)
        np.array([self.draw(theta_star, context) for context in contexts.flat]).reshape(shape)

    def is_nonzero(self):
        raise NotImplementedError("This method is_nonzero() has to be implemented in the class inheriting from ContextualArm.")
