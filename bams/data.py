"""Classes for handling data."""

import matplotlib.pyplot as plt
import numpy as np


class Data(object):

    def update(self, x, y):
        raise NotImplementedError


class VectorData(Data):

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def update(self, x, y):
        if self.x is None:
            self.x, self.y = np.array([x]), np.array([y])
        else:
            self.x = np.vstack((self.x, x))
            self.y = np.concatenate((self.y, [y]))

    def plot(self):
        plt.scatter(self.x, self.y)
        plt.show()

    def __repr__(self):
        if not self:
            return 'x : [] and y : []'
        else:
            return 'features x:\n{}\n outcomes y:\n{}'.format(self.x, self.y)

    def __bool__(self):
        return self.x is not None
    __nonzero__ = __bool__
