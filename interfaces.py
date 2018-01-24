"""
Interfaces for using BAMS Learner
"""


class QueryStrategy(object):

    def next(self):
        """Select the next point"""
        raise NotImplementedError

    def update(self, x):
        """Update the pool of points"""
        raise NotImplementedError


class Models(object):

    def update(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class Data(object):
    def update(self, x, y):
        raise NotImplemented
