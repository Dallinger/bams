"""Learners."""

import sys

import matplotlib.pyplot as plt
import numpy as np

from data import VectorData
from models import GrammarModels
from query_strategies import BALD


class ActiveLearner(object):
    """An active learner."""

    def __init__(self, data=None, models=None, query_strategy=None,
                 budget=sys.maxsize, base_kernels=None, max_depth=2):

        if not data:
            self.data = VectorData()
        else:
            self.data = data

        if models and base_kernels:
            raise ValueError("Do not specify both models AND base kernels.")

        if not models:
            ndim = self.data.x.shape[1] if self.data.x else 1
            if base_kernels:
                models = GrammarModels(
                    ndim=ndim,
                    base_kernels=base_kernels,
                    max_depth=max_depth,
                    data=self.data
                )
            else:
                models = GrammarModels(ndim=ndim, data=self.data)

        self.models = models

        if not query_strategy:
            query_strategy = BALD
        self.query_strategy = query_strategy

        self.budget = budget

    def next_query(self):
        """Select a point to query using the learner's query strategy."""
        return self.query_strategy.next(self.models)

    def query(self, oracle, point):
        """Query the oracle at the given point."""
        self.budget -= 1
        return oracle(point)

    def update(self, x, y):
        """Store an observation and update the models."""
        self.data.update(x, y)
        self.models.update()

    def learn(self, oracle):
        """Query the oracle until the budget is depleted."""
        while self.budget > 0:
            x = self.next_query()
            y = self.query(oracle, x)
            self.update(x, y)

    def __repr__(self):
        return 'models={}\nqs={}\ndata:\n{}'.format(
            self.models,
            self.query_strategy,
            self.data
        )

    @property
    def posteriors(self):
        return self.models.posteriors()

    @property
    def map_model(self):
        """The maximum a posteriori model."""
        return self.models[np.argmax(self.posteriors)]

    def predict(self, x):
        """TODO: Should we be doing some sort of model averaging?"""
        return self.map_model.predict(x)

    def plot_predictions(self, x):
        """Plot the learner's predictions."""
        plt.scatter(self.data.x[:, 0], self.data.y)
        (predictions, uncertainty) = self.predict(x)
        plt.scatter(x[:, 0], predictions)
        plt.show()
