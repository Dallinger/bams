"""Learners."""

import matplotlib.pyplot as plt
import numpy as np

from data import VectorData
from models import GrammarModels
from query_strategies import BAMS


class ActiveLearner(object):
    """An active learner."""

    def __init__(self, data=VectorData(), models=None, query_strategy=None):
        self.data = data

        if not models:
            models = GrammarModels().models
        self.models = models

        if not query_strategy:
            query_strategy = BAMS
        self.query_strategy = query_strategy

    def update(self, x, y):
        self.data.update(x, y)

    def query(self):
        return self.query_strategy.next()

    def __repr__(self):
        return 'models={}\nqs={}\ndata:\n{}'.format(
            self.models,
            self.query_strategy,
            self.data)

    @property
    def posteriors(self):
        """Compute posterior probabilities of the models.

        TODO: Don't hardcode in the zeroth dimension of the input when
        computing log evidence of the models.
        """
        # Compute the log model evidence.
        print(self.data.y.flatten())
        log_evidences = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            model.compute(self.data.x[:, 0].flatten(), yerr=0.1)
            log_evidences[i] = model.log_likelihood(self.data.y.flatten())

        # Compute the model posteriors.
        model_posterior = np.exp(log_evidences - np.max(log_evidences))
        return model_posterior / np.sum(model_posterior)

    @property
    def map_model(self):
        """The maximum a posteriori model."""
        return self.models[np.argmax(self.posteriors)]

    def predict(self, x, dim=0):
        """TODO: Should we be doing some sort of model averaging?"""
        return self.map_model.predict(
            self.data.y.flatten(),
            x[:, dim],
            return_var=True
        )

    def plot_predictions(self, x, dim=0):
        """Plot the learner's predictions."""
        plt.scatter(self.data.x[:, dim], self.data.y)
        (predictions, uncertainty) = self.predict(x, dim=dim)
        plt.scatter(x[:, dim], predictions)
        plt.show()
