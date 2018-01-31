"""Test script for BAMS port."""

import random

import numpy as np

from learners import ActiveLearner
from query_strategies import HyperCubePool, RandomStrategy
from models import SimpleModelBag
from data import VectorData


s = 5822646
random.seed(s)
np.random.seed(s)


def f1(x):
    """Define a test oracle function."""
    return np.sin(10 * x[0]) + 0.1 * np.random.randn()


def f2(x):
    """Define another test oracle function."""
    return x[0]


def f3(x):
    """Define a third test oracle function."""
    return max(0, min(0.25 + 0.75 * x[0]**3, 1))


class Runner(object):

    def __init__(self, oracle):
        self.oracle = oracle

    def run(self, learner):
        """Query the oracle until the learner's budget runs out."""
        while learner.budget > 0:
            x = learner.next_query()
            y = self.oracle(x)
            learner.update(x, y)
            learner.budget -= 1

        return {
            "posterior": learner.posteriors,
        }


if __name__ == '__main__':

    # Initial fake data
    x = np.sort(np.random.rand(5, 2))
    y = np.sin(x).sum(axis=1)

    ndim = 2
    pool_size = 200
    budget = 20
    data = VectorData(x, y)

    pool = HyperCubePool(ndim, pool_size)
    qs = RandomStrategy(pool=pool)
    models = SimpleModelBag(ndim=ndim, data=data)

    learner = ActiveLearner(data=data,
                            models=models,
                            query_strategy=qs,
                            budget=budget)

    runner = Runner(oracle=f1)

    results = runner.run(learner=learner)
    print(results['posterior'])

    x = np.array([np.linspace(0, 1, 50), np.linspace(0, 1, 50)]).T
    learner.plot_predictions(x)
