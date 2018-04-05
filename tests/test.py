"""Test script for BAMS port."""

import random

import numpy as np

from bams.learners import ActiveLearner
from bams.query_strategies import (
    HyperCubePool,
    RandomStrategy,
)


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
    return max(0, min(0.25 + 0.75 * x[2]**3, 1))


if __name__ == '__main__':

    """DEMO 1."""

    ndim = 3
    pool_size = 200
    budget = 50

    pool = HyperCubePool(ndim, pool_size)
    qs = RandomStrategy(pool=pool)

    learner = ActiveLearner(
        query_strategy=qs,
        budget=budget,
        base_kernels=["PER", "LIN"],
        max_depth=2,
        ndim=ndim,
    )

    print(learner.models)

    # TODO: Don't raise exception when there is no data — use prior.

    learner.learn(oracle=f3)

    # # Alternative API for use in external experiment scripts.
    # for i in range(20):
    #     x = learner.next_query()
    #     y = f2(x)
    #     learner.update(x, y)

    print(learner.posteriors)
    print(learner.map_model)

    # Plot predictions
    # x = np.array([np.linspace(0, 1, 50)]).T
    x = np.random.rand(50, ndim)
    learner.plot_predictions(x, dim=2)

# TODO: Test that when candidate models are identical, the posteriors match.
