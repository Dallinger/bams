"""Test script for BAMS port."""

import random

import numpy as np

from learners import ActiveLearner
from query_strategies import (
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
    return max(0, min(0.25 + 0.75 * x[0]**3, 1))


if __name__ == '__main__':

    """DEMO 1."""

    ndim = 1
    pool_size = 200
    budget = 50

    pool = HyperCubePool(ndim, pool_size)
    qs = RandomStrategy(pool=pool)

    learner = ActiveLearner(
        query_strategy=qs,
        budget=budget,
        base_kernels=["PER", "K", "LIN"],
        max_depth=3,
    )

    print(learner.models)

    # TODO: Don't raise exception when there is no data — use prior.

    learner.learn(oracle=f1)

    # # Alternative API for use in external experiment scripts.
    # for i in range(20):
    #     x = learner.next_query()
    #     y = f2(x)
    #     learner.update(x, y)

    print(learner.posteriors)
    print(learner.map_model)

    # Plot predictions
    x = np.array([np.linspace(0, 1, 50)]).T
    learner.plot_predictions(x)

# TODO: Test that when candidate models are identical, the posteriors match.
