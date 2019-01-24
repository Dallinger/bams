"""Breaking BALD test script."""

import random

import numpy as np

from bams.learners import ActiveLearner
from bams.query_strategies import (
    BALD,
    HyperCubePool,
)


s = 5822646
random.seed(s)
np.random.seed(s)


def f1(x):
    """Define a test oracle function."""
    return np.sin(10 * x[0]) + 0.1 * np.random.randn()


def f2(x):
    """Define another test oracle function."""
    return x[2]


def f3(x):
    """Define a third test oracle function."""
    return max(0, min(0.25 + 0.75 * x[2]**3, 1))


if __name__ == '__main__':

    ndim = 3
    pool_size = 200
    budget = 20

    pool = HyperCubePool(ndim, pool_size)
    qs = BALD(pool=pool)

    learner = ActiveLearner(
        query_strategy=qs,
        budget=budget,
        base_kernels=["PER", "LIN", "SE", "LG"],
        max_depth=2,
        ndim=ndim,
    )

    learner.learn(oracle=f2)

    #print(learner.posteriors)
    #print(learner.map_model)

    # Plot predictions
    #x = np.random.rand(50, ndim)
    #learner.plot_predictions(x, dim=2)
