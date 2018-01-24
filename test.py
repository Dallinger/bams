"""Test script for BAMS port."""

import random

import numpy as np

from learners import ActiveLearner
from query_strategies import HyperCubePool, RandomStrategy

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


qs = RandomStrategy(pool=HyperCubePool(2, 20))
# qs = RandomStrategy(dim=2)
learner = ActiveLearner(query_strategy=qs)

for i in range(100):
    x = learner.query()
    y = f1(x)
    learner.update(x, y)

print(learner.posteriors)
x = np.array([np.linspace(0, 1, 50), np.linspace(0, 1, 50)]).T
# x = np.array([np.linspace(0, 1, 50)],).T
learner.predict(x)
learner.plot_predictions(x)
