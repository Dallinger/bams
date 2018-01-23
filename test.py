"""Test script for BAMS port."""

import numpy as np
from active_learner import *
from query_strategies import HyperCubePool, RandomStrategy

# import bams
s = 5822646
random.seed(s)
np.random.seed(s)


def f1(x):
    """A test oracle function."""
    return np.sum(np.sin(x) + 0.2 * np.random.randn())


def f2(x):
    """Another test oracle function."""
    return x


def f3(x):
    """A third test oracle function."""
    return max(0, min(0.25 + 0.75 * x**3, 1))


# learner = bams.Learner(
#     kernels=[
#         "LIN",
#         "PER",
#         "SE",
#         "RQ",
#         "M32",
#         "M52",
#         "E",
#         "ES2",
#         "K",
#         "DP",
#         "LG",
#         "POLY",
#     ],
#     depth=2,
#     query_strategy="random",
# )

pool = HyperCubePool(2, 10)
qs = RandomStrategy(pool)
learner = ActiveLearner(query_strategy=qs)

for i in range(10):
    x = learner.query()
    y = f1(x)
    learner.update(x, y)

# y_predict = np.linspace(0, 1, 50)
# print(learner.predict(y_predict))
# learner.plot_predictions(y_predict)
# print(learner.map_model_kernel)
