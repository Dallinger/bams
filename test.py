"""Test script for BAMS port."""

import numpy as np

# import bams

np.random.seed(5822646)


def f1(x):
    """A test oracle function."""
    return np.sin(x) + 0.2 * np.random.randn()


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

# # for i in range(500):
# #     x = learner.query()
# #     y = f1(x)
# #     learner.train(x, y)

# # y_predict = np.linspace(0, 1, 50)
# # print(learner.predict(y_predict))
# # learner.plot_predictions(y_predict)
# # print(learner.map_model_kernel)

def my_loop(generator):
    for i in generator:
        print(i)


my_loop(range(10))
