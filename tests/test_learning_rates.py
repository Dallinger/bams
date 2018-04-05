"""Test script for BAMS port."""

import random

import matplotlib.pyplot as plt
import numpy as np

from bams.learners import ActiveLearner
from bams.models import GrammarModels
from bams.query_strategies import (
    BALD,
    HyperCubePool,
    RandomStrategy,
)


s = 5822646
random.seed(s)
np.random.seed(s)


def f1(x):
    """Define a test oracle function."""
    return np.sin(10 * x[2]) + 0.1 * np.random.randn()


def f2(x):
    """Define another test oracle function."""
    return x[2]


def f3(x):
    """Define a third test oracle function."""
    return max(0, min(0.25 + 0.75 * x[2]**3, 1))


if __name__ == '__main__':

    """Compare learning rates of a random and BALD learners."""

    NUM_ROUNDS = 100
    NDIM = 5
    POOL_SIZE = 500
    BUDGET = 40
    BASE_KERNELS = ["PER", "LIN"]
    DEPTH = 1

    def grammar_oracle_sampler(ndim=1, max_depth=1, base_kernels=None, pool_size=POOL_SIZE):
        """Construct an oracle by sampling from a grammar over GPs."""
        if not base_kernels:
            base_kernels = ["LIN", "PER"]

        models = GrammarModels(
            base_kernels=base_kernels,
            max_depth=max_depth,
            ndim=ndim,
        )

        model_idx = random.randint(0, len(models) - 1)
        # print("Model {}".format(model_idx))
        model = models[model_idx]

        print(model)

        pool = HyperCubePool(ndim, pool_size)
        oracle_ys = model.sample(pool)

        def f(x, noise_level=0.00):
            """An oracle that looks up value from the sampled function."""
            idx = np.where((pool == x).all(axis=1))
            if not idx[0].size:
                raise ValueError
            else:
                return oracle_ys[idx][0] + np.random.normal(scale=noise_level)

        return f, models, model_idx

    def learning_curve(learner_factory):

        learning_curves = np.zeros((NUM_ROUNDS, BUDGET))
        for i in range(NUM_ROUNDS):
            print(i)

            learner = learner_factory()

            f0, models, model_idx = grammar_oracle_sampler(
                base_kernels=BASE_KERNELS,
                ndim=NDIM,
                max_depth=DEPTH,
            )

            print(model_idx)

            j = 0
            while learner.budget > 0:
                print(j)
                x = learner.next_query()
                y = learner.query(f0, x)
                learner.update(x, y)
                learning_curves[i, j] = learner.posteriors[model_idx]
                j += 1

        return learning_curves.mean(axis=0)

    def learner_factory_random():
        pool = HyperCubePool(NDIM, POOL_SIZE)
        qs = RandomStrategy(pool=pool)

        learner = ActiveLearner(
            query_strategy=qs,
            budget=BUDGET,
            base_kernels=BASE_KERNELS,
            max_depth=DEPTH,
            ndim=NDIM,
        )
        return learner

    def learner_factory_bald():
        pool = HyperCubePool(NDIM, POOL_SIZE)
        qs = BALD(pool=pool)

        learner = ActiveLearner(
            query_strategy=qs,
            budget=BUDGET,
            base_kernels=BASE_KERNELS,
            max_depth=DEPTH,
            ndim=NDIM,
        )
        return learner

    curve_random = learning_curve(learner_factory_random)
    curve_bald = learning_curve(learner_factory_bald)
    plot_rnd = plt.scatter(1 + np.arange(BUDGET), curve_random, label="Random")
    plot_bald = plt.scatter(1 + np.arange(BUDGET), curve_bald, label="BALD")
    plt.legend(handles=[plot_rnd, plot_bald], loc='upper right')
    plt.ylim([0, 1])
    plt.xlabel("Sample number")
    plt.ylabel("Posterior probability of generating model")
    plt.show()
