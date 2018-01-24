"""Query strategies for a learner."""

import random

import numpy as np
import scipy.integrate as integrate
import sobol_seq


class HyperCubePool(object):

    def __init__(self, dim, num_points):
        self.dim = dim
        self.num_points = num_points
        self._hypercube = sobol_seq.i4_sobol_generate(dim, num_points)

    def __getitem__(self, index):
        return self._hypercube[index]

    def __repr__(self):
        return 'Pool (Sobol). Dim={}, num_points={}'.format(
            self.dim,
            self.num_points)


class QueryStrategy(object):

    def __init__(self, pool=None, dim=2):
        if pool:
            self.pool = pool
        else:
            self.pool = HyperCubePool(dim, 100)

    def next(self, models=None):
        """Select the point with the highest score."""
        scores = np.array([self.score(point, models) for point in self.pool._hypercube])
        return self.pool._hypercube[np.argmax(scores)]

    def score(self, point):
        raise NotImplementedError


class RandomStrategy(QueryStrategy):

    def score(self, point, models):
        return random.random()

    def __repr__(self):
        return 'RandomStrategy. Active_points={}.'.format(self.num_points)


class BAMS(QueryStrategy):

    def __init__(self, **kwargs):
        self.replacement = kwargs.get('replacement', False)
        super(BAMS, self).__init__(**kwargs)

    def score(self, point, models):
        # If there's no data, query at random.
        if self.models == 0:
            return RandomStrategy().next()

        for model in self.models:
            # Compute the individual entropy.
            pass

        # Compute the model posterior.
        # posteriors = self.posteriors

        # x_cand = np.random.rand(100)

        # # # Compute the marginal expected entropy.
        # expected_marginal_entropy = self.mee(
        #     self.data_x,
        #     x_cand,
        #     self.models,
        #     self.posteriors
        # )
        #
        # # Compute the individual expected entropy.
        #
        # # Return bald.
        # bald = expected_marginal_entropy - expected_individual_entropy
        raise NotImplementedError

    def mee(self, x_train, x_cand, models, model_posterior):  # marginal_model_entropy
        # TODO: Figure out where this is most logically placed.
        r = 4
        # Compute predictions and means.
        predictions_means = np.zeros((len(models), len(x_cand)))
        predictions_stds = np.ones((len(models), len(x_cand)))
        for i, model in enumerate(models):
            (mean, var) = model.predict(self.data_y, x_cand, return_var=True)
            predictions_means[i, :] = mean
            predictions_stds[i, :] = np.sqrt(var)

        # TODO: Compute min_vs, max_vs.
        min_vs = np.zeros(len(models))
        max_vs = np.zeros(len(models))

        def entropy(x):
            """From gpr_marginal_expected_entropy.m.

            TODO: Implement entropy function.
            """
            return 0

        mee = np.zeros(len(x_cand))
        for i in range(len(x_cand)):
            mee[i] = integrate.quad(entropy, min_vs[i], max_vs[i])[0]

        return mee


class QBC(QueryStrategy):

    def score(self, point, models):
        raise NotImplementedError


class UncertaintySampling(QueryStrategy):

    def score(self, point, models):
        raise NotImplementedError
