import random

import numpy as np
import scipy.integrate as integrate
import sobol_seq


class QueryStrategy(object):

    def __init__(self, pool=None, dim=2):
        if pool:
            self.pool = pool
        else:
            self.pool = HyperCubePool(dim, 100)

    def next(self):
        """Select the next point"""
        raise NotImplementedError


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


class RandomStrategy(QueryStrategy):

    def __init__(self, **kwargs):
        super(RandomStrategy, self).__init__(**kwargs)
        self.num_points = self.pool.num_points if self.pool else 0
        self.active_points = [i for i in range(self.num_points)]

    def next(self):
        index = random.sample(self.active_points, 1)
        return self.pool[index[0]]

    def __repr__(self):
        return 'RandomStrategy. Active_points={}.'.format(self.num_points)


class BAMS(QueryStrategy):

    def __init__(self, **kwargs):
        self.replacement = kwargs.get('replacement', False)
        super(BAMS, self).__init__(**kwargs)
        self.num_points = self.pool.num_points if self.pool else 0
        self.active_points = [i for i in range(self.num_points)]

    def next(self):
        # If there's no data, query at random.
        if self.models == 0:
            return random.sample(self.active_points, 1)

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
        # # Compute bald.
        # bald = expected_marginal_entropy - expected_individual_entropy
        #
        # # Return the index where bald is maximized.
        # return x_pool[np.argmax(bald)]
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
    pass


class UncertaintySampling(QueryStrategy):
    pass
