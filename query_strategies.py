from interfaces import QueryStrategy
import sobol_seq
from random import *


class HyperCubePool():
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

    def __init__(self, pool=None):
        self.pool = pool
        self.num_points = pool.num_points if pool else 0
        self.active_points = [i for i in range(self.num_points)]

    def next(self):
        index = sample(self.active_points, 1)
        return self.pool[index[0]]

    def __repr__(self):
        return 'RandomStrategy. Active_points={}.'.format(self.num_points)


class BAMS(QueryStrategy):

    def __init__(self, pool=None, replacement=False):
        self.num_points = pool.num_points if pool else 0
        self.replacement = replacement
        self.active_points = [i for i in range(self.num_points)]

    def next(self):
        # If there's no data, query at random.
        if self.models == 0:
            return sample(self.active_points, 1)

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


class QBC(QueryStrategy):
    pass


class UncertaintySampling(QueryStrategy):
    pass
