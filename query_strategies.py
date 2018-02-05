"""Query strategies for a learner."""

import numpy as np
import sobol_seq


class HyperCubePool(object):

    def __init__(self, dim, num_points):
        self.dim = dim
        self.num_points = num_points
        self._hypercube = sobol_seq.i4_sobol_generate(dim, num_points)

    def __getitem__(self, index):
        return self._hypercube[index]

    def __len__(self):
        return self.num_points

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
        # scores = np.array([self.score(point, models)
        #                    for point in self.pool._hypercube])
        scores = np.array(self.score(self.pool, models))
        return self.pool._hypercube[np.argmax(scores)]

    def score(self, point):
        raise NotImplementedError


class RandomStrategy(QueryStrategy):

    def score(self, points, models):
        return np.random.rand(len(points))


class BALD(QueryStrategy):
    """Bayesian Active Learning by Disagreement (BALD).

    BALD is a score function based on mutual information.
    Here, we use the formulation presented in [1], which
    is the mutual information applied to model selection
    I(y; M | x, D) = H[y|x,D] - E_M[H[y|x,D,M]],
    where the first term is the model marginalized entropy
    and the second term is the expected entropy of each model."""

    def __init__(self, **kwargs):
        self.replacement = kwargs.get('replacement', False)
        super(BALD, self).__init__(**kwargs)

    def score(self, points, models):
        # If there's no data, query at random.
        if not models.data:
            return RandomStrategy().score(points, models)

        # For each model compute (an approximation to)
        # the model evdience and H[y|x,D,M]
        log_evidences = np.zeros(len(models))
        model_entropies = np.zeros((len(points), len(models)))

        for i, model in enumerate(models):
            log_evidences[i] = model.log_evidence()
            model_entropies[:, i] = model.entropy(points)

        # Compute the model posterior
        model_posterior = np.exp(log_evidences - np.max(log_evidences))

        # Compute the expected model entropy
        expected_model_entropies = np.dot(model_entropies, model_posterior)

        # Compute the model-marginal entropy
        marginal_entropies = models.marginal_entropy(points, model_posterior)

        # BALD is H[y | x, D] - E_model[H[y |x,D,M] ]
        return marginal_entropies - expected_model_entropies


class QBC(QueryStrategy):

    def score(self, point, models):
        raise NotImplementedError


class UncertaintySampling(QueryStrategy):

    def score(self, point, models):
        raise NotImplementedError
