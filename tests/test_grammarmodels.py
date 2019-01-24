from bams.models import GrammarModels
from george.kernels import LinearKernel
from bams.data import VectorData
import numpy as np


class TestGrammarModels(object):
    def test_create_grammarModels_linear_data(self):

        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50) + 0.01 * np.random.randn()
        xs = np.linspace(1, 2, 50)
        data = VectorData(x, y)

        base_kernels = ["LIN", "PER", "K", "RQ"]
        models = GrammarModels(base_kernels=base_kernels,
                               data=data, ndim=1, max_depth=1)
        models.update()

        model_posterior = models.posteriors()
        assert abs(model_posterior.sum() - 1.0) < 1e-10

        best_model = models[model_posterior.argmax()]
        assert isinstance(best_model.kernel, LinearKernel)

        # the last point's label should be the most uncertainty label
        marginal_entropy = models.marginal_entropy(xs, model_posterior)
        assert marginal_entropy.argmax() == len(marginal_entropy) - 1
