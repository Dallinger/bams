from bams.models import GrammarModels
from george.kernels import LinearKernel
from bams.data import VectorData
import numpy as np
import pytest


class TestGrammarModels(object):

    @pytest.fixture
    def linear_data(self):
        x = np.linspace(0, 1, 50)
        x = x.reshape((50, 1))
        y = np.linspace(0, 1, 50) + 0.01 * np.random.randn(len(x))
        test_data = np.linspace(1, 2, 50)
        test_data = test_data.reshape((50, 1))
        data = VectorData(x, y)
        return data, test_data

    def test_data_update(self, linear_data):
        data, test_data = linear_data

        base_kernels = ["LIN", "PER", "K", "RQ"]
        models = GrammarModels(base_kernels=base_kernels,
                               data=data, ndim=1, max_depth=1)

        old_data_len = len(data)
        old_parameters = [m.gp.get_parameter_vector() for m in models]
        models.update()
        new_parameters = [m.gp.get_parameter_vector() for m in models]

        assert any(not all(i == j)
                   for i, j in zip(old_parameters, new_parameters))

        x_new, y_new = [0.2], 0.2
        data.update(x_new, y_new)
        assert old_data_len + 1 == len(data)
        for m in models:
            assert len(m.data) == len(data)

    def test_create_grammarModels_linear_data(self, linear_data):

        data, test_data = linear_data

        base_kernels = ["LIN", "PER", "K", "RQ"]
        models = GrammarModels(base_kernels=base_kernels,
                               data=data, ndim=1, max_depth=1)
        models.update()

        model_posterior = models.posteriors()
        assert abs(model_posterior.sum() - 1.0) < 1e-10

        best_model = models[model_posterior.argmax()]
        assert isinstance(best_model.kernel, LinearKernel)

        # the last point's label should be the most uncertainty label
        marginal_entropy = models.marginal_entropy(test_data, model_posterior)
        assert marginal_entropy.argmax() == len(marginal_entropy) - 1
