import numpy as np
import pytest
from bams.models import GPModel
from bams.data import VectorData
from george.kernels import ExpSquaredKernel, RationalQuadraticKernel
from scipy import integrate


class TestGpModels(object):

    @pytest.fixture
    def training_data(self):
        # Training X
        x = np.array([0.0855, 0.9631, 0.2625, 0.5468, 0.8010, 0.5211, 0.0292,
                      0.2316, 0.9289, 0.4889, 0.7303, 0.6241, 0.4886, 0.6791,
                      0.5785, 0.3955, 0.2373, 0.3674, 0.4588, 0.9880])
        x = x.reshape(10, 2)

        # Training Y
        y = np.array([0.0377, 0.8852, 0.9133, 0.7962, 0.0987,
                      0.2619, 0.3354, 0.6797, 0.1366, 0.7212])
        return VectorData(x, y)

    @pytest.fixture
    def test_data(self):
        # Test X
        xs = np.array([0.6596, 0.4538, 0.5186, 0.4324, 0.9730,
                       0.8253, 0.6490, 0.0835, 0.8003, 0.1332])
        return xs.reshape(5, 2)

    @pytest.fixture
    def models(self, training_data):
        models = []
        kernel = 1.0 * ExpSquaredKernel(1.0, ndim=2)
        se_model = GPModel(kernel=kernel, data=training_data, yerr=1)
        models.append(se_model)

        kernel = 1.0 * \
            RationalQuadraticKernel(log_alpha=0.0, metric=1.0, ndim=2)
        rq_model = GPModel(kernel=kernel, data=training_data, yerr=1)
        models.append(rq_model)

        return models

    def test_create_gp_model(self, training_data):
        kernel = 1.0 * ExpSquaredKernel(1.0, ndim=2)
        GPModel(kernel=kernel, data=training_data, yerr=1)

    def test_loglikelihoods(self, models):
        # values computed using MATLAB implementation
        expected_log_likeloods = [-11.4466, -11.4256]
        tolerance = 0.001

        for model, expected_ll in zip(models, expected_log_likeloods):
            model.update(hyperparameter_optimization=False)
            assert abs(model.log_likelihood() - expected_ll) < tolerance
            model.update()
            assert model.log_likelihood() > expected_ll

    def test_marginal_entropy(self, models, test_data):
        # Compute the marginal model entropy
        num_models = len(models)
        max_range = 4

        # Compute predictions and model posterior
        means = np.zeros((num_models, len(test_data)))
        stds = np.ones((num_models, len(test_data)))
        log_evidences = np.zeros(num_models)
        for i, model in enumerate(models):
            model.update(hyperparameter_optimization=False)
            log_evidences[i] = model.log_evidence(bic=True)
            (mean, var) = model.predict(test_data)
            means[i, :] = mean
            stds[i, :] = np.sqrt(var)

        model_posterior = np.exp(log_evidences - np.max(log_evidences))
        model_posterior = model_posterior / np.sum(model_posterior)

        assert abs(model_posterior.sum() - 1.0) < 1e-10

        upper_values = means.max(0) + max_range * stds.max(0)
        lower_values = means.min(0) - max_range * stds.max(0)

        def entropy(y, mu, sigma, model_posterior):
            sqrt_2pi = 2.5066282746310002
            prob = np.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sqrt_2pi * sigma)
            prob = np.dot(model_posterior, prob)
            eps = np.spacing(1)
            return -prob * np.log(prob + eps)

        y_entropy = np.zeros(len(test_data))
        for i in range(len(test_data)):
            def func(x):
                return entropy(x, means[:, i], stds[:, i], model_posterior)
            y_entropy[i] = integrate.quad(
                func, lower_values[i], upper_values[i])[0]

        # value computed using MATLAB implementation
        expected_total_entropy = 2.9411
        assert abs(y_entropy.sum() - expected_total_entropy) < 0.01
