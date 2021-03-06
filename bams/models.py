from __future__ import absolute_import
from six.moves import range

import george
import numpy as np
from scipy import integrate, optimize

HALF_LOG_2PI = 0.9189385
SQRT_2PI = 2.5066282


class Model(object):

    def update(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class GPModel(Model):
    """
    TODO: add error if no data is attached
    """

    def __init__(self, data=None, kernel=None, yerr=0.1):
        self.kernel = kernel
        self.gp = george.GP(kernel)
        self.data = data
        self.yerr = yerr

    def nll(self, params):
        if np.any((params[:] < -10) + (params[:] > 11)):
            return np.inf
        self.gp.set_parameter_vector(params)
        ll = self.gp.log_likelihood(self.data.y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(self, params):
        self.gp.set_parameter_vector(params)
        return -self.gp.grad_log_likelihood(self.data.y, quiet=True)

    def optimize(self):
        # method="L-BFGS-B"
        initial_parameter = self.gp.get_parameter_vector()
        random_parameters = [
            np.random.uniform(-5, 5, len(self.gp)) for _ in range(3)]
        parameters = [initial_parameter] + random_parameters
        results = []

        for p in parameters:
            try:
                result = optimize.minimize(
                    self.nll, p, jac=self.grad_nll,  # method="BFGS"
                )
                chol_hess_inv = np.linalg.cholesky(result.hess_inv)
            except np.linalg.LinAlgError:
                continue

            results.append((result, chol_hess_inv))

        results = sorted(results, key=lambda x: x[0].fun)
        result, chol_hess_inv = results[0]
        self.gp.set_parameter_vector(result.x)
        self.chol_hess_inv = chol_hess_inv

    def update(self, hyperparameter_optimization=True):
        if not self.data:
            raise ValueError('Data is None')
        self.gp.compute(self.data.x, yerr=self.yerr)
        if hyperparameter_optimization:
            self.optimize()

    def predict(self, x):
        return self.gp.predict(self.data.y, x, return_var=True)

    def log_likelihood(self):
        return self.gp.log_likelihood(self.data.y) + self.gp.log_prior()

    def log_evidence(self, bic=False):
        if self.data:
            n = len(self.data.y)     # number of observations
        else:
            n = 1
        k = len(self.gp.parameter_vector)    # number of parameters

        # Negative of Bayesian information criterion (BIC)
        if bic:
            return 2 * self.log_likelihood() - k * np.log(n)

        # Laplace Approximation to the model evidence
        chol_hess_inv = self.chol_hess_inv
        half_log_det_hess_inv = np.sum(np.log(np.diag(chol_hess_inv)))
        return self.log_likelihood() + k * HALF_LOG_2PI + half_log_det_hess_inv

    def entropy(self, points):
        (mean, covariance) = self.predict(points)
        if any(covariance <= 0):
            # try to recover by recomputing the precomputations
            # and adding more noise
            self.gp.compute(self.data.x, yerr=self.yerr + 5)
            (mean, covariance) = self.predict(points)
        if any(covariance <= 0):
            import warnings
            warnings.warn(
                'Predictions are negative for model: {}'.format(self.kernel)
            )
            covariance[covariance <= 0] = 1e2

        return 0.5 + HALF_LOG_2PI + np.log(covariance) * 0.5

    def sample(self, points):
        return self.gp.sample(points)

    def __repr__(self):
        return str(self.gp.kernel)


class GrammarModels(object):
    """A collections of models from the grammar of kernels."""

    def __init__(self, base_kernels=["LIN", "PER", "K"], ndim=1, max_depth=2, data=None):
        self.max_depth = max_depth
        self.base_kernels = base_kernels
        self.kernels = self._build_kernels(self.base_kernels, ndim)
        self.data = data
        self._models = [GPModel(kernel=k, data=self.data)
                        for k in self.kernels]

    @property
    def _kernel_lookup(self):
        return {
            "RQ": (george.kernels.RationalQuadraticKernel,
                   {"metric": 5.0, "log_alpha": 2}),
            "M32": (george.kernels.Matern32Kernel, {"metric": 5.0}),
            "M52": (george.kernels.Matern52Kernel, {"metric": 5.0}),
            "E": (george.kernels.ExpKernel, {"metric": 5.0}),
            "SE": (george.kernels.ExpSquaredKernel, {"metric": 5.0}),
            "ES2": (george.kernels.ExpSine2Kernel,
                    {"gamma": 0.1, "log_period": -1}),
            "PER": (george.kernels.CosineKernel, {"log_period": 0.25}),
            "K": (george.kernels.ConstantKernel, {"log_constant": 0}),
            "LIN": (george.kernels.LinearKernel,
                    {"order": 1, "log_gamma2": 1}),
            "DP": (george.kernels.DotProductKernel, {}),
            "LG": (george.kernels.LocalGaussianKernel,
                   {"location": 0.5, "log_width": -1}),
            "POLY": (george.kernels.PolynomialKernel,
                     {"order": 2, "log_sigma2": 2}),
        }

    def _build_kernels(self, kernel_names, ndim=1):
        # TODO as suggestion: change for namedtuple
        # TODO: Remove duplicates due to commutativity of + and *.

        kernels = [self._kernel_lookup[name] for name in kernel_names]

        operators = [
            george.kernels.Product,
            george.kernels.Sum,
        ]

        models = []

        # Add base kernels.
        for kernel in kernels:
            for dim in range(ndim):
                models.append(kernel[0](ndim=ndim, axes=dim, **kernel[1]))

        # Add all compositions of the base kernels up to the max depth.
        for _ in range(1, self.max_depth):
            previous_level_models = models[:]
            for model in previous_level_models:
                for operator in operators:
                    for kernel in kernels:
                        for dim in range(ndim):
                            models.append(
                                operator(
                                    kernel[0](
                                        ndim=ndim,
                                        axes=dim,
                                        **kernel[1]
                                    ), model))

        return models

    def update(self):
        for model in self._models:
            model.update()

    def posteriors(self):
        """Compute posterior probabilities of the models."""
        # Compute log model evidence.
        log_evidences = np.zeros(len(self._models))
        for i, model in enumerate(self._models):
            model.update()
            log_evidences[i] = model.log_evidence()

        # Compute model posteriors.
        model_posterior = np.exp(log_evidences - np.max(log_evidences))
        model_posterior = model_posterior / np.sum(model_posterior)
        return model_posterior

    def marginal_entropy(self, points, model_posterior):

        # Compute predictions and means for the test points
        means = np.zeros((len(self._models), len(points)))
        stds = np.ones((len(self._models), len(points)))
        for i, model in enumerate(self._models):
            # model.update()
            (mean, var) = model.predict(points)
            means[i, :] = mean
            stds[i, :] = np.sqrt(var)

        # Compute an upper and lower bounds for y
        max_range = 4
        upper_values = means.max(0) + max_range * stds.max(0)
        lower_values = means.min(0) - max_range * stds.max(0)

        # Compute the entropy of a mixture of Gaussians for a single y
        def entropy(y, mu, sigma, model_posterior):
            if any(sigma <= 0):
                sigma[sigma <= 0] = 1e-6
            prob = np.exp(-0.5 * ((y - mu) / sigma) ** 2) / (SQRT_2PI * sigma)
            prob = np.dot(model_posterior, prob)
            eps = np.spacing(1)
            return -prob * np.log(prob + eps)

        # Numerically compute the entropy of y for each test point
        y_entropy = np.zeros(len(points))
        for i in range(len(points)):

            def func(x):
                return entropy(x, means[:, i], stds[:, i], model_posterior)

            y_entropy[i] = integrate.quad(
                func,
                lower_values[i],
                upper_values[i],
                full_output=1,
            )[0]

        return y_entropy

    def __getitem__(self, index):
        return self._models[index]

    def __len__(self):
        return len(self._models)

    def __repr__(self):
        return str(self.kernels)
