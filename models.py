import george
import numpy as np
from scipy import integrate
from data import VectorData


class Model(object):

    def update(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class GPModel(Model):

    """TODO: Optimize hyperparameters.
    http://george.readthedocs.io/en/latest/tutorials/hyper/#hyperparameter-optimization
    TODO: add error if no data is attached
    """

    def __init__(self, data=None, kernel=None, yerr=0.1):
        self.kernel = kernel
        self.gp = george.GP(kernel)
        self.data = data
        self.yerr = yerr

    def update(self):
        if self.data:
            self.gp.compute(self.data.x, yerr=self.yerr)

    def predict(self, x):
        return self.gp.predict(self.data.y, x, return_var=True)

    def log_likelihood(self):
        return self.gp.log_likelihood(self.data.y)

    def log_evidence(self):
        if self.data:
            n = len(self.data.y)     # number of observations
        else:
            n = 1
        k = len(self.gp.parameter_vector)    # number of parameters
        # Negative of BIC - Bayesian information criterion
        return 2 * self.log_likelihood() - k * np.log(n)

    def entropy(self, points):
        half_log_2pi = 0.9189385332046727
        (mean, covariance) = self.predict(points)
        return 0.5 + half_log_2pi + np.log(covariance) * 0.5

    def __repr__(self):
        return str(self.gp.kernel)


class GrammarModels(object):
    """A collections of models from the grammar of kernels."""

    def __init__(self, base_kernels=["LIN", "PER", "K"], ndim=1, max_depth=2, data=None):
        self.max_depth = max_depth
        self.base_kernels = base_kernels
        self.kernels = self.build_kernels(self.base_kernels, ndim)
        self.data = data
        self._models = [GPModel(kernel=k, data=self.data) for k in self.kernels]

    def build_kernels(self, kernel_names, ndim=1):
        # TODO as suggestion: change for namedtuple
        #      fix kernel_lookup K_d where K is the kernel name and
        #      d is the dimension
        # TODO: Remove duplicates due to commutativity of + and *.

        kernel_lookup = {
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
                   {"location": 10, "log_width": 3}),
            "POLY": (george.kernels.PolynomialKernel,
                     {"order": 2, "log_sigma2": 2}),
        }

        kernels = [kernel_lookup[name] for name in kernel_names]

        operators = [
            george.kernels.Product,
            george.kernels.Sum,
        ]

        models = []
        # Add base kernels.
        for kernel in kernels:
            models.append(kernel[0](**kernel[1], ndim=ndim))

        # Add all compositions up to the max depth.
        for _ in range(1, self.max_depth):
            previous_level_models = models[:]
            for model in previous_level_models:
                for operator in operators:
                    for kernel in kernels:
                        models.append(operator(kernel[0](**kernel[1]), model))

        return models

    def update(self):
        for model in self._models:
            model.update()

    def posteriors(self):
        """Compute posterior probabilities of the models.
        """
        # Compute the log model evidence.
        log_evidences = np.zeros(self.num_models)
        for i, model in enumerate(self._models):
            model.update()
            log_evidences[i] = model.log_evidence()

        # Compute the model posteriors.
        model_posterior = np.exp(log_evidences - np.max(log_evidences))
        model_posterior = model_posterior / np.sum(model_posterior)
        return model_posterior

    def marginal_entropy(self, points, model_posterior):

        # Compute predictions and means for the test points
        num_models = self.num_models
        means = np.zeros((num_models, len(points)))
        stds = np.ones((num_models, len(points)))
        for i, model in enumerate(self._models):
            model.update()
            (mean, var) = model.predict(points)
            means[i, :] = mean
            stds[i, :] = np.sqrt(var)

        # compute an upper and lower bounds for y
        max_range = 4
        upper_values = means.max(0) + max_range * stds.max(0)
        lower_values = means.min(0) - max_range * stds.max(0)

        # compute the entropy of a mixture of Gaussians for a single y
        def entropy(y, mu, sigma, model_posterior):
            sqrt_2pi = 2.5066282746310002
            prob = np.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sqrt_2pi * sigma)
            prob = np.dot(model_posterior, prob)
            eps = np.spacing(1)
            return -prob * np.log(prob + eps)

        # numerically compute the entropy of y for each test point
        y_entropy = np.zeros(len(points))
        for i in range(len(points)):
            def func(x):
                return entropy(x, means[:, i], stds[:, i], model_posterior)
            y_entropy[i] = integrate.quad(func, lower_values[i],
                                          upper_values[i])[0]

        return y_entropy

    def __getitem__(self, index):
        return self._models[index]

    def __len__(self):
        return len(self._models)

    def __repr__(self):
        return str(self.kernels)
