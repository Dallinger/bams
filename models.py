import george
import numpy as np

from data import VectorData


class Model(object):

    def update(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class GPModel(Model):

    """TODO: Optimize hyperparameters.
    http://george.readthedocs.io/en/latest/tutorials/hyper/#hyperparameter-optimization
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

    def log_likelihood(self, y):
        return self.gp.log_likelihood(y)

    def log_evidence(self):
        n = len(data.y)     # number of observations
        k = len(gp.parameter_vector)    # number of parameters
        # Negative of BIC - Bayesian information criterion
        return 2 * self.log_likelihood(self, y) - k * np.log(n)

    def entropy(self, points):
        half_log_2pi = 0.9189385332046727
        (mean, covariance) = self.predict(points)
        return 1 / 2 + half_log_2pi + np.log(covariance) / 2

    def __repr__(self):
        return str(self.gp.kernel)


class SimpleModelBag(object):
    """A simple collection of models: SE, Mater32 and LIN"""

    def __init__(self, data=None, ndim=1):
        k1 = george.kernels.ExpSquaredKernel(metric=1.0, ndim=ndim)
        k2 = george.kernels.Matern32Kernel(metric=1.0, ndim=ndim)
        k3 = george.kernels.LinearKernel(order=1, log_gamma2=2, ndim=ndim)
        self.data = data
        self.kernels = [k1, k2, k3]
        self._models = [GPModel(kernel=k, data=data) for k in self.kernels]

    def update(self):
        for model in self._models:
            model.update()

    def marginal_entropy(self, points, model_posterior):
        r = 4
        # Compute predictions and means.
        predictions_means = np.zeros((len(models), len(points)))
        predictions_stds = np.ones((len(models), len(points)))
        for i, model in enumerate(models):
            (mean, var) = model.predict(points)
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

        entropy = np.zeros(len(pool))
        for i in range(len(pool)):
            entropy[i] = integrate.quad(entropy, min_vs[i], max_vs[i])[0]

        return entropy

    def __getitem__(self, index):
        return self._models[index]

    def __len__(self):
        return len(self._models)

    def __repr__(self):
        return str(self.kernels)


class GrammarModels(object):
    """A collections of models from the grammar of kernels."""

    def __init__(self, base_kernels=["LIN", "PER"], ndim=1, max_depth=2):
        self.max_depth = max_depth
        self.base_kernels = base_kernels
        self.kernels = self.build_kernels(self.base_kernels, ndim)
        self.models = [GPModel(kernel=k) for k in self.kernels]
        self.probabilities = np.ones(len(self.models)) / len(self.models)

    def build_kernels(self, kernel_names, ndim=1):
        # TODO as suggestion: change for namedtuple
        #      fix kernel_lookup K_d where K is the kernel name and
        #      d is the dimension

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
