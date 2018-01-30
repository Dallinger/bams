import george
import numpy as np

from data import VectorData


class Model(object):

    def update(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class GPModel(Model):

    """TODO: Optimize hyperparameters.
    http://george.readthedocs.io/en/latest/tutorials/hyper/#hyperparameter-optimization
    """

    def __init__(self, kernel=None):
        self.kernel = kernel
        self.gp = george.GP(kernel)
        self.data = VectorData()

    def add_new_data(self, x, y):
        self.data.update(x, y)
        #self.gp.compute(self.data.x[:, 0].flatten(), yerr=0.1)

    def precompute(self):
        self.gp.compute(self.data.x, yerr=0.1)

    def update(self, x, y):
        self.add_new_data(x, y)
        self.precompute()

    def predict(self, x):
        # return self.gp.predict(
        #     self.data.y.flatten(),
        #     x[:, 0],
        #     return_var=True
        # )
        return self.gp.predict(self.data.y, x, return_var=True)

    def log_likelihood(self, y):
        return self.gp.log_likelihood(y)

    def __repr__(self):
        return str(self.gp.kernel)


class SimpleModelBag(object):
    """A simple collection of models: SE, Mater32 and LIN"""

    def __init__(self, ndim=1):
        k1 = george.kernels.ExpSquaredKernel(metric=1.0, ndim=ndim)
        k2 = george.kernels.Matern32Kernel(metric=1.0, ndim=ndim)
        k3 = george.kernels.LinearKernel(order=1, log_gamma2=2, ndim=ndim)
        self.kernels = [k1, k2, k3]
        self._models = [GPModel(kernel=k) for k in self.kernels]

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
