import george
import numpy as np


class Model(object):

    def update(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class GrammarModels(object):
    """A collections of models from the grammar of kernels."""

    def __init__(self, base_kernels=["LIN", "PER"], max_depth=2):
        self.max_depth = max_depth
        self.base_kernels = base_kernels
        self.kernels = self.build_kernels(self.base_kernels)
        self.models = [george.GP(k) for k in self.kernels]
        self.probabilities = np.ones(len(self.models)) / len(self.models)

    def build_kernels(self, kernel_names):
        # TODO as suggestion: change for namedtuple
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
            models.append(kernel[0](**kernel[1]))

        # Add all compositions up to the max depth.
        for _ in range(1, self.max_depth):
            previous_level_models = models[:]
            for model in previous_level_models:
                for operator in operators:
                    for kernel in kernels:
                        models.append(operator(kernel[0](**kernel[1]), model))

        return models
