import george
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm


class Learner(object):
    """An active learner."""
    def __init__(self, kernels=["LIN", "PER"], depth=2, query_strategy="BALD"):
        self.base_kernels = kernels
        self.depth = depth
        self.query_strategy = query_strategy
        self.data_x = np.array([])
        self.data_y = np.array([])

        # Build model space.
        self.kernels = self.build_kernels(self.base_kernels)
        self.models = [george.GP(k) for k in self.kernels]
        self.probabilities = np.ones(len(self.models)) / len(self.models)

    def build_kernels(self, kernel_names):
        kernel_lookup = {
            "RQ":   (george.kernels.RationalQuadraticKernel,
                     {"metric": 5.0, "log_alpha": 2}),
            "M32":  (george.kernels.Matern32Kernel, {"metric": 5.0}),
            "M52":  (george.kernels.Matern52Kernel, {"metric": 5.0}),
            "E":    (george.kernels.ExpKernel, {"metric": 5.0}),
            "SE":   (george.kernels.ExpSquaredKernel, {"metric": 5.0}),
            "ES2":  (george.kernels.ExpSine2Kernel, {"gamma": 0.1, "log_period": -1}),
            "PER":  (george.kernels.CosineKernel, {"log_period": 0.25}),
            "K":    (george.kernels.ConstantKernel, {"log_constant": 0}),
            "LIN":  (george.kernels.LinearKernel,
                     {"order": 1, "log_gamma2": 1}),
            "DP":   (george.kernels.DotProductKernel, {}),
            "LG":   (george.kernels.LocalGaussianKernel, {"location": 10, "log_width": 3}),
            "POLY": (george.kernels.PolynomialKernel, {"order": 2, "log_sigma2": 2}),
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

        # Add all compositions up to the given depth.
        for _ in range(1, self.depth):
            previous_level_models = models[:]
            for model in previous_level_models:
                for operator in operators:
                    for kernel in kernels:
                        models.append(operator(kernel[0](**kernel[1]), model))

        return models

    def train(self, x, y):
        self.data_x = np.append(self.data_x, x)
        self.data_y = np.append(self.data_y, y)

    def query(self):
        """Recommend a query."""
        query_strategies = {
            "bald": self.query_bald,
            "qbc": self.query_qbc,
            "random": self.query_random,
            "uncertainty_sampling": self.query_uncertainty_sampling,
        }
        return query_strategies[self.query_strategy]()

    def mee(self, x_train, x_cand, models, model_posterior):

        r = 4

        # Compute predictions and means.
        predictions_means = np.zeros((len(models), len(x_cand)))
        predictions_stds = np.ones((len(models), len(x_cand)))
        for i, model in enumerate(models):
            (mean, var) = model.predict(self.data_y, x_cand, return_var=True)
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

        mee = np.zeros(len(x_cand))
        for i in range(len(x_cand)):
            mee[i] = integrate.quad(entropy, min_vs[i], max_vs[i])[0]

        return mee

    @property
    def posteriors(self):
        # Compute the log model evidence.
        log_evidences = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            model.compute(self.data_x, yerr=0.1)
            log_evidences[i] = model.log_likelihood(self.data_y)

        # Compute the model posteriors.
        model_posterior = np.exp(log_evidences - np.max(log_evidences))
        return model_posterior / np.sum(model_posterior)

    def query_bald(self):
        # If there's no data, query at random.
        if len(self.data_x) == 0:
            return self.query_random()

        for model in self.models:
            # Compute the individual entropy.
            pass

        # Compute the model posterior.
        posteriors = self.posteriors

        x_cand = np.random.rand(100)

        # # Compute the marginal expected entropy.
        expected_marginal_entropy = self.mee(
            self.data_x,
            x_cand,
            self.models,
            self.posteriors
        )
        #
        # # Compute the individual expected entropy.
        #
        # # Compute bald.
        # bald = expected_marginal_entropy - expected_individual_entropy
        #
        # # Return the index where bald is maximized.
        # return x_pool[np.argmax(bald)]
        raise NotImplementedError

    def query_qbc(self):
        raise NotImplementedError

    def query_random(self):
        """Query at a randomly selected location."""
        return np.random.rand()

    def query_uncertainty_sampling(self):
        raise NotImplementedError

    def predict(self, x):
        """Should we be doing some sort of model averaging?"""
        return self.map_model.predict(self.data_y, x, return_var=True)

    @property
    def map_model(self):
        return self.models[np.argmax(self.posteriors)]

    @property
    def map_model_kernel(self):
        return self.kernels[np.argmax(self.posteriors)]

    def plot_predictions(self, x):
        plt.scatter(self.data_x, self.data_y)
        (predictions, uncertainty) = self.predict(x)
        plt.scatter(x, predictions)
        plt.show()
