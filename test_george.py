from models import GPModel, SimpleModelBag
from data import VectorData
from george.kernels import ExpSquaredKernel, RationalQuadraticKernel
from scipy import integrate
import numpy as np

# Training X
x = np.array([0.0855, 0.9631, 0.2625, 0.5468, 0.8010, 0.5211, 0.0292,
              0.2316, 0.9289, 0.4889, 0.7303, 0.6241, 0.4886, 0.6791,
              0.5785, 0.3955, 0.2373, 0.3674, 0.4588, 0.9880])
x = x.reshape(10, 2)

# Training Y
y = np.array([0.0377, 0.8852, 0.9133, 0.7962, 0.0987,
              0.2619, 0.3354, 0.6797, 0.1366, 0.7212])

# Test X
xs = np.array([0.6596, 0.4538, 0.5186, 0.4324, 0.9730,
               0.8253, 0.6490, 0.0835, 0.8003, 0.1332])
xs = xs.reshape(5, 2)

# Data
data = VectorData(x, y)

# First kernel
kernel = 1.0 * ExpSquaredKernel(1.0, ndim=2)
model1 = GPModel(kernel=kernel, data=data, yerr=1)
model1.update()
assert abs(model1.log_likelihood() + 11.4466) < 0.001

# Second kernel
kernel = 1.0 * RationalQuadraticKernel(log_alpha=0.0, metric=1.0, ndim=2)
model2 = GPModel(kernel=kernel, data=data, yerr=1)
model2.update()
assert abs(model2.log_likelihood() + 11.4256) < 0.001

# Compute the log model evidence.
log_evidences = np.zeros(2)
log_evidences[0] = model1.log_evidence()
log_evidences[1] = model2.log_evidence()

# Compute the model posteriors.
model_posterior = np.exp(log_evidences - np.max(log_evidences))
model_posterior = model_posterior / np.sum(model_posterior)
print(model_posterior)

points = xs
models = [model1, model2]
num_models = len(models)
max_range = 4

# Compute predictions and means.
means = np.zeros((num_models, len(points)))
stds = np.ones((num_models, len(points)))
for i, model in enumerate(models):
    (mean, var) = model.predict(points)
    means[i, :] = mean
    stds[i, :] = np.sqrt(var)

upper_values = means.max(0) + max_range * stds.max(0)
lower_values = means.min(0) - max_range * stds.max(0)


# def entropy(x, means, stds, model_posterior):

#     return 0


# entropy = np.zeros(len(points))
# for i in range(len(points)):
#     entropy[i] = integrate.quad(entropy, lower_values[i], upper_values[i])[0]


# Test simple model bag

# x = np.array([0.])
# y = np.array([0.0377, 0.8852, 0.9133, 0.7962, 0.0987,
#               0.2619, 0.3354, 0.6797, 0.1366, 0.7212])

data = VectorData(x, y)
models = SimpleModelBag(data=data, ndim=2)
models.update()
model_posterior = models.posteriors()
# models.marginal_entropy(xs, model_posterior)
