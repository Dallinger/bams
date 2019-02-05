from bams.models import GrammarModels
from bams.data import VectorData

import george
from george import kernels
import numpy as np


def generate_data(kernel, N=50, rng=(-5, 5)):
    var = np.random.randn() * 0.5
    if kernel == 'SE':
        gp = george.GP(0.1 * kernels.ExpSquaredKernel(5 + var))
    elif kernel == 'M32':
        gp = george.GP(0.1 * kernels.Matern32Kernel(5 + var))
    elif kernel == 'PER':
        gp = george.GP(0.1 * kernels.CosineKernel(log_period=0.25 + var / 8))
    elif kernel == 'LIN':
        gp = george.GP(
            0.1 * kernels.LinearKernel(order=1, log_gamma2=1.25 + var))
    else:
        gp = None
    x = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))
    y = gp.sample(x)
    return x, y


kernel_names = ["SE", "M32", "PER", "LIN"]
num_exp = 500
num_kernels = len(kernel_names)
kernel_dict = dict(zip(range(num_kernels), kernel_names))

for i in range(num_kernels):
    success_rate = 0
    for j in range(num_exp):
        x, y = generate_data(kernel=kernel_dict[i])
        data = VectorData(x, y)
        models = GrammarModels(data=data, ndim=1, max_depth=1,
                               base_kernels=kernel_names)
        model_posterior = models.posteriors()
        success_rate += i == model_posterior.argmax()
        # print(model_posterior)
    print("Model {} success_rate {}".format(
        kernel_names[i], success_rate / num_exp))
