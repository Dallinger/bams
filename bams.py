import numpy as np
import GrammarModels
import Bald
import matplotlib.pyplot as plt


class Learner(object):
    """An active learner."""

    def __init__(self, models=None, query_strategy=None, data=None):
        self.models = models if models else GrammarModels()
        self.query_strategy = query_strategy if query_strategy else Bald()
        self.data_x = np.array([])
        self.data_y = np.array([])

    def update(self, x, y):
        self.data_x = np.append(self.data_x, x)
        self.data_y = np.append(self.data_y, y)
        self.models.update(x, y)
        self.query_strategy.update(x)

    def query(self):
        return self.query_strategy.next()

    def run(self, gen_iterations, oracle):
        # gen_iterations can be range(budget) for example
        # TODO: we can use gen_iteratios as a callback function
        for i in gen_iterations:
            x_next = self.query()
            y_next = oracle(x_next)
            self.update(x_next, y_next)
            # create a generator class that allows user
            # to call control the stopping criterion and
            # to implement a callback function
            # if isinstance(gen_iterations, generator):
            #     gen_iterations.

    def plot_predictions(self, x):
        plt.scatter(self.data_x, self.data_y)
        (predictions, uncertainty) = self.predict(x)
        plt.scatter(x, predictions)
        plt.show()
