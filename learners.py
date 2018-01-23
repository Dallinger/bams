from query_strategies import BAMS
from data import VectorData
from models import GrammarModels


class ActiveLearner(object):
    """An active learner."""

    def __init__(self, data=None, model=None, query_strategy=None):
        self.data = data if data else VectorData()
        self.model = model  # if model else GrammarModels()
        # if query_strategy else BAMS(pool)
        self.query_strategy = query_strategy

    def update(self, x, y):
        self.data.update(x, y)
        if self.model:
            self.model.update(x, y)

    def query(self):
        return self.query_strategy.next()

    def run(self, gen_iterations, oracle):
        # gen_iterations can be range(budget) for example
        # TODO: we can use gen_iteratios as a callback function
        for i in gen_iterations:
            index = self.query()
            x = self.pool[index]
            y = oracle(x)
            self.update(index, x, y)
            # create a generator class that allows user
            # to call control the stopping criterion and
            # to implement a callback function
            # if isinstance(gen_iterations, generator):
            #     gen_iterations.

    def __repr__(self):
        return 'model={}\nqs={}\ndata:\n{}'.format(
            self.model,
            self.query_strategy,
            self.data)
