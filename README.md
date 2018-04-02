# Active model selection

## Installation
```
pip install bams
```

## Usage
```Python
import bams
import oracle

learner = bams.learners.ActiveLearner(
    query_strategy=bams.strategies.BALD,
    budget=10,
    base_kernels=["PER", "LIN", "SE", "LG"],
    max_depth=2,
    ndim=3,
)

while learner.budget > 0:
  x = learner.next_query()
  y = learner.query(oracle, x)
  learner.update(x, y)
```


## References

Duvenaud, D., Lloyd, J. R., Grosse, R., Tenenbaum, J. B., & Ghahramani, Z. (2013). Structure discovery in nonparametric regression through compositional kernel search. arXiv preprint arXiv:1302.4922.

Duvenaud, D. (2014). Automatic model construction with Gaussian processes (Doctoral dissertation, University of Cambridge).

Gardner, J., Malkomes, G., Garnett, R., Weinberger, K. Q., Barbour, D., & Cunningham, J. P. (2015). Bayesian active model selection with an application to automated audiometry. *In Advances in Neural Information Processing Systems* (pp. 2386-2394).

Malkomes, G., Schaff, C., & Garnett, R. (2016). Bayesian optimization for automated model selection. In *Advances in Neural Information Processing Systems* (pp. 2900-2908).
