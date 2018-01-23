class QueryStrategy:

    def __init__(query_name='BALD'):

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
