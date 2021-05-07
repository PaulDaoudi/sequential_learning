import numpy as np


class Bernoulli:
    """
    Bernoulli arm.
    """
    def __init__(self, p):
        self.p = p

    def sample(self):
        return np.random.binomial(1, self.p)

    def mean(self):
        return self.p