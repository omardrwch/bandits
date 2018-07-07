import numpy as np


class Bernoulli():
    """
    Bernoulli arm.
    X_t = 1 with probability p 
    X_t = 0 with probability (1-p)
    """
    def __init__(self, p):
        self.p = p

    def sample(self):
        return np.random.binomial(1, self.p)

    def mean(self):
        return self.p


class Uniform():
    """
    Uniform distribution (on [0,1]) arm.
    Prob(X_t in [a,b] )  = (b-a), 0 <= a <= b <= 1
    """
    def __init__(self):
        pass

    def sample(self):
        return np.random.uniform(low=0.0, high=1.0)

    def mean(self):
        return 0.5
