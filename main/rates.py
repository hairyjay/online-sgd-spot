import numpy as np
import numpy.random as random

class FixedRates(object):
    def __init__(self, t):
        self.t = t

    def get_t(self, size):
        return np.ones(size) * self.t, 1 / self.t

    def get_stats(self):
        return {"distribution": "fixed",
                "t": self.t}

class UniformRates(object):
    def __init__(self, t):
        self.rate = 1 / t
        print(self.rate)

    def get_t(self, size):
        rates = np.random.uniform(1, self.rate * 2, size=size)
        return np.reciprocal(rates), rates

    def get_stats(self):
        return {"distribution": "uniform",
                "min_rate": 1,
                "max_rate": self.rate * 2}

class DirichletRates(object):
    def __init__(self, t):
        self.rate = 1 / t
        print(self.rate)

    def get_t(self, size):
        rates = np.random.dirichlet(np.ones(size) * 3.0)
        rates = rates * size * self.rate
        print(rates, np.mean(rates))
        return np.reciprocal(rates), rates

    def get_stats(self):
        return {"distribution": "dirichlet",
                "alpha": 3.0,
                "mean_rate": self.rate}

class BoundedUniformRates(object):
    def __init__(self, t, std):
        self.rate = 1 / t
        self.std = std

    def get_t(self, size):
        rates = np.random.uniform(np.min(1, self.rate - np.sqrt(3)*self.std), self.rate + np.sqrt(3)*self.std, size=size)
        return np.reciprocal(rates)

    def get_stats(self):
        return {"distribution": "bounded_uniform",
                "min_rate": np.min(1, self.rate - np.sqrt(3)*self.std),
                "max_rate": self.rate + np.sqrt(3)*self.std}

class GaussianRates(object):
    def __init__(self, t, var):
        self.rate = 1 / t
        self.var = var

    def get_t(self, size):
        rates = np.random.normal(self.rate, self.var, size=size)
        rates[rates < 1] = 1
        return np.reciprocal(rates)

    def get_stats(self):
        return {"distribution": "gaussian",
                "mean_rate": self.rate,
                "std_rate": self.var}
