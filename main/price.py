import numpy as np
import numpy.random as random
import asyncio
import time
import os

class Uniform_Pricing(object):
    def __init__(self, min_p, max_p, update_mean, update_var=None):
        self.min_p = min_p
        self.max_p = max_p
        self.update_mean = update_mean
        self.update_var = update_var

    def start_price(self):
        return random.uniform(self.min_p, self.max_p)

    def get_price(self):
        interval = self.update_mean
        if self.update_var:
            interval = random.normal(self.update_mean, self.update_var)
            while interval <= 0:
                interval = random.normal(self.update_mean, self.update_var)

        return random.uniform(self.min_p, self.max_p), interval

    def get_stats(self):
        return {"distribution": "uniform",
                "price_min": self.min_p,
                "price_max": self.max_p,
                "update_time_mean": self.update_mean,
                "update_time_var": self.update_var if self.update_var else 0}

class Gaussian_Pricing(object):
    def __init__(self, p_mean, p_var, update_mean, update_var=None):
        self.p_mean = p_mean
        self.p_var = p_var
        self.update_mean = update_mean
        self.update_var = update_var

    def start_price(self):
        return random.normal(self.p_mean, self.p_var)

    def get_price(self):
        interval = self.update_mean
        if self.update_var:
            interval = random.normal(self.update_mean, self.update_var)
            while interval <= 0:
                interval = random.normal(self.update_mean, self.update_var)

        return random.normal(self.p_mean, self.p_var), interval

    def get_stats(self):
        return {"distribution": "gaussian",
                "price_mean": self.p_mean,
                "price_var": self.p_var,
                "update_time_mean": self.update_mean,
                "update_time_var": self.update_var if self.update_var else 0}

class Trace_Pricing(object):
    def __init__(self, filename, scale=1):
        with open(os.path.join("price-trace", filename), 'rb') as f:
            self.trace = np.load(f)
            self.trace[1, :] /= scale
        self.filename = filename
        self.scale = scale
        self.i = 0

    def start_price(self):
        return self.trace[0, 0]

    def get_price(self):
        self.i += 1
        if self.i >= self.trace.shape[1]:
            self.i = 0
        return self.trace[0, self.i], self.trace[1, self.i]

    def get_stats(self):
        return {"distribution": "trace",
                "filename": self.filename,
                "scale": self.scale}

class Fixed_Pricing(object):
    def __init__(self, p):
        self.p = p

    def start_price(self):
        return self.p

    def get_price(self):
        return self.p, False

    def get_stats(self):
        return {"distribution": "fixed",
                "price": self.p}

class Binomial_Preemption(object):
    def __init__(self, q):
        self.q = q

    def choose(self, items):
        choice = random.binomial(size=len(items), n=1, p=self.q)
        return [i for i, c in zip(items, choice) if c]

    def get_info(self):
        return {"distribution": "binomial",
                "q": self.q}

class Uniform_Preemption(object):
    def __init__(self):
        return

    def choose(self, items):
        n = random.randint(low=1, high=len(items))
        return random.choice(items, n, replace=False)

    def get_info(self):
        return {"distribution": "uniform"}
