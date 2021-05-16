import numpy as np
import numpy.random as random
import asyncio
import time

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
