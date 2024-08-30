import numpy as np
import numpy.random as random
import asyncio
import time
import os

class TracePricing(object):
    def __init__(self, filename, on_demand, scale=1):
        with open(os.path.join("price-trace", filename), 'rb') as f:
            self.trace = np.load(f)
            self.trace[1, :] /= scale
        self.on_demand = on_demand
        self.filename = filename
        self.scale = scale
        self.i = -1

    def start_price(self):
        return self.trace[0, 0]

    def get_price(self):
        self.i += 1
        if self.i >= self.trace.shape[1]:
            self.i = 0
        return self.trace[0, self.i], self.trace[1, self.i]

    def get_on_demand(self):
        return self.on_demand

    def get_stats(self):
        return {"distribution": "trace",
                "filename": self.filename,
                "scale": self.scale,
                "on_demand": self.on_demand}

class FixedPricing(object):
    def __init__(self, p):
        self.p = p

    def start_price(self):
        return self.p

    def get_price(self):
        return self.p, False

    def get_on_demand(self):
        return self.p

    def get_stats(self):
        return {"distribution": "fixed",
                "price": self.p}

class InstanceAllocation(object):
    def __init__(self, args, cycles=500):
        self.J = args.J
        self.b = args.bs
        self.N = args.size
        self.t = args.d
        self.a = args.a
        self.cycles = cycles
        if self.a >= 1.0:
            self.q_turn_off = 0
            self.q_turn_on = 1
        else:
            self.q_turn_off = 1 / (self.a * cycles)
            self.q_turn_on = 1 / ((1 - self.a) * cycles)

    def allocate(self, l, p_spot, p_on_demand, arrived=0, elapsed=0, a=None):
        spot = np.zeros(self.N)
        t = max(self.t - elapsed, 1.0)
        J = max(self.J - arrived, 1)
        if a == None:
            a = self.a

        if a >= 1:
            return np.ones(self.N)

        if p_spot >= p_on_demand or not p_on_demand:
            pass

        elif np.isscalar(l):
            #FIXED RATE
            NS = np.floor((self.N - (J * self.b) / (l * t)) / (1 - a)).astype(int)
            NS = min(NS, self.N)
            spot[:NS] = 1

        elif isinstance(l, np.ndarray):
            #VARIABLE RATE
            l_arg = np.argsort(l)
            threshold = (np.sum(l) - (J * self.b / t)) / (1 - a)
            cost = np.inf
            rate_sum = 0
            spot_indices = []
            for ns, idx in enumerate(l_arg):
                new_cost = self.compute_cost(l, ns, rate_sum, p_spot, p_on_demand, J, a)
                rate_sum += l[idx]
                #if new_cost > cost:
                    #print("local min found")
                #if rate_sum > threshold:
                    #print("threshold {} reached".format(threshold))
                if new_cost > cost or rate_sum >= threshold:
                    break
                else:
                    spot_indices.append(idx)
                    cost = new_cost

            spot = np.zeros(self.N)
            #print(len(spot_indices))
            spot[spot_indices] = 1

        else:
            raise TypeError

        return spot

    def compute_cost(self, l, ns, rate_sum, p_spot, p_on_demand, J, a):
        return J * self.b * (self.N * p_spot + (a * p_spot - p_on_demand) * ns) / (np.sum(l) - (1 - a) * rate_sum)

    def preempt(self, state):
        if self.q_turn_off <= 0:
            return np.zeros(state.shape)
        else:
            probs = state * self.q_turn_off
            probs[probs == 0] = self.q_turn_on
            return random.binomial(1, probs)

    def get_stats(self):
        return {"expected_interations": self.J,
                "deadline": self.t,
                "availability": self.a,
                "cycles": self.cycles,
                "p_preempt": self.q_turn_off,
                "p_restart": self.q_turn_on}
