import numpy as np
from scipy.stats import norm, uniform

R = 0.1052974043089158
phi = 400000
theta_no_int = R * phi

time_delay = 1.5
theta = theta_no_int * time_delay

#bid = norm.ppf(1/time_delay, loc=0.6, scale=0.175)
bid = uniform.ppf(1/time_delay, loc=0.2, scale=0.8)
print(bid)
