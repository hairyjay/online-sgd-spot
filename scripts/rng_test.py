import numpy as np

rng = np.random.default_rng()

for j in range(64):
    n = 0
    for i in range(100):
        n += rng.gamma(256, 0.008)
    print(n/(100*256))